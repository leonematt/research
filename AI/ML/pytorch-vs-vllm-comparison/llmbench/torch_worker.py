"""One PyTorch config, all its workloads, in its own process (fresh CUDA context + compile
cache, mirroring one `vllm serve` per vLLM config). Launched by torch_runner; not a CLI.

Static batching (`mode=generate`): concurrency C = batch size. Prompts run in batches of C; a
request's TTFT is its batch's first token and its E2E is the batch's end (generate() returns
the whole batch at once). Variable-length prompts are left-padded; with variable output
lengths the batch runs until its longest member is done, and only each request's own tokens
count as output: the wasted slots are exactly what continuous batching removes.

Continuous batching (`mode=cb`): transformers' ContinuousBatchingManager with
max_requests_per_batch=C; latency is measured from when each request was first scheduled,
matching `vllm bench serve --max-concurrency C`.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import sys
import time
import zlib
from pathlib import Path

from .metrics import parse_slo, summarize
from .results import write_row
from .sampling import Policy, cv, good, is_quiet, need_more
from .util import write_json
from .workloads import PromptSet, Workload, make_prompts, regular_vocab

COMPILE_WARN_S = 1.0


class CompileLog:
    """Wall-clock spans of torch.compile work, from Dynamo's compile callbacks. Used to tell
    whether a compiled config was still recompiling inside the measured window."""

    def __init__(self):
        self.spans: list[tuple[float, float]] = []
        self._t0 = None
        try:
            from torch._dynamo.callback import callback_handler
            callback_handler.register_start_callback(self._start)
            callback_handler.register_end_callback(self._end)
            self.ok = True
        except Exception:  # noqa: BLE001  (older torch: no callbacks)
            self.ok = False

    def _start(self, *_):
        self._t0 = time.perf_counter()

    def _end(self, *_):
        if self._t0 is not None:
            self.spans.append((self._t0, time.perf_counter()))
            self._t0 = None

    def within(self, t0: float, t1: float) -> tuple[int, float]:
        hit = [(s, e) for s, e in self.spans if e > t0 and s < t1]
        return len(hit), sum(min(e, t1) - max(s, t0) for s, e in hit)


def has_accelerate() -> bool:
    import importlib.util
    return importlib.util.find_spec("accelerate") is not None


# ----------------------------------------------------------------------------
# static batching
# ----------------------------------------------------------------------------

def run_static(model, draft, pt: dict, w: Workload, prompts: PromptSet, pad_id: int, max_batches: int,
               sync, dev, profile_fn=None) -> dict:
    import torch
    from transformers.generation.streamers import BaseStreamer

    class Clock(BaseStreamer):
        def __init__(self):
            self.n, self.times = 0, []

        def put(self, value):
            sync()
            self.n += 1
            if self.n == 1:            # the prompt itself
                return
            k = value.shape[-1] if value.dim() > 1 else 1
            self.times += [time.perf_counter()] * k    # assisted generation emits several tokens per step

        def end(self):
            pass

    B = w.concurrency
    n = min(len(prompts) - B, B * max_batches)
    gen = dict(do_sample=False, pad_token_id=pad_id)
    if pt.get("cache") == "static":
        gen["cache_implementation"] = "static"
    if pt.get("compile"):
        from transformers import CompileConfig
        gen["compile_config"] = CompileConfig(mode=pt["compile"], fullgraph=True)
        gen["disable_compile"] = False
    else:
        gen["disable_compile"] = True
    if draft is not None:
        gen["assistant_model"] = draft

    def one(idx: list[int], keep: bool = True):
        seqs = [prompts.ids[i] for i in idx]
        outs = [prompts.out_lens[i] for i in idx]
        L = max(len(s) for s in seqs)
        ids = torch.tensor([[pad_id] * (L - len(s)) + s for s in seqs], device=dev)       # left padding
        mask = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
        steps = max(outs)
        clock = Clock()
        sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.generate(input_ids=ids, attention_mask=mask, streamer=clock, max_new_tokens=steps,
                           min_new_tokens=steps, **gen)
        sync()
        t1 = time.perf_counter()
        if not keep:
            return []
        times = clock.times or [t1]
        # every request leaves with the batch (t1); only its own tokens count as output
        return [dict(submit=t0, first=times[0], token_times=times[:o], end=t1, n_in=len(s), n_out=o)
                for s, o in zip(seqs, outs)]

    tw = time.perf_counter()
    one(list(range(B)), keep=False)                        # warmup: compile + graph capture
    warmup_s = time.perf_counter() - tw
    reqs, t_start, batches = [], time.perf_counter(), 0
    for i in range(B, B + n, B):
        reqs += one(list(range(i, min(i + B, B + n))))
        batches += 1
    duration = time.perf_counter() - t_start
    if profile_fn is not None:
        profile_fn(lambda: one(list(range(B, min(2 * B, len(prompts)))), keep=False))
    return dict(reqs=reqs, duration=duration, warmup_s=warmup_s, batches=batches,
                compiled=bool(pt.get("compile")) and dev == "cuda")


# ----------------------------------------------------------------------------
# continuous batching
# ----------------------------------------------------------------------------

def cb_config(cb_opts: dict, C: int, kv_tokens: int, dev: str, facts: dict):
    """ContinuousBatchingConfig with only the options this transformers version accepts; the rest
    are recorded in facts['cb_dropped'] so the functional check can flag a knob that didn't apply."""
    try:
        from transformers.generation import ContinuousBatchingConfig
    except ImportError:
        from transformers.generation.continuous_batching import ContinuousBatchingConfig
    params = inspect.signature(ContinuousBatchingConfig).parameters
    kw = dict(cb_opts)
    kw.setdefault("default_compile_level", 0)
    kw["max_requests_per_batch"] = C
    size_kw = "page_size" if "page_size" in params else "block_size"    # renamed in transformers 5.19
    if kv_tokens:            # same KV capacity (tokens) as the vLLM default had: a fair fight
        kw.update({size_kw: 256, "num_blocks": max(1, math.ceil(kv_tokens / 256))})
    if dev == "cpu":         # the auto-sizer reads free GPU memory; size explicitly for CPU tests
        kw.update({size_kw: 256, "num_blocks": kw.get("num_blocks", 512)})
        kw.setdefault("max_batch_tokens", int(os.environ.get("PT_CPU_MBT", 8192)))
    accepts_any = any(p.kind == p.VAR_KEYWORD for p in params.values())
    dropped = [k for k in kw if k not in params and not accepts_any]
    for k in dropped:
        kw.pop(k)
    cfg = ContinuousBatchingConfig(**kw)
    asked = [k for k in dropped if k in cb_opts]          # only the config's own knobs count as "not applied"
    if asked:
        facts["cb_dropped"] = asked
    for k in ("allow_block_sharing", "use_cuda_graph", "use_async_batching", "max_batch_tokens", "default_compile_level"):
        if k in cb_opts or k == "default_compile_level":
            v = getattr(cfg, k, None)
            facts[f"cb_{k}"] = v if k not in dropped else f"not supported ({cb_opts.get(k)})"
    return cfg


def run_cb(model, pt: dict, w: Workload, prompts: PromptSet, kv_tokens: int, dev: str, facts: dict) -> dict:
    from transformers import GenerationConfig
    try:
        from transformers.generation.continuous_batching.utils import WorkloadHints
    except ImportError:
        WorkloadHints = None

    if dev == "cpu":   # plumbing tests only: the cache sizer probes free GPU memory, which is 0 on CPU
        from transformers.generation.continuous_batching import cache as _cbc
        _cbc.PagedAttentionMemoryHandler.get_available_memory = lambda self: 2 * 2**30

    C = w.concurrency
    n_meas = len(prompts) - C
    gcfg = GenerationConfig(max_new_tokens=max(prompts.out_lens), do_sample=False)
    cbc = cb_config(pt.get("cb", {}), C, kv_tokens, dev, facts)
    init_kw = dict(generation_config=gcfg, continuous_batching_config=cbc)
    if WorkloadHints is not None:
        init_kw["workload_hints"] = WorkloadHints(max_prompt_length=max(len(p) for p in prompts.ids),
                                                  max_generated_length=max(prompts.out_lens), num_requests=len(prompts))
    mgr = model.init_continuous_batching(**init_kw)
    tw = time.perf_counter()
    if hasattr(mgr, "warmup"):
        mgr.warmup()                                      # pre-capture CUDA graphs
    # Every request is queued BEFORE the manager starts (adding requests while the background loop
    # schedules races with it and can produce an empty step in transformers 5.19).
    for i, (p, o) in enumerate(zip(prompts.ids, prompts.out_lens)):
        mgr.add_request(input_ids=p, request_id=f"{'w' if i < C else 'm'}{i}", max_new_tokens=o,
                        record_timestamps=True, eos_token_id=-1)
    mgr.start()
    results = {}
    try:
        while len(results) < len(prompts):
            res = mgr.get_result(timeout=900)
            if res is None:
                if not mgr.is_running():
                    raise RuntimeError("continuous batching manager stopped unexpectedly")
                continue
            if getattr(res, "error", None):
                raise RuntimeError(f"request failed: {res.error}")
            results[res.request_id] = res
    finally:
        try:
            mgr.stop(block=True, timeout=120)
        except TypeError:
            mgr.stop()
    reqs = []
    for rid, r in results.items():
        if not rid.startswith("m"):
            continue
        ts = r.timestamps or []
        start = r.lifespan[0] if r.lifespan and r.lifespan[0] > 0 else r.created_time
        end = ts[-1] if ts else (r.lifespan[1] if r.lifespan and r.lifespan[1] > 0 else start)
        reqs.append(dict(submit=start, first=ts[0] if ts else end, token_times=ts, end=end,
                         n_in=len(r.prompt_ids) if r.prompt_ids else 0, n_out=len(r.generated_tokens)))
    if len(reqs) < n_meas:
        raise RuntimeError(f"only {len(reqs)} of {n_meas} measured requests returned")
    warm_end = max((r.lifespan[1] for rid, r in results.items()
                    if rid.startswith("w") and r.lifespan and r.lifespan[1] > 0), default=tw)
    duration = max(r["end"] for r in reqs) - min(r["submit"] for r in reqs)
    return dict(reqs=reqs, duration=duration, warmup_s=warm_end - tw,
                compiled=(pt.get("cb", {}).get("default_compile_level", 0) or 0) > 0 and dev == "cuda")


# ----------------------------------------------------------------------------
# synthetic engine for orchestration tests (LLMBENCH_FAKE_TORCH=1, no torch needed)
# ----------------------------------------------------------------------------

_FAKE_SPEED = {"pt_eager": 0.35, "pt_sdpa": 0.45, "pt_static": 0.5, "pt_compile_nograph": 0.6, "pt_compile": 0.75,
               "pt_optimized": 0.95, "pt_quant_int4": 0.9, "pt_kernels": 0.55, "pt_assisted": 0.6, "pt_cb": 0.7,
               "pt_cb_noprefix": 0.66, "pt_cb_chunk_small": 0.68, "pt_cb_compile": 0.8, "pt_cb_graphs": 0.85,
               "pt_serve_optimized": 1.0}


def fake_result(cfg_name: str, w: Workload, rep: int, seed: int) -> dict:
    rng = random.Random(zlib.crc32(f'{cfg_name}|{w.name}|{rep}|{seed}'.encode()))
    if os.environ.get("LLMBENCH_FAKE_FAIL") == f"{cfg_name}:{w.name}" and rep == 0:
        raise RuntimeError("simulated failure")
    noisy = os.environ.get("LLMBENCH_FAKE_NOISY") == f"{cfg_name}:{w.name}"
    speed = _FAKE_SPEED.get(cfg_name, 0.5) * (1 + (rng.random() - 0.5) * (0.6 if noisy else 0.02))
    static = not cfg_name.startswith("pt_cb") and cfg_name != "pt_serve_optimized"
    c = w.concurrency
    tpot = 0.02 / speed * (1 + 0.02 * c)
    ttft = 0.001 * (w.input_len + w.prefix_len) / speed / 50 * (c if static else 1)
    n = min(w.num_prompts, 4 * c)
    reqs, t = [], 0.0
    for i in range(n):
        o = w.output_len
        first = t + ttft
        times = [first + k * tpot for k in range(o)]
        reqs.append(dict(submit=t, first=first, token_times=times, end=times[-1], n_in=w.input_len, n_out=o))
        if (i + 1) % c == 0:
            t = times[-1]
    return dict(reqs=reqs, duration=max(r["end"] for r in reqs), warmup_s=0.1, batches=n // c, compiled="compile" in cfg_name)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config-json", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--workloads-json", required=True)
    p.add_argument("--targets-json", default="")
    p.add_argument("--max-batches", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--draft-model", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--kv-tokens", type=int, default=0)
    p.add_argument("--slo", default="")
    p.add_argument("--min-repeats", type=int, default=1)
    p.add_argument("--max-repeats", type=int, default=3)
    p.add_argument("--target-cv", type=float, default=0.05)
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()

    from .configs import Config
    from .results import base_row

    cfg = Config.from_dict(json.loads(a.config_json))
    pt = cfg.pt
    run_dir = Path(a.run_dir)
    raw = run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    wls = [Workload.from_dict(w) for w in json.loads(a.workloads_json)]
    targets = json.loads(a.targets_json) if a.targets_json else None
    slo = parse_slo(a.slo)
    fake = os.environ.get("LLMBENCH_FAKE_TORCH") == "1"
    facts: dict = dict(status="ok", mode=pt.get("mode"), attn=pt.get("attn"), cache=pt.get("cache"))

    def write_facts():
        write_json(raw / f"{cfg.name}__facts.json", facts)

    gpu_count = a.tp if a.device == "cuda" else 1
    if fake:
        torch = None
        sync = lambda: None  # noqa: E731
        dev = "cpu"
        model = draft = tok = None
        vocab = list(range(100, 32000))
        pad_id = 100
        facts.update(torch="fake", compiled=bool(pt.get("compile") or pt.get("cb", {}).get("default_compile_level")),
                     quant=pt.get("quant"), kernels=bool(pt.get("kernels")))
        for k, v in pt.get("cb", {}).items():
            facts[f"cb_{k}"] = v
        clog = None
    else:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
        sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(a.dtype, torch.bfloat16)
        facts.update(torch=torch.__version__, transformers=transformers.__version__)
        clog = CompileLog()
        t0 = time.perf_counter()
        # device_map needs `accelerate`; without it (single GPU only) load on CPU, then move
        direct = dev == "cuda" and (a.tp > 1 or has_accelerate())

        def load(path: str, attn: str, kernels: bool = False):
            kw = dict(attn_implementation=attn)
            if direct:
                kw["device_map"] = "auto" if a.tp > 1 else "cuda"
            if kernels:
                kw["use_kernels"] = True
            try:
                m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **kw)
            except TypeError as e:   # transformers < 4.56 spells it torch_dtype
                if "use_kernels" in str(e):
                    raise
                m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **kw)
            if not direct:
                m.to(dev)
            return m.eval()

        facts["load_path"] = "device_map" if direct else f"cpu->{dev}"
        try:
            model = load(a.model, pt.get("attn", "sdpa"), kernels=bool(pt.get("kernels")))
            if pt.get("kernels"):
                facts["kernels"] = bool(getattr(model, "use_kernels", True))
            draft = load(a.draft_model, "sdpa") if pt.get("assisted") else None
            if pt.get("quant"):
                from torchao.quantization import quantize_
                cc = torch.cuda.get_device_capability() if dev == "cuda" else (0, 0)
                if pt["quant"] == "int4":
                    from torchao.quantization import Int4WeightOnlyConfig
                    q = Int4WeightOnlyConfig(group_size=128)
                elif cc >= (8, 9):
                    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
                    q = Float8DynamicActivationFloat8WeightConfig()
                else:
                    from torchao.quantization import Int8WeightOnlyConfig
                    q = Int8WeightOnlyConfig()
                # Linear layers only, lm_head excluded: vLLM's --quantization fp8 leaves lm_head in BF16
                # too, and on tied-embedding models (Qwen3 <= 4B) quantizing it breaks the tie and needs
                # a 1.45 GiB fp32 temporary for Qwen3-4B's 152k x 2560 head (OOMed a 16 GB card).
                if dev == "cuda":
                    torch.cuda.empty_cache()
                quantize_(model, q, filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear)
                          and not fqn.endswith("lm_head"))
                facts["quant"] = type(q).__name__ + " (Linear layers, lm_head kept BF16)"
        except Exception as e:  # noqa: BLE001
            facts.update(status="failed", error=f"load: {type(e).__name__}: {str(e)[:300]}")
            write_facts()
            print(f"PT| !! {facts['error']}", flush=True)
            return
        facts["weights_load_s"] = facts["startup_s"] = round(time.perf_counter() - t0, 1)
        facts["attn_resolved"] = getattr(model.config, "_attn_implementation", pt.get("attn"))
        if dev == "cuda":
            facts["weights_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
        try:
            tok = AutoTokenizer.from_pretrained(a.model)
        except Exception:  # noqa: BLE001
            tok = None
        tc = getattr(model.config, "text_config", model.config)
        vocab = regular_vocab(tok, tc.vocab_size)
        pad_id = vocab[0]

        def kv_gib(B: int, L: int) -> float:
            heads = tc.num_attention_heads
            kvh = getattr(tc, "num_key_value_heads", None) or heads
            hd = getattr(tc, "head_dim", None) or tc.hidden_size // heads
            return 2 * tc.num_hidden_layers * kvh * hd * dtype.itemsize * B * L / 2**30

    policy = Policy(a.min_repeats, a.max_repeats, a.target_cv)
    quiet_cvs: list[float] = []
    compiled_any = False
    compiles_total, compile_s_total = 0, 0.0
    for w in wls:
        if pt.get("assisted") and w.concurrency != 1:
            continue    # HF assisted generation is batch-size-1 only
        if os.environ.get("PT_TEST_CRASH_ON") == w.name:   # test hook: simulate a hard crash
            import signal
            os.kill(os.getpid(), signal.SIGKILL)
        have = {}
        for f in raw.glob(f"{cfg.name}__{w.name}__r*.row.json"):
            r = json.loads(f.read_text())
            have[int(r.get("repeat") or 0)] = r
        if targets is None:
            pol, attempted0 = policy, len(have)
            if is_quiet(quiet_cvs, policy):
                pol = Policy(1, policy.max_repeats, policy.target_cv)
        else:
            base_good = sum(1 for r in have.values() if good(r))
            extra = max(1, int(targets.get(w.name, 1)))
            pol, attempted0 = Policy(base_good + extra, base_good + extra + 1, policy.target_cv), base_good
        attempted = 0
        while need_more(list(have.values()), attempted0 + attempted, pol):
            rep = max(have, default=-1) + 1
            row = base_row(cfg, w, a.model, rep, gpu_count)
            t_wall = time.perf_counter()
            prof_path = None
            try:
                if not fake and dev == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                n = w.num_prompts + w.concurrency                    # + one warmup batch / wave
                prompts = make_prompts(w, n, vocab, a.seed + 1000 * rep, tokenizer=tok) if not fake else None
                profile_fn = None
                if a.profile and w.kind == "reference" and rep == 0 and not fake:
                    prof_path = run_dir / "profiles" / cfg.name / "trace.json"
                    profile_fn = make_profiler(prof_path, dev)
                if fake:
                    res = fake_result(cfg.name, w, rep, a.seed)
                elif pt["mode"] == "generate":
                    B, L = w.concurrency, w.total_len
                    if dev == "cuda":
                        free = torch.cuda.mem_get_info()[0] / 2**30
                        need = kv_gib(B, L)
                        if need > free * 0.95:
                            raise MemoryError(f"static batch {B}x{L} needs ~{need:.1f} GiB KV, {free:.1f} GiB free")
                    res = run_static(model, draft, pt, w, prompts, pad_id, a.max_batches, sync, dev, profile_fn)
                else:
                    # transformers' CB scheduler occasionally emits an empty step ("cannot reshape tensor
                    # of 0 elements"), nondeterministically; retry with a fresh manager, and record it
                    for attempt in range(3):
                        try:
                            res = run_cb(model, pt, w, prompts, a.kv_tokens, dev, facts)
                            row["cb_retries"] = attempt
                            break
                        except RuntimeError as e:
                            if "0 elements" not in str(e) or attempt == 2:
                                raise
                            print(f"PT| {w.label:<34} transformers CB empty-step error; retry {attempt + 1}/2", flush=True)
                    if profile_fn is not None:
                        profile_fn(lambda: run_cb(model, pt, w, PromptSet(prompts.ids[:2 * w.concurrency], [],
                                                                       prompts.out_lens[:2 * w.concurrency]),
                                                  a.kv_tokens, dev, {}))
                row.update(summarize(res["reqs"], res["duration"], slo))
                row.update(warmup_s=round(res["warmup_s"], 1), batches_measured=res.get("batches"),
                           num_prompts=len(res["reqs"]), bench_rc=0.0, client="pytorch")
                compiled_any |= bool(res.get("compiled"))
                if clog is not None and clog.ok and res["reqs"]:
                    nc, cs = clog.within(min(r["submit"] for r in res["reqs"]), max(r["end"] for r in res["reqs"]))
                    row.update(compiles_in_measure=nc, compile_s_in_measure=round(cs, 1))
                    compiles_total, compile_s_total = compiles_total + nc, compile_s_total + cs
            except Exception as e:  # noqa: BLE001  (includes torch.OutOfMemoryError)
                oom = isinstance(e, MemoryError) or "out of memory" in str(e).lower() \
                    or type(e).__name__ == "OutOfMemoryError"
                row.update(bench_rc=1.0, bench_error=("OOM: " if oom else f"{type(e).__name__}: ") + str(e)[:200])
                if not fake and dev == "cuda":
                    torch.cuda.empty_cache()
                if oom:
                    attempted = 10 ** 6            # deterministic: don't repeat an OOM
            if not fake and dev == "cuda":
                row["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            row["bench_wall_s"] = time.perf_counter() - t_wall
            if targets is not None:
                row["retry"] = 1.0
            for k in ("preemptions", "prefix_hit_rate", "spec_accept_rate"):
                row.setdefault(k, float("nan"))
            write_row(raw, row)
            have[rep] = row
            attempted += 1
            if prof_path is not None and prof_path.exists():
                try:
                    from .profiling import breakdown
                    bd = breakdown(prof_path)
                    bd.update(config=cfg.name, engine="pytorch", workload=w.label,
                              trace=str(prof_path.relative_to(run_dir)))
                    write_json(raw / f"{cfg.name}__profile.json", bd)
                    facts["gpu_busy_pct"] = bd.get("gpu_busy_pct")
                except Exception as e:  # noqa: BLE001
                    print(f"PT| profile parse failed: {e}", flush=True)
            thr = row.get("output_throughput", float("nan"))
            spread = cv([r.get("output_throughput") for r in have.values() if good(r)])
            if row["bench_rc"]:
                print(f"PT| {w.label:<36} r{rep} {row['bench_error'][:110]}", flush=True)
            else:
                rc = (f"  !! {row['compiles_in_measure']} compiles ({row['compile_s_in_measure']:.1f}s) inside the "
                      "measured window" if (row.get("compile_s_in_measure") or 0) >= COMPILE_WARN_S else "")
                print(f"PT| {w.label:<36} r{rep} out={thr:8.1f} tok/s  ttft50={row['median_ttft_ms']:7.1f}ms  "
                      f"tpot50={row['median_tpot_ms']:6.2f}ms" + (f"  cv={spread:.1%}" if spread == spread else "")
                      + f"  ({row['bench_wall_s']:.0f}s){rc}", flush=True)

        sw = cv([r.get("output_throughput") for r in have.values() if good(r)])
        if sw == sw:
            quiet_cvs.append(sw)

    facts["compiled"] = compiled_any if not fake else facts.get("compiled")
    if clog is not None and clog.ok:
        facts.update(compiles_in_measure=compiles_total, compile_s_in_measure=round(compile_s_total, 1))
    if not fake and dev == "cuda":
        facts["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    bits = [f"attn={facts.get('attn_resolved', pt.get('attn'))}"]
    if pt.get("mode") == "generate":
        bits += [f"kv={pt.get('cache')}", "batching=static", f"compile={pt.get('compile') or 'off'}"]
    else:
        bits += ["kv=paged", "batching=continuous"] + [f"{k}={v}" for k, v in pt.get("cb", {}).items()]
    if facts.get("quant"):
        bits.append(f"quant={facts['quant']}")
    if pt.get("assisted"):
        bits.append(f"draft={a.draft_model}")
    facts["resolved"] = " ".join(bits)
    facts["gpu_count"] = gpu_count
    write_facts()


def make_profiler(path: Path, dev: str):
    """Returns fn(run_once): profiles one extra (unmeasured) pass and writes a chrome trace."""
    def go(run_once):
        import torch
        from torch.profiler import ProfilerActivity, profile
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev == "cuda" else [])
        path.parent.mkdir(parents=True, exist_ok=True)
        with profile(activities=acts, record_shapes=False, with_stack=False) as prof:
            run_once()
            if dev == "cuda":
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(path))
    return go


if __name__ == "__main__":
    sys.exit(main())
