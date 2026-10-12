#!/usr/bin/env python3
"""
pytorch_bench.py - the PyTorch/transformers side of the vLLM comparison.

Runs the SAME model on the SAME workload grid as vllm_ablation.py, using only
features PyTorch itself offers, and writes rows in the same schema so one
report.md compares everything. Each config runs in its own subprocess
(fresh CUDA context and compile cache), mirroring one-vllm-serve-per-config.

PyTorch ladder (each step adds one shared feature):
  pt_eager        generate(): eager attention, dynamic KV cache, static batching
  pt_sdpa         + fused attention (F.scaled_dot_product_attention)
  pt_static       + preallocated static KV cache (fixed shapes)
  pt_compile      + torch.compile (Inductor) + CUDA graphs on the decode step
  pt_compile_fp8  + torchao FP8 W8A8 (int8 weight-only below sm_89)  ~ vLLM fp8_online
  pt_assisted     speculative decoding with a draft model (batch 1 only) ~ vLLM spec_draft
PyTorch-native serving (transformers continuous batching, paged attention):
  pt_cb           paged KV + continuous batching + prefix caching + chunked prefill
  pt_cb_compile   + torch.compile (default_compile_level=1: Inductor, dynamic shapes)

Static-batching semantics: concurrency C = batch size; prompts run in batches
of C, so a request's TTFT is its batch's prefill and its E2E is the batch's
completion - the static-batching analog of `vllm bench serve --max-concurrency C`.
Continuous batching uses a closed-loop client (a new request is submitted the
moment one finishes), exactly like vllm bench's max-concurrency semaphore.

Usage:
  python pytorch_bench.py --attach <run_id>        # add PyTorch rows to an existing vLLM run
  python pytorch_bench.py --quick                  # standalone (same auto model/grid as vLLM)
  python pytorch_bench.py --attach <id> --pt-configs pt_eager,pt_compile,pt_cb
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vllm_ablation as V  # noqa: E402

PT_CONFIGS = [
    dict(name="pt_eager", group="pytorch", mode="generate", attn="eager", cache="dynamic", compile=False,
         knob="HF generate(): eager attention, dynamic KV cache, static batching"),
    dict(name="pt_sdpa", group="pytorch", mode="generate", attn="sdpa", cache="dynamic", compile=False,
         knob="+ fused attention (SDPA / FlashAttention kernels)"),
    dict(name="pt_static", group="pytorch", mode="generate", attn="sdpa", cache="static", compile=False,
         knob="+ preallocated static KV cache"),
    dict(name="pt_compile", group="pytorch", mode="generate", attn="sdpa", cache="static", compile=True,
         knob="+ torch.compile (Inductor) + CUDA graphs on decode"),
    dict(name="pt_compile_fp8", group="pytorch", mode="generate", attn="sdpa", cache="static", compile=True,
         quant=True, knob="+ torchao FP8 W8A8 (int8 weight-only below sm_89)"),
    dict(name="pt_assisted", group="pytorch", mode="generate", attn="sdpa", cache="dynamic", compile=False,
         assisted=True, knob="speculative decoding with a draft model (HF assisted generation, batch 1)"),
    dict(name="pt_cb", group="pytorch-cb", mode="cb", attn="paged|sdpa", compile_level=0,
         knob="transformers continuous batching: paged KV, prefix caching, chunked prefill"),
    # Level 1 = Inductor default mode, dynamic shapes. Level 3 compiles static shapes with
    # max-autotune: every new batch/sequence shape recompiles, which ran ~3x slower than eager CB
    # and then hit Dynamo's recompile limit (8) on an RTX 5070 Ti.
    dict(name="pt_cb_compile", group="pytorch-cb", mode="cb", attn="paged|sdpa", compile_level=1,
         knob="+ torch.compile (default_compile_level=1: Inductor, dynamic shapes)"),
]
PT_BY_NAME = {c["name"]: c for c in PT_CONFIGS}


# ----------------------------------------------------------------------------
# metrics (same field names as `vllm bench serve` result JSON)
# ----------------------------------------------------------------------------

def _pctl(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q / 100.0
    f = math.floor(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(reqs: list[dict], duration: float, slo: dict | None) -> dict:
    """reqs: submit, first, token_times, end, n_in, n_out (perf_counter seconds)."""
    ttft = [(r["first"] - r["submit"]) * 1e3 for r in reqs]
    e2el = [(r["end"] - r["submit"]) * 1e3 for r in reqs]
    tpot = [(r["end"] - r["first"]) / (r["n_out"] - 1) * 1e3 for r in reqs if r["n_out"] > 1]
    itl = []
    for r in reqs:
        t = [r["first"]] + list(r["token_times"][1:])
        itl += [(b - a) * 1e3 for a, b in zip(t, t[1:])]
    tin = sum(r["n_in"] for r in reqs)
    tout = sum(r["n_out"] for r in reqs)
    row = dict(duration=duration, completed=float(len(reqs)), total_input_tokens=float(tin),
               total_output_tokens=float(tout),
               request_throughput=len(reqs) / duration if duration else float("nan"),
               output_throughput=tout / duration if duration else float("nan"),
               total_token_throughput=(tin + tout) / duration if duration else float("nan"))
    for name, xs in (("ttft", ttft), ("tpot", tpot), ("itl", itl), ("e2el", e2el)):
        row[f"mean_{name}_ms"] = statistics.fmean(xs) if xs else float("nan")
        row[f"median_{name}_ms"] = _pctl(xs, 50)
        row[f"std_{name}_ms"] = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        for q in (50, 90, 99):
            row[f"p{q}_{name}_ms"] = _pctl(xs, q)
    if slo and duration:
        ok = sum(1 for a, b in zip(ttft, tpot) if a <= slo.get("ttft", 1e18) and b <= slo.get("tpot", 1e18))
        row["request_goodput"] = ok / duration
    return row


def make_prompts(w: V.Workload, n: int, vocab: list[int], seed: int) -> list[list[int]]:
    rng = random.Random(seed ^ zlib.crc32(w.name.encode()))
    prefix = [rng.choice(vocab) for _ in range(w.prefix_len)]
    return [prefix + [rng.choice(vocab) for _ in range(w.input_len)] for _ in range(n)]


# ----------------------------------------------------------------------------
# worker: one config, all workloads, in its own process
# ----------------------------------------------------------------------------

class CompileLog:
    """Wall-clock spans of torch.compile work (Dynamo + Inductor), from Dynamo's compile
    callbacks. Used to tell whether a "compiled" config was still recompiling inside the
    measured window, in which case its numbers measure the compiler, not the model."""

    def __init__(self):
        self.spans: list[tuple[float, float]] = []
        self._t0 = None
        try:
            from torch._dynamo.callback import callback_handler
            callback_handler.register_start_callback(self._start)
            callback_handler.register_end_callback(self._end)
            self.ok = True
        except Exception:  # noqa: BLE001  (older torch: no callbacks, nothing recorded)
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


def worker(a: argparse.Namespace) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    cfg = PT_BY_NAME[a.config]
    run_dir = Path(a.run_dir)
    raw = run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    wls = [V.Workload(**w) for w in json.loads(a.workloads_json)]
    dev = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(a.dtype, torch.bfloat16)
    slo = dict(kv.split(":") for kv in a.slo.split()) if a.slo else None
    slo = {k: float(v) for k, v in slo.items()} if slo else None
    facts = dict(status="ok", torch=torch.__version__, transformers=transformers.__version__,
                 attn=cfg["attn"], cache=cfg.get("cache"), mode=cfg["mode"])

    def write_facts():
        (raw / f"{cfg['name']}__facts.json").write_text(json.dumps(facts, indent=1, default=str))

    clog = CompileLog()
    compiles_total, compile_s_total = 0, 0.0

    # ---- load
    t0 = time.perf_counter()
    # device_map needs `accelerate`; without it (single GPU only) load on CPU, then move
    direct = dev == "cuda" and (a.tp > 1 or has_accelerate())

    def load(path: str, attn: str):
        kw = dict(attn_implementation=attn)
        if direct:
            kw["device_map"] = "auto" if a.tp > 1 else "cuda"
        try:
            m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **kw)
        except TypeError:   # transformers < 4.56 spells it torch_dtype
            m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **kw)
        if not direct:
            m.to(dev)
        return m.eval()

    facts["load_path"] = "device_map" if direct else f"cpu->{dev}"
    try:
        model = load(a.model, cfg["attn"])
        draft = load(a.draft_model, "sdpa") if cfg.get("assisted") else None
        if cfg.get("quant"):
            from torchao.quantization import quantize_
            cc = torch.cuda.get_device_capability() if dev == "cuda" else (0, 0)
            if cc >= (8, 9):
                from torchao.quantization import Float8DynamicActivationFloat8WeightConfig as Q
            else:
                from torchao.quantization import Int8WeightOnlyConfig as Q
            # Linear layers only, lm_head excluded: vLLM's --quantization fp8 leaves lm_head in BF16
            # too, and on tied-embedding models (Qwen3 <= 4B) quantizing it breaks the tie and
            # needs a 1.45 GiB fp32 temporary for Qwen3-4B's 152k x 2560 head, which OOMed a 16 GB card.
            if dev == "cuda":
                torch.cuda.empty_cache()
            quantize_(model, Q(), filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear)
                      and not fqn.endswith("lm_head"))
            facts["quant"] = Q.__name__ + " (Linear layers, lm_head kept BF16)"
    except Exception as e:  # noqa: BLE001
        facts.update(status="failed", error=f"load: {type(e).__name__}: {str(e)[:300]}")
        write_facts()
        print(f"PT| !! {facts['error']}", flush=True)
        return
    facts["weights_load_s"] = facts["startup_s"] = round(time.perf_counter() - t0, 1)
    facts["attn_resolved"] = getattr(model.config, "_attn_implementation", cfg["attn"])
    if dev == "cuda":
        facts["weights_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)

    # ---- prompt vocabulary: regular tokens only (like vllm bench's random dataset)
    vocab_n = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        vocab_n = tok.vocab_size
        special = set(tok.all_special_ids)
    except Exception:  # noqa: BLE001
        special = set()
    tc = getattr(model.config, "text_config", model.config)
    if not vocab_n or vocab_n < 1000:          # no real tokenizer (v5 can return a 1-token stub)
        vocab_n, special = tc.vocab_size, set()
    vocab_n = min(vocab_n, tc.vocab_size)
    vocab = [i for i in range(min(100, vocab_n // 2), vocab_n) if i not in special]
    pad_id = vocab[0]

    def kv_gib(B: int, L: int) -> float:
        heads = tc.num_attention_heads
        kvh = getattr(tc, "num_key_value_heads", None) or heads
        hd = getattr(tc, "head_dim", None) or tc.hidden_size // heads
        return 2 * tc.num_hidden_layers * kvh * hd * dtype.itemsize * B * L / 2**30

    compiled_any = False
    for w in wls:
        if w.dataset != "random":
            continue
        base = dict(config=cfg["name"], group=cfg["group"], knob=cfg["knob"], model=a.model,
                    workload=w.name, workload_label=w.label, dataset=w.dataset,
                    input_len=w.input_len, output_len=w.output_len, prefix_len=w.prefix_len,
                    concurrency=w.concurrency, num_prompts=w.num_prompts, repeat=0)
        if cfg.get("assisted") and w.concurrency != 1:
            continue  # HF assisted generation is batch-size-1 only
        if os.environ.get("PT_TEST_CRASH_ON") == w.name:   # test hook: simulate a hard crash
            import signal
            os.kill(os.getpid(), signal.SIGKILL)
        row = dict(base)
        t_wall = time.perf_counter()
        try:
            if dev == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            if cfg["mode"] == "generate":
                B, L = w.concurrency, w.total_len
                if dev == "cuda":
                    free = torch.cuda.mem_get_info()[0] / 2**30
                    need = kv_gib(B, L)
                    if need > free * 0.95:
                        raise MemoryError(f"static batch {B}x{L} needs ~{need:.1f} GiB KV, "
                                          f"{free:.1f} GiB free")
                res = run_static(model, draft, cfg, w, vocab, pad_id, a, sync, dev)
            else:
                # transformers' CB scheduler occasionally emits an empty step ("cannot reshape tensor
                # of 0 elements"), nondeterministically; retry with a fresh manager, and record it
                for attempt in range(3):
                    try:
                        res = run_cb(model, cfg, w, vocab, a, sync, dev)
                        row["cb_retries"] = attempt
                        break
                    except RuntimeError as e:
                        if "0 elements" not in str(e) or attempt == 2:
                            raise
                        print(f"PT| {w.label:<34} transformers CB empty-step error; retry {attempt + 1}/2",
                              flush=True)
            row.update(summarize(res["reqs"], res["duration"], slo))
            row.update(warmup_s=round(res["warmup_s"], 1), batches_measured=res.get("batches"),
                       num_prompts=len(res["reqs"]), bench_rc=0.0)
            compiled_any |= bool(res.get("compiled"))
            if clog.ok and res["reqs"]:
                nc, cs = clog.within(min(r["submit"] for r in res["reqs"]), max(r["end"] for r in res["reqs"]))
                row.update(compiles_in_measure=nc, compile_s_in_measure=round(cs, 1))
                compiles_total, compile_s_total = compiles_total + nc, compile_s_total + cs
        except Exception as e:  # noqa: BLE001  (includes torch.OutOfMemoryError)
            oom = isinstance(e, MemoryError) or "out of memory" in str(e).lower() \
                or type(e).__name__ == "OutOfMemoryError"
            row.update(bench_rc=1.0, bench_error=("OOM: " if oom else f"{type(e).__name__}: ") + str(e)[:200])
            if dev == "cuda":
                torch.cuda.empty_cache()
        if dev == "cuda":
            row["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        row["bench_wall_s"] = time.perf_counter() - t_wall
        for k in ("preemptions", "prefix_hit_rate", "spec_accept_rate"):
            row.setdefault(k, float("nan"))
        (raw / f"{cfg['name']}__{w.name}__r0.row.json").write_text(json.dumps(row, default=float))
        thr = row.get("output_throughput", float("nan"))
        if row["bench_rc"]:
            print(f"PT| {w.label:<34} {row['bench_error'][:110]}", flush=True)
        else:
            rc = (f"  !! {row['compiles_in_measure']} compiles ({row['compile_s_in_measure']:.1f}s) "
                  "inside the measured window" if (row.get("compile_s_in_measure") or 0) >= V.COMPILE_WARN_S else "")
            print(f"PT| {w.label:<34} out={thr:8.1f} tok/s  ttft50={row['median_ttft_ms']:7.1f}ms  "
                  f"tpot50={row['median_tpot_ms']:6.2f}ms  warmup={row['warmup_s']:.0f}s  "
                  f"({row['bench_wall_s']:.0f}s){rc}", flush=True)

    facts["compiled"] = compiled_any
    if clog.ok:
        facts.update(compiles_in_measure=compiles_total, compile_s_in_measure=round(compile_s_total, 1))
    if dev == "cuda":
        facts["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    bits = [f"attn={facts['attn_resolved']}"]
    if cfg["mode"] == "generate":
        bits += [f"kv={cfg['cache']}", "batching=static",
                 f"compile={'inductor+cudagraphs' if compiled_any else 'off'}"]
    else:
        bits += ["kv=paged", "batching=continuous", f"compile_level={cfg['compile_level']}"]
    if facts.get("quant"):
        bits.append(f"quant={facts['quant']}")
    if cfg.get("assisted"):
        bits.append(f"draft={a.draft_model}")
    facts["resolved"] = " ".join(bits)
    write_facts()


def run_static(model, draft, cfg, w, vocab, pad_id, a, sync, dev) -> dict:
    """Static batching through model.generate(); per-token times via a streamer."""
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
            now = time.perf_counter()
            self.times += [now] * k    # assisted generation emits several tokens per step

        def end(self):
            pass

    B = w.concurrency
    n = min(w.num_prompts, B * a.max_batches)
    prompts = make_prompts(w, n + B, vocab, a.seed)     # + one warmup batch
    gen = dict(max_new_tokens=w.output_len, min_new_tokens=w.output_len, do_sample=False,
               pad_token_id=pad_id)
    if cfg["cache"] == "static":
        gen["cache_implementation"] = "static"
    if cfg.get("compile"):
        from transformers import CompileConfig
        gen["compile_config"] = CompileConfig(mode="reduce-overhead", fullgraph=True)
        gen["disable_compile"] = False
    else:
        gen["disable_compile"] = True
    if draft is not None:
        gen["assistant_model"] = draft

    def one(batch):
        ids = torch.tensor(batch, device=dev)
        clock = Clock()
        sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), streamer=clock, **gen)
        sync()
        t1 = time.perf_counter()
        n_out = out.shape[1] - ids.shape[1]
        first = clock.times[0] if clock.times else t1
        return [dict(submit=t0, first=first, token_times=clock.times, end=t1,
                     n_in=ids.shape[1], n_out=n_out) for _ in batch]

    tw = time.perf_counter()
    one(prompts[:B])                                     # warmup: compile + graph capture
    warmup_s = time.perf_counter() - tw
    reqs, t_start = [], time.perf_counter()
    batches = 0
    for i in range(B, B + n, B):
        reqs += one(prompts[i:i + B])
        batches += 1
    compiled = bool(cfg.get("compile")) and dev == "cuda"
    return dict(reqs=reqs, duration=time.perf_counter() - t_start, warmup_s=warmup_s,
                batches=batches, compiled=compiled)


def run_cb(model, cfg, w, vocab, a, sync, dev) -> dict:
    """transformers continuous batching at concurrency C.

    Every request is queued BEFORE the manager starts and concurrency is capped with
    max_requests_per_batch=C. (Adding requests from this thread while the background loop
    schedules races with it and can produce an empty step in transformers 5.19.) Latency is
    measured from lifespan[0] - when a request was first scheduled - so queue wait is
    excluded, matching vllm bench's closed-loop --max-concurrency semantics. The first C
    requests are warmup (compile / graph capture) and are not measured.
    """
    from transformers import GenerationConfig
    try:
        from transformers.generation import ContinuousBatchingConfig
    except ImportError:
        from transformers.generation.continuous_batching import ContinuousBatchingConfig
    try:
        from transformers.generation.continuous_batching.utils import WorkloadHints
    except ImportError:
        WorkloadHints = None

    if dev == "cpu":   # plumbing tests only: the cache sizer probes free GPU memory, which is 0 on CPU
        from transformers.generation.continuous_batching import cache as _cbc
        _cbc.PagedAttentionMemoryHandler.get_available_memory = lambda self: 2 * 2**30

    C, n = w.concurrency, w.num_prompts
    prompts = make_prompts(w, n + C, vocab, a.seed)
    gcfg = GenerationConfig(max_new_tokens=w.output_len, do_sample=False)
    cb_kw = dict(default_compile_level=cfg["compile_level"], max_requests_per_batch=C)
    import inspect
    block = 256
    size_kw = "page_size" if "page_size" in inspect.signature(ContinuousBatchingConfig).parameters \
        else "block_size"     # renamed in transformers 5.19
    if a.kv_tokens:            # same KV capacity (tokens) as the vLLM baseline had: a fair fight
        cb_kw.update({size_kw: block, "num_blocks": max(1, math.ceil(a.kv_tokens / block))})
    if dev == "cpu":           # the auto-sizer reads free GPU memory; size explicitly for CPU tests
        cb_kw.update({size_kw: block, "num_blocks": cb_kw.get("num_blocks", 512),
                      "max_batch_tokens": int(os.environ.get("PT_CPU_MBT", 8192))})
    cbc = ContinuousBatchingConfig(**cb_kw)
    init_kw = dict(generation_config=gcfg, continuous_batching_config=cbc)
    if WorkloadHints is not None:
        init_kw["workload_hints"] = WorkloadHints(max_prompt_length=w.prefix_len + w.input_len,
                                                  max_generated_length=w.output_len, num_requests=n + C)
    mgr = model.init_continuous_batching(**init_kw)
    tw = time.perf_counter()
    if hasattr(mgr, "warmup"):
        mgr.warmup()                                      # pre-capture CUDA graphs
    for i, p in enumerate(prompts):
        mgr.add_request(input_ids=p, request_id=f"{'w' if i < C else 'm'}{i}",
                        max_new_tokens=w.output_len, record_timestamps=True, eos_token_id=-1)
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
                         n_in=len(r.prompt_ids) if r.prompt_ids else w.prefix_len + w.input_len,
                         n_out=len(r.generated_tokens)))
    warm_end = max((r.lifespan[1] for rid, r in results.items()
                    if rid.startswith("w") and r.lifespan and r.lifespan[1] > 0), default=tw)
    duration = max(r["end"] for r in reqs) - min(r["submit"] for r in reqs)
    return dict(reqs=reqs, duration=duration, warmup_s=warm_end - tw,
                compiled=cfg["compile_level"] > 0 and dev == "cuda")


# ----------------------------------------------------------------------------
# orchestrator
# ----------------------------------------------------------------------------

def alloc_env() -> dict:
    """Expandable segments: fewer fragmentation OOMs when weights are rewritten in place
    (torchao quantization) on a GPU with little headroom. A user-set allocator config wins."""
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or os.environ.get("PYTORCH_ALLOC_CONF"):
        return {}
    return {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def has_accelerate() -> bool:
    import importlib.util
    return importlib.util.find_spec("accelerate") is not None


def available_configs(a, model: str, hw: V.Hardware | None, has_c1: bool) -> list[tuple[dict, str | None]]:
    """(config, skip_reason) for every PyTorch config."""
    # Loading without accelerate goes CPU -> GPU: fine on one discrete GPU, but multi-GPU needs
    # device_map, and on unified memory the CPU copy and the GPU copy share the same RAM.
    no_acc = None
    if a.device == "cuda" and not has_accelerate():
        if a.tp > 1:
            no_acc = "multi-GPU loading needs accelerate (./run.sh setup, or uv pip install accelerate)"
        elif hw is not None and hw.unified:
            no_acc = ("unified memory needs accelerate to load straight to the GPU; the CPU->GPU path "
                      "holds two copies (./run.sh setup, or uv pip install accelerate)")
    try:
        from transformers import GenerationMixin
        has_cb = hasattr(GenerationMixin, "init_continuous_batching")
    except Exception:  # noqa: BLE001
        has_cb = False
    try:
        import torchao  # noqa: F401
        has_ao = True
    except Exception:  # noqa: BLE001
        has_ao = False
    fam = V.FAMILIES.get(V.family_of(model) or "")
    draft = (fam["draft"](model) or {}).get("model") if fam else None
    out = []
    for c in PT_CONFIGS:
        skip = no_acc
        if skip:
            pass
        elif c.get("quant") and not has_ao:
            skip = "torchao not installed (uv pip install torchao)"
        elif c.get("assisted") and not draft:
            skip = "no draft model for this model family"
        elif c.get("assisted") and not has_c1:
            skip = "needs a concurrency-1 workload"
        elif c["mode"] == "cb" and not has_cb:
            skip = "transformers has no continuous batching (needs v5.8+)"
        elif c.get("compile") and a.tp > 1:
            skip = "compile path is single-GPU here (multi-GPU uses device_map='auto' layer split)"
        out.append((c, skip))
    return out


def merge_summary(run_dir: Path, base: dict | None, cfgs: list[dict], facts: dict,
                  skipped: dict | None = None) -> None:
    p = run_dir / "summary.json"
    s = json.loads(p.read_text()) if p.exists() else (base or {})
    s.setdefault("configs", [])
    s.setdefault("server_facts", {})
    s.setdefault("skipped", {}).update(skipped or {})
    names = {c["name"] for c in s["configs"]}
    for c in cfgs:
        if c["name"] not in names:
            s["configs"].append(dict(name=c["name"], group=c["group"], knob=c["knob"], args=[], env={},
                                     model_override=None, note=c.get("note", "")))
    s["server_facts"].update(facts)
    p.write_text(json.dumps(s, indent=1, default=str))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attach", metavar="RUN_ID", help="add PyTorch rows to this existing vllm_ablation run")
    p.add_argument("--out", default="results")
    p.add_argument("--run-id")
    p.add_argument("--run-tag", default="pytorch")
    p.add_argument("--family", choices=list(V.FAMILIES), default="qwen3")
    p.add_argument("--model")
    p.add_argument("--tp", type=int)
    p.add_argument("--gpu-mem", type=float)
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--dtype")
    p.add_argument("--grid", choices=list(V.GRIDS))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--workloads", help="comma-separated workload names (default: all random-token ones)")
    p.add_argument("--pt-configs", help=f"comma-separated subset of: {','.join(PT_BY_NAME)}")
    p.add_argument("--max-batches", type=int, default=4,
                   help="static batching: measured batches per workload (default 4; e.g. 4 x C prompts)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--slo", help='e.g. "ttft:500 tpot:50" (ms) for a goodput column')
    p.add_argument("--allow-shared", action="store_true")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--device", default="cuda", help=argparse.SUPPRESS)  # cpu: plumbing tests only
    # worker-only
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--config", help=argparse.SUPPRESS)
    p.add_argument("--run-dir", help=argparse.SUPPRESS)
    p.add_argument("--workloads-json", help=argparse.SUPPRESS)
    p.add_argument("--draft-model", default="", help=argparse.SUPPRESS)
    p.add_argument("--kv-tokens", type=int, default=0,
                   help="continuous batching KV capacity in tokens (default: vLLM baseline's, when attached)")
    a = p.parse_args()

    if a.worker:
        worker(a)
        return
    V.exit_on_term()

    # ---- resolve model / grid: from an existing run, or exactly as vllm_ablation would
    base_summary = None
    if a.attach:
        run_dir = Path(a.out) / a.attach
        if not (run_dir / "summary.json").exists():
            raise SystemExit(f"[attach] {run_dir} has no summary.json (did that vLLM run start?)")
        s = json.loads((run_dir / "summary.json").read_text())
        model, dtype = s["model"], s.get("dtype", "bfloat16")
        tp = a.tp or s.get("tp", 1)
        wls = [V.Workload(**{k: w[k] for k in ("name", "input_len", "output_len", "concurrency",
                                                "num_prompts", "prefix_len", "dataset")})
               for w in s["workloads"]]
        hw = V.detect_hardware()
        kv = (s.get("server_facts", {}).get("baseline") or {}).get("kv_cache_tokens")
        if kv and not a.kv_tokens:
            a.kv_tokens = int(kv)
        print(f"[attach] {run_dir}: model={model} tp={tp} dtype={dtype} ({len(wls)} workloads)"
              + (f"; continuous-batching KV matched to vLLM baseline: {a.kv_tokens:,} tokens" if a.kv_tokens else ""))
    else:
        ns = argparse.Namespace(
            family=a.family, model=a.model, tp=a.tp, gpu_mem=a.gpu_mem, max_model_len=a.max_model_len,
            dtype=a.dtype, mm_off=None, quick=a.quick, grid=a.grid, drop_caches=None,
            workloads=a.workloads, sharegpt_path=None, spec_config=None, w4a16_model=None)
        hw = V.resolve(ns)
        V.print_setup(ns, hw)
        model, dtype, tp = ns.model, ns.dtype, ns.tp
        wls = V.build_workloads(ns)
        run_id = a.run_id or V.default_run_id(hw, a.run_tag)
        run_dir = Path(a.out) / run_id
        ref = V.GRIDS[ns.grid]["reference"]
        base_summary = dict(
            run_id=run_id, model=model, family=V.family_of(model), tp=tp, dtype=dtype,
            weights_gib=ns.weights_gib, autosize_why=ns.autosize_why, grid=ns.grid,
            max_model_len=ns.max_model_len, gpu_mem=ns.gpu_mem, repeats=1, reference=list(ref),
            reference_label=next(w.label for w in wls if w.name == "reference"),
            configs=[], workloads=[dict(name=w.name, label=w.label, input_len=w.input_len,
                                        output_len=w.output_len, concurrency=w.concurrency,
                                        prefix_len=w.prefix_len, num_prompts=w.num_prompts,
                                        dataset=w.dataset) for w in wls],
            server_facts={}, environment=V.environment(hw) if a.device == "cuda" else
            dict(hardware=hw.to_dict()))
    if a.workloads and a.attach:
        want = set(a.workloads.split(","))
        wls = [w for w in wls if w.name in want]
    wls = [w for w in wls if w.dataset == "random"]
    a.tp = tp
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "server_logs").mkdir(parents=True, exist_ok=True)

    # ---- preflight (GPU must be free; same rules as the vLLM side)
    if a.device == "cuda":
        pf = argparse.Namespace(model=model, port=8031, drop_caches=hw.unified, gpu_mem=0.0, tp=tp,
                                allow_shared=a.allow_shared, nvcc_check=False)
        V.preflight(pf, hw, [])

    if a.device == "cuda" and not has_accelerate() and tp == 1 and not hw.unified:
        print("[pytorch] accelerate not installed: loading each model on CPU, then moving it to the GPU "
              "(same result, slower load; `uv pip install accelerate` loads straight to the GPU)")
    want = set(a.pt_configs.split(",")) if a.pt_configs else None
    plan = [(c, s) for c, s in available_configs(a, model, hw, any(w.concurrency == 1 for w in wls))
            if want is None or c["name"] in want]
    fam = V.FAMILIES.get(V.family_of(model) or "")
    draft = (fam["draft"](model) or {}).get("model", "") if fam else ""
    run = [c for c, s in plan if not s]
    for c, s in plan:
        if s:
            print(f"[select] skipping {c['name']}: {s}")
    merge_summary(run_dir, base_summary, [c for c, s_ in plan if not s_], {},
                  skipped={c["name"]: s_ for c, s_ in plan if s_})
    print(f"\n[pytorch] {len(run)} configs x {len(wls)} workloads -> {run_dir}\n")

    wl_json = json.dumps([dict(name=w.name, input_len=w.input_len, output_len=w.output_len,
                               concurrency=w.concurrency, num_prompts=w.num_prompts,
                               prefix_len=w.prefix_len, dataset=w.dataset) for w in wls])
    all_wls = json.loads(wl_json)
    for i, c in enumerate(run, 1):
        print(f"=== [{i}/{len(run)}] {c['name']}  ({c['group']}: {c['knob']})", flush=True)
        log = run_dir / "server_logs" / f"{c['name']}.log"
        fpath = run_dir / "raw" / f"{c['name']}__facts.json"
        fpath.unlink(missing_ok=True)
        row_path = lambda w: run_dir / "raw" / f"{c['name']}__{w['name']}__r0.row.json"  # noqa: E731
        todo = [w for w in all_wls if w["dataset"] == "random"
                and not (c.get("assisted") and w["concurrency"] != 1)]
        for w in todo:                                   # fresh results for this config
            row_path(w).unlink(missing_ok=True)
        t0, attempt, idle, crashes = time.perf_counter(), 0, 0, []
        with open(log, "w") as lf:
            while todo:
                attempt += 1
                if hw.unified and a.device == "cuda":
                    V.drop_page_cache()   # unified memory: cached safetensors compete with CUDA allocations
                cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config", c["name"],
                       "--run-dir", str(run_dir), "--model", model, "--dtype", dtype, "--tp", str(tp),
                       "--workloads-json", json.dumps(todo), "--max-batches", str(a.max_batches),
                       "--seed", str(a.seed), "--draft-model", draft, "--device", a.device,
                       "--kv-tokens", str(a.kv_tokens or 0)] + (["--slo", a.slo] if a.slo else [])
                lf.write(f"\n### attempt {attempt}: {len(todo)} workloads\n")
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        env={**os.environ, "TOKENIZERS_PARALLELISM": "false", **alloc_env()})
                try:
                    for line in proc.stdout:             # tee: progress lines to console, all to log
                        lf.write(line)
                        if line.startswith("PT| "):
                            print("  " + line[4:], end="", flush=True)
                    rc = proc.wait()
                except BaseException:                    # Ctrl-C / kill: don't leave the worker on the GPU
                    proc.kill()
                    proc.wait()
                    raise
                lf.flush()
                done = [w for w in todo if row_path(w).exists()]
                todo = [w for w in todo if not row_path(w).exists()]
                if rc == 0 or not todo:
                    break
                # A hard crash (host OOM-kill, CUDA context poisoned, segfault) lost the workload in
                # flight. Record it as failed and continue the rest in a fresh process.
                idle = 0 if done else idle + 1
                if idle >= 2:                            # crashes before producing anything: give up
                    for w in todo:
                        crashes.append(w["name"])
                        V.write_row(run_dir / "raw", _crash_row(c, w, model, rc, "gave up after repeated crashes"))
                    todo = []
                    break
                w = todo.pop(0)
                crashes.append(w["name"])
                V.write_row(run_dir / "raw", _crash_row(c, w, model, rc, "worker crashed on this workload"))
                print(f"  {V.Workload(**w).label:<34} worker crashed (exit {rc}); continuing in a fresh process",
                      flush=True)
        facts = json.loads(fpath.read_text()) if fpath.exists() else \
            dict(status="failed", error=f"worker exited {rc} before writing facts")
        if crashes:
            facts["crashed_workloads"] = crashes
        if facts.get("status") != "ok":
            print(f"  !! {facts.get('error')}  (log: {log})")
            print(V.explain_failure(log))
        print(f"  ({time.perf_counter() - t0:.0f}s)")
        merge_summary(run_dir, None, [], {c["name"]: facts})
        V.aggregate(run_dir)

    V.aggregate(run_dir)
    V.write_report(run_dir, a.plots)
    print(f"\n[pytorch] done -> {run_dir}/report.md")


def _crash_row(c: dict, w: dict, model: str, rc: int, why: str) -> dict:
    wl = V.Workload(**w)
    hint = " (signal 9: likely host OOM-kill)" if rc in (-9, 137) else ""
    return dict(config=c["name"], group=c["group"], knob=c["knob"], model=model, workload=wl.name,
                workload_label=wl.label, dataset=wl.dataset, input_len=wl.input_len, output_len=wl.output_len,
                prefix_len=wl.prefix_len, concurrency=wl.concurrency, num_prompts=wl.num_prompts, repeat=0,
                bench_rc=1.0, bench_error=f"{why}: exit {rc}{hint}", preemptions=float("nan"),
                prefix_hit_rate=float("nan"), spec_accept_rate=float("nan"))


if __name__ == "__main__":
    main()
