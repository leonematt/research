"""vLLM side: one fresh `vllm serve` per config, its workloads, Prometheus deltas, profiling.

Nothing from vllm is imported: everything goes through the CLI and HTTP, so the sweep survives
vLLM version churn as long as the flags do (and the functional check catches flags that don't).
"""
from __future__ import annotations

import argparse
import json
import math
import numbers
import os
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

from . import client, profiling, sampling, tailor
from .configs import Config, select_workloads
from .hardware import nvcc_env, vllm_version, wait_gpu_free, wait_gpu_settle, gpu_mem_state
from .metrics import parse_slo, summarize
from .results import aggregate, base_row, write_row
from .util import drop_page_cache, http_get, http_post_json, meminfo_gib, write_json
from .workloads import NATIVE_KINDS, Workload, make_prompts, regular_vocab, resolve_rates

# ----------------------------------------------------------------------------
# server log facts (what the engine actually resolved; proves each knob took effect)
# ----------------------------------------------------------------------------

LOG_FACTS = {
    "kv_cache_tokens": r"KV cache size:\s*([\d,]+)\s*tokens",
    "max_concurrency_x": r"Maximum concurrency for\s*[\d,]+\s*tokens per request:\s*([\d.]+)x",
    "weights_load_s": r"Loading weights took\s*([\d.]+)\s*seconds",
    "model_load_gib": r"Model loading took\s*([\d.]+)\s*Gi?B",
    "compile_s": r"torch\.compile (?:takes|took)\s*([\d.]+)\s*s",
    "engine_init_s": r"init engine \([^)]*\) took\s*([\d.]+)\s*s",
    "cudagraph_capture_s": r"Graph capturing finished in\s*([\d.]+)\s*secs",
    # from the "Initializing a V1 LLM engine" config dump
    "cfg_compile_mode": r"'mode': <CompilationMode\.(\w+)",
    "cfg_cudagraph_mode": r"'cudagraph_mode': <CUDAGraphMode\.(\w+)",
    "cfg_custom_ops": r"'custom_ops': \[([^\]]*)\]",
    "cfg_fuse_norm_quant": r"'fuse_norm_quant': (\w+)",
    "cfg_enforce_eager": r"enforce_eager=(\w+)",
    "cfg_kv_cache_dtype": r"kv_cache_dtype=(\w+)",
    "cfg_prefix_caching": r"enable_prefix_caching=(\w+)",
    "cfg_chunked_prefill": r"(?:enable_chunked_prefill|chunked_prefill_enabled)=(\w+)",
    # engine config dump ("async_scheduling=True"), or the scheduler's own startup line
    "cfg_async_scheduling": r"async_scheduling['\"]?\s*[=:]\s*['\"]?(\w+)|(?:Async|Asynchronous) scheduling (?:is )?(enabled|disabled)",
    "cfg_quantization": r"\bquantization=(\w+)",
    "cfg_speculative": r"speculative_config=(\w+)",
    "cfg_tp": r"tensor_parallel_size=(\d+)",
    "cfg_pp": r"pipeline_parallel_size=(\d+)",
    "cfg_mnbt": r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)",
    # "Using FlashInfer backend", "Using FLASH_ATTN attention backend out of potential backends",
    # "Using AttentionBackendEnum.FLASHINFER backend." (vLLM >= 0.15, chosen with --attention-backend)
    "attention_backend": r"Using\s+(?:AttentionBackendEnum\.)?([A-Za-z0-9_]+(?: Attention)?)\s+(?:attention\s+)?backend",
}


def parse_server_log(path: Path) -> dict:
    facts: dict = {}
    try:
        txt = path.read_text(errors="replace")
    except Exception:  # noqa: BLE001
        return facts
    for key, pat in LOG_FACTS.items():
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if not m:
            continue
        raw = next(g for g in m.groups() if g is not None)
        raw = {"enabled": "True", "disabled": "False"}.get(raw.lower(), raw)
        val = raw.replace(",", "") if key != "cfg_custom_ops" else raw
        if key == "attention_backend" or key.startswith("cfg_"):
            facts[key] = val.strip().strip("'\"") if key != "cfg_custom_ops" else val.replace("'", "").strip()
        else:
            try:
                facts[key] = float(val)
            except ValueError:
                facts[key] = val
    # compiled graph loaded from vLLM's on-disk cache (fast start) vs compiled fresh
    facts["compile_cached"] = bool(re.search(
        r"Directly load AOT compilation|from standalone compile artifacts|Loading .*compil.* from cache",
        txt, flags=re.IGNORECASE))
    return facts


def tail(path: Path, n: int = 25) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except Exception:  # noqa: BLE001
        return ""


# (pattern in log, human hint) for the failures that actually happen
KNOWN_FAILURES = [
    (r"GatedRepoError|401 Client Error|Cannot access gated repo|Access to model .* is restricted",
     "Gated model: accept the license on the model's HF page, then `export HF_TOKEN=hf_...` "
     "or run `hf auth login`."),
    (r"RepositoryNotFoundError|404 Client Error.*huggingface",
     "Model id not found on the Hub: check the id, or pass MODEL=<id or local path>."),
    (r"Could not find nvcc|nvcc: not found|No such file or directory: .nvcc",
     "FlashInfer needs the CUDA toolkit (nvcc) to JIT kernels for this GPU: `./run.sh cuda`, "
     "or point CUDA_HOME at an existing toolkit."),
    (r"Free memory on device .* is less than desired GPU memory utilization",
     "Not enough free GPU memory at server start: vLLM needs gpu_mem x total free AFTER creating "
     "its own CUDA context. Something else holds memory (desktop/browser on a display GPU, a "
     "leftover process; on unified-memory boxes also the page cache). Lower GPU_MEM by ~0.05, or "
     "close it (`nvidia-smi` lists GPU processes)."),
    (r"CUDA error: out of memory|CUDA out of memory|OutOfMemoryError|cudaErrorMemoryAllocation"
     r"|No available memory for the cache blocks",
     "OOM. Check nothing else holds memory (`nvidia-smi`; on unified-memory boxes `free -g` and "
     "the page cache, which --drop-caches clears). Otherwise lower GPU_MEM / --max-model-len, "
     "or pick a smaller --model."),
    (r"unrecognized arguments|error: argument|invalid choice|Extra inputs are not permitted|validation error",
     "A CLI flag or config key isn't accepted by this vLLM version; see the line in the log and "
     "adjust the config in llmbench/configs.py."),
    (r"FlashInfer backend is not available",
     "vLLM treats FlashInfer as missing: its has_flashinfer() needs `nvcc` on PATH (or the "
     "flashinfer-cubin package). Run via ./run.sh (which adds the toolkit to PATH) or "
     "`export PATH=/usr/local/cuda/bin:$PATH`."),
    (r"recompile limit|Hard failure due to (fullgraph=True|fail_on_recompile_limit_hit)",
     "torch.compile hit Dynamo's recompile limit: every new input shape forced a recompile, so the "
     "compiled path never settles. Typical of static-shape compile on varying batch/sequence sizes."),
    (r"requires `?accelerate`?|No module named '(accelerate|torchao|kernels)'",
     "Missing Python package in the venv: `./run.sh setup` (re)installs it."),
    # keep this generic one last: explain_failure() drops it when a specific hint matched
    (r"not supported|NotImplementedError|Unsupported",
     "Feature/arch not supported by this build (check the line above for which)."),
]


def explain_failure(path: Path, n: int = 30) -> str:
    try:
        txt = path.read_text(errors="replace")
    except Exception:  # noqa: BLE001
        return "  (no log)"
    lines = txt.splitlines()
    err_re = re.compile(r"Error|Exception|Traceback|error:|FAILED")
    err_lines = [ln for ln in lines if err_re.search(ln)]
    out = []
    first_err = next((i for i, ln in enumerate(lines) if err_re.search(ln)), None)
    # match causes against the failure itself, not startup WARNINGs ("X not supported, falling back")
    failure = "\n".join(lines[first_err:]) if first_err is not None else "\n".join(lines[-n:])
    hints = [h for pat, h in KNOWN_FAILURES if re.search(pat, failure, flags=re.IGNORECASE)]
    if len(hints) > 1:
        hints = [h for h in hints if h != KNOWN_FAILURES[-1][1]]
    if first_err is not None:
        stage = next((lines[i] for i in range(first_err - 1, -1, -1) if re.search(r"(^|\s)INFO\s", lines[i])), None)
        if stage:
            out.append(f"  last step before failure: {stage.strip()[:200]}")
    if hints:
        out.append("  likely cause:")
        out += [f"    -> {h}" for h in hints[:2]]
    out.append(f"  error lines ({len(err_lines)}):" if err_lines else f"  last {n} log lines:")
    out += ["    " + ln[:240] for ln in (err_lines[-12:] if err_lines else lines[-n:])]
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Prometheus counters
# ----------------------------------------------------------------------------

METRIC_RE = re.compile(r"^(vllm:[A-Za-z0-9_]+)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|[+-]?Inf)\s*$")


def scrape_metrics(port: int) -> dict[str, float]:
    """Sum every vllm:* sample over its label sets (counters and gauges only)."""
    status, txt = http_get(f"http://127.0.0.1:{port}/metrics", timeout=10)
    out: dict[str, float] = {}
    if status != 200:
        return out
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        m = METRIC_RE.match(line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        if not math.isnan(v):
            out[m.group(1)] = out.get(m.group(1), 0.0) + v
    return out


def _metric(metrics: dict[str, float], *candidates: str) -> float:
    for c in candidates:
        if c in metrics:
            return metrics[c]
    for c in candidates:            # renamed metrics: substring fallback
        stem = c.replace("vllm:", "")
        for k, v in metrics.items():
            if stem in k:
                return v
    return float("nan")


def metric_deltas(before: dict, after: dict) -> dict:
    def d(*names):
        return _metric(after, *names) - _metric(before, *names)
    pq, ph = d("vllm:prefix_cache_queries_total"), d("vllm:prefix_cache_hits_total")
    drafts = d("vllm:spec_decode_num_drafts_total")
    draft_tok = d("vllm:spec_decode_num_draft_tokens_total")
    acc_tok = d("vllm:spec_decode_num_accepted_tokens_total")
    return dict(
        preemptions=d("vllm:num_preemptions_total"),
        prefix_cache_queries=pq, prefix_cache_hits=ph,
        prefix_hit_rate=(ph / pq) if pq and pq > 0 else float("nan"),
        spec_drafts=drafts, spec_draft_tokens=draft_tok, spec_accepted_tokens=acc_tok,
        spec_accept_rate=(acc_tok / draft_tok) if draft_tok and draft_tok > 0 else float("nan"),
        spec_accepted_per_draft=(acc_tok / drafts) if drafts and drafts > 0 else float("nan"),
    )


# ----------------------------------------------------------------------------
# server lifecycle
# ----------------------------------------------------------------------------

class Server:
    def __init__(self, cfg: Config, model: str, a: argparse.Namespace, log_path: Path, profile_dir: Optional[Path]):
        self.cfg, self.model, self.a, self.log_path = cfg, model, a, log_path
        self.profile_dir = profile_dir
        self.proc: Optional[subprocess.Popen] = None
        self.logf = None
        self.port = a.port
        self._reset_warned = False

    @property
    def n_gpus(self) -> int:
        return self.cfg.gpus(self.a.tp)

    def command(self) -> list[str]:
        p = self.cfg.parallel
        # pre-quantized checkpoints carry their own activation dtype (AWQ is fp16)
        dtype = "auto" if self.cfg.model_override else self.a.dtype
        cmd = ["vllm", "serve", self.model, "--host", "127.0.0.1", "--port", str(self.port),
               "--dtype", dtype, "--max-model-len", str(self.a.max_model_len), "--seed", "0",
               "--tensor-parallel-size", str(p.get("tp", self.a.tp))]
        if int(p.get("pp", 1)) > 1:
            cmd += ["--pipeline-parallel-size", str(p["pp"])]
        if int(p.get("dp", 1)) > 1:
            cmd += ["--data-parallel-size", str(p["dp"])]
        if "--gpu-memory-utilization" not in self.cfg.args:
            cmd += ["--gpu-memory-utilization", str(self.a.gpu_mem)]
        # multimodal families: text-only benchmarks shouldn't pay for the encoders
        if self.a.mm_off == "language-model-only":
            cmd += ["--language-model-only"]
        elif self.a.mm_off == "limit":
            cmd += ["--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0})]
        if self.profile_dir is not None and _profiler_flag():
            cmd += ["--profiler-config", json.dumps({
                "profiler": "torch", "torch_profiler_dir": str(self.profile_dir.resolve()),
                "torch_profiler_with_stack": False, "torch_profiler_record_shapes": False})]
        cmd += self.cfg.args
        if self.a.extra_server_args:
            cmd += shlex.split(self.a.extra_server_args)
        return cmd

    def gpu_mem(self) -> float:
        args = self.cfg.args
        if "--gpu-memory-utilization" in args:
            return float(args[args.index("--gpu-memory-utilization") + 1])
        return float(self.a.gpu_mem)

    def env(self) -> dict:
        e = dict(os.environ)
        e.update(nvcc_env(e))
        # /reset_prefix_cache is a dev-only route (VLLM_SERVER_DEV_MODE=1); without it the reset
        # silently 404s and repeat 2 re-sends repeat 1's prompts into a warm cache
        e.setdefault("VLLM_SERVER_DEV_MODE", "1")
        if self.profile_dir is not None and not _profiler_flag():
            e["VLLM_TORCH_PROFILER_DIR"] = str(self.profile_dir.resolve())     # vLLM < 0.13
        e.update(self.cfg.env)
        e.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        return e

    def start(self) -> dict:
        if self.a.drop_caches:
            before = meminfo_gib()
            dropped = drop_page_cache()
            after = meminfo_gib()
            mem = dict(page_cache_dropped=dropped, page_cache_gib_before=before.get("page_cache_gib"), **after)
            if before and after:
                print(f"  [mem] available {before['mem_available_gib']:.0f} -> {after['mem_available_gib']:.0f} GiB "
                      f"of {after['mem_total_gib']:.0f} (page cache {before['page_cache_gib']:.0f} -> "
                      f"{after['page_cache_gib']:.0f} GiB)")
        else:
            mem = meminfo_gib()
        hw = self.a.hw
        if hw is not None and not hw.unified:
            ids = hw.smi_ids(self.n_gpus)
            warn = wait_gpu_free(self.gpu_mem(), ids)
            if warn:
                print(f"  [mem] {warn}")
            st = gpu_mem_state(ids)
            if st:
                mem["gpu_free_gib_at_start"] = round(min(t - u for u, t in st.values()) / 1024, 2)
        t0 = time.time()
        self.logf = open(self.log_path, "w")
        self.logf.write("$ " + shlex.join(self.command()) + "\n")
        self.logf.write("env: " + json.dumps(self.cfg.env) + "\n\n")
        self.logf.flush()
        self.proc = subprocess.Popen(self.command(), stdout=self.logf, stderr=subprocess.STDOUT,
                                     env=self.env(), start_new_session=True)
        deadline = t0 + self.a.startup_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return dict(status="failed", startup_s=round(time.time() - t0, 1),
                            error=f"vllm serve exited with {self.proc.returncode}", log_tail=tail(self.log_path), **mem)
            st, _ = http_get(f"http://127.0.0.1:{self.port}/health", timeout=3)
            if st == 200:
                break
            time.sleep(2)
        else:
            self.stop()
            return dict(status="failed", startup_s=round(time.time() - t0, 1), error="startup timeout",
                        log_tail=tail(self.log_path), **mem)
        ok = self._probe()
        facts = dict(status="ok" if ok else "failed", startup_s=round(time.time() - t0, 1), **mem)
        if not ok:
            facts["error"] = "health ok but completion probe failed"
            facts["log_tail"] = tail(self.log_path)
        facts.update(parse_server_log(self.log_path))
        return facts

    def _probe(self) -> bool:
        payload = {"model": self.model, "prompt": "Hello", "max_tokens": 4, "temperature": 0}
        for _ in range(10):
            st, _ = http_post_json(f"http://127.0.0.1:{self.port}/v1/completions", payload, timeout=120)
            if st == 200:
                return True
            time.sleep(3)
        return False

    def reset_prefix_cache(self) -> bool:
        """True once vLLM confirms the cache is empty. It answers success=false while blocks are
        still held (requests finishing), so retry briefly; a 404 means dev mode is off."""
        st = 0
        for _ in range(20):
            st, body = http_post_json(f"http://127.0.0.1:{self.port}/reset_prefix_cache", timeout=30)
            if st == 200 and (not isinstance(body, dict) or body.get("success", True)):
                return True
            if st in (404, 405):
                break
            time.sleep(1)
        if not self._reset_warned:
            self._reset_warned = True
            print(f"  [warn] prefix-cache reset failed (HTTP {st}): runs after the first can hit cached "
                  "prompts; rows are marked prefix_reset_ok=0")
        return False

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=30)
        self.proc = None
        if self.logf:
            self.logf.close()
            self.logf = None
        hw = self.a.hw
        wait_gpu_settle(self.a.settle_s, hw.smi_ids(self.n_gpus) if hw and not hw.unified else None)


def _profiler_flag() -> bool:
    v = vllm_version()
    return v is None or v >= (0, 13)


# ----------------------------------------------------------------------------
# one workload
# ----------------------------------------------------------------------------

def bench_command(a: argparse.Namespace, model: str, tokenizer: str, wl: Workload, out_dir: Path, fname: str,
                  num_prompts: Optional[int] = None, meta: Optional[dict] = None, rep: int = 0) -> list[str]:
    cmd = ["vllm", "bench", "serve", "--backend", "vllm", "--host", "127.0.0.1", "--port", str(a.port),
           "--model", model, "--tokenizer", tokenizer,
           "--num-prompts", str(num_prompts or wl.num_prompts),
           "--ignore-eos",
           "--seed", str(a.seed + 1000 * rep),     # repeat r gets its own prompts; every config the same ones
           "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
           "--save-result", "--result-dir", str(out_dir), "--result-filename", fname]
    if wl.closed_loop:
        cmd += ["--max-concurrency", str(wl.concurrency), "--request-rate", "inf"]
    else:
        cmd += ["--request-rate", str(wl.request_rate)]       # Poisson arrivals (burstiness 1)
    cmd += wl.bench_args()
    if a.slo:
        cmd += ["--goodput"] + a.slo.split()
    if meta:
        cmd += ["--metadata"] + [f"{k}={v}" for k, v in meta.items()]
    if a.extra_bench_args:
        cmd += shlex.split(a.extra_bench_args)
    return cmd


def flatten_bench_json(path: Path) -> dict:
    """Every scalar numeric field of a `vllm bench serve` result."""
    row: dict = {}
    try:
        data = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        return {"bench_parse_error": str(e)}
    for k, v in data.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, numbers.Number):
            row[k] = float(v)
        elif k == "errors" and isinstance(v, list):
            row["n_errors"] = float(len([x for x in v if x]))
    return row


class PromptCache:
    """Tokenizer + vocab loaded once per run (built-in client workloads)."""

    def __init__(self, model: str):
        self.model = model
        self._tok = None
        self._vocab = None
        self._loaded = False

    def get(self):
        if not self._loaded:
            self._loaded = True
            self._tok = client.load_tokenizer(self.model)
            self._vocab = regular_vocab(self._tok, client.model_vocab_size(self.model))
        return self._tok, self._vocab


def run_workload(a: argparse.Namespace, server: Server, cfg: Config, model: str, wl: Workload, rep: int,
                 raw_dir: Path, prompts_cache: PromptCache) -> dict:
    fname = f"{cfg.name}__{wl.name}__r{rep}.json"
    json_path = raw_dir / fname
    row_path = raw_dir / f"{cfg.name}__{wl.name}__r{rep}.row.json"
    base = base_row(cfg, wl, model, rep, cfg.gpus(a.tp))
    native = cfg.client == "waves" or wl.kind in NATIVE_KINDS

    if a.resume and row_path.exists():
        row = json.loads(row_path.read_text())
        if not row.get("bench_rc"):
            row["resumed"] = 1.0
            return row

    reset_ok = server.reset_prefix_cache()
    before = scrape_metrics(a.port)
    t0 = time.time()
    row = dict(base)
    if native:
        tok, vocab = prompts_cache.get()
        prompts = make_prompts(wl, wl.num_prompts, vocab, a.seed + 1000 * rep, tokenizer=tok)
        try:
            res = client.run("127.0.0.1", a.port, model, prompts, wl.concurrency,
                             mode="waves" if cfg.client == "waves" else "closed", timeout=a.bench_timeout)
            row.update(summarize(res["reqs"], res["duration"], parse_slo(a.slo)))
            row.update(n_errors=float(res["n_errors"]), bench_rc=0.0, client="waves" if cfg.client == "waves" else "native")
            if res["n_errors"]:
                row["bench_error"] = "; ".join(res["errors"])
        except Exception as e:  # noqa: BLE001
            row.update(bench_rc=1.0, bench_error=f"{type(e).__name__}: {e}"[:300], client="native")
            print(f"      client failed: {row['bench_error'][:160]}")
    else:
        cmd = bench_command(a, model, a.model, wl, raw_dir, fname, rep=rep,
                            meta=dict(config=cfg.name, workload=wl.label, repeat=rep, prefix_reset_ok=float(reset_ok)))
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=a.bench_timeout)
            rc, err_tail = res.returncode, "\n".join((res.stderr or "").splitlines()[-15:])
        except subprocess.TimeoutExpired:
            rc, err_tail = -1, "bench timeout"
        row["bench_rc"] = float(rc)
        row["client"] = "vllm-bench"
        if rc == 0 and json_path.exists():
            row.update(flatten_bench_json(json_path))
        else:
            row["bench_error"] = err_tail
            print(f"      bench failed rc={rc}: {err_tail.splitlines()[-1] if err_tail else ''}")
    after = scrape_metrics(a.port)
    row.update(metric_deltas(before, after))
    row.update(bench_wall_s=time.time() - t0, prefix_reset_ok=float(reset_ok))
    write_row(raw_dir, row)
    return row


def warmup(a: argparse.Namespace, model: str, wl: Workload, tmp_dir: Path) -> None:
    cmd = bench_command(a, model, a.model, wl, tmp_dir, "warmup.json", num_prompts=a.warmup_prompts)
    subprocess.run(cmd, capture_output=True, text=True, timeout=a.bench_timeout)
    (tmp_dir / "warmup.json").unlink(missing_ok=True)


def profile_pass(a: argparse.Namespace, server: Server, cfg: Config, model: str, ref: Workload,
                 run_dir: Path, prompts_cache: PromptCache) -> Optional[dict]:
    """One short profiled run at the reference shape, AFTER the measured runs (so profiler overhead
    never touches a measured number). Returns the kernel breakdown, or None."""
    if server.profile_dir is None:
        return None
    tok, vocab = prompts_cache.get()
    n = max(2, min(ref.concurrency * 2, 64))
    w = Workload(ref.name, ref.input_len, min(ref.output_len, 64), ref.concurrency, n, ref.prefix_len, kind="grid")
    prompts = make_prompts(w, n, vocab, a.seed + 777, tokenizer=tok)
    st, _ = http_post_json(f"http://127.0.0.1:{a.port}/start_profile", {}, timeout=120)
    if st != 200:
        print(f"  [profile] /start_profile answered HTTP {st}; skipping")
        return None
    t0 = time.perf_counter()
    try:
        client.run("127.0.0.1", a.port, model, prompts, ref.concurrency, timeout=a.bench_timeout)
    finally:
        window_s = time.perf_counter() - t0
        http_post_json(f"http://127.0.0.1:{a.port}/stop_profile", {}, timeout=1800)
    traces = profiling.find_traces(server.profile_dir)
    if not traces:
        print(f"  [profile] no trace files under {server.profile_dir}")
        return None
    bd = profiling.breakdown(traces[0], window_s=window_s)
    bd.update(config=cfg.name, engine="vllm", workload=w.label, trace=str(traces[0].relative_to(run_dir)))
    write_json(run_dir / "raw" / f"{cfg.name}__profile.json", bd)
    print(f"  [profile] {bd.get('n_kernels', 0)} kernels, GPU busy {bd.get('gpu_busy_pct', float('nan')):.0f}% "
          f"-> {traces[0].name}")
    return bd


# ----------------------------------------------------------------------------
# the vLLM sweep
# ----------------------------------------------------------------------------

def reference_capacity(rows: list[dict], ref: tuple[int, int, int]) -> Optional[float]:
    """req/s of the vLLM default at the reference shape and its highest measured concurrency:
    the capacity that rate workloads are expressed as fractions of."""
    ri, ro, _ = ref
    pts = [(r["concurrency"], r.get("request_throughput")) for r in rows
           if r.get("input_len") == ri and r.get("output_len") == ro and r.get("kind") in ("grid", "reference")
           and not r.get("bench_rc") and r.get("request_throughput")]
    if not pts:
        return None
    return float(max(pts)[1])


def existing_reps(raw_dir: Path, cfg: str, wl: str) -> dict[int, dict]:
    out = {}
    for f in raw_dir.glob(f"{cfg}__{wl}__r*.row.json"):
        try:
            r = json.loads(f.read_text())
            out[int(r.get("repeat") or 0)] = r
        except Exception:  # noqa: BLE001
            pass
    return out


def retire_failed(raw_dir: Path, cfg: str, wl: str) -> int:
    """Move failed repeats aside (raw/retired/) so a retry replaces them; kept for the audit."""
    n = 0
    ret = raw_dir / "retired"
    for rep, r in existing_reps(raw_dir, cfg, wl).items():
        if not sampling.good(r):
            ret.mkdir(exist_ok=True)
            for f in raw_dir.glob(f"{cfg}__{wl}__r{rep}.*"):
                f.replace(ret / f.name)
            n += 1
    return n


REJECTED_FLAG = re.compile(r"unrecognized arguments:\s*(--?[\w-]+)|Extra inputs are not permitted.*?\b(\w+)\b|"
                           r"unexpected keyword argument '(\w+)'", re.S)


def rebuild_without_rejected(a: argparse.Namespace, cfg: Config, log_path: Path) -> Optional[Config]:
    """A composite (vllm_minimal, ladder, optimized) whose server rejected one component's flag is
    rebuilt without that component and run again, instead of losing the whole anchor."""
    if not cfg.components or getattr(a, "rebuild_composite", None) is None or cfg.note.count("rebuilt") > 0:
        return None
    txt = log_path.read_text(errors="replace") if log_path.exists() else ""
    m = REJECTED_FLAG.search(txt)
    if not m:
        return None
    bad = next(g for g in m.groups() if g)
    culprit = a.rebuild_composite_lookup(cfg, bad)
    if not culprit:
        return None
    new = a.rebuild_composite(cfg.name, {culprit})
    if new is None or new.components == cfg.components:
        return None
    new.note = (new.note + "; " if new.note else "") + f"rebuilt: this vLLM rejected `{bad}` ({culprit})"
    print(f"  [rebuild] {cfg.name}: vLLM rejected `{bad}`; retrying without {culprit} "
          f"(components now: {', '.join(new.components)})")
    log_path.replace(log_path.with_name(f"{cfg.name}.rejected.log"))
    return new


def run_vllm(a: argparse.Namespace, run_dir: Path, cfgs: list[Config], wls: list[Workload], summary: dict,
             save_summary, targets: Optional[dict[str, dict[str, int]]] = None) -> list[dict]:
    """Run every config's workloads. `targets` = retry mode: {config: {workload: extra good repeats}};
    only those pairs run, failed repeats are retired first, and the new repeats are appended."""
    raw_dir, log_dir = run_dir / "raw", run_dir / "server_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ref_t = tuple(summary["reference"])
    ref_wl = next(w for w in wls if w.kind == "reference")
    prompts_cache = PromptCache(a.model)
    policy = sampling.Policy(a.min_repeats, a.max_repeats, a.target_cv)
    if summary.get("rate_base_rps"):
        resolve_rates(wls, summary["rate_base_rps"])
    plan = []
    for c in cfgs:
        cw = select_workloads(c, wls, ref_t, a.all_workloads)
        if targets is not None:
            cw = [w for w in cw if w.name in targets.get(c.name, {})]
        if cw:
            plan.append((c, cw))
    n_runs = sum(len(p) for _, p in plan) * (policy.min_repeats if targets is None else 1)
    eta = sampling.ETA(n_runs + len(plan))          # + one "run" of time per server start
    mode = "retry" if targets is not None else "sweep"
    print(f"\n[vllm] {mode}: {len(plan)} configs, >= {n_runs} measured runs "
          f"(repeats {policy.min_repeats}..{policy.max_repeats}, target CV {policy.target_cv:.0%}) -> {run_dir}\n")
    rows: list[dict] = []
    oom_tries: dict[str, int] = {}
    ci = 0
    while ci < len(plan):
        cfg, cwls = plan[ci]
        ci += 1
        model = cfg.model_override or a.model
        log_path = log_dir / f"{cfg.name}.log"
        prof_dir = (run_dir / "profiles" / cfg.name) if (a.profile and targets is None) else None
        if prof_dir is not None:
            prof_dir.mkdir(parents=True, exist_ok=True)
        server = Server(cfg, model, a, log_path, prof_dir)
        print(f"=== [{ci}/{len(plan)}] {cfg.name}  ({cfg.tier}: {cfg.knob})  {eta}")
        if cfg.note:
            print(f"  note: {cfg.note}")
        if a.dry_run:
            print("  $ " + shlex.join(server.command()))
            if cfg.env:
                print("    env: " + json.dumps(cfg.env))
            for w in cwls:
                how = ("built-in client" + (" (waves)" if cfg.client == "waves" else "")) \
                    if (cfg.client == "waves" or w.kind in NATIVE_KINDS) else \
                    shlex.join(bench_command(a, model, a.model, w, raw_dir, "x.json"))
                print(f"    {w.label:<36} n={w.num_prompts:<4} {how if len(how) < 70 else how[:67] + '...'}")
            continue

        def satisfied(w: Workload) -> bool:
            have = list(existing_reps(raw_dir, cfg.name, w.name).values())
            return not sampling.need_more(have, len(have), policy)

        if targets is None and a.resume and all(satisfied(w) for w in cwls if w.kind != "rate" or w.request_rate):
            print("  all results present, reusing (no server start)")
            continue
        if targets is not None:
            for w in cwls:
                n = retire_failed(raw_dir, cfg.name, w.name)
                if n:
                    print(f"  retired {n} failed repeat(s) of {w.label}")
        if targets is not None and log_path.exists():          # keep the sweep's log next to the retry's
            log_path.replace(log_path.with_name(f"{cfg.name}.sweep.log"))
        try:
            facts = server.start()
            # A start that compiles fresh measures less free memory during profiling and gets a ~20%
            # smaller KV cache than one that loads the compiled graph from cache (RTX 5070 Ti: 21.8k vs
            # 25.9k tokens for configs with identical memory needs). For KV-limited workloads that
            # would masquerade as a knob effect, so every measured server starts warm.
            if (a.warm_start and facts.get("status") == "ok" and not facts.get("compile_cached")
                    and (facts.get("compile_s") or 0) >= 1.0):
                cold = facts
                print(f"  cold start: compiled for {cold['compile_s']}s, KV {int(cold.get('kv_cache_tokens') or 0):,} "
                      "tokens; restarting from the compile cache so every config is measured warm")
                server.stop()
                log_path.replace(log_path.with_name(f"{cfg.name}.cold.log"))
                facts = server.start()
                facts["cold_start"] = {k: cold.get(k) for k in ("startup_s", "compile_s", "kv_cache_tokens")}
        except BaseException:          # Ctrl-C / kill during startup: don't orphan the server
            server.stop()
            raise
        eta.tick()
        facts["gpu_count"] = cfg.gpus(a.tp)
        facts["gpu_mem_used"] = server.gpu_mem()
        if oom_tries.get(cfg.name):
            facts["oom_retries"] = oom_tries[cfg.name]
        if targets is not None:
            prev = summary["server_facts"].get(cfg.name, {})
            facts["retry_of"] = {k: prev.get(k) for k in ("status", "kv_cache_tokens", "startup_s") if k in prev}
        summary["server_facts"][cfg.name] = facts
        save_summary()
        if facts.get("status") != "ok":
            print(f"  !! server failed: {facts.get('error')}  (full log: {log_path})")
            print(explain_failure(log_path))
            server.stop()
            log_txt = log_path.read_text(errors="replace") if log_path.exists() else ""
            kind = tailor.oom_kind(log_txt)
            if kind and oom_tries.get(cfg.name, 0) < tailor.OOM_RETRIES:
                oom_tries[cfg.name] = oom_tries.get(cfg.name, 0) + 1
                new_mem = tailor.lower_gpu_mem(cfg, a.gpu_mem, raise_=kind == "starved")
                if new_mem is not None:
                    what = ("no memory left for the KV cache: memory fraction raised" if kind == "starved"
                            else "out of memory at startup: memory fraction lowered")
                    cfg.note = (cfg.note + "; " if cfg.note else "") + f"{what} to {new_mem}"
                    print(f"  [tailor] {what} to {new_mem}; retrying {cfg.name} "
                          f"(attempt {oom_tries[cfg.name]}/{tailor.OOM_RETRIES})")
                    log_path.replace(log_path.with_name(f"{cfg.name}.oom{oom_tries[cfg.name]}.log"))
                    summary["configs"] = [cfg.to_dict() if c["name"] == cfg.name else c for c in summary["configs"]]
                    save_summary()
                    plan.insert(ci, (cfg, cwls))
                    continue
            rebuilt = rebuild_without_rejected(a, cfg, log_path)
            if rebuilt is not None:
                gone = set(cfg.components) - set(rebuilt.components)
                plan.insert(ci, (rebuilt, cwls))      # runs next, under the same name
                for j in range(ci + 1, len(plan)):    # pending composites with the same component too
                    c2, w2 = plan[j]
                    if gone & set(c2.components):
                        plan[j] = (a.rebuild_composite(c2.name, set()) or c2, w2)
                        plan[j][0].note = (plan[j][0].note + "; " if plan[j][0].note else "") + \
                            f"rebuilt without {', '.join(sorted(gone))} (rejected by this vLLM)"
                fixed = {c.name: c for c, _ in plan[ci:] if gone & (set(cfg.components) - set(c.components)) or c is rebuilt}
                summary["configs"] = [fixed[c["name"]].to_dict() if c["name"] in fixed else c for c in summary["configs"]]
                save_summary()
            continue
        print(f"  server up in {facts['startup_s']}s"
              + (f", KV cache {int(facts['kv_cache_tokens']):,} tokens" if "kv_cache_tokens" in facts else "")
              + (f", compile {facts['compile_s']}s" if "compile_s" in facts else "")
              + (f", graphs {facts['cudagraph_capture_s']}s" if "cudagraph_capture_s" in facts else ""))
        try:
            warmup(a, model, ref_wl, run_dir)
            # closed-loop workloads first: the default's capacity sizes the rate workloads
            ordered = [w for w in cwls if w.kind != "rate"] + [w for w in cwls if w.kind == "rate"]
            cfg_rows: list[dict] = []
            quiet_cvs: list[float] = []          # run-to-run spread seen on this config so far
            for w in ordered:
                if w.kind == "rate" and not w.request_rate:
                    if cfg.name == "vllm_default":
                        cap = reference_capacity(cfg_rows, ref_t)
                        if cap:
                            summary["rate_base_rps"] = cap
                            resolve_rates(wls, cap)
                            save_summary()
                            print(f"  [rates] vLLM-default capacity at the reference shape: {cap:.2f} req/s; "
                                  f"rate workloads: {', '.join(f'{x.request_rate:g}' for x in wls if x.kind == 'rate')} req/s")
                    if not w.request_rate:
                        print(f"  {w.label:<36} skipped: no vLLM-default capacity measured yet (pass --rate-base)")
                        continue
                have = existing_reps(raw_dir, cfg.name, w.name)
                if targets is None and not a.resume:
                    have = {}
                force = targets[cfg.name][w.name] if targets is not None else 0
                base_good = sum(1 for r in have.values() if sampling.good(r)) if targets is not None else 0
                p = sampling.Policy(base_good + max(1, force) if targets is not None else policy.min_repeats,
                                    base_good + max(1, force) + 1 if targets is not None else policy.max_repeats,
                                    policy.target_cv)
                if targets is None and sampling.is_quiet(quiet_cvs, policy):
                    p = sampling.Policy(1, policy.max_repeats, policy.target_cv)
                    eta.add(-(policy.min_repeats - 1))
                attempted = 0
                while sampling.need_more(list(have.values()), len(have) if targets is None else base_good + attempted, p):
                    rep = max(have, default=-1) + 1
                    t0 = time.time()
                    row = run_workload(a, server, cfg, model, w, rep, raw_dir, prompts_cache)
                    if targets is not None:
                        row["retry"] = 1.0
                        write_row(raw_dir, row)
                    have[rep] = row
                    attempted += 1
                    rows.append(row)
                    cfg_rows.append(row)
                    eta.tick()
                    if attempted > (p.min_repeats if targets is None else 1):
                        eta.add(1)
                    spread = sampling.cv([r.get("output_throughput") for r in have.values() if sampling.good(r)])
                    print(f"  {w.label:<36} r{rep} out={row.get('output_throughput', float('nan')):8.1f} tok/s  "
                          f"ttft50={row.get('median_ttft_ms', float('nan')):7.1f}ms  "
                          f"tpot50={row.get('median_tpot_ms', float('nan')):6.2f}ms  "
                          f"pre={row.get('preemptions', float('nan')):.0f}"
                          + (f"  cv={spread:.1%}" if spread == spread else "")
                          + f"  ({time.time() - t0:.0f}s, {eta})")
                spread_w = sampling.cv([r.get("output_throughput") for r in have.values() if sampling.good(r)])
                if spread_w == spread_w:
                    was = sampling.is_quiet(quiet_cvs, policy)
                    quiet_cvs.append(spread_w)
                    if not was and sampling.is_quiet(quiet_cvs, policy) and policy.min_repeats > 1:
                        print(f"  [sampling] {cfg.name} is quiet (run-to-run spread "
                              f"{max(quiet_cvs):.1%} <= {policy.target_cv / 2:.1%}): 1 repeat for its remaining "
                              "workloads; the anomaly walk still re-checks outliers")
                aggregate(run_dir)
            if prof_dir is not None:
                try:
                    bd = profile_pass(a, server, cfg, model, ref_wl, run_dir, prompts_cache)
                    if bd:
                        facts["gpu_busy_pct"] = bd.get("gpu_busy_pct")
                except Exception as e:  # noqa: BLE001
                    print(f"  [profile] failed: {type(e).__name__}: {e}")
        except KeyboardInterrupt:
            print("\n[vllm] interrupted; stopping server and writing partial results")
            server.stop()
            break
        finally:
            server.stop()
        save_summary()
    return rows
