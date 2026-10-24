"""PyTorch side orchestrator: one worker process per config (see torch_worker.py), crash
recovery, and the same row files / summary facts as the vLLM side."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from . import sampling
from .configs import Config, select_workloads
from .hardware import Hardware
from .models import derived
from .results import aggregate, base_row, write_row
from .util import drop_page_cache
from .vllm_runner import explain_failure
from .workloads import Workload


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def fake_mode() -> bool:
    return os.environ.get("LLMBENCH_FAKE_TORCH") == "1"


def apply_skips(cfgs: list[Config], a: argparse.Namespace, hw: Optional[Hardware], has_c1: bool) -> None:
    """Set cfg.skip for PyTorch configs this environment can't run, with the reason."""
    fake = fake_mode()
    no_acc = None
    if not fake and a.device == "cuda" and not _has("accelerate"):
        if a.tp > 1:
            no_acc = "multi-GPU loading needs accelerate (./run.sh setup)"
        elif hw is not None and hw.unified:
            no_acc = ("unified memory needs accelerate to load straight to the GPU; the CPU->GPU path holds "
                      "two copies (./run.sh setup)")
    has_cb = fake
    if not fake:
        try:
            from transformers import GenerationMixin
            has_cb = hasattr(GenerationMixin, "init_continuous_batching")
        except Exception:  # noqa: BLE001
            has_cb = False
    draft = (derived(a.model, "draft") or {}).get("model")
    for c in cfgs:
        pt = c.pt
        skip = no_acc
        if skip:
            pass
        elif pt.get("quant") and not fake and not _has("torchao"):
            skip = "torchao not installed (./run.sh setup)"
        elif pt.get("quant") == "int4" and not fake and not _has("mslk"):
            skip = "torchao int4 needs the mslk package (try: uv pip install mslk)"
        elif pt.get("kernels") and not fake and not _has("kernels"):
            skip = "the `kernels` package isn't installed (uv pip install kernels)"
        elif pt.get("assisted") and not draft:
            skip = "no draft model for this model family"
        elif pt.get("assisted") and not has_c1:
            skip = "needs a concurrency-1 workload"
        elif pt.get("mode") == "cb" and not has_cb:
            skip = "transformers has no continuous batching (needs v5.8+)"
        elif pt.get("compile") and a.tp > 1:
            skip = "compile path is single-GPU here (multi-GPU uses device_map='auto' layer split)"
        c.skip = c.skip or skip


def alloc_env() -> dict:
    """Expandable segments: fewer fragmentation OOMs when weights are rewritten in place
    (torchao quantization) on a GPU with little headroom. A user-set allocator config wins."""
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or os.environ.get("PYTORCH_ALLOC_CONF"):
        return {}
    return {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def _next_rep(raw: Path, c: Config, w: Workload) -> int:
    reps = [int(f.name.rsplit("__r", 1)[1].split(".")[0]) for f in raw.glob(f"{c.name}__{w.name}__r*.row.json")]
    return max(reps, default=-1) + 1


def _crash_row(c: Config, w: Workload, model: str, rc: int, why: str, rep: int, gpus: int) -> dict:
    hint = " (signal 9: likely host OOM-kill)" if rc in (-9, 137) else ""
    row = base_row(c, w, model, rep, gpus)
    row.update(bench_rc=1.0, bench_error=f"{why}: exit {rc}{hint}", preemptions=float("nan"),
               prefix_hit_rate=float("nan"), spec_accept_rate=float("nan"))
    return row


def run_pytorch(a: argparse.Namespace, run_dir: Path, cfgs: list[Config], wls: list[Workload], summary: dict,
                save_summary, targets: Optional[dict[str, dict[str, int]]] = None) -> None:
    raw = run_dir / "raw"
    logs = run_dir / "server_logs"
    raw.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    hw = a.hw
    ref_t = tuple(summary["reference"])
    kv = (summary.get("server_facts", {}).get("vllm_default") or {}).get("kv_cache_tokens")
    kv_tokens = a.kv_tokens or (int(kv) if kv else 0)
    if kv_tokens and targets is None:
        print(f"[pytorch] continuous-batching KV capacity matched to vLLM default: {kv_tokens:,} tokens")
    draft = (derived(a.model, "draft") or {}).get("model", "")
    plan = []
    for c in cfgs:
        cw = [w for w in select_workloads(c, wls, ref_t, a.all_workloads)
              if not (c.pt.get("assisted") and w.concurrency != 1)]
        if targets is not None:
            want = targets.get(c.name, {})
            cw = [w for w in cw if w.name in want or "*" in want]
        if cw:
            plan.append((c, cw))
    eta = sampling.ETA(sum(len(w) for _, w in plan) * (a.min_repeats if targets is None else 1) + len(plan))
    print(f"\n[pytorch] {'retry' if targets is not None else 'sweep'}: {len(plan)} configs -> {run_dir}\n")
    for i, (c, cwls) in enumerate(plan, 1):
        print(f"=== [{i}/{len(plan)}] {c.name}  ({c.tier}: {c.knob})  {eta}", flush=True)
        log = logs / f"{c.name}.log"
        fpath = raw / f"{c.name}__facts.json"
        fpath.unlink(missing_ok=True)
        gpus = a.tp if a.device == "cuda" else 1
        if targets is None and not a.resume:              # fresh results for this config
            for w in cwls:
                for f in raw.glob(f"{c.name}__{w.name}__r*.row.json"):
                    f.unlink()
        elif targets is not None:                          # retry: failed repeats are replaced
            ret = raw / "retired"
            for w in cwls:
                for f in raw.glob(f"{c.name}__{w.name}__r*.row.json"):
                    r = json.loads(f.read_text())
                    if not sampling.good(r) and not str(r.get("bench_error", "")).startswith("OOM"):
                        ret.mkdir(exist_ok=True)
                        f.replace(ret / f.name)
        todo = list(cwls)
        t0, attempt, idle, crashes, rc = time.perf_counter(), 0, 0, [], 0
        n_before = len(list(raw.glob(f"{c.name}__*.row.json")))
        with open(log, "a" if targets is not None else "w") as lf:
            while todo:
                attempt += 1
                if hw is not None and hw.unified and a.device == "cuda":
                    drop_page_cache()   # unified memory: cached safetensors compete with CUDA allocations
                tgt = {w.name: targets[c.name].get(w.name, targets[c.name].get("*", 1)) for w in todo} \
                    if targets is not None else None
                cmd = [sys.executable, "-m", "llmbench.torch_worker", "--config-json", json.dumps(c.to_dict()),
                       "--run-dir", str(run_dir), "--model", a.model, "--dtype", a.dtype, "--tp", str(a.tp),
                       "--workloads-json", json.dumps([w.to_dict() for w in todo]),
                       "--max-batches", str(a.max_batches), "--seed", str(a.seed), "--draft-model", draft,
                       "--device", a.device, "--kv-tokens", str(kv_tokens or 0),
                       "--min-repeats", str(a.min_repeats), "--max-repeats", str(a.max_repeats),
                       "--target-cv", str(a.target_cv)]
                if tgt is not None:
                    cmd += ["--targets-json", json.dumps(tgt)]
                if a.slo:
                    cmd += ["--slo", a.slo]
                if a.profile and targets is None:
                    cmd += ["--profile"]
                lf.write(f"\n### attempt {attempt}: {len(todo)} workloads\n")
                pkg_root = str(Path(__file__).resolve().parent.parent)
                env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", **alloc_env(),
                       "PYTHONPATH": pkg_root + os.pathsep + os.environ.get("PYTHONPATH", "")}
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
                try:
                    for line in proc.stdout:             # tee: progress lines to console, everything to the log
                        lf.write(line)
                        if line.startswith("PT| "):
                            print("  " + line[4:], end="", flush=True)
                            if " r" in line and " out=" in line or "OOM" in line:
                                eta.tick()
                    rc = proc.wait()
                except BaseException:                    # Ctrl-C / kill: don't leave the worker on the GPU
                    proc.kill()
                    proc.wait()
                    raise
                lf.flush()

                def finished(w: Workload) -> bool:
                    return any(raw.glob(f"{c.name}__{w.name}__r*.row.json"))

                done = [w for w in todo if finished(w)]
                if rc == 0:
                    break
                # A hard crash (host OOM-kill, CUDA context poisoned, segfault) lost the workload in flight.
                # Record it as failed and continue the rest in a fresh process.
                todo = [w for w in todo if not finished(w)] if targets is None else \
                    [w for w in todo if w not in done]
                if not todo:
                    break
                idle = 0 if done else idle + 1
                if idle >= 2:
                    for w in todo:
                        crashes.append(w.name)
                        write_row(raw, _crash_row(c, w, a.model, rc, "gave up after repeated crashes", _next_rep(raw, c, w), gpus))
                    break
                w = todo.pop(0)
                crashes.append(w.name)
                write_row(raw, _crash_row(c, w, a.model, rc, "worker crashed on this workload", _next_rep(raw, c, w), gpus))
                print(f"  {w.label:<34} worker crashed (exit {rc}); continuing in a fresh process", flush=True)
        eta.tick()
        facts = json.loads(fpath.read_text()) if fpath.exists() else \
            dict(status="failed", error=f"worker exited {rc} before writing facts")
        if crashes:
            facts["crashed_workloads"] = crashes
        if targets is not None:
            facts["retried"] = True
        if facts.get("status") != "ok":
            print(f"  !! {facts.get('error')}  (log: {log})")
            print(explain_failure(log))
        n_after = len(list(raw.glob(f"{c.name}__*.row.json")))
        print(f"  ({time.perf_counter() - t0:.0f}s, {n_after - n_before} new rows)")
        summary["server_facts"][c.name] = facts
        save_summary()
        aggregate(run_dir)
