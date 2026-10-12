"""torch-profiler chrome trace -> GPU time by kernel class.

Both engines produce the same trace format (vLLM's --profiler-config torch, and
torch.profiler in the PyTorch worker), so one parser classifies both. The breakdown answers
"where does the GPU time go" per config: attention vs GEMM vs norms/activations vs
communication vs memory traffic, plus how busy the GPU was overall (CUDA graphs and async
scheduling show up as fewer gaps, i.e. higher gpu_busy_pct).
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Optional

# first match wins, so the specific classes go before the broad ones
KERNEL_CLASSES: list[tuple[str, re.Pattern]] = [(n, re.compile(p, re.I)) for n, p in [
    ("communication", r"nccl|all_?reduce|all_?gather|reduce_?scatter|cross_device|custom_ar|one_?shot|two_?shot"),
    ("attention", r"flash|fmha|attn|attention|paged|flashinfer|xformers|sdpa|mha|reshape_and_cache|unified_attention"
                  r"|batch_decode|batch_prefill|merge_state"),
    ("moe", r"moe|expert|grouped_?gemm|topk_softmax|align_block"),
    ("quant", r"quant|fp8|scaled_mm|cvt_|marlin|awq|gptq|int4|int8|dequant"),
    ("gemm", r"gemm|matmul|cutlass|cublas|xmma|wgmma|sm\d+_|ampere_|hopper|splitk|gemv|_mm_|nvjet|cijk|bmm|addmm"),
    ("norm", r"rms|layer_?norm|norm_kernel|_norm"),
    ("activation", r"silu|gelu|act_and_mul|swiglu|activation|sigmoid|tanh"),
    ("rope", r"rotary|rope|pos_enc"),
    ("sampling", r"sampl|softmax|argmax|top_?k|top_?p|logits|multinomial|penalt"),
    ("fused_elementwise", r"^triton_|fused_"),
    ("memory", r"memcpy|memset|copy|cat_|index|gather|scatter|fill|transpose|contiguous|clone|zero"),
    ("elementwise", r"elementwise|vectorized|unrolled|reduce_kernel|reduction"),
]]


def classify(name: str) -> str:
    for cls, pat in KERNEL_CLASSES:
        if pat.search(name):
            return cls
    return "other"


def find_traces(d: Path) -> list[Path]:
    """Trace files under d, rank 0 first, newest first."""
    fs = [p for p in d.rglob("*") if p.is_file() and (p.name.endswith(".json") or p.name.endswith(".json.gz"))
          and ("trace" in p.name or "pt.trace" in p.name)]
    return sorted(fs, key=lambda p: ("rank0" not in p.name and "rank_0" not in p.name, -p.stat().st_mtime))


def _load(path: Path) -> list[dict]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data.get("traceEvents", data) if isinstance(data, dict) else data


def _union_ms(intervals: list[tuple[float, float]]) -> float:
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot / 1e3


def _top_level(events: list[dict]) -> list[dict]:
    """CPU ops: keep only those not nested inside another op on the same thread."""
    out = []
    by_tid: dict = {}
    for e in sorted(events, key=lambda e: (e.get("tid"), e["ts"], -e.get("dur", 0))):
        stack = by_tid.setdefault(e.get("tid"), [])
        end = e["ts"] + e.get("dur", 0)
        while stack and stack[-1] <= e["ts"]:
            stack.pop()
        if not stack:
            out.append(e)
        stack.append(end)
    return out


def breakdown(path: Path, window_s: Optional[float] = None) -> dict:
    events = [e for e in _load(path) if isinstance(e, dict) and e.get("ph") == "X" and "dur" in e and "ts" in e]
    kernels = [e for e in events if str(e.get("cat", "")).lower() in ("kernel", "gpu_memcpy", "gpu_memset")]
    device = "cuda"
    if not kernels:                         # CPU-only trace (tests, or a CPU run): top-level ops instead
        kernels = _top_level([e for e in events if str(e.get("cat", "")).lower() == "cpu_op"])
        device = "cpu"
    classes: dict[str, dict] = {}
    for e in kernels:
        name = str(e.get("name", ""))
        cls = "memory" if str(e.get("cat", "")).lower() in ("gpu_memcpy", "gpu_memset") else classify(name)
        d = classes.setdefault(cls, {"ms": 0.0, "n": 0, "_top": {}})
        ms = float(e["dur"]) / 1e3
        d["ms"] += ms
        d["n"] += 1
        d["_top"][name] = d["_top"].get(name, 0.0) + ms
    total = sum(d["ms"] for d in classes.values()) or 1.0
    for d in classes.values():
        d["pct"] = 100.0 * d["ms"] / total
        d["top"] = max(d["_top"].items(), key=lambda kv: kv[1])[0][:120] if d["_top"] else ""
        d["ms"] = round(d["ms"], 3)
        del d["_top"]
    busy = _union_ms([(float(e["ts"]), float(e["ts"]) + float(e["dur"])) for e in kernels])
    if window_s:
        window_ms = window_s * 1e3
    elif events:
        window_ms = (max(e["ts"] + e["dur"] for e in events) - min(e["ts"] for e in events)) / 1e3
    else:
        window_ms = float("nan")
    return dict(classes=dict(sorted(classes.items(), key=lambda kv: -kv[1]["ms"])), n_kernels=len(kernels),
                gpu_ms=round(total, 3), window_ms=round(window_ms, 3),
                gpu_busy_pct=min(100.0, 100.0 * busy / window_ms) if window_ms and window_ms == window_ms else float("nan"),
                device=device)
