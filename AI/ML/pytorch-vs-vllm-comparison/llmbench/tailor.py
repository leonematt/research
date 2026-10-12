"""Fit the run to the card: classify it, reserve memory for configs that need extra, and drop
workload shapes the card's KV cache can't meaningfully hold.

Why this exists: vLLM sizes its KV cache from a memory-profiling pass, and some allocations
happen after that pass (the draft model's attention warmup under speculative decoding, FlashInfer
workspaces). On an 80 GB card the leftover slack absorbs them; on a 16 GB card it doesn't, and
the server dies at startup with an OOM. So configs that load extra things get an explicit memory
reservation (a lower --gpu-memory-utilization), and any start that still OOMs is retried with a
slightly lower fraction. Both are recorded in the server facts and the config note, because a
smaller KV pool is part of what that optimization costs on this card.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from .hardware import Hardware, weights_gib
from .util import GIB

OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError", "cudaErrorMemoryAllocation",
               "No available memory for the cache blocks", "Free memory on device")
OOM_STEP = 0.04            # each OOM retry lowers the memory fraction by this much
OOM_RETRIES = 2
OVERHEAD_GIB = 1.5         # activations, CUDA graphs, allocator slack (same figure autosize uses)


@dataclass
class CardProfile:
    name: str
    tier: str              # small (< 20 GiB) | medium (< 60 GiB) | large
    per_gpu_gib: float
    n_gpus: int
    unified: bool
    cc: float
    gpu_mem: float
    kv_tokens_est: Optional[int]
    bytes_per_token: Optional[int]

    @property
    def tight(self) -> bool:
        """Little slack after the KV pool: post-profiling allocations can OOM."""
        return self.tier == "small" or (self.unified and self.per_gpu_gib < 64)

    def lines(self) -> list[str]:
        arch = f"sm_{str(self.cc).replace('.', '')}"
        out = [f"{self.name}: {self.tier} tier ({self.per_gpu_gib:.0f} GiB x {self.n_gpus}, {arch}"
               + (", unified memory" if self.unified else "") + f", memory fraction {self.gpu_mem})"]
        if self.kv_tokens_est:
            out.append(f"estimated KV capacity ~{self.kv_tokens_est:,} tokens "
                       f"({self.bytes_per_token / 1024:.0f} KiB/token)")
        return out


def kv_bytes_per_token(model: str, dtype_bytes: int = 2) -> Optional[int]:
    """K + V bytes per token across all layers, from the model's config.json."""
    import os
    if os.environ.get("LLMBENCH_KV_BYTES_PER_TOKEN"):        # tests (no transformers)
        return int(os.environ["LLMBENCH_KV_BYTES_PER_TOKEN"])
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model)
        tc = getattr(cfg, "text_config", cfg)
        heads = tc.num_attention_heads
        kvh = getattr(tc, "num_key_value_heads", None) or heads
        hd = getattr(tc, "head_dim", None) or tc.hidden_size // heads
        return int(2 * tc.num_hidden_layers * kvh * hd * dtype_bytes)
    except Exception:  # noqa: BLE001
        return None


def profile(a, hw: Hardware) -> CardProfile:
    per = hw.per_gpu_gib
    tier = "small" if per < 20 else "medium" if per < 60 else "large"
    bpt = kv_bytes_per_token(a.model)
    kv = None
    if bpt and a.weights_gib:
        room = per * a.gpu_mem * a.tp - a.weights_gib - OVERHEAD_GIB
        kv = int(max(0.0, room) * GIB / bpt) if room > 0 else 0
    return CardProfile(hw.name, tier, per, hw.n, hw.unified, hw.min_cc, a.gpu_mem, kv, bpt)


def headroom_gib(cfg, card: CardProfile) -> tuple[float, str]:
    """Memory a vLLM config allocates AFTER its profiling pass, i.e. outside --gpu-memory-utilization.
    Draft / EAGLE weights are NOT in here: vLLM loads them before profiling and counts them in the
    budget. What escapes is the draft's attention warmup (FlashInfer workspace) and its CUDA graphs,
    ~0.75-1 GiB, which on a 16 GB card was enough to push the process past the device limit."""
    args = cfg.args
    if "--speculative-config" not in args:
        return 0.0, ""
    try:
        spec = json.loads(args[args.index("--speculative-config") + 1])
    except (ValueError, IndexError):
        spec = {}
    if spec.get("method") == "ngram":
        return 0.25, "n-gram drafting buffers"
    warm = 1.0 if card.tight else 0.5
    return warm, f"draft attention warmup + CUDA graphs ~{warm:.1f} GiB, allocated after vLLM's profiling"


def apply_headroom(cfgs: list, a, card: CardProfile, quiet: bool = False) -> None:
    """Lower --gpu-memory-utilization for configs that need room the profiler doesn't see."""
    total = card.per_gpu_gib * max(1, a.tp)
    for c in cfgs:
        if c.engine != "vllm" or "--gpu-memory-utilization" in c.args:
            continue
        need, why = headroom_gib(c, card)
        if need <= 0:
            continue
        frac = round(max(0.3, a.gpu_mem - need / total), 2)
        c.args = c.args + ["--gpu-memory-utilization", str(frac)]
        c.note = (c.note + "; " if c.note else "") + f"memory fraction {a.gpu_mem} -> {frac} to reserve {need:.1f} GiB ({why})"
        if not quiet:
            print(f"[tailor] {c.name}: memory fraction {a.gpu_mem} -> {frac} ({why})")


def kv_demand(w) -> int:
    """KV tokens in flight for a closed-loop workload. A shared prefix is stored once (that's what
    prefix caching is for); every request holds its own unique input + output."""
    if w.kind == "prefix":
        return w.prefix_len + w.concurrency * (w.input_len + w.output_len)
    return w.concurrency * w.total_len


def fit_workloads(wls: list, card: CardProfile, oversub: float, quiet: bool = False) -> tuple[list, list]:
    """Make closed-loop shapes fit the card's KV cache (at most `oversub` x the estimated capacity
    in flight); past that point vLLM just queues and preempts and the run measures a waiting line.
      * grid points that don't fit are dropped: the rest of the grid still covers the trend;
      * the one-of-a-kind workloads an optimization is judged on (shared prefix, mixed) are never
        dropped: their concurrency is halved until they fit (minimum 4);
      * the reference workload is always kept."""
    if not card.kv_tokens_est or oversub <= 0:
        return wls, []
    from .workloads import num_prompts_for

    cap = card.kv_tokens_est * oversub
    kept, dropped = [], []
    for w in wls:
        if w.kind == "reference" or not w.closed_loop or w.kind in ("text", "sharegpt") or kv_demand(w) <= cap:
            kept.append(w)
            continue
        if w.kind in ("prefix", "mixed"):
            old = w.concurrency
            while w.concurrency > 4 and kv_demand(w) > cap:
                w.concurrency //= 2
            w.num_prompts = max(num_prompts_for(w.concurrency, w.output_len), 2 * w.concurrency)
            if not quiet:
                print(f"[tailor] {w.name}: concurrency {old} -> {w.concurrency} to fit ~{card.kv_tokens_est:,} KV tokens "
                      f"({kv_demand(w):,} in flight)")
            kept.append(w)
            continue
        dropped.append(w)
        if not quiet:
            print(f"[tailor] dropping {w.label}: {kv_demand(w):,} KV tokens in flight "
                  f"> {oversub:g} x ~{card.kv_tokens_est:,} capacity")
    return kept, dropped


CACHE_STARVED = "No available memory for the cache blocks"


def oom_kind(log_text: str) -> Optional[str]:
    """'starved': the budget left nothing for KV (raise the fraction); 'device': the process outgrew
    the GPU (lower it); None: not a memory failure."""
    if CACHE_STARVED in log_text:
        return "starved"
    return "device" if any(m in log_text for m in OOM_MARKERS if m != CACHE_STARVED) else None


def is_oom(log_text: str) -> bool:
    return any(m in log_text for m in OOM_MARKERS)


def lower_gpu_mem(cfg, base: float, raise_: bool = False) -> Optional[float]:
    """Next memory fraction for a memory-failure retry (lower, or raise back toward the run's base
    fraction when the KV cache was starved), or None when there's no room left to move."""
    args = cfg.args
    cur = float(args[args.index("--gpu-memory-utilization") + 1]) if "--gpu-memory-utilization" in args else base
    new = round(min(base, cur + OOM_STEP) if raise_ else cur - OOM_STEP, 2)
    if new < 0.3 or new == cur:
        return None
    if "--gpu-memory-utilization" in args:
        i = args.index("--gpu-memory-utilization")
        cfg.args = args[:i + 1] + [str(new)] + args[i + 2:]
    else:
        cfg.args = args + ["--gpu-memory-utilization", str(new)]
    return new
