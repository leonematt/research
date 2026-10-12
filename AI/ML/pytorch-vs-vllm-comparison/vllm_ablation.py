#!/usr/bin/env python3
"""
vllm_ablation.py - one-knob-at-a-time vLLM ablation sweep on any NVIDIA box.

Design
------
* Detects the GPUs (count, memory, compute capability, discrete vs unified
  memory) and picks the largest open-source model that fits, the smallest TP
  that fits it, a context length that leaves KV room, and a safe memory fraction.
* Every configuration is `baseline` + exactly one change (the "knob").
* Every configuration gets its own fresh `vllm serve` process (isolation), the
  workload grid is run against it with `vllm bench serve`, then the server is
  torn down before the next configuration starts.
* The workload grid is input_len x output_len x concurrency, plus a
  shared-prefix workload (for prefix caching) and, optionally, a ShareGPT
  workload (real text, so speculative decoding has something to speculate on).
* Prometheus counters on /metrics are diffed around every run, so preemptions,
  prefix-cache hit rate and spec-decode acceptance rate land in the tables next
  to throughput and latency percentiles.
* /reset_prefix_cache is called before every measured run so cache state never
  leaks between workloads.

Outputs (under --out/<run_id>/)
-------------------------------
  raw/             one JSON per (config, workload, repeat) from `vllm bench serve`
  server_logs/     stdout+stderr of each `vllm serve` process
  results_all.csv  one row per repeat
  results.csv      one row per (config, workload): medians across repeats
  summary.json     environment, hardware, auto-sizing decisions, config catalog, server facts
  report.md        headline deltas vs baseline, pivots, scaling tables
  plots/           optional PNGs (--plots, needs matplotlib)

Usage
-----
  python vllm_ablation.py --list                       # what this box would run
  python vllm_ablation.py --quick --dry-run
  python vllm_ablation.py --quick
  python vllm_ablation.py --family gemma4 --groups graphs,kv --repeats 3
  python vllm_ablation.py --model meta-llama/Llama-3.1-8B-Instruct --configs baseline,kv_fp8
  python vllm_ablation.py --report-only --run-id <run_id>

Requirements: a working `vllm` CLI on PATH (same env this script runs in),
`pandas` for the report, optionally `matplotlib` for plots. Nothing from vllm
is imported here; everything is driven through subprocesses so the sweep
survives vllm version churn as long as the CLI flags do.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import numbers
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ----------------------------------------------------------------------------
# Hardware presets and workload grids
# ----------------------------------------------------------------------------

EPILOG = """
Hardware is auto-detected (any NVIDIA GPU(s), discrete or unified-memory like
DGX Spark / Jetson). The largest model from --family that fits is chosen, with
the smallest tensor-parallel size that fits across the visible GPUs. Override
with --model / --tp / --max-model-len / --gpu-mem; restrict GPUs with
CUDA_VISIBLE_DEVICES. Preview what any box would pick with --list.

Families (largest first; first that fits wins):
  qwen3  (default; Apache-2.0, ungated)  Qwen3-30B-A3B (MoE) > 14B > 8B > 4B > 1.7B > 0.6B
         spec_draft uses Qwen3-0.6B as draft; fp8_ckpt / w4a16 use Qwen's -FP8 / -AWQ repos
  gemma4 (license-gated on HF)           gemma-4-26B-A4B-it (MoE) > 12B > E4B > E2B
         spec_draft uses <id>-assistant; w4a16 uses <id>-qat-w4a16-ct (none for 26B)
Any other HF id or local path works with --model; family-specific configs are
skipped unless you pass --spec-config / --w4a16-model.
"""


def _gb(params_b: float) -> float:
    """BF16 weight size in GiB for a parameter count in billions."""
    return params_b * 2e9 / 2 ** 30


# Each ladder entry: (HF id, BF16 weights GiB). Derivation helpers return None
# when the family has no such checkpoint for that model.
FAMILIES = {
    "qwen3": dict(
        ladder=[("Qwen/Qwen3-30B-A3B", _gb(30.5)), ("Qwen/Qwen3-14B", _gb(14.8)),
                ("Qwen/Qwen3-8B", _gb(8.2)), ("Qwen/Qwen3-4B", _gb(4.0)),
                ("Qwen/Qwen3-1.7B", _gb(2.0)), ("Qwen/Qwen3-0.6B", _gb(0.75))],
        extra=[("Qwen/Qwen3-32B", _gb(32.8))],
        draft=lambda m: None if m.endswith("0.6B") else
            {"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 4},
        fp8_ckpt=lambda m: f"{m}-FP8",
        w4a16=lambda m: None if ("A3B" in m or m.endswith(("0.6B", "1.7B"))) else f"{m}-AWQ",
        mm_off="none",
    ),
    "gemma4": dict(
        # Google's published BF16 figures carry ~10% over raw weights
        ladder=[("google/gemma-4-26B-A4B-it", 57.7e9 / 1.1 / 2 ** 30),
                ("google/gemma-4-12B-it", 26.7e9 / 1.1 / 2 ** 30),
                ("google/gemma-4-E4B-it", 17.9e9 / 1.1 / 2 ** 30),
                ("google/gemma-4-E2B-it", 11.4e9 / 1.1 / 2 ** 30)],
        extra=[("google/gemma-4-31B-it", 69.9e9 / 1.1 / 2 ** 30)],
        draft=lambda m: {"model": f"{m}-assistant",
                         "num_speculative_tokens": 2 if "E2B" in m else 4},
        fp8_ckpt=lambda m: None,
        w4a16=lambda m: None if "26B-A4B" in m else f"{m}-qat-w4a16-ct",
        mm_off="language-model-only",   # skip vision/audio encoders
    ),
}


def family_of(model: str) -> Optional[str]:
    for fam, spec in FAMILIES.items():
        if any(model == m for m, _ in spec["ladder"] + spec["extra"]):
            return fam
    return None

# Each grid: full factorial of input_lens x output_lens x concurrencies, plus
# `reference` (always included; used for the headline table) and `prefix`
# = (prefix_len, extra_input_len, output_len, concurrency).
#
# The reference point must lie on the grid (its input_len, output_len and
# concurrency each appear in the lists) so the scaling tables in the report
# have full rows; build_workloads() asserts this.
GRIDS = {
    "quick": dict(input_lens=[256, 2048], output_lens=[256],
                  concurrencies=[1, 16, 64],
                  reference=(2048, 256, 16),
                  prefix=(2048, 256, 128, 16)),
    "standard": dict(input_lens=[256, 1024, 4096], output_lens=[256, 1024],
                     concurrencies=[1, 8, 32, 128],
                     reference=(1024, 256, 32),
                     prefix=(4096, 256, 128, 32)),
    "long": dict(input_lens=[1024, 4096, 16384], output_lens=[256, 1024],
                 concurrencies=[1, 8, 32, 64],
                 reference=(4096, 256, 32),
                 prefix=(8192, 512, 256, 32)),
}


def num_prompts_for(concurrency: int) -> int:
    """Enough requests to reach steady state without making c=1 take an hour."""
    return max(8, min(256, 4 * concurrency))


# ----------------------------------------------------------------------------
# Hardware detection and auto-sizing
# ----------------------------------------------------------------------------

GIB = 2 ** 30

# Run in a subprocess so this process never holds a CUDA context. torch respects
# CUDA_VISIBLE_DEVICES and knows whether a device is integrated (unified memory).
_TORCH_PROBE = r"""
import json
try:
    import torch
    out = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append(dict(name=p.name, total_gib=p.total_memory / 2**30,
                            cc=float(f"{p.major}.{p.minor}"),
                            integrated=bool(getattr(p, "is_integrated", False)),
                            uuid=str(getattr(p, "uuid", "") or "")))
    print(json.dumps(out))
except Exception as e:
    print(json.dumps({"error": repr(e)}))
"""

UNIFIED_NAME_HINTS = ("GB10", "THOR", "ORIN", "XAVIER", "TEGRA")


@dataclass
class Hardware:
    gpus: list[dict]            # visible devices in CUDA order: name, total_gib, cc, uuid, smi_index
    unified: bool               # CPU and GPU share one memory pool (GB10, Jetson)
    mem_total_gib: float        # system RAM
    source: str                 # torch | nvidia-smi | none

    @property
    def n(self) -> int:
        return len(self.gpus)

    @property
    def min_cc(self) -> float:
        return min((g["cc"] for g in self.gpus if g.get("cc")), default=0.0)

    @property
    def per_gpu_gib(self) -> float:
        return min((g["total_gib"] for g in self.gpus if g.get("total_gib")), default=0.0)

    @property
    def slug(self) -> str:
        if not self.gpus:
            return "nogpu"
        drop = {"nvidia", "geforce", "tesla", "quadro"}
        toks = [t for t in re.split(r"[\s\-_]+", self.gpus[0]["name"].lower())
                if t and t not in drop and not re.fullmatch(r"\d+gb|hbm\d*e?|pcie|sxm\d*|nvl\d*|oam", t)]
        s = re.sub(r"[^a-z0-9]", "", "".join(toks)) or "gpu"
        return f"{self.n}x{s}" if self.n > 1 else s

    def describe(self) -> str:
        if not self.gpus:
            return "no CUDA GPU detected"
        g = self.gpus[0]
        mem = (f"{self.mem_total_gib:.0f} GiB unified" if self.unified
               else f"{g['total_gib']:.0f} GiB each" if self.n > 1 else f"{g['total_gib']:.0f} GiB")
        return f"{self.n}x {g['name']} (sm_{str(g['cc']).replace('.', '')}, {mem})"

    def smi_ids(self, k: Optional[int] = None) -> list[str]:
        return [str(g["smi_index"]) for g in self.gpus[:k] if g.get("smi_index") is not None]

    def to_dict(self) -> dict:
        return dict(gpus=self.gpus, unified=self.unified, mem_total_gib=self.mem_total_gib,
                    source=self.source, slug=self.slug, describe=self.describe())


def _smi_rows() -> list[dict]:
    out = sh(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,compute_cap",
              "--format=csv,noheader,nounits"])
    rows = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 6:
            continue
        num = lambda s: float(s) if re.fullmatch(r"[\d.]+", s) else None  # noqa: E731
        rows.append(dict(smi_index=int(p[0]), uuid=p[1], name=p[2],
                         total_mib=num(p[3]), used_mib=num(p[4]), cc=num(p[5])))
    return rows


def detect_hardware() -> Hardware:
    fake = os.environ.get("VLLM_ABLATION_FAKE_HW")   # preview another box: '{"name":..,"gib":..,"n":..}'
    if fake:
        d = json.loads(fake)
        gpus = [dict(name=d["name"], total_gib=float(d["gib"]), cc=float(d.get("cc", 9.0)),
                     integrated=bool(d.get("unified")), uuid="", smi_index=i) for i in range(int(d.get("n", 1)))]
        return Hardware(gpus=gpus, unified=bool(d.get("unified")),
                        mem_total_gib=float(d.get("ram_gib", d["gib"] if d.get("unified") else 256)),
                        source="fake")
    mem_total = meminfo_gib().get("mem_total_gib", 0.0)
    smi = _smi_rows()
    gpus, source = [], "none"
    raw = sh([sys.executable, "-c", _TORCH_PROBE])
    try:
        probed = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        probed = None
    if isinstance(probed, list) and probed:
        gpus, source = probed, "torch"
    elif smi:
        # no torch: honour CUDA_VISIBLE_DEVICES ourselves (indices or UUIDs)
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        sel = smi
        if cvd is not None:
            want = [x.strip() for x in cvd.split(",") if x.strip()]
            sel = [r for w in want for r in smi if w == str(r["smi_index"]) or r["uuid"].endswith(w)]
        gpus = [dict(name=r["name"], total_gib=(r["total_mib"] or 0) / 1024, cc=r["cc"] or 0.0,
                     integrated=False, uuid=r["uuid"]) for r in sel]
        source = "nvidia-smi"
    # attach nvidia-smi index (for memory/process queries) by UUID, else by position
    cvd_idx = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip().isdigit()]
    for i, g in enumerate(gpus):
        u = g.get("uuid", "").lower().replace("gpu-", "")
        hit = next((r for r in smi if u and r["uuid"].lower().replace("gpu-", "") == u), None)
        if hit is None and smi:
            j = int(cvd_idx[i]) if i < len(cvd_idx) else i
            hit = next((r for r in smi if r["smi_index"] == j), None)
        g["smi_index"] = hit["smi_index"] if hit else None
        if hit and not g.get("total_gib") and hit["total_mib"]:
            g["total_gib"] = hit["total_mib"] / 1024
    unified = any(g.get("integrated") for g in gpus) \
        or any(h in g["name"].upper() for g in gpus for h in UNIFIED_NAME_HINTS) \
        or (len(gpus) == 1 and mem_total > 0 and gpus[0].get("total_gib")
            and abs(gpus[0]["total_gib"] - mem_total) / mem_total < 0.06)
    for g in gpus:
        if unified and not g.get("total_gib"):
            g["total_gib"] = mem_total
        g["total_gib"] = round(float(g.get("total_gib") or 0.0), 1)
    return Hardware(gpus=gpus, unified=bool(unified), mem_total_gib=mem_total, source=source)


_DTYPE_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
                "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def weights_gib(model: str) -> Optional[float]:
    """Best-effort size of the checkpoint's weights: local safetensors, the Gemma 4
    table, or the Hub's safetensors metadata. None if unknown."""
    p = Path(model).expanduser()
    if p.is_dir():
        sz = sum(f.stat().st_size for f in p.glob("*.safetensors"))
        return sz / GIB if sz else None
    for spec in FAMILIES.values():
        for m, w in spec["ladder"] + spec["extra"]:
            if m == model:
                return w
    try:
        from huggingface_hub import get_safetensors_metadata
        md = get_safetensors_metadata(model)
        return sum(n * _DTYPE_BYTES.get(dt_, 2) for dt_, n in md.parameter_count.items()) / GIB
    except Exception:
        return None


def kv_and_overhead(w: float) -> float:
    """Minimum non-weight memory to start sensibly: a few GiB of KV + graphs/activations."""
    return max(2.0, 0.10 * w) + 1.5


def fits(w: float, hw: Hardware, gpu_mem: float, tp: int) -> bool:
    return w + kv_and_overhead(w) <= hw.per_gpu_gib * gpu_mem * tp


def autosize(hw: Hardware, family: str, model: Optional[str], gpu_mem: float,
             tp: Optional[int]) -> dict:
    """Pick model (if not given), TP (if not given) and a max_model_len that leaves KV room."""
    tps = [t for t in (1, 2, 4, 8) if t <= max(hw.n, 1)]
    why = []
    if model is None:
        ladder = FAMILIES[family]["ladder"]
        for m, w in ladder:
            t = tp or next((t for t in tps if fits(w, hw, gpu_mem, t)), None)
            if t and fits(w, hw, gpu_mem, t):
                model, tp = m, t
                why.append(f"{m} (~{w:.0f} GiB BF16 weights) is the largest {family} model that fits "
                           f"{hw.per_gpu_gib:.0f} GiB x {gpu_mem} x TP{t}")
                break
        else:
            model, tp = ladder[-1][0], tp or tps[-1]
            why.append(f"nothing fits comfortably; falling back to {model} (expect OOM / tiny KV)")
    w = weights_gib(model)
    if tp is None:
        tp = next((t for t in tps if w is None or fits(w, hw, gpu_mem, t)), tps[-1])
        if w is not None and tp > 1:
            why.append(f"TP{tp}: ~{w:.0f} GiB of weights needs >1 GPU at {hw.per_gpu_gib:.0f} GiB each")
    if w is None:
        mml = 16384
        why.append("weights size unknown; max_model_len 16384")
    else:
        kv_room = hw.per_gpu_gib * gpu_mem * tp - w - 1.5
        mml = 32768 if kv_room >= 16 else 16384 if kv_room >= 6 else 8192
        why.append(f"~{kv_room:.0f} GiB left for KV -> max_model_len {mml}")
    return dict(model=model, tp=tp, max_model_len=mml, weights_gib=w, why=why)


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------

@dataclass
class Workload:
    name: str
    input_len: int
    output_len: int
    concurrency: int
    num_prompts: int
    prefix_len: int = 0
    dataset: str = "random"            # random | sharegpt
    dataset_path: Optional[str] = None

    @property
    def label(self) -> str:
        if self.dataset == "sharegpt":
            return f"sharegpt/out{self.output_len}/c{self.concurrency}"
        s = f"in{self.input_len}/out{self.output_len}/c{self.concurrency}"
        if self.prefix_len:
            s += f"/pfx{self.prefix_len}"
        return s

    @property
    def total_len(self) -> int:
        return self.prefix_len + self.input_len + self.output_len

    def bench_args(self) -> list[str]:
        if self.dataset == "sharegpt":
            return ["--dataset-name", "sharegpt",
                    "--dataset-path", str(self.dataset_path),
                    "--sharegpt-output-len", str(self.output_len)]
        return ["--dataset-name", "random",
                "--random-input-len", str(self.input_len),
                "--random-output-len", str(self.output_len),
                "--random-range-ratio", "0.0",
                "--random-prefix-len", str(self.prefix_len)]


@dataclass
class ServerConfig:
    name: str
    group: str
    knob: str                                   # the single change vs baseline
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    model_override: Optional[str] = None
    skip: Optional[str] = None                  # reason this config can't run here
    extra_models: list[str] = field(default_factory=list)   # HF repos it also pulls
    note: str = ""

    def to_dict(self) -> dict:
        return dict(name=self.name, group=self.group, knob=self.knob,
                    args=self.args, env=self.env,
                    model_override=self.model_override, note=self.note)


# ----------------------------------------------------------------------------
# CONFIG CATALOG — baseline + one knob each
# ----------------------------------------------------------------------------

def build_configs(a: argparse.Namespace, hw: Optional[Hardware] = None) -> list[ServerConfig]:
    mml = str(a.max_model_len)
    C = ServerConfig
    fam = FAMILIES.get(family_of(a.model) or "")
    cc = hw.min_cc if hw else 0.0

    # speculative decoding with a draft model sharing the tokenizer
    if a.spec_config:
        spec_json, spec_skip = a.spec_config, None
    elif fam and fam["draft"](a.model):
        spec_json, spec_skip = json.dumps(fam["draft"](a.model)), None
    else:
        spec_json = ""
        spec_skip = "no draft model for this model" if fam else "unknown family; pass --spec-config '{...}'"
    try:
        spec_models = [json.loads(spec_json)["model"]] if spec_json else []
    except (ValueError, KeyError, TypeError):
        spec_models = []

    # vendor pre-quantized checkpoints of the same model (different weights!)
    if a.w4a16_model:
        w4, w4_skip = a.w4a16_model, None
    elif fam and fam["w4a16"](a.model):
        w4, w4_skip = fam["w4a16"](a.model), None
    else:
        w4 = None
        w4_skip = "no vendor W4A16 checkpoint for this model" if fam else "pass --w4a16-model <repo>"
    fp8c = fam["fp8_ckpt"](a.model) if fam else None
    fp8c_skip = None if fp8c else "no vendor FP8 checkpoint for this model"

    # starve the KV pool to ~40% of baseline's KV budget (exposes preemption)
    w = a.weights_gib
    if w and hw and hw.per_gpu_gib:
        total = hw.per_gpu_gib * a.tp
        kv = total * a.gpu_mem - w - 1.5
        low = round((w + 1.5 + 0.4 * max(kv, 0)) / total, 2)
    else:
        low = round(a.gpu_mem - 0.15, 2)
    low_skip = None if low < a.gpu_mem - 0.02 else "baseline KV budget already minimal"

    fp8_skip = None if cc == 0 or cc >= 8.0 else f"FP8 needs sm_80+ (have sm_{str(cc).replace('.', '')})"

    cfgs = [
        C("baseline", "baseline", "vLLM defaults (V1: chunked prefill on, prefix caching on, "
          "piecewise torch.compile + CUDA graphs, FA/FlashInfer attention)"),

        # --- graphs / compile ------------------------------------------------
        C("enforce_eager", "graphs", "no CUDA graphs, no torch.compile",
          ["--enforce-eager"]),
        C("compile_O0", "graphs", "-O0: compilation disabled (graphs per version semantics)",
          ["-O0"], note="verify -O level semantics on your pinned version"),
        C("compile_O1", "graphs", "-O1: quick optimizations", ["-O1"]),
        C("compile_O3", "graphs", "-O3: full optimization incl. extra fusion passes", ["-O3"]),
        C("cudagraph_none", "graphs", "torch.compile kept, CUDA graphs off (isolates graph replay)",
          ["--compilation-config", json.dumps({"cudagraph_mode": "NONE"})]),

        # --- KV cache / memory ---------------------------------------------
        C("no_prefix_cache", "kv", "automatic prefix caching off",
          ["--no-enable-prefix-caching"]),
        C("kv_fp8", "kv", "FP8 KV cache (2x KV capacity, decode reads half the bytes)",
          ["--kv-cache-dtype", "fp8"], skip=fp8_skip),
        C("gpu_mem_low", "kv", f"gpu_memory_utilization={low} (~40% of baseline KV -> preemptions)",
          ["--gpu-memory-utilization", str(low)], skip=low_skip),
        C("block_size_32", "kv", "KV block size 32 (vs 16)", ["--block-size", "32"]),

        # --- scheduler -----------------------------------------------------
        C("mnbt_2048", "scheduler", "max_num_batched_tokens=2048 (small prefill chunks)",
          ["--max-num-batched-tokens", "2048"]),
        C("mnbt_max_model_len", "scheduler",
          f"max_num_batched_tokens={mml} (approximates chunked prefill OFF)",
          ["--max-num-batched-tokens", mml]),
        C("max_seqs_32", "scheduler", "max_num_seqs=32 (cap concurrency)",
          ["--max-num-seqs", "32"]),
        C("max_seqs_512", "scheduler", "max_num_seqs=512", ["--max-num-seqs", "512"]),
        C("async_scheduling", "scheduler", "async scheduling (overlap CPU scheduling with GPU)",
          ["--async-scheduling"]),

        # --- speculative decoding -----------------------------------------
        C("spec_ngram", "spec", "n-gram speculative decoding (no draft model)",
          ["--speculative-config",
           json.dumps({"method": "ngram", "num_speculative_tokens": 4,
                       "prompt_lookup_max": 4, "prompt_lookup_min": 2})],
          note="random-token prompts give ~0 acceptance; judge on the sharegpt workload"),
        C("spec_draft", "spec", f"draft-model speculative decoding ({', '.join(spec_models) or '-'})",
          ["--speculative-config", spec_json], skip=spec_skip, extra_models=spec_models,
          note="acceptance is low on random tokens; judge on the sharegpt workload"),

        # --- quantization --------------------------------------------------
        C("fp8_online", "quant", "online FP8 weight quantization (W8A8 on sm>=8.9, "
          "Marlin weight-only on Ampere)", ["--quantization", "fp8"], skip=fp8_skip),
        C("fp8_ckpt", "quant", f"vendor FP8 checkpoint ({fp8c or '-'})",
          [], model_override=fp8c, skip=fp8_skip or fp8c_skip, extra_models=[fp8c] if fp8c else []),
        C("w4a16", "quant", f"vendor 4-bit weight-only checkpoint ({w4 or '-'}; Marlin kernels)",
          [], model_override=w4, skip=w4_skip, extra_models=[w4] if w4 else []),

        # --- parallelism (only meaningful with TP > 1) ---------------------
        C("no_custom_allreduce", "parallel", "NCCL all-reduce instead of vLLM's custom one-shot kernel",
          ["--disable-custom-all-reduce"], skip=None if a.tp > 1 else "TP=1"),

        # --- attention backend -------------------------------------------
        C("attn_flashinfer", "attn", "VLLM_ATTENTION_BACKEND=FLASHINFER",
          env={"VLLM_ATTENTION_BACKEND": "FLASHINFER"}),
        C("attn_flash_attn", "attn", "VLLM_ATTENTION_BACKEND=FLASH_ATTN",
          env={"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"}),
        C("attn_triton", "attn", "VLLM_ATTENTION_BACKEND=TRITON_ATTN",
          env={"VLLM_ATTENTION_BACKEND": "TRITON_ATTN"},
          note="backend name differs across versions (TRITON_ATTN / TRITON_ATTN_VLLM_V1)"),
    ]
    return cfgs


def build_workloads(a: argparse.Namespace) -> list[Workload]:
    g = GRIDS[a.grid]
    wls: list[Workload] = []
    seen = set()

    def add(inp, out, conc, prefix=0, name=None):
        key = (inp, out, conc, prefix)
        if key in seen:
            return
        seen.add(key)
        nm = name or f"i{inp}_o{out}_c{conc}" + (f"_p{prefix}" if prefix else "")
        wls.append(Workload(nm, inp, out, conc, num_prompts_for(conc), prefix))

    ri, ro, rc = g["reference"]
    assert ri in g["input_lens"] and ro in g["output_lens"] and rc in g["concurrencies"], \
        f"grid {a.grid}: reference {g['reference']} must be a point on the grid"
    add(ri, ro, rc, name="reference")
    for inp in g["input_lens"]:
        for out in g["output_lens"]:
            for conc in g["concurrencies"]:
                add(inp, out, conc)
    p_len, p_in, p_out, p_c = g["prefix"]
    add(p_in, p_out, p_c, prefix=p_len, name="shared_prefix")
    if a.sharegpt_path:
        wls.append(Workload("sharegpt", 0, ro, rc, num_prompts_for(rc),
                            dataset="sharegpt", dataset_path=a.sharegpt_path))

    want = set(a.workloads.split(",")) if a.workloads else None
    kept = []
    for w in wls:
        if want is not None and w.name not in want:
            continue
        if w.dataset == "random" and w.total_len > a.max_model_len:
            print(f"[grid] dropping {w.label}: needs {w.total_len} > max_model_len {a.max_model_len}")
            continue
        kept.append(w)
    if not any(w.name == "reference" for w in kept):
        raise SystemExit("the `reference` workload must be included (used for warmup and the headline table)")
    return kept


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def http_get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:
        return 0, ""


def http_post(url: str, payload: Optional[dict] = None, timeout: float = 30.0) -> int:
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def environment(hw: Hardware) -> dict:
    return dict(
        timestamp=dt.datetime.now().isoformat(timespec="seconds"),
        host=os.uname().nodename, arch=os.uname().machine,
        python=sys.version.split()[0],
        vllm=sh(["vllm", "--version"]) or sh([sys.executable, "-c", "import vllm;print(vllm.__version__)"]),
        torch=sh([sys.executable, "-c", "import torch;print(torch.__version__)"]),
        cuda=sh([sys.executable, "-c", "import torch;print(torch.version.cuda)"]),
        driver=sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).split("\n")[0],
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        hardware=hw.to_dict(),
    )


METRIC_RE = re.compile(r"^(vllm:[A-Za-z0-9_]+)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|[+-]?Inf)\s*$")


def scrape_metrics(port: int) -> dict[str, float]:
    """Sum every vllm:* sample over its label sets. Only counters/gauges are used."""
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
        if math.isnan(v):
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + v
    return out


def metric(metrics: dict[str, float], *candidates: str) -> float:
    for c in candidates:
        if c in metrics:
            return metrics[c]
    # substring fallback for renamed metrics
    for c in candidates:
        stem = c.replace("vllm:", "")
        for k, v in metrics.items():
            if stem in k:
                return v
    return float("nan")


def metric_deltas(before: dict, after: dict) -> dict:
    def d(*names):
        return metric(after, *names) - metric(before, *names)
    preempt = d("vllm:num_preemptions_total")
    pq = d("vllm:prefix_cache_queries_total")
    ph = d("vllm:prefix_cache_hits_total")
    drafts = d("vllm:spec_decode_num_drafts_total")
    draft_tok = d("vllm:spec_decode_num_draft_tokens_total")
    acc_tok = d("vllm:spec_decode_num_accepted_tokens_total")
    return dict(
        preemptions=preempt,
        prefix_cache_queries=pq,
        prefix_cache_hits=ph,
        prefix_hit_rate=(ph / pq) if pq and pq > 0 else float("nan"),
        spec_drafts=drafts,
        spec_draft_tokens=draft_tok,
        spec_accepted_tokens=acc_tok,
        spec_accept_rate=(acc_tok / draft_tok) if draft_tok and draft_tok > 0 else float("nan"),
        spec_accepted_per_draft=(acc_tok / drafts) if drafts and drafts > 0 else float("nan"),
    )


LOG_FACTS = {
    "kv_cache_tokens": r"KV cache size:\s*([\d,]+)\s*tokens",
    "max_concurrency_x": r"Maximum concurrency for\s*[\d,]+\s*tokens per request:\s*([\d.]+)x",
    "weights_load_s": r"Loading weights took\s*([\d.]+)\s*seconds",
    "model_load_gib": r"Model loading took\s*([\d.]+)\s*Gi?B",
    "compile_s": r"torch\.compile takes\s*([\d.]+)\s*s",
    "cudagraph_capture_s": r"Graph capturing finished in\s*([\d.]+)\s*secs",
    # "Using FlashInfer backend", "Using Flash Attention backend on V1 engine",
    # "Using TRITON_ATTN backend" ...
    "attention_backend": r"Using\s+([A-Za-z0-9_]+(?: Attention)?)\s+backend",
}


def parse_server_log(path: Path) -> dict:
    facts: dict = {}
    try:
        txt = path.read_text(errors="replace")
    except Exception:
        return facts
    for key, pat in LOG_FACTS.items():
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if not m:
            continue
        val = m.group(1).replace(",", "")
        try:
            facts[key] = float(val) if key != "attention_backend" else val
        except ValueError:
            facts[key] = val
    return facts


def tail(path: Path, n: int = 25) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


# (pattern in log, human hint) for the failures that actually happen on a Spark
KNOWN_FAILURES = [
    (r"GatedRepoError|401 Client Error|Cannot access gated repo|Access to model .* is restricted",
     "Gated model: accept the Gemma license on the model's HF page, then `export HF_TOKEN=hf_...` "
     "or run `hf auth login`."),
    (r"RepositoryNotFoundError|404 Client Error.*huggingface",
     "Model id not found on the Hub: check the id, or pass MODEL=<id or local path>."),
    (r"Free memory on device .* is less than desired GPU memory utilization",
     "Unified memory already in use (page cache / other processes). Lower --gpu-mem "
     "(e.g. GPU_MEM=0.6) or free memory: `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`."),
    (r"CUDA error: out of memory|CUDA out of memory|OutOfMemoryError|cudaErrorMemoryAllocation"
     r"|No available memory for the cache blocks",
     "OOM. Check nothing else holds memory (`nvidia-smi`; on unified-memory boxes `free -g` and "
     "the page cache, which --drop-caches clears). Otherwise lower GPU_MEM / --max-model-len, "
     "or pick a smaller --model."),
    (r"unrecognized arguments|error: argument|invalid choice",
     "A CLI flag isn't accepted by this vLLM version; see the argparse line in the log and "
     "adjust build_configs() / Server.command()."),
    (r"not supported|NotImplementedError|Unsupported",
     "Feature/arch not supported by this build (check the line above for which)."),
]


def explain_failure(path: Path, n: int = 30) -> str:
    try:
        txt = path.read_text(errors="replace")
    except Exception:
        return "  (no server log)"
    hints = [h for pat, h in KNOWN_FAILURES if re.search(pat, txt, flags=re.IGNORECASE)]
    lines = txt.splitlines()
    err_re = re.compile(r"Error|Exception|Traceback|error:|FAILED")
    err_lines = [l for l in lines if err_re.search(l)]
    out = []
    # the last INFO line before the first error says which startup stage died
    first_err = next((i for i, l in enumerate(lines) if err_re.search(l)), None)
    if first_err is not None:
        stage = next((lines[i] for i in range(first_err - 1, -1, -1)
                      if re.search(r"(^|\s)INFO\s", lines[i])), None)
        if stage:
            out.append(f"  last step before failure: {stage.strip()[:200]}")
    if hints:
        out.append("  likely cause:")
        out += [f"    -> {h}" for h in hints[:2]]
    out.append(f"  error lines ({len(err_lines)}):" if err_lines else f"  last {n} log lines:")
    out += ["    " + l[:240] for l in (err_lines[-12:] if err_lines else lines[-n:])]
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Server lifecycle
# ----------------------------------------------------------------------------

class Server:
    def __init__(self, cfg: ServerConfig, model: str, a: argparse.Namespace, log_path: Path):
        self.cfg, self.model, self.a, self.log_path = cfg, model, a, log_path
        self.proc: Optional[subprocess.Popen] = None
        self.logf = None
        self.port = a.port

    def command(self) -> list[str]:
        # pre-quantized checkpoints carry their own activation dtype (AWQ is fp16)
        dtype = "auto" if self.cfg.model_override else self.a.dtype
        cmd = ["vllm", "serve", self.model,
               "--host", "127.0.0.1", "--port", str(self.port),
               "--dtype", dtype,
               "--max-model-len", str(self.a.max_model_len),
               "--seed", "0",
               "--tensor-parallel-size", str(self.a.tp)]
        if "--gpu-memory-utilization" not in self.cfg.args:
            cmd += ["--gpu-memory-utilization", str(self.a.gpu_mem)]
        # Gemma 4 is multimodal; text-only benchmarks shouldn't pay for the encoders.
        if self.a.mm_off == "language-model-only":
            cmd += ["--language-model-only"]
        elif self.a.mm_off == "limit":
            cmd += ["--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0})]
        cmd += self.cfg.args
        if self.a.extra_server_args:
            cmd += shlex.split(self.a.extra_server_args)
        return cmd

    def env(self) -> dict:
        e = dict(os.environ)
        e.update(self.cfg.env)
        e.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        return e

    def start(self) -> dict:
        mem = {}
        if self.a.drop_caches:
            before = meminfo_gib()
            dropped = drop_page_cache()
            after = meminfo_gib()
            mem = dict(page_cache_dropped=dropped,
                       page_cache_gib_before=before.get("page_cache_gib"))
            mem.update(after)
            if before and after:
                print(f"  [mem] available {before['mem_available_gib']:.0f} -> "
                      f"{after['mem_available_gib']:.0f} GiB of {after['mem_total_gib']:.0f} "
                      f"(page cache {before['page_cache_gib']:.0f} -> {after['page_cache_gib']:.0f} GiB)")
        else:
            mem = meminfo_gib()
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
                return dict(status="failed", startup_s=time.time() - t0,
                            error=f"vllm serve exited with {self.proc.returncode}",
                            log_tail=tail(self.log_path), **mem)
            st, _ = http_get(f"http://127.0.0.1:{self.port}/health", timeout=3)
            if st == 200:
                break
            time.sleep(2)
        else:
            self.stop()
            return dict(status="failed", startup_s=time.time() - t0,
                        error="startup timeout", log_tail=tail(self.log_path), **mem)
        # readiness: one real completion
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
            if http_post(f"http://127.0.0.1:{self.port}/v1/completions", payload, timeout=120) == 200:
                return True
            time.sleep(3)
        return False

    def reset_prefix_cache(self) -> None:
        http_post(f"http://127.0.0.1:{self.port}/reset_prefix_cache", timeout=30)

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
        hw = getattr(self.a, "hw", None)
        wait_gpu_settle(self.a.settle_s, hw.smi_ids(self.a.tp) if hw and not hw.unified else None)


def meminfo_gib() -> dict:
    """MemTotal / MemAvailable / page cache from /proc/meminfo, in GiB."""
    vals = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable", "Cached", "Buffers"):
                vals[k] = int(v.split()[0]) / 1048576
    except Exception:
        return {}
    return dict(mem_total_gib=round(vals.get("MemTotal", 0), 1),
                mem_available_gib=round(vals.get("MemAvailable", 0), 1),
                page_cache_gib=round(vals.get("Cached", 0) + vals.get("Buffers", 0), 1))


_DROP_WARNED = False


def drop_page_cache() -> bool:
    """sync + drop the page cache. On unified-memory parts (GB10, Jetson) the cache holds a
    second copy of the safetensors after every model load and CUDA can't reclaim it.
    Needs passwordless sudo for `tee /proc/sys/vm/drop_caches` (run.sh primes it)."""
    global _DROP_WARNED
    subprocess.run(["sync"], check=False)
    r = subprocess.run(["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"], input="3\n",
                       capture_output=True, text=True)
    if r.returncode != 0 and not _DROP_WARNED:
        _DROP_WARNED = True
        user = os.environ.get("USER", "<you>")
        print("  [mem] couldn't drop page cache (sudo needs a password). For unattended sweeps add:\n"
              f"        echo '{user} ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches' "
              "| sudo tee /etc/sudoers.d/drop-caches")
    return r.returncode == 0


_VLLM_PROC = re.compile(
    r"^(?:\S*/)?(?:python[\d.]*\s+)?(?:\S*/)?vllm\s+serve\b"   # vllm serve / python .../vllm serve
    r"|^VLLM::\w+"                                               # VLLM::EngineCore / workers
    r"|^(?:\S*/)?python[\d.]*\s+(?:-\S+\s+)*-m\s+vllm\.entrypoints")


def other_vllm_processes() -> list[str]:
    """Running vLLM servers/engines, matched on the executable, not on any mention of it."""
    out = sh(["ps", "-eo", "pid=,user=,etime=,args="])
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and _VLLM_PROC.search(parts[3]):
            rows.append(line.strip()[:220])
    return rows


def gpu_compute_apps(smi_ids: list[str]) -> list[str]:
    """Processes holding memory on the GPUs we're about to use (discrete GPUs)."""
    if not smi_ids:
        return []
    uuids = sh(["nvidia-smi", "-i", ",".join(smi_ids), "--query-gpu=uuid",
                "--format=csv,noheader"]).split()
    out = sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
              "--format=csv,noheader"])
    rows = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4 and p[0] in uuids:
            rows.append(f"pid {p[1]} {p[2]} ({p[3]})")
    return rows


def hf_access(repo: str) -> tuple[str, Optional[str]]:
    """("ok"|"local"|"unchecked"|"fail", reason). Only "fail" blocks."""
    if Path(repo).expanduser().exists():
        return "local", None
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return "unchecked", "HF_HUB_OFFLINE=1"
    try:
        from huggingface_hub import auth_check
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except Exception:
        return "unchecked", "huggingface_hub too old for auth_check"
    try:
        auth_check(repo)
        return "ok", None
    except GatedRepoError:
        return "fail", (f"{repo} is gated: accept its license at https://huggingface.co/{repo} "
                        f"and `export HF_TOKEN=hf_...` (or `hf auth login`)")
    except RepositoryNotFoundError:
        return "fail", f"{repo} not found on the Hub (or private)"
    except Exception as e:
        return "unchecked", f"{type(e).__name__}"   # offline / transient: let vllm try


def preflight(a: argparse.Namespace, hw: Hardware, cfgs: list[ServerConfig]) -> list[ServerConfig]:
    """Refuse to benchmark on a busy machine, verify model access, and drop configs
    whose extra checkpoints can't be pulled. Another server on the box both steals the
    memory vLLM needs and shares memory bandwidth, contaminating every decode number."""
    if not hw.gpus:
        raise SystemExit("[preflight] no CUDA GPU visible (check nvidia-smi / CUDA_VISIBLE_DEVICES)")

    # model access: main model is fatal, extras just drop their config
    status, why = hf_access(a.model)
    if status == "fail":
        raise SystemExit(f"[preflight] {why}")
    print(f"[preflight] model {a.model}: " + {"ok": "HF access ok", "local": "local path"}.get(
        status, f"access not verified ({why}); vllm will find out"))
    kept = []
    for c in cfgs:
        bad = next((r for s, r in (hf_access(m) for m in c.extra_models) if s == "fail"), None)
        if bad:
            print(f"[preflight] dropping {c.name}: {bad}")
            continue
        kept.append(c)

    problems = []
    others = other_vllm_processes()
    if others:
        problems.append("another vLLM server is running:\n" + "\n".join("      " + r for r in others))
    st, _ = http_get(f"http://127.0.0.1:{a.port}/health", timeout=2)
    if st:
        problems.append(f"something is already answering on port {a.port} (use --port / PORT=)")
    if hw.unified:
        if a.drop_caches:
            drop_page_cache()
        m = meminfo_gib()
        if m:
            need = a.gpu_mem * hw.per_gpu_gib
            if m["mem_available_gib"] < need:
                problems.append(f"only {m['mem_available_gib']:.0f} GiB available but "
                                f"--gpu-mem {a.gpu_mem} asks for {need:.0f} GiB of {hw.per_gpu_gib:.0f}")
    else:
        ids = hw.smi_ids(a.tp)
        apps = gpu_compute_apps(ids)
        if apps:
            problems.append(f"processes already on GPU(s) {','.join(ids)}:\n"
                            + "\n".join("      " + r for r in apps))
        for r in _smi_rows():
            if str(r["smi_index"]) in ids and r["total_mib"] and r["used_mib"] is not None:
                free = (r["total_mib"] - r["used_mib"]) / 1024
                need = a.gpu_mem * r["total_mib"] / 1024
                if free < need:
                    problems.append(f"GPU {r['smi_index']}: {free:.0f} GiB free, --gpu-mem {a.gpu_mem} "
                                    f"needs {need:.0f} GiB")
    if problems:
        print("\n[preflight] machine isn't free for benchmarking:")
        for p_ in problems:
            print("  - " + p_)
        print("  The sweep needs these GPUs to itself: anything else running there shares memory and\n"
              "  bandwidth and skews every number. Stop it for the sweep (`docker ps` -> `docker stop`),\n"
              "  pick other GPUs with CUDA_VISIBLE_DEVICES, or rerun with --allow-shared.")
        if not a.allow_shared:
            raise SystemExit(2)
        print("  --allow-shared set: continuing; results are not clean.\n")
    return kept


def wait_gpu_settle(max_s: float, smi_ids: Optional[list[str]] = None) -> None:
    """Best-effort: wait for the GPUs we used to drain before the next launch.
    On unified-memory parts nvidia-smi usage is not meaningful, so this
    degrades to a fixed settle delay."""
    t0 = time.time()
    sel = ["-i", ",".join(smi_ids)] if smi_ids else []
    while time.time() - t0 < max_s:
        used = sh(["nvidia-smi", *sel, "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        try:
            vals = [float(x) for x in used.splitlines() if x.strip()]
        except ValueError:
            vals = []
        if vals and max(vals) < 3000:
            time.sleep(3)
            return
        time.sleep(3)


# ----------------------------------------------------------------------------
# Benchmark runs
# ----------------------------------------------------------------------------

def bench_command(a: argparse.Namespace, model: str, wl: Workload, out_dir: Path,
                  fname: str, num_prompts: Optional[int] = None,
                  meta: Optional[dict] = None) -> list[str]:
    cmd = ["vllm", "bench", "serve",
           "--backend", "vllm", "--host", "127.0.0.1", "--port", str(a.port),
           "--model", model, "--tokenizer", model,
           "--num-prompts", str(num_prompts or wl.num_prompts),
           "--max-concurrency", str(wl.concurrency),
           "--request-rate", "inf",
           "--ignore-eos",
           "--seed", str(a.seed),
           "--percentile-metrics", "ttft,tpot,itl,e2el",
           "--metric-percentiles", "50,90,99",
           "--save-result", "--result-dir", str(out_dir), "--result-filename", fname]
    cmd += wl.bench_args()
    if a.slo:
        cmd += ["--goodput"] + a.slo.split()
    if meta:
        cmd += ["--metadata"] + [f"{k}={v}" for k, v in meta.items()]
    if a.extra_bench_args:
        cmd += shlex.split(a.extra_bench_args)
    return cmd


def flatten_bench_json(path: Path) -> dict:
    """Pull every scalar numeric field out of a `vllm bench serve` result."""
    row: dict = {}
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return {"bench_parse_error": str(e)}
    for k, v in data.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, numbers.Number):
            row[k] = float(v)
        elif k == "errors" and isinstance(v, list):
            row["n_errors"] = float(len([x for x in v if x]))
    return row


def run_bench(a: argparse.Namespace, server: Server, cfg: ServerConfig, model: str,
              wl: Workload, rep: int, raw_dir: Path) -> dict:
    fname = f"{cfg.name}__{wl.name}__r{rep}.json"
    meta_path = raw_dir / (fname[:-5] + ".meta.json")
    json_path = raw_dir / fname
    base = dict(config=cfg.name, group=cfg.group, knob=cfg.knob, model=model,
                workload=wl.name, workload_label=wl.label, dataset=wl.dataset,
                input_len=wl.input_len, output_len=wl.output_len,
                prefix_len=wl.prefix_len, concurrency=wl.concurrency,
                num_prompts=wl.num_prompts, repeat=rep)

    if a.resume and json_path.exists():
        row = dict(base)
        row.update(flatten_bench_json(json_path))
        if meta_path.exists():
            row.update(json.loads(meta_path.read_text()))
        row["resumed"] = 1.0
        return row

    server.reset_prefix_cache()
    before = scrape_metrics(a.port)
    cmd = bench_command(a, model, wl, raw_dir, fname,
                        meta=dict(config=cfg.name, workload=wl.label, repeat=rep))
    t0 = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=a.bench_timeout)
        rc = res.returncode
        err_tail = "\n".join((res.stderr or "").splitlines()[-15:])
    except subprocess.TimeoutExpired:
        rc, err_tail = -1, "bench timeout"
    wall = time.time() - t0
    after = scrape_metrics(a.port)

    row = dict(base)
    row["bench_wall_s"] = wall
    row["bench_rc"] = float(rc)
    deltas = metric_deltas(before, after)
    row.update(deltas)
    if rc == 0 and json_path.exists():
        row.update(flatten_bench_json(json_path))
    else:
        row["bench_error"] = err_tail
        print(f"      bench failed rc={rc}: {err_tail.splitlines()[-1] if err_tail else ''}")
    meta_path.write_text(json.dumps(dict(deltas, bench_wall_s=wall, bench_rc=rc), indent=1))
    return row


def warmup(a: argparse.Namespace, model: str, wl: Workload, tmp_dir: Path) -> None:
    cmd = bench_command(a, model, wl, tmp_dir, "warmup.json", num_prompts=a.warmup_prompts)
    subprocess.run(cmd, capture_output=True, text=True, timeout=a.bench_timeout)


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def fmt_cell(v, floatfmt: str) -> str:
    if v is None:
        return "–"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "–"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, numbers.Integral):
        return f"{int(v):,}"
    if isinstance(v, numbers.Real):
        return floatfmt.format(float(v))
    return str(v)


def md_table(df, floatfmt: str = "{:,.1f}", index: bool = True, index_name: str = "") -> str:
    if df is None or len(df) == 0:
        return "_no data_\n"
    cols = list(df.columns)
    headers = ([index_name or (df.index.name or "")] if index else []) + [str(c) for c in cols]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)] if index else []
        cells += [fmt_cell(row[c], floatfmt) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def pct(new, base):
    try:
        if base is None or base == 0 or math.isnan(base) or math.isnan(new):
            return float("nan")
        return (new - base) / base * 100.0
    except TypeError:
        return float("nan")


def write_report(run_dir: Path, make_plots: bool = False) -> None:
    import pandas as pd

    summary = json.loads((run_dir / "summary.json").read_text())
    df = pd.read_csv(run_dir / "results.csv")
    cfg_order = [c["name"] for c in summary["configs"]]
    df["config"] = pd.Categorical(df["config"], categories=cfg_order, ordered=True)
    df = df.sort_values(["config", "workload"]).reset_index(drop=True)
    wl_order = [w["label"] for w in summary["workloads"]]

    def pivot(metric_col, rows="config", cols="workload_label", sub=None):
        d = df if sub is None else df[sub]
        if metric_col not in d.columns:
            return None
        p = d.pivot_table(index=rows, columns=cols, values=metric_col, aggfunc="median",
                          observed=False)
        p = p.reindex([c for c in cfg_order if c in p.index])
        if cols == "workload_label":
            p = p[[c for c in wl_order if c in p.columns]]
        else:
            p = p[sorted(p.columns)]
        return p

    ref = summary["reference_label"]
    env = summary["environment"]
    out = []
    hwd = env.get("hardware", {})
    out.append(f"# vLLM ablation sweep — `{run_dir.name}`\n")
    out.append(f"Model: `{summary['model']}` · {hwd.get('describe', '?')} · TP={summary.get('tp', 1)} "
               f"· dtype={summary.get('dtype', '?')}\n")
    out.append(f"vLLM {env.get('vllm','?')} · torch {env.get('torch','?')} · CUDA {env.get('cuda','?')} "
               f"· driver {env.get('driver','?')}\n")
    out.append(f"max_model_len={summary['max_model_len']} · gpu_memory_utilization={summary['gpu_mem']} "
               f"· grid=`{summary['grid']}` · repeats={summary['repeats']} · reference workload=`{ref}`\n")
    if summary.get("autosize_why"):
        out.append("Auto-sizing: " + "; ".join(summary["autosize_why"]) + "\n")
    out.append("Each row/config = baseline + exactly one change. Each config ran in a fresh `vllm serve` "
               "process; `/reset_prefix_cache` was called before every measured run; values are medians "
               "across repeats. Latency in ms; throughput in tokens/s.\n")

    # 1. Config catalog + server facts
    out.append("## 1. Configurations and server facts\n")
    facts_rows = []
    for c in summary["configs"]:
        f = summary["server_facts"].get(c["name"], {})
        kv = f.get("kv_cache_tokens")
        facts_rows.append(dict(config=c["name"], group=c["group"], knob=c["knob"],
                               status=f.get("status", "skipped"),
                               startup_s=f.get("startup_s"), weights_load_s=f.get("weights_load_s"),
                               compile_s=f.get("compile_s"), cudagraph_capture_s=f.get("cudagraph_capture_s"),
                               kv_cache_tokens=f"{int(kv):,}" if kv else None,
                               max_concurrency_x=f.get("max_concurrency_x"),
                               mem_avail_gib=f.get("mem_available_gib"),
                               attn=f.get("attention_backend"),
                               note=(f.get("error") or c.get("note") or "")[:80]))
    fdf = pd.DataFrame(facts_rows).set_index("config")
    out.append(md_table(fdf))
    out.append("`compile_s` reflects the torch.compile cache state on disk (first run is cold).\n")

    # 2. Headline table at the reference workload
    out.append(f"## 2. Headline: reference workload `{ref}` — deltas vs baseline\n")
    r = df[df["workload_label"] == ref].set_index("config")
    if len(r):
        base = r.loc["baseline"] if "baseline" in r.index else None
        cols = [("output_throughput", "out tok/s", True),
                ("request_throughput", "req/s", True),
                ("median_ttft_ms", "TTFT p50", False),
                ("p99_ttft_ms", "TTFT p99", False),
                ("median_tpot_ms", "TPOT p50", False),
                ("p99_tpot_ms", "TPOT p99", False),
                ("p99_itl_ms", "ITL p99", False),
                ("median_e2el_ms", "E2E p50", False)]
        rows = []
        for cfg in [c for c in cfg_order if c in r.index]:
            row = {}
            for col, label, _ in cols:
                v = r.loc[cfg, col] if col in r.columns else float("nan")
                row[label] = v
                # always emit the Δ% column (NaN for baseline) so column order is stable
                row[label + " Δ%"] = (pct(v, base[col]) if base is not None and cfg != "baseline"
                                      and col in r.columns else float("nan"))
            row["preempt"] = r.loc[cfg, "preemptions"] if "preemptions" in r.columns else float("nan")
            row["prefix hit%"] = r.loc[cfg, "prefix_hit_rate"] * 100 if "prefix_hit_rate" in r.columns else float("nan")
            row["spec acc%"] = r.loc[cfg, "spec_accept_rate"] * 100 if "spec_accept_rate" in r.columns else float("nan")
            if "request_goodput" in r.columns:
                row["goodput req/s"] = r.loc[cfg, "request_goodput"]
            rows.append(pd.Series(row, name=cfg))
        hdf = pd.DataFrame(rows)
        hdf.index.name = "config"
        out.append(md_table(hdf))
        out.append("Positive Δ% on throughput is better; negative Δ% on latency is better.\n")
    else:
        out.append("_reference workload not found in results_\n")

    # 3. Pivots across the whole grid
    out.append("## 3. Full grid pivots (rows = config, columns = workload)\n")
    for col, title, ff in [("output_throughput", "Output throughput (tok/s)", "{:,.0f}"),
                           ("median_ttft_ms", "TTFT p50 (ms)", "{:,.1f}"),
                           ("p99_ttft_ms", "TTFT p99 (ms)", "{:,.1f}"),
                           ("median_tpot_ms", "TPOT p50 (ms)", "{:,.2f}"),
                           ("p99_itl_ms", "ITL p99 (ms)", "{:,.2f}"),
                           ("median_e2el_ms", "E2E latency p50 (ms)", "{:,.0f}"),
                           ("preemptions", "Preemptions (count)", "{:,.0f}")]:
        p = pivot(col)
        if p is not None and p.notna().any().any():
            out.append(f"### {title}\n")
            out.append(md_table(p, ff))

    # 4. Scaling: concurrency and input length
    rin, rout, rc = summary["reference"]
    out.append(f"## 4. Scaling sweeps\n")
    sub = (df["input_len"] == rin) & (df["output_len"] == rout) & (df["prefix_len"] == 0) & (df["dataset"] == "random")
    for col, title, ff in [("output_throughput", f"Throughput vs concurrency (in={rin}, out={rout})", "{:,.0f}"),
                           ("median_tpot_ms", f"TPOT p50 vs concurrency (in={rin}, out={rout})", "{:,.2f}"),
                           ("median_ttft_ms", f"TTFT p50 vs concurrency (in={rin}, out={rout})", "{:,.1f}")]:
        p = pivot(col, cols="concurrency", sub=sub)
        if p is not None and p.notna().any().any():
            p.columns = [f"c={int(c)}" for c in p.columns]
            out.append(f"### {title}\n")
            out.append(md_table(p, ff))
    sub = (df["concurrency"] == rc) & (df["output_len"] == rout) & (df["prefix_len"] == 0) & (df["dataset"] == "random")
    for col, title, ff in [("output_throughput", f"Throughput vs input length (c={rc}, out={rout})", "{:,.0f}"),
                           ("median_ttft_ms", f"TTFT p50 vs input length (c={rc}, out={rout})", "{:,.1f}"),
                           ("median_tpot_ms", f"TPOT p50 vs input length (c={rc}, out={rout})", "{:,.2f}")]:
        p = pivot(col, cols="input_len", sub=sub)
        if p is not None and p.notna().any().any():
            p.columns = [f"in={int(c)}" for c in p.columns]
            out.append(f"### {title}\n")
            out.append(md_table(p, ff))
    sub = (df["input_len"] == rin) & (df["concurrency"] == rc) & (df["prefix_len"] == 0) & (df["dataset"] == "random")
    p = pivot("output_throughput", cols="output_len", sub=sub)
    if p is not None and p.notna().any().any():
        p.columns = [f"out={int(c)}" for c in p.columns]
        out.append(f"### Throughput vs output length (in={rin}, c={rc})\n")
        out.append(md_table(p, "{:,.0f}"))

    # 5. Prefix caching workload
    pw = df[df["prefix_len"] > 0]
    if len(pw):
        out.append("## 5. Shared-prefix workload (prefix caching)\n")
        keep = [c for c in ["workload_label", "output_throughput", "median_ttft_ms", "p99_ttft_ms",
                            "median_tpot_ms", "prefix_hit_rate", "preemptions"] if c in pw.columns]
        t = pw.set_index("config")[keep].copy()
        if "prefix_hit_rate" in t.columns:
            t["prefix_hit_rate"] = t["prefix_hit_rate"] * 100
            t = t.rename(columns={"prefix_hit_rate": "prefix hit%"})
        out.append(md_table(t))

    # 6. ShareGPT / spec decode
    sg = df[df["dataset"] == "sharegpt"]
    if len(sg):
        out.append("## 6. ShareGPT workload (real text; speculative decoding acceptance)\n")
        keep = [c for c in ["output_throughput", "median_ttft_ms", "median_tpot_ms", "p99_itl_ms",
                            "spec_accept_rate", "spec_accepted_per_draft", "preemptions"] if c in sg.columns]
        t = sg.set_index("config")[keep].copy()
        if "spec_accept_rate" in t.columns:
            t["spec_accept_rate"] = t["spec_accept_rate"] * 100
            t = t.rename(columns={"spec_accept_rate": "spec acc%"})
        out.append(md_table(t))

    # 7. Failures (per-repeat rows, so a single failed repeat is listed as such)
    fails = [(n, f) for n, f in summary["server_facts"].items() if f.get("status") != "ok"]
    all_path = run_dir / "results_all.csv"
    bench_fail = pd.DataFrame()
    if all_path.exists():
        adf = pd.read_csv(all_path)
        if "bench_rc" in adf.columns:
            bench_fail = adf[adf["bench_rc"].fillna(0) != 0]
    if fails or len(bench_fail):
        out.append("## 7. Failures\n")
        for n, f in fails:
            out.append(f"- **{n}**: {f.get('error', f.get('status'))} — see `server_logs/{n}.log`")
        for _, rw in bench_fail.iterrows():
            out.append(f"- bench `{rw['config']}` / `{rw['workload_label']}` repeat {int(rw['repeat'])} "
                       f"rc={int(rw['bench_rc'])}: {str(rw.get('bench_error', ''))[:160]}")
        out.append("")

    (run_dir / "report.md").write_text("\n".join(out))
    print(f"[report] wrote {run_dir / 'report.md'}")

    if make_plots:
        try:
            make_plot_files(run_dir, df, summary, cfg_order)
        except Exception as e:  # matplotlib missing etc.
            print(f"[plots] skipped: {e}")


def make_plot_files(run_dir: Path, df, summary, cfg_order) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdir = run_dir / "plots"
    pdir.mkdir(exist_ok=True)
    rin, rout, rc = summary["reference"]
    sub = df[(df["input_len"] == rin) & (df["output_len"] == rout) & (df["prefix_len"] == 0) & (df["dataset"] == "random")]
    if len(sub):
        fig, ax = plt.subplots(figsize=(9, 5))
        for cfg in cfg_order:
            d = sub[sub["config"] == cfg].sort_values("concurrency")
            if len(d):
                ax.plot(d["concurrency"], d["output_throughput"], marker="o", label=cfg)
        ax.set_xscale("log", base=2); ax.set_xlabel("concurrency"); ax.set_ylabel("output tok/s")
        ax.set_title(f"Throughput vs concurrency (in={rin}, out={rout})"); ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(pdir / "throughput_vs_concurrency.png", dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for cfg in cfg_order:
            d = sub[sub["config"] == cfg]
            if len(d):
                ax.scatter(d["output_throughput"], d["median_tpot_ms"], label=cfg, s=28)
        ax.set_xlabel("output tok/s"); ax.set_ylabel("TPOT p50 (ms)")
        ax.set_title("Latency/throughput frontier across concurrency"); ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(pdir / "tpot_vs_throughput.png", dpi=130)
        plt.close(fig)
    print(f"[plots] wrote {pdir}")


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def aggregate(rows: list[dict], run_dir: Path) -> None:
    import pandas as pd

    all_df = pd.DataFrame(rows)
    all_df.to_csv(run_dir / "results_all.csv", index=False)
    key = ["config", "workload"]
    num_cols = [c for c in all_df.columns if c not in key and pd.api.types.is_numeric_dtype(all_df[c])]
    str_cols = [c for c in all_df.columns if c not in key and c not in num_cols]
    g = all_df.groupby(key, sort=False)
    med = g[num_cols].median()
    first = g[str_cols].first()
    n = g.size().rename("n_repeats")
    parts = [first, med, n]
    if "bench_rc" in all_df.columns:
        parts.append(g["bench_rc"].apply(lambda s: int((s.fillna(0) != 0).sum())).rename("n_failed"))
        med.drop(columns=["bench_rc"], inplace=True, errors="ignore")
    agg = pd.concat(parts, axis=1).reset_index()
    agg.to_csv(run_dir / "results.csv", index=False)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=EPILOG)
    p.add_argument("--family", choices=list(FAMILIES), default="qwen3",
                   help="model ladder for auto-selection (default qwen3: Apache-2.0, ungated)")
    p.add_argument("--model", help="HF id or local path; skips auto-selection")
    p.add_argument("--max-model-len", type=int, help="default: sized to the KV room left after weights")
    p.add_argument("--gpu-mem", type=float,
                   help="--gpu-memory-utilization (default 0.90 discrete, 0.75 unified memory)")
    p.add_argument("--tp", type=int, help="tensor-parallel size (default: smallest that fits)")
    p.add_argument("--dtype", help="default bfloat16 on sm_80+, float16 below")
    p.add_argument("--mm-off", choices=["language-model-only", "limit", "none"],
                   help="how to keep multimodal encoders out of a text benchmark (default per family)")
    p.add_argument("--grid", choices=list(GRIDS), help="workload grid (default standard)")
    p.add_argument("--quick", action="store_true", help="use the quick grid")
    p.add_argument("--configs", help="comma-separated config names to run")
    p.add_argument("--groups", help="comma-separated groups: graphs,kv,scheduler,spec,quant,parallel,attn")
    p.add_argument("--skip", help="comma-separated config names to skip")
    p.add_argument("--workloads", help="comma-separated workload names to run (see --list); "
                   "`reference` is always required")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=8031)
    p.add_argument("--out", default="results")
    p.add_argument("--run-id", help="default <hw>_<tag>_<timestamp>; pass an existing one with --resume")
    p.add_argument("--run-tag", default="run", help="middle part of the default run id")
    p.add_argument("--resume", action="store_true", help="skip (config, workload, repeat) with existing raw JSON")
    p.add_argument("--report-only", action="store_true", help="rebuild report.md for --run-id")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--slo", help='goodput SLO for vllm bench, e.g. "ttft:500 tpot:50" (ms)')
    p.add_argument("--sharegpt-path", help="ShareGPT_V3_unfiltered_cleaned_split.json; adds a real-text workload")
    p.add_argument("--spec-config", help="JSON for the spec_draft config's --speculative-config "
                   "(default derived from the family)")
    p.add_argument("--w4a16-model", help="4-bit checkpoint for the w4a16 config (default derived from the family)")
    p.add_argument("--warmup-prompts", type=int, default=8)
    p.add_argument("--startup-timeout", type=int, default=1800, help="seconds to wait for vllm serve")
    p.add_argument("--bench-timeout", type=int, default=3600)
    p.add_argument("--settle-s", type=float, default=20)
    p.add_argument("--drop-caches", dest="drop_caches", action="store_true", default=None,
                   help="sync + drop the page cache before each vllm serve (default on for unified "
                        "memory; needs passwordless sudo for tee /proc/sys/vm/drop_caches)")
    p.add_argument("--no-drop-caches", dest="drop_caches", action="store_false")
    p.add_argument("--allow-shared", action="store_true",
                   help="run even if the GPUs are busy or memory is short (results not clean)")
    p.add_argument("--extra-server-args", default="", help="appended to every vllm serve")
    p.add_argument("--extra-bench-args", default="", help="appended to every vllm bench serve")
    p.add_argument("--list", action="store_true",
                   help="show detected hardware, auto choices, config catalog and grid, then exit")
    p.add_argument("--dry-run", action="store_true", help="print commands without running")
    p.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)  # for run.sh
    return p.parse_args()


def resolve(a: argparse.Namespace) -> Hardware:
    """Fill every unset knob from the detected hardware (CLI values always win)."""
    hw = detect_hardware()
    a.hw = hw
    if a.gpu_mem is None:
        a.gpu_mem = (0.75 if hw.per_gpu_gib >= 64 else 0.6) if hw.unified else 0.90
    auto = autosize(hw, a.family, a.model, a.gpu_mem, a.tp)
    a.model, a.tp, a.weights_gib = auto["model"], auto["tp"], auto["weights_gib"]
    a.max_model_len = a.max_model_len or auto["max_model_len"]
    a.autosize_why = auto["why"]
    a.dtype = a.dtype or ("float16" if 0 < hw.min_cc < 8.0 else "bfloat16")
    fam = FAMILIES.get(family_of(a.model) or "")
    a.mm_off = a.mm_off or (fam["mm_off"] if fam else "none")
    a.grid = "quick" if a.quick else (a.grid or "standard")
    if a.drop_caches is None:
        a.drop_caches = hw.unified
    return hw


def default_run_id(hw: Hardware, tag: str) -> str:
    return f"{hw.slug}_{tag}_{dt.datetime.now():%Y%m%d_%H%M%S}"


def select_configs(a: argparse.Namespace, cfgs: list[ServerConfig], quiet: bool = False) -> list[ServerConfig]:
    want = set(a.configs.split(",")) if a.configs else None
    groups = set(a.groups.split(",")) if a.groups else None
    skip = set(a.skip.split(",")) if a.skip else set()
    out = []
    for c in cfgs:
        if c.name in skip:
            continue
        if want is not None and c.name not in want:
            continue
        if groups is not None and c.group not in groups and c.name != "baseline":
            continue
        if c.skip:
            if not quiet:
                print(f"[select] skipping {c.name}: {c.skip}")
            continue
        out.append(c)
    if out and not any(c.name == "baseline" for c in out):
        out.insert(0, next(c for c in cfgs if c.name == "baseline"))
        print("[select] baseline added so deltas can be computed")
    return out


def print_setup(a: argparse.Namespace, hw: Hardware) -> None:
    print(f"[hw]   {hw.describe()}  (via {hw.source}"
          + (f", CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
             if os.environ.get("CUDA_VISIBLE_DEVICES") else "") + ")")
    for w in a.autosize_why:
        print(f"[auto] {w}")
    print(f"[cfg]  model={a.model} tp={a.tp} dtype={a.dtype} max_model_len={a.max_model_len} "
          f"gpu_mem={a.gpu_mem} mm_off={a.mm_off} drop_caches={a.drop_caches}")
    if 0 < hw.min_cc < 8.0:
        print(f"[warn] sm_{str(hw.min_cc).replace('.', '')} is below sm_80: no BF16/FP8, vLLM V1 support "
              "is limited; numbers may not be representative.")


def main() -> None:
    a = parse_args()

    if a.report_only:
        if not a.run_id:
            raise SystemExit("--report-only needs --run-id")
        write_report(Path(a.out) / a.run_id, a.plots)
        return

    if a.probe:   # cheap: just enough for run.sh to name the run and prime sudo
        hw = detect_hardware()
        print(a.run_id or default_run_id(hw, a.run_tag), int(hw.unified))
        return

    hw = resolve(a)
    cfgs = build_configs(a, hw)
    wls = build_workloads(a)

    if a.list:
        print()
        print_setup(a, hw)
        print("\nCONFIG CATALOG (baseline + one knob each)\n")
        print(f"{'name':<22}{'group':<11}knob")
        for c in cfgs:
            print(f"{c.name:<22}{c.group:<11}{c.knob}" + (f"  [skip: {c.skip}]" if c.skip else ""))
        print(f"\nWORKLOAD GRID `{a.grid}` ({len(wls)} workloads, max_model_len={a.max_model_len})\n")
        for w in wls:
            print(f"  {w.name:<18}{w.label:<34}num_prompts={w.num_prompts}")
        return

    sel = select_configs(a, cfgs)
    run_id = a.run_id or default_run_id(hw, a.run_tag)
    run_dir = Path(a.out) / run_id
    print()
    print_setup(a, hw)

    if not a.dry_run:
        sel = preflight(a, hw, sel)

    raw_dir, log_dir = run_dir / "raw", run_dir / "server_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ref = GRIDS[a.grid]["reference"]
    ref_label = next(w.label for w in wls if w.name == "reference")
    summary = dict(run_id=run_id, model=a.model, family=family_of(a.model), tp=a.tp, dtype=a.dtype,
                   weights_gib=a.weights_gib, autosize_why=a.autosize_why, grid=a.grid,
                   max_model_len=a.max_model_len, gpu_mem=a.gpu_mem, repeats=a.repeats,
                   reference=list(ref), reference_label=ref_label,
                   configs=[c.to_dict() for c in sel],
                   workloads=[dict(name=w.name, label=w.label, input_len=w.input_len,
                                   output_len=w.output_len, concurrency=w.concurrency,
                                   prefix_len=w.prefix_len, num_prompts=w.num_prompts,
                                   dataset=w.dataset) for w in wls],
                   server_facts={},
                   environment=environment(hw) if not a.dry_run else dict(hardware=hw.to_dict()))
    summary_path = run_dir / "summary.json"
    prev_facts: dict = {}
    if a.resume and summary_path.exists():
        try:
            prev_facts = json.loads(summary_path.read_text()).get("server_facts", {})
        except Exception:
            prev_facts = {}

    def save_summary():
        summary_path.write_text(json.dumps(summary, indent=1))

    total_runs = len(sel) * len(wls) * a.repeats
    print(f"\n[run] {run_id}: {len(sel)} configs x {len(wls)} workloads x {a.repeats} repeats = {total_runs} bench runs")
    print(f"[run] outputs -> {run_dir}\n")

    # On --resume, rows are rebuilt from raw/*.json inside run_bench (it short-
    # circuits when the result file exists), so nothing is preloaded here.
    rows: list[dict] = []
    done = 0
    for ci, cfg in enumerate(sel, 1):
        model = cfg.model_override or a.model
        log_path = log_dir / f"{cfg.name}.log"
        server = Server(cfg, model, a, log_path)
        print(f"=== [{ci}/{len(sel)}] {cfg.name}  ({cfg.group}: {cfg.knob})")
        if a.resume and all((raw_dir / f"{cfg.name}__{w.name}__r{rep}.json").exists()
                            for w in wls for rep in range(a.repeats)):
            print("  all results present, reusing (no server start)")
            for w in wls:
                for rep in range(a.repeats):
                    done += 1
                    rows.append(run_bench(a, server, cfg, model, w, rep, raw_dir))
            if cfg.name in prev_facts:
                summary["server_facts"][cfg.name] = prev_facts[cfg.name]
            aggregate(rows, run_dir)
            continue
        if a.dry_run:
            print("  $ " + shlex.join(server.command()))
            if cfg.env:
                print("    env: " + json.dumps(cfg.env))
            for w in wls:
                print("  $ " + shlex.join(bench_command(a, model, w, raw_dir, f"{cfg.name}__{w.name}__r0.json")))
            continue

        facts = server.start()
        summary["server_facts"][cfg.name] = facts
        save_summary()
        if facts.get("status") != "ok":
            print(f"  !! server failed: {facts.get('error')}  (full log: {log_path})")
            print(explain_failure(log_path))
            server.stop()
            continue
        print(f"  server up in {facts['startup_s']}s"
              + (f", KV cache {int(facts['kv_cache_tokens']):,} tokens" if "kv_cache_tokens" in facts else "")
              + (f", compile {facts['compile_s']}s" if "compile_s" in facts else "")
              + (f", graphs {facts['cudagraph_capture_s']}s" if "cudagraph_capture_s" in facts else ""))

        try:
            warm_wl = next(w for w in wls if w.name == "reference")
            warmup(a, model, warm_wl, run_dir)
            for w in wls:
                for rep in range(a.repeats):
                    done += 1
                    t0 = time.time()
                    row = run_bench(a, server, cfg, model, w, rep, raw_dir)
                    rows.append(row)
                    thr = row.get("output_throughput", float("nan"))
                    tpot = row.get("median_tpot_ms", float("nan"))
                    ttft = row.get("median_ttft_ms", float("nan"))
                    print(f"  [{done}/{total_runs}] {w.label:<34} out={thr:8.1f} tok/s  "
                          f"ttft50={ttft:7.1f}ms  tpot50={tpot:6.2f}ms  "
                          f"pre={row.get('preemptions', float('nan')):.0f}  ({time.time()-t0:.0f}s)")
                    aggregate(rows, run_dir)
        except KeyboardInterrupt:
            print("\n[run] interrupted; stopping server and writing partial results")
            server.stop()
            break
        finally:
            server.stop()

    if rows:
        aggregate(rows, run_dir)
        save_summary()
        write_report(run_dir, a.plots)
    print(f"\n[run] done -> {run_dir}")


if __name__ == "__main__":
    main()
