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
import functools
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
    # functional check: smallest grid that still exercises every path
    "check": dict(input_lens=[256], output_lens=[64], concurrencies=[1, 4],
                  reference=(256, 64, 4),
                  prefix=(512, 128, 32, 4)),
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
    expect: dict[str, str] = field(default_factory=dict)    # engine-reported fact -> value the knob must produce

    def to_dict(self) -> dict:
        return dict(name=self.name, group=self.group, knob=self.knob,
                    args=self.args, env=self.env,
                    model_override=self.model_override, note=self.note, expect=self.expect)


# ----------------------------------------------------------------------------
# CONFIG CATALOG — baseline + one knob each
# ----------------------------------------------------------------------------

def build_configs(a: argparse.Namespace, hw: Optional[Hardware] = None) -> list[ServerConfig]:
    mml = str(a.max_model_len)
    C = ServerConfig
    d_mnbt, d_seqs = vllm_default_batching(hw)
    mnbt_small = max(256, d_mnbt // 4)
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
        C("compile_O0", "graphs", "-O0: no torch.compile (graphs per version semantics)",
          ["-O0"], note="from vLLM 0.12 -O0 also drops CUDA graphs, i.e. equals enforce_eager: the check "
                        "marks it NOOP then"),
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

        # --- scheduler (relative to vLLM's real defaults on this GPU/version) ---
        C("mnbt_small", "scheduler",
          f"max_num_batched_tokens={mnbt_small} (default here {d_mnbt}): smaller prefill chunks",
          ["--max-num-batched-tokens", str(mnbt_small)], expect={"cfg_mnbt": str(mnbt_small)}),
        C("mnbt_max_model_len", "scheduler",
          f"max_num_batched_tokens={mml} (default here {d_mnbt}; approximates chunked prefill OFF)",
          ["--max-num-batched-tokens", mml], expect={"cfg_mnbt": mml},
          skip=None if int(mml) > d_mnbt else f"max_model_len {mml} <= the default budget {d_mnbt}"),
        C("max_seqs_32", "scheduler", f"max_num_seqs=32 (default here {d_seqs}): caps concurrency",
          ["--max-num-seqs", "32"]),
        C("max_seqs_512", "scheduler",
          f"max_num_seqs=512 (default here {d_seqs}; "
          + ("more CUDA-graph sizes, less KV)" if d_seqs < 512 else "lower cap)"),
          ["--max-num-seqs", "512"], skip=None if d_seqs != 512 else "512 is already the default here"),
        async_cfg(),

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
        attn_cfg("attn_flashinfer", "FLASHINFER"),
        attn_cfg("attn_flash_attn", "FLASH_ATTN"),
        attn_cfg("attn_triton", "TRITON_ATTN"),
    ]
    return cfgs


def attn_cfg(name: str, backend: str) -> "ServerConfig":
    """vLLM < 0.13 selects the backend with VLLM_ATTENTION_BACKEND; 0.13-0.14 accept both; from
    0.15 the env var is gone and silently ignored, so only --attention-backend works."""
    v = vllm_version()
    if v is not None and v < (0, 13):
        return ServerConfig(name, "attn", f"VLLM_ATTENTION_BACKEND={backend}", env={"VLLM_ATTENTION_BACKEND": backend})
    return ServerConfig(name, "attn", f"--attention-backend {backend}", ["--attention-backend", backend])


def vllm_default_batching(hw: Optional[Hardware]) -> tuple[int, int]:
    """vLLM's `vllm serve` defaults for (max_num_batched_tokens, max_num_seqs) on this GPU
    (EngineArgs._set_default_max_num_seqs_and_batched_tokens_args, v0.31): >=160 GiB -> 16384/1024,
    >=70 GiB and not A100 -> 8192/1024, else 2048/256. A knob set to the default is a no-op."""
    mem = hw.per_gpu_gib if hw and hw.gpus else 0.0
    name = str(hw.gpus[0].get("name", "")).lower() if hw and hw.gpus else ""
    if mem >= 160:
        return 16384, 1024
    if mem >= 70 and "a100" not in name:
        return 8192, 1024
    return 2048, 256


def async_cfg() -> "ServerConfig":
    """Async scheduling is ON by default from vLLM 0.12 (unless something incompatible is
    configured), so from then on the ablation turns it off; before that it turns it on."""
    v = vllm_version()
    if v is None or v >= (0, 12):
        return ServerConfig("no_async_scheduling", "scheduler",
                            "async scheduling OFF (on by default since vLLM 0.12)", ["--no-async-scheduling"])
    return ServerConfig("async_scheduling", "scheduler", "async scheduling ON (overlap CPU scheduling with GPU)",
                        ["--async-scheduling"])


@functools.lru_cache(maxsize=1)
def vllm_version() -> Optional[tuple[int, int]]:
    """(major, minor) of the vllm on PATH, or None if it can't be read."""
    out = sh(["vllm", "--version"]) or sh([sys.executable, "-c", "import vllm;print(vllm.__version__)"])
    m = re.search(r"^\s*v?(\d+)\.(\d+)\.\d+", out or "", flags=re.M)   # a bare version line, not log noise
    return (int(m.group(1)), int(m.group(2))) if m else None


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


def http_post_json(url: str, payload: Optional[dict] = None, timeout: float = 30.0) -> tuple[int, object]:
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:
                return r.status, json.loads(raw or b"null")
            except ValueError:
                return r.status, None
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return 0, None


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
    "compile_s": r"torch\.compile (?:takes|took)\s*([\d.]+)\s*s",
    "engine_init_s": r"init engine \([^)]*\) took\s*([\d.]+)\s*s",
    "cudagraph_capture_s": r"Graph capturing finished in\s*([\d.]+)\s*secs",
    # what the engine actually resolved (from the "Initializing a V1 LLM engine" config dump),
    # so the report can prove each knob took effect
    "cfg_compile_mode": r"'mode': <CompilationMode\.(\w+)",
    "cfg_cudagraph_mode": r"'cudagraph_mode': <CUDAGraphMode\.(\w+)",
    "cfg_enforce_eager": r"enforce_eager=(\w+)",
    "cfg_kv_cache_dtype": r"kv_cache_dtype=(\w+)",
    "cfg_prefix_caching": r"enable_prefix_caching=(\w+)",
    "cfg_quantization": r"\bquantization=(\w+)",
    "cfg_speculative": r"speculative_config=(\w+)",
    "cfg_mnbt": r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)",
    # "Using FlashInfer backend", "Using Flash Attention backend on V1 engine",
    # "Using FLASH_ATTN attention backend out of potential backends: [...]" (auto-selected),
    # "Using AttentionBackendEnum.FLASHINFER backend." (vLLM >= 0.15, chosen with --attention-backend)
    "attention_backend": r"Using\s+(?:AttentionBackendEnum\.)?([A-Za-z0-9_]+(?: Attention)?)\s+(?:attention\s+)?backend",
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
            facts[key] = float(val) if not (key == "attention_backend" or key.startswith("cfg_")) else val
        except ValueError:
            facts[key] = val
    # compiled graph loaded from vLLM's on-disk cache (fast start) vs compiled fresh
    facts["compile_cached"] = bool(re.search(
        r"Directly load AOT compilation|from standalone compile artifacts|Loading .*compil.* from cache",
        txt, flags=re.IGNORECASE))
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
    (r"Could not find nvcc|nvcc: not found|No such file or directory: .nvcc",
     "FlashInfer needs the CUDA toolkit (nvcc) to JIT kernels for this GPU. Install it "
     "(`./run.sh setup` prints the command for your distro), or point CUDA_HOME at an existing one."),
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
    (r"unrecognized arguments|error: argument|invalid choice",
     "A CLI flag isn't accepted by this vLLM version; see the argparse line in the log and "
     "adjust build_configs() / Server.command()."),
    (r"FlashInfer backend is not available",
     "vLLM treats FlashInfer as missing: its has_flashinfer() needs `nvcc` on PATH (or the "
     "flashinfer-cubin package), even when FlashInfer's own JIT finds /usr/local/cuda. Run via "
     "./run.sh (which adds the toolkit to PATH) or `export PATH=/usr/local/cuda/bin:$PATH`."),
    (r"recompile limit|Hard failure due to (fullgraph=True|fail_on_recompile_limit_hit)",
     "torch.compile hit Dynamo's recompile limit (8 by default): every new input shape forced a "
     "recompile, so the compiled path never settles. Typical of static-shape compile on varying "
     "batch/sequence sizes."),
    (r"requires `?accelerate`?|No module named '(accelerate|torchao)'",
     "Missing Python package in the vLLM venv: `./run.sh setup` (re)installs it, or "
     "`uv pip install accelerate torchao`."),
    # keep this generic one last: explain_failure() drops it when a specific hint matched
    (r"not supported|NotImplementedError|Unsupported",
     "Feature/arch not supported by this build (check the line above for which)."),
]


def explain_failure(path: Path, n: int = 30) -> str:
    try:
        txt = path.read_text(errors="replace")
    except Exception:
        return "  (no server log)"
    lines = txt.splitlines()
    err_re = re.compile(r"Error|Exception|Traceback|error:|FAILED")
    err_lines = [l for l in lines if err_re.search(l)]
    out = []
    # the last INFO line before the first error says which startup stage died
    first_err = next((i for i, l in enumerate(lines) if err_re.search(l)), None)
    # match causes against the failure itself, not startup WARNINGs ("X not supported, falling back")
    failure = "\n".join(lines[first_err:]) if first_err is not None else "\n".join(lines[-n:])
    hints = [h for pat, h in KNOWN_FAILURES if re.search(pat, failure, flags=re.IGNORECASE)]
    if len(hints) > 1:                       # the generic "not supported" hint only when nothing better
        hints = [h for h in hints if h != KNOWN_FAILURES[-1][1]]
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

    def gpu_mem(self) -> float:
        """The memory fraction this server actually asks for (a config may override the base)."""
        args = self.cfg.args
        if "--gpu-memory-utilization" in args:
            return float(args[args.index("--gpu-memory-utilization") + 1])
        return float(self.a.gpu_mem)

    def env(self) -> dict:
        e = dict(os.environ)
        e.update(nvcc_env(e))
        # /reset_prefix_cache is a dev-only route (vLLM registers it only with VLLM_SERVER_DEV_MODE=1);
        # without it the reset silently 404s and repeat 2 re-sends repeat 1's prompts into a warm cache
        e.setdefault("VLLM_SERVER_DEV_MODE", "1")
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
        hw = getattr(self.a, "hw", None)
        if hw is not None and not hw.unified and not getattr(self.a, "dry_run", False):
            ids = hw.smi_ids(self.a.tp)
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

    def reset_prefix_cache(self) -> bool:
        """True once vLLM confirms the cache is empty. It answers success=false while blocks are
        still held (requests finishing), so retry briefly; a 404 means dev mode is off."""
        for _ in range(20):
            st, body = http_post_json(f"http://127.0.0.1:{self.port}/reset_prefix_cache", timeout=30)
            if st == 200 and (not isinstance(body, dict) or body.get("success", True)):
                return True
            if st in (404, 405):
                break
            time.sleep(1)
        if not getattr(self, "_reset_warned", False):
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


def gpu_compute_apps(smi_ids: list[str]) -> list[dict]:
    """Processes holding memory on the GPUs we're about to use (discrete GPUs)."""
    if not smi_ids:
        return []
    uuids = sh(["nvidia-smi", "-i", ",".join(smi_ids), "--query-gpu=uuid",
                "--format=csv,noheader"]).split()
    out = sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
              "--format=csv,noheader,nounits"])
    rows = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4 and p[0] in uuids:
            mib = float(p[3]) if re.fullmatch(r"[\d.]+", p[3]) else None
            rows.append(dict(pid=p[1], name=p[2], mib=mib))
    return rows


# Processes that block a clean benchmark: anything ML-shaped, or anything holding real
# memory. Desktop clients (file manager, browser compositor, ...) at tens of MiB don't.
_ML_PROC = re.compile(r"python|vllm|ollama|llama|sglang|triton|text-generation|comfy|torch|"
                      r"jupyter|lm-?studio|kobold|exllama|tensorrt|trtllm|nemo", re.I)
BUSY_MIB = 1024
# vLLM's startup check (free >= gpu_mem x total) runs after its own CUDA context exists
# (~0.5-0.8 GiB), and a GPU that drives a desktop drifts by a few hundred MiB over a sweep.
CUDA_CTX_MIB = 768
DISCRETE_HEADROOM_MIB = 1536
# memory.used per GPU when the sweep started (preflight): "settled" means back near this
_IDLE_USED_MIB: dict[int, float] = {}


def gpu_mem_state(smi_ids: Optional[list[str]]) -> dict[int, tuple[float, float]]:
    """{smi_index: (used_mib, total_mib)} for the given GPUs (all when smi_ids is None)."""
    want = {int(i) for i in smi_ids} if smi_ids else None
    return {r["smi_index"]: (r["used_mib"], r["total_mib"]) for r in _smi_rows()
            if r["used_mib"] is not None and r["total_mib"] and (want is None or r["smi_index"] in want)}


def _fmt_app(r: dict) -> str:
    mib = "?" if r["mib"] is None else "%.0f" % r["mib"]
    return "pid %s %s (%s MiB)" % (r["pid"], r["name"], mib)


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


def find_nvcc() -> Optional[str]:
    """Where FlashInfer's JIT will look: $CUDA_HOME, PATH, then /usr/local/cuda."""
    for c in (os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "nvcc") if os.environ.get("CUDA_HOME") else None,
              sh(["which", "nvcc"]) or None, "/usr/local/cuda/bin/nvcc"):
        if c and os.path.exists(c):
            return c
    return None


def nvcc_env(env: dict) -> dict:
    """PATH/CUDA_HOME additions so vLLM sees the toolkit. FlashInfer's JIT falls back to
    /usr/local/cuda on its own, but vLLM's has_flashinfer() checks `which nvcc`: with nvcc off
    PATH it treats FlashInfer as missing, and anything routed to it (FP8 KV cache on sm_120,
    --attention-backend FLASHINFER) dies with "FlashInfer backend is not available"."""
    nvcc = find_nvcc()
    if not nvcc:
        return {}
    bindir = os.path.dirname(nvcc)
    path = env.get("PATH", "")
    out = {}
    if bindir not in path.split(os.pathsep):
        out["PATH"] = bindir + os.pathsep + path
    if not env.get("CUDA_HOME"):
        out["CUDA_HOME"] = os.path.dirname(bindir)
    return out


def preflight(a: argparse.Namespace, hw: Hardware, cfgs: list[ServerConfig]) -> list[ServerConfig]:
    """Refuse to benchmark on a busy machine, verify model access, and drop configs
    whose extra checkpoints can't be pulled. Another server on the box both steals the
    memory vLLM needs and shares memory bandwidth, contaminating every decode number."""
    if not hw.gpus:
        raise SystemExit("[preflight] no CUDA GPU visible (check nvidia-smi / CUDA_VISIBLE_DEVICES)")

    # Blackwell (sm_100+) has FlashInfer kernels that are JIT-compiled at startup: no nvcc,
    # every config dies ~2 min into its server start. Older archs mostly ship prebuilt.
    nvcc = find_nvcc()
    if nvcc:
        added = " (not on PATH: adding its directory for vLLM, whose FlashInfer check uses PATH)" \
            if "PATH" in nvcc_env(dict(os.environ)) else ""
        print(f"[preflight] nvcc: {nvcc}{added}")
    elif a.nvcc_check and hw.min_cc >= 10.0:
        raise SystemExit(
            f"[preflight] no CUDA toolkit (nvcc) found, and this GPU (sm_{str(hw.min_cc).replace('.', '')}) "
            "needs it: FlashInfer JIT-compiles its Blackwell kernels at server start.\n"
            "  `./run.sh setup` prints the install command for this distro, or set CUDA_HOME to an\n"
            "  existing toolkit. --no-nvcc-check to try anyway.")
    else:
        print("[preflight] warning: no nvcc found; fine unless a backend needs to JIT kernels")

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
        busy = [r for r in apps if _ML_PROC.search(r["name"]) or (r["mib"] or 0) >= BUSY_MIB]
        minor = [r for r in apps if r not in busy]
        if busy:
            problems.append(f"processes already on GPU(s) {','.join(ids)}:\n"
                            + "\n".join("      " + _fmt_app(r) for r in busy))
        if minor:
            print(f"[preflight] ignoring {len(minor)} small non-ML GPU process(es) (desktop etc.): "
                  + ", ".join(_fmt_app(r) for r in minor[:4]))
        for idx, (used, total) in gpu_mem_state(ids).items():
            _IDLE_USED_MIB[idx] = used
            if a.gpu_mem <= 0:
                continue
            free = (total - used) / 1024
            need = (a.gpu_mem * total + CUDA_CTX_MIB) / 1024
            if free < need:
                fit = math.floor((total - used - DISCRETE_HEADROOM_MIB) / total * 100) / 100
                problems.append(f"GPU {idx}: {free:.1f} GiB free, but --gpu-mem {a.gpu_mem} needs "
                                f"{need:.1f} GiB ({a.gpu_mem} x {total / 1024:.1f} GiB + ~{CUDA_CTX_MIB} MiB "
                                f"CUDA context)" + (f"; GPU_MEM={fit} would fit" if fit >= 0.3 else ""))
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
    while time.time() - t0 < max_s:
        st = gpu_mem_state(smi_ids)
        # settled = back within 512 MiB of what was in use before the sweep (desktop etc.)
        if st and all(used <= _IDLE_USED_MIB.get(i, 2488) + 512 for i, (used, _) in st.items()):
            time.sleep(3)
            return
        time.sleep(3)


def wait_gpu_free(gpu_mem: float, smi_ids: list[str], max_s: float = 60) -> Optional[str]:
    """Before a launch on discrete GPUs: wait until each GPU has room for gpu_mem x total plus
    vLLM's CUDA context. Returns a warning if it never does (something else grew)."""
    t0, short = time.time(), {}
    while True:
        st = gpu_mem_state(smi_ids)
        short = {i: (t - u, gpu_mem * t + CUDA_CTX_MIB) for i, (u, t) in st.items()
                 if t - u < gpu_mem * t + CUDA_CTX_MIB}
        if not short or time.time() - t0 >= max_s:
            break
        time.sleep(3)
    if not short:
        return None
    return "; ".join(f"GPU {i}: {f / 1024:.1f} GiB free < {n / 1024:.1f} GiB needed (gpu_mem {gpu_mem} "
                     f"+ CUDA context)" for i, (f, n) in short.items()) + \
        ". Something else grew on the GPU since the sweep started (browser/desktop?); starting anyway."


# ----------------------------------------------------------------------------
# Benchmark runs
# ----------------------------------------------------------------------------

def bench_command(a: argparse.Namespace, model: str, wl: Workload, out_dir: Path,
                  fname: str, num_prompts: Optional[int] = None,
                  meta: Optional[dict] = None, rep: int = 0) -> list[str]:
    cmd = ["vllm", "bench", "serve",
           "--backend", "vllm", "--host", "127.0.0.1", "--port", str(a.port),
           "--model", model, "--tokenizer", model,
           "--num-prompts", str(num_prompts or wl.num_prompts),
           "--max-concurrency", str(wl.concurrency),
           "--request-rate", "inf",
           "--ignore-eos",
           "--seed", str(a.seed + 1000 * rep),     # repeat r gets its own prompts; every config the same ones
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
        write_row(raw_dir, row)
        return row

    reset_ok = server.reset_prefix_cache()
    before = scrape_metrics(a.port)
    cmd = bench_command(a, model, wl, raw_dir, fname, rep=rep,
                        meta=dict(config=cfg.name, workload=wl.label, repeat=rep,
                                  prefix_reset_ok=float(reset_ok)))
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
    meta_path.write_text(json.dumps(dict(deltas, bench_wall_s=wall, bench_rc=rc,
                                         prefix_reset_ok=float(reset_ok)), indent=1))
    write_row(raw_dir, row)
    return row


def write_row(raw_dir: Path, row: dict) -> None:
    """One file per (config, workload, repeat): the on-disk source of truth that lets several
    processes (vLLM sweep, PyTorch bench) contribute to one run directory."""
    name = f"{row['config']}__{row['workload']}__r{int(row.get('repeat') or 0)}.row.json"
    (raw_dir / name).write_text(json.dumps(row, default=float))


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


# Which serving features each config has. PyTorch rows climb the shared-feature ladder;
# vLLM rows are the baseline plus the ablations that switch one shared feature off/on.
FEATURE_COLS = ["fused attn", "KV cache", "torch.compile", "CUDA graphs", "batching",
                "prefix cache", "chunked prefill", "quant", "spec decode"]
FEATURE_MATRIX = {
    "pt_eager":           ["–", "dynamic", "–", "–", "static", "–", "–", "–", "–"],
    "pt_sdpa":            ["✓", "dynamic", "–", "–", "static", "–", "–", "–", "–"],
    "pt_static":          ["✓", "static", "–", "–", "static", "–", "–", "–", "–"],
    "pt_compile":         ["✓", "static", "✓", "✓", "static", "–", "–", "–", "–"],
    "pt_compile_fp8":     ["✓", "static", "✓", "✓", "static", "–", "–", "fp8/int8", "–"],
    "pt_assisted":        ["✓", "dynamic", "–", "–", "bs=1", "–", "–", "–", "draft"],
    "pt_cb":              ["✓", "paged", "–", "auto", "continuous", "✓", "✓", "–", "–"],
    "pt_cb_compile":      ["✓", "paged", "✓", "auto", "continuous", "✓", "✓", "–", "–"],
    "enforce_eager":      ["✓", "paged", "–", "–", "continuous", "✓", "✓", "–", "–"],
    "cudagraph_none":     ["✓", "paged", "✓", "–", "continuous", "✓", "✓", "–", "–"],
    "no_prefix_cache":    ["✓", "paged", "✓", "✓", "continuous", "–", "✓", "–", "–"],
    "mnbt_max_model_len": ["✓", "paged", "✓", "✓", "continuous", "✓", "≈off", "–", "–"],
    "baseline":           ["✓", "paged", "✓", "✓", "continuous", "✓", "✓", "–", "–"],
    "fp8_online":         ["✓", "paged", "✓", "✓", "continuous", "✓", "✓", "fp8", "–"],
    "spec_draft":         ["✓", "paged", "✓", "✓", "continuous", "✓", "✓", "–", "draft"],
}


def pytorch_vs_vllm_section(df, summary: dict, cfg_order: list, ref: str) -> list[str]:
    import pandas as pd

    present = set(df["config"].astype(str))
    order = [c for c in FEATURE_MATRIX if c in present]
    out = ["## 2b. PyTorch vs vLLM (shared features)\n",
           "PyTorch rows use only what PyTorch/transformers provide, climbing one shared feature at a "
           "time; vLLM rows are the baseline and the ablations that switch the same features off. "
           "Static-batching PyTorch rows run concurrency C as batches of C (TTFT = the batch's "
           "prefill), measuring up to `--max-batches` batches; continuous-batching and vLLM rows use "
           "a closed-loop client at concurrency C over all prompts.\n"]
    fm = pd.DataFrame([FEATURE_MATRIX[c] for c in order], index=order, columns=FEATURE_COLS)
    fm.index.name = "config"
    out.append(md_table(fm))

    r = df[df["workload_label"] == ref].copy()
    r["config"] = r["config"].astype(str)
    r = r.set_index("config")
    rows = []
    eager = r.loc["pt_eager", "output_throughput"] if "pt_eager" in r.index else float("nan")
    base = r.loc["baseline", "output_throughput"] if "baseline" in r.index else float("nan")
    for c in order:
        if c not in r.index:
            continue
        x = r.loc[c]
        thr = x.get("output_throughput", float("nan"))
        err = x.get("bench_error")
        err = err if isinstance(err, str) else ""          # NaN when the run succeeded
        rows.append(pd.Series({
            "engine": "PyTorch" if c.startswith("pt_") else "vLLM",
            "out tok/s": thr if not err else float("nan"),
            "TTFT p50": x.get("median_ttft_ms", float("nan")),
            "TPOT p50": x.get("median_tpot_ms", float("nan")),
            "E2E p50": x.get("median_e2el_ms", float("nan")),
            "peak GiB": x.get("peak_mem_gib", float("nan")),
            "× pt_eager": thr / eager if eager == eager and eager and not err else float("nan"),
            "% of vLLM": 100 * thr / base if base == base and base and not err else float("nan"),
            "note": "OOM" if err.startswith("OOM") else (err[:40] if err else ""),
        }, name=c))
    if rows:
        t = pd.DataFrame(rows)
        t.index.name = "config"
        out.append(f"### Head-to-head at the reference workload `{ref}`\n")
        out.append(md_table(t, "{:,.1f}"))
        pt = t[t["engine"] == "PyTorch"]["out tok/s"].dropna()
        if len(pt) and base == base and base:
            best = pt.idxmax()
            out.append(f"Best PyTorch config: **{best}** at {pt.max():,.0f} tok/s = "
                       f"{100 * pt.max() / base:.0f}% of the vLLM baseline ({base:,.0f} tok/s).\n")

    rin, rout, _ = summary["reference"]
    sub = df[(df["input_len"] == rin) & (df["output_len"] == rout) & (df["prefix_len"] == 0)
             & (df["dataset"] == "random") & (df["config"].astype(str).isin(order))]
    if len(sub):
        p = sub.pivot_table(index="config", columns="concurrency", values="output_throughput",
                            aggfunc="median", observed=True)
        p.index = p.index.astype(str)
        p = p.reindex([c for c in order if c in p.index])
        p = p[sorted(p.columns)]
        p.columns = [f"c={int(c)}" for c in p.columns]
        out.append(f"### Output throughput (tok/s) vs concurrency (in={rin}, out={rout})\n")
        out.append(md_table(p, "{:,.0f}"))
        out.append("Missing cells: the config can't run that shape (static batch OOM, or assisted "
                   "generation at batch > 1).\n")
    if "compiles_in_measure" in df.columns:
        cm = df[(df["config"].astype(str).str.startswith("pt_"))
                & (df.get("compile_s_in_measure", pd.Series(0, index=df.index)).fillna(0) >= COMPILE_WARN_S)]
        if len(cm):
            g = cm.groupby(cm["config"].astype(str)).agg(workloads=("workload_label", "nunique"),
                                                         compiles=("compiles_in_measure", "sum"),
                                                         compile_s=("compile_s_in_measure", "sum"))
            out.append("**Caution: still compiling while being measured.** These PyTorch configs ran "
                       "torch.compile inside the measured window on some workloads (new shapes forcing "
                       "recompiles), so those numbers include compiler time: "
                       + "; ".join(f"`{c}` {int(x.compiles)} compiles / {x.compile_s:.0f}s over "
                                   f"{int(x.workloads)} workloads" for c, x in g.iterrows()) + ".\n")
    return out


# What each config's own engine must report for the knob to count as applied (a flag that is
# silently ignored would otherwise "pass" and produce baseline numbers under another name).
def _is(x):
    return lambda v: str(v).lower() == x


def _has(x):
    return lambda v: x in str(v).lower()


def _set(v):
    return str(v) not in ("None", "", "nan")


# Compile work inside a measured window only matters if it takes real time: Dynamo also fires its
# compile callbacks for near-instant cache hits / guard rechecks (seen: 3 "compiles", 0.0 s).
COMPILE_WARN_S = 1.0

# Configs that can silently collapse into another one on some GPU/version: if every listed
# engine-reported fact matches the other config's, the check marks it NOOP and the suite drops it.
SAME_AS = {
    "compile_O0": ("enforce_eager", ["cfg_compile_mode", "cfg_cudagraph_mode"]),   # -O0 = no compile, no graphs
    "mnbt_small": ("baseline", ["cfg_mnbt"]),
    "mnbt_max_model_len": ("baseline", ["cfg_mnbt"]),
}

KNOB_EXPECT = {
    "enforce_eager": ("cfg_enforce_eager", _is("true"), "enforce_eager=True"),
    "cudagraph_none": ("cfg_cudagraph_mode", _is("none"), "cudagraph_mode=NONE"),
    "no_prefix_cache": ("cfg_prefix_caching", _is("false"), "prefix caching off"),
    "kv_fp8": ("cfg_kv_cache_dtype", _has("fp8"), "kv_cache_dtype=fp8*"),
    "fp8_online": ("cfg_quantization", _has("fp8"), "quantization=fp8"),
    "fp8_ckpt": ("cfg_quantization", _set, "quantized checkpoint detected"),
    "w4a16": ("cfg_quantization", _set, "quantized checkpoint detected"),
    "spec_ngram": ("cfg_speculative", _set, "speculative_config set"),
    "spec_draft": ("cfg_speculative", _set, "speculative_config set"),
    "attn_flashinfer": ("attention_backend", _has("flashinfer"), "FlashInfer backend in log"),
    "attn_flash_attn": ("attention_backend",
                        lambda v: any(s in str(v).lower() for s in ("flash_attn", "flash attention"))
                        and "flashinfer" not in str(v).lower(), "FlashAttention backend in log"),
    "attn_triton": ("attention_backend", _has("triton"), "Triton backend in log"),
    "pt_compile": ("compiled", _is("true"), "torch.compile applied"),
    "pt_compile_fp8": ("quant", _set, "torchao quantization applied"),
    "pt_cb_compile": ("compiled", _is("true"), "torch.compile applied"),
}


def harness_fingerprint() -> str:
    """Hash of the harness code, stored with a check so `suite` never reuses a check that was
    made with different config definitions."""
    import hashlib
    h = hashlib.sha1()
    for f in ("vllm_ablation.py", "pytorch_bench.py"):
        p = Path(__file__).resolve().parent / f
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()[:12]


def refresh_log_facts(run_dir: Path, summary: dict) -> dict:
    """Re-parse each vLLM server log with the current LOG_FACTS, so a run made with an older
    harness (e.g. a log format it didn't know) picks up fixes at report time."""
    changed = False
    for name, f in summary.get("server_facts", {}).items():
        log = run_dir / "server_logs" / f"{name}.log"
        if name.startswith("pt_") or not isinstance(f, dict) or not log.exists():
            continue
        for k, v in parse_server_log(log).items():
            if f.get(k) != v:
                f[k], changed = v, True
    if changed:
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def write_checklist(run_dir: Path) -> dict:
    """PASS / WARN / FAIL / SKIP per config from a (check-grid) run; writes check.md and
    check_passed.json (configs safe to include in the full sweep)."""
    import pandas as pd

    if not (run_dir / "summary.json").exists():
        raise SystemExit(f"[check] {run_dir} has no summary.json: nothing ran")
    summary = refresh_log_facts(run_dir, json.loads((run_dir / "summary.json").read_text()))
    facts_all = summary.get("server_facts", {})
    df = pd.read_csv(run_dir / "results_all.csv") if (run_dir / "results_all.csv").exists() else None
    rows, passed = [], {"vllm": [], "pytorch": []}
    names = [c["name"] for c in summary["configs"]]
    cfg_by_name = {c["name"]: c for c in summary["configs"]}
    for name in names + [n for n in summary.get("skipped", {}) if n not in names]:
        engine = "pytorch" if name.startswith("pt_") else "vllm"
        f = facts_all.get(name, {})
        skip = summary.get("skipped", {}).get(name)
        sub = df[df["config"] == name] if df is not None and len(df) else None
        n_ok = int((sub["bench_rc"].fillna(0) == 0).sum()) if sub is not None and len(sub) else 0
        n_all = int(len(sub)) if sub is not None else 0
        knob, detail = "–", ""
        if skip:
            status, detail = "SKIP", skip
        elif f.get("status") != "ok":
            status, detail = "FAIL", str(f.get("error") or "no server facts (did it run?)")[:120]
        elif n_all == 0 or n_ok < n_all:
            bad = sub[sub["bench_rc"].fillna(0) != 0] if n_all else None
            err = str(bad["bench_error"].iloc[0])[:120] if bad is not None and len(bad) else "no workload rows"
            status, detail = "FAIL", err
        else:
            status = "PASS"
            exp = cfg_by_name.get(name, {}).get("expect") or {}
            bad = {k: (v, f.get(k)) for k, v in exp.items() if f.get(k) is not None and str(f.get(k)) != str(v)}
            dup = SAME_AS.get(name)
            other = facts_all.get(dup[0], {}) if dup else {}
            if bad:
                k, (want, got) = next(iter(bad.items()))
                status, knob, detail = "FAIL", "✗", f"knob not applied: expected {k}={want}, engine reported {got}"
            elif dup and other.get("status") == "ok" and all(
                    f.get(k) is not None and f.get(k) == other.get(k) for k in dup[1]):
                status, knob = "NOOP", "="
                detail = (f"same engine settings as {dup[0]} here ({', '.join(f'{k}={f.get(k)}' for k in dup[1])}): "
                          f"its numbers would duplicate {dup[0]}'s")
            elif exp and all(f.get(k) is not None for k in exp):
                knob = "✓ " + ", ".join(f"{k}={v}" for k, v in exp.items())
            elif name in KNOB_EXPECT:
                key, ok, what = KNOB_EXPECT[name]
                val = f.get(key)
                if val is None:
                    status, knob, detail = "WARN", "?", f"couldn't read `{key}` from the engine to confirm {what}"
                elif ok(val):
                    knob = f"✓ {what}"
                else:   # the flag was accepted but did nothing: its numbers would just be baseline's
                    status, knob, detail = "FAIL", "✗", f"knob not applied: expected {what}, engine reported {key}={val}"
            elif f.get("resolved"):
                knob = f.get("resolved")
            if status == "PASS" and (f.get("compile_s_in_measure") or 0) >= COMPILE_WARN_S:
                status, detail = "WARN", (f"torch.compile still compiling inside the measured window "
                                          f"({f['compiles_in_measure']} compiles, "
                                          f"{f.get('compile_s_in_measure', 0):.0f}s): numbers include compile time")
        if status in ("PASS", "WARN"):
            passed[engine].append(name)
        rows.append(dict(config=name, engine=engine, status=status,
                         workloads=f"{n_ok}/{n_all}" if n_all else "–", knob=knob, detail=detail))

    counts = {s: sum(1 for r in rows if r["status"] == s) for s in ("PASS", "WARN", "FAIL", "NOOP", "SKIP")}
    t = pd.DataFrame(rows).set_index("config")
    md = [f"# Functional check — `{run_dir.name}`\n",
          f"{summary.get('model')} · {summary.get('environment', {}).get('hardware', {}).get('describe', '')}\n",
          f"**{counts['PASS']} pass · {counts['WARN']} warn · {counts['FAIL']} fail · {counts['NOOP']} no-op · "
          f"{counts['SKIP']} skip**\n",
          "PASS: server/model started, every check workload produced a row, and (where checkable) the engine "
          "reported the knob as applied. WARN: ran fine but the knob couldn't be read back to confirm it. "
          "FAIL: didn't start, a workload failed, or the engine contradicts the knob (its numbers would just "
          "be baseline's) - excluded from `./run.sh suite`'s full run. NOOP: ran, but on this GPU/version "
          "the engine ends up with the same settings as another config (e.g. a value that is already the "
          "default) - excluded, since it would only re-measure that config. SKIP: not applicable here.\n",
          md_table(t, index_name="config")]
    (run_dir / "check.md").write_text("\n".join(md))
    (run_dir / "check_passed.json").write_text(json.dumps({**passed, "harness": harness_fingerprint()}, indent=1))

    icon = {"PASS": "ok  ", "WARN": "warn", "FAIL": "FAIL", "NOOP": "noop", "SKIP": "skip"}
    print(f"\n[check] {run_dir.name}: {counts['PASS']} pass, {counts['WARN']} warn, "
          f"{counts['FAIL']} fail, {counts['NOOP']} no-op, {counts['SKIP']} skip")
    for r in rows:
        print(f"  {icon[r['status']]}  {r['config']:<22} {r['workloads']:>5}  "
              f"{(r['detail'] or (r['knob'] if r['knob'] != '–' else ''))[:110]}")
    print(f"[check] wrote {run_dir / 'check.md'}")
    return passed


def write_report(run_dir: Path, make_plots: bool = False) -> None:
    import pandas as pd

    summary = refresh_log_facts(run_dir, json.loads((run_dir / "summary.json").read_text()))
    if not (run_dir / "results.csv").exists():
        (run_dir / "report.md").write_text(f"# vLLM ablation sweep — `{run_dir.name}`\n\nNo results yet: "
                                           "every config failed before producing a row. See server_logs/.\n")
        print(f"[report] no results yet; wrote stub {run_dir / 'report.md'}")
        return
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
        resolved = "/".join(str(f[k]) for k in ("cfg_compile_mode", "cfg_cudagraph_mode") if f.get(k))
        extras = [f"{lbl}={f[k]}" for k, lbl in (("cfg_kv_cache_dtype", "kv"), ("cfg_prefix_caching", "apc"),
                                                 ("cfg_quantization", "quant"), ("cfg_speculative", "spec"))
                  if f.get(k) is not None]
        comp = f.get("compile_s")
        cold = f.get("cold_start") or {}
        if cold.get("compile_s"):          # measured warm; show what the cold compile cost
            comp_txt = f"{cold['compile_s']:.1f} cold / {comp or 0:.1f} warm"
        else:
            comp_txt = f"{comp:.1f} (cached)" if comp is not None and f.get("compile_cached") else comp
        facts_rows.append(dict(config=c["name"], group=c["group"], knob=c["knob"],
                               status=f.get("status", "skipped"),
                               resolved=f.get("resolved") or " ".join([resolved] + extras).strip() or None,
                               startup_s=f.get("startup_s"), engine_init_s=f.get("engine_init_s"),
                               weights_load_s=f.get("weights_load_s"),
                               compile_s=comp_txt,
                               cudagraph_capture_s=f.get("cudagraph_capture_s"),
                               kv_cache_tokens=f"{int(kv):,}" if kv else None,
                               max_concurrency_x=f.get("max_concurrency_x"),
                               mem_avail_gib=f.get("mem_available_gib"),
                               peak_mem_gib=f.get("peak_mem_gib"),
                               attn=f.get("attention_backend"),
                               note=(f.get("error") or c.get("note") or "")[:80]))
    fdf = pd.DataFrame(facts_rows).set_index("config")
    out.append(md_table(fdf))
    out.append("`resolved` is what the engine reported in its config dump (compile mode / CUDA-graph mode, "
               "KV dtype, prefix caching, quantization, speculative decoding), i.e. proof the knob took. "
               "`compile_s` marked (cached) loaded a graph compiled by an earlier start; startup time "
               "never affects the benchmark numbers, which are measured after warmup.\n")

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

    # 2b. PyTorch vs vLLM on shared features
    if any(str(c.get("group", "")).startswith("pytorch") for c in summary["configs"]):
        out += pytorch_vs_vllm_section(df, summary, cfg_order, ref)

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

def aggregate(run_dir: Path, extra_rows: Optional[list[dict]] = None) -> None:
    """Rebuild results_all.csv / results.csv from every row on disk (row files win over a
    legacy results_all.csv from runs made before row files existed)."""
    import pandas as pd

    raw = run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for r in extra_rows or []:
        write_row(raw, r)
    rows: dict = {}
    legacy = run_dir / "results_all.csv"
    if legacy.exists():
        for r in pd.read_csv(legacy).to_dict("records"):
            rep = r.get("repeat")
            rows[(r["config"], r["workload"], int(rep) if rep == rep and rep is not None else 0)] = r
    for f in sorted(raw.glob("*.row.json")):
        r = json.loads(f.read_text())
        rows[(r["config"], r["workload"], int(r.get("repeat") or 0))] = r
    try:      # a run resumed with fewer --repeats drops the rows of repeats it no longer has
        n_rep = int(json.loads((run_dir / "summary.json").read_text()).get("repeats") or 0)
    except Exception:  # noqa: BLE001
        n_rep = 0
    if n_rep:
        rows = {k: v for k, v in rows.items() if k[2] < n_rep}
    if not rows:
        return
    all_df = pd.DataFrame(list(rows.values()))
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
    p.add_argument("--checklist", action="store_true",
                   help="write check.md / check_passed.json (PASS/WARN/FAIL/SKIP per config) for --run-id")
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
    p.add_argument("--no-warm-start", dest="warm_start", action="store_false",
                   help="measure a server even if its start compiled fresh (smaller KV cache than a warm start)")
    p.add_argument("--drop-caches", dest="drop_caches", action="store_true", default=None,
                   help="sync + drop the page cache before each vllm serve (default on for unified "
                        "memory; needs passwordless sudo for tee /proc/sys/vm/drop_caches)")
    p.add_argument("--no-drop-caches", dest="drop_caches", action="store_false")
    p.add_argument("--allow-shared", action="store_true",
                   help="run even if the GPUs are busy or memory is short (results not clean)")
    p.add_argument("--no-nvcc-check", dest="nvcc_check", action="store_false",
                   help="don't require the CUDA toolkit on Blackwell GPUs")
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
    a.gpu_mem_why = None
    if a.gpu_mem is None:
        a.gpu_mem = (0.75 if hw.per_gpu_gib >= 64 else 0.6) if hw.unified else 0.90
        if not hw.unified:
            # A GPU that also drives a desktop: stay under what's actually free. vLLM refuses to
            # start if free < gpu_mem x total, measured AFTER its own CUDA context exists, and the
            # budget must also survive desktop drift over a long sweep -> DISCRETE_HEADROOM_MIB.
            ids = {g["smi_index"] for g in hw.gpus if g.get("smi_index") is not None}
            rows = [r for r in _smi_rows() if r["smi_index"] in ids and r["total_mib"]
                    and r["used_mib"] is not None]
            fracs = [(r["total_mib"] - r["used_mib"] - DISCRETE_HEADROOM_MIB) / r["total_mib"] for r in rows]
            if fracs and 0.5 <= min(fracs) < 0.90:
                a.gpu_mem = math.floor(min(fracs) * 100) / 100
                r = rows[fracs.index(min(fracs))]
                a.gpu_mem_why = (f"GPU {r['smi_index']} has {r['used_mib'] / 1024:.1f} GiB in use by other "
                                 f"processes; leaving {DISCRETE_HEADROOM_MIB / 1024:.1f} GiB more for vLLM's "
                                 f"CUDA context and desktop drift -> gpu_mem {a.gpu_mem}")
    auto = autosize(hw, a.family, a.model, a.gpu_mem, a.tp)
    a.model, a.tp, a.weights_gib = auto["model"], auto["tp"], auto["weights_gib"]
    a.max_model_len = a.max_model_len or auto["max_model_len"]
    a.autosize_why = ([a.gpu_mem_why] if a.gpu_mem_why else []) + auto["why"]
    a.dtype = a.dtype or ("float16" if 0 < hw.min_cc < 8.0 else "bfloat16")
    fam = FAMILIES.get(family_of(a.model) or "")
    a.mm_off = a.mm_off or (fam["mm_off"] if fam else "none")
    a.grid = "quick" if a.quick else (a.grid or "standard")
    if a.drop_caches is None:
        a.drop_caches = hw.unified
    return hw


def default_run_id(hw: Hardware, tag: str) -> str:
    return f"{hw.slug}_{tag}_{dt.datetime.now():%Y%m%d_%H%M%S}"


def select_configs(a: argparse.Namespace, cfgs: list[ServerConfig], quiet: bool = False,
                   skipped: Optional[dict] = None) -> list[ServerConfig]:
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
            if skipped is not None:
                skipped[c.name] = c.skip
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


def exit_on_term() -> None:
    """`kill <pid>` / a closed terminal should still run the finally: blocks that stop the vLLM
    server (it runs in its own session, so it would otherwise outlive us and hold the GPU)."""
    def bye(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, bye)
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:   # keep nohup's ignore for BG=1
        signal.signal(signal.SIGHUP, bye)


def main() -> None:
    exit_on_term()
    a = parse_args()

    if a.report_only or a.checklist:
        if not a.run_id:
            raise SystemExit("--report-only / --checklist need --run-id")
        if a.report_only:
            aggregate(Path(a.out) / a.run_id)        # rebuild from the row files first
            write_report(Path(a.out) / a.run_id, a.plots)
        if a.checklist:
            write_checklist(Path(a.out) / a.run_id)
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

    skipped: dict = {}
    sel = select_configs(a, cfgs, skipped=skipped)
    run_id = a.run_id or default_run_id(hw, a.run_tag)
    run_dir = Path(a.out) / run_id
    print()
    print_setup(a, hw)

    if not a.dry_run:
        kept = preflight(a, hw, sel)
        for c in sel:
            if c not in kept:
                skipped[c.name] = "checkpoint not accessible on the Hub"
        sel = kept

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
                   server_facts={}, skipped=skipped,
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
            aggregate(run_dir)
            continue
        if a.dry_run:
            print("  $ " + shlex.join(server.command()))
            if cfg.env:
                print("    env: " + json.dumps(cfg.env))
            for w in wls:
                print("  $ " + shlex.join(bench_command(a, model, w, raw_dir, f"{cfg.name}__{w.name}__r0.json")))
            continue

        try:
            facts = server.start()
            # A start that compiles fresh measures less free memory during profiling and gets a
            # ~20% smaller KV cache than a start that loads the compiled graph from cache (seen on an
            # RTX 5070 Ti: 21.8k vs 25.9k tokens for configs with identical memory needs). For
            # KV-limited workloads that would masquerade as a knob effect, so every measured server
            # starts warm: if this start compiled, restart once from the now-populated cache.
            if (a.warm_start and facts.get("status") == "ok" and not facts.get("compile_cached")
                    and (facts.get("compile_s") or 0) >= 1.0):
                cold = facts
                print(f"  cold start: compiled for {cold['compile_s']}s, KV {int(cold.get('kv_cache_tokens') or 0):,} "
                      f"tokens; restarting from the compile cache so every config is measured warm")
                server.stop()
                log_path.replace(log_path.with_name(f"{cfg.name}.cold.log"))
                facts = server.start()
                facts["cold_start"] = {k: cold.get(k) for k in ("startup_s", "compile_s", "kv_cache_tokens")}
        except BaseException:          # Ctrl-C / kill during startup: don't orphan the server
            server.stop()
            raise
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
                    aggregate(run_dir)
        except KeyboardInterrupt:
            print("\n[run] interrupted; stopping server and writing partial results")
            server.stop()
            break
        finally:
            server.stop()

    if rows:
        aggregate(run_dir)
        save_summary()
        write_report(run_dir, a.plots)
    print(f"\n[run] done -> {run_dir}")


if __name__ == "__main__":
    main()
