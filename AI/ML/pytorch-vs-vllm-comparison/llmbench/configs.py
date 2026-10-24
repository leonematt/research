"""Config catalogs for both engines.

Tiers
-----
baseline   the unoptimized starting point: `pt_eager` (HF generate, eager attention) and
           `vllm_minimal` (no compile, no CUDA graphs, no prefix cache, no chunked prefill,
           no async scheduling; paging and continuous batching can't be switched off)
default    `vllm_default`: what `vllm serve` does out of the box. Component deltas are vs this.
optimized  `pt_optimized` (static path: SDPA + static KV + compile + CUDA graphs + FP8),
           `pt_serve_optimized` (continuous batching + compile + CUDA graphs + async batching),
           `vllm_optimized` (defaults + FP8 weights + FP8 KV), `vllm_optimized_spec` (+ spec decoding)
ladder     cumulative steps between baseline and optimized, one feature per step
component  one knob vs `vllm_default` (vLLM) or vs its parent on the PyTorch side

Composite configs (vllm_minimal, the ladder, vllm_optimized*) are built from the component
configs' own arguments, so a component that fails the functional check is left out of every
composite automatically (and the composite says so in its note).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from .hardware import Hardware, vllm_default_batching, vllm_version
from .models import derived, family, is_moe
from .workloads import Workload

TIER_ORDER = ["baseline", "default", "ladder", "optimized", "component"]
# ladder steps measure where each step should matter, not the whole grid (anchors do that)
LADDER_FOCUS = ("core", "inputs", "prefix", "mixed", "text")


@dataclass
class Config:
    name: str
    engine: str                                   # vllm | pytorch
    tier: str                                     # baseline | default | ladder | optimized | component
    categories: list[str]
    knob: str                                     # what this config changes, in one line
    focus: list[str] = field(default_factory=lambda: ["core"])   # which workloads it runs (select_workloads)
    # --- vLLM ---
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    model_override: Optional[str] = None          # a different checkpoint (vendor FP8 / 4-bit)
    extra_models: list[str] = field(default_factory=list)   # HF repos it also pulls (draft, EAGLE)
    client: str = "default"                       # default | waves (static-batch admission)
    parallel: dict = field(default_factory=dict)  # tp / pp / dp / ep overrides
    # --- PyTorch ---
    pt: dict = field(default_factory=dict)
    # --- verification / bookkeeping ---
    verify: list[list] = field(default_factory=list)   # [fact_key, op, value, description]
    same_as: Optional[list] = None                # [other_config, [fact keys]]: equal facts -> NOOP
    components: list[str] = field(default_factory=list)  # composites: what went in
    skip: Optional[str] = None
    note: str = ""

    def gpus(self, base_tp: int) -> int:
        p = self.parallel
        return int(p.get("tp", base_tp)) * int(p.get("pp", 1)) * int(p.get("dp", 1))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Config":
        keys = Config.__dataclass_fields__
        return Config(**{k: v for k, v in d.items() if k in keys})


def V(name, tier, cats, knob, args=None, **kw) -> Config:
    return Config(name, "vllm", tier, list(cats), knob, args=list(args or []), **kw)


def _cc(d: dict) -> list[str]:
    return ["--compilation-config", json.dumps(d)]


# ----------------------------------------------------------------------------
# argument composition
# ----------------------------------------------------------------------------

def merge_args(*arglists: list[str]) -> list[str]:
    """Merge vLLM CLI argument lists: later flags win, --compilation-config JSON is deep-merged,
    and -O levels / attention backends keep only the last one."""
    flags: dict[str, Optional[str]] = {}
    cc: dict = {}
    for args in arglists:
        i = 0
        while i < len(args):
            tok = args[i]
            val = None
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                val = args[i + 1]
                i += 1
            i += 1
            if tok == "--compilation-config":
                _deep_merge(cc, json.loads(val))
                continue
            if tok.startswith("-O") and len(tok) == 3:          # -O0 .. -O3
                for k in [k for k in flags if k.startswith("-O") and len(k) == 3]:
                    del flags[k]
            # --enable-x / --no-enable-x: the later one wins
            opposite = tok.replace("--no-", "--", 1) if tok.startswith("--no-") else "--no-" + tok[2:]
            flags.pop(opposite, None)
            flags[tok] = val
    out: list[str] = []
    for k, v in flags.items():
        out.append(k)
        if v is not None:
            out.append(v)
    if cc:
        out += _cc(cc)
    return out


def _deep_merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def compose(name: str, tier: str, cats: list[str], knob: str, parts: list, comps: dict[str, Config],
            exclude: set[str], focus=("all",), note: str = "") -> Config:
    """Build a composite config from component configs. `parts` entries are names or tuples of
    alternatives (the first usable one is taken)."""
    used, missing, args, verify, extra = [], [], [], [], []
    model_override = None
    env: dict = {}
    for p in parts:
        alts = p if isinstance(p, tuple) else (p,)
        pick = next((comps[a] for a in alts if a in comps and not comps[a].skip and a not in exclude), None)
        if pick is None:
            missing.append("/".join(alts))
            continue
        used.append(pick.name)
        args.append(pick.args)
        verify += pick.verify
        extra += pick.extra_models
        env.update(pick.env)
        model_override = pick.model_override or model_override
    n = note
    if missing:
        n = (n + "; " if n else "") + "built without " + ", ".join(missing) + " (not available / failed the check)"
    return Config(name, "vllm", tier, list(cats), knob, focus=list(focus), args=merge_args(*args), env=env,
                  model_override=model_override, extra_models=extra, verify=verify, components=used, note=n)


# ----------------------------------------------------------------------------
# vLLM catalog
# ----------------------------------------------------------------------------

def _attn(name: str, backend: str) -> Config:
    """vLLM < 0.13 selects the backend with VLLM_ATTENTION_BACKEND; 0.13-0.14 accept both; from
    0.15 the env var is gone and silently ignored, so only --attention-backend works."""
    pat = {"FLASHINFER": "flashinfer", "FLASH_ATTN": "flash", "TRITON_ATTN": "triton"}[backend]
    verify = [["attention_backend", "has", pat, f"{backend} backend in log"]]
    if backend == "FLASH_ATTN":
        verify.append(["attention_backend", "not_has", "flashinfer", "not FlashInfer"])
    v = vllm_version()
    if v is not None and v < (0, 13):
        return V(name, "component", ["model_kernels"], f"VLLM_ATTENTION_BACKEND={backend}",
                 env={"VLLM_ATTENTION_BACKEND": backend}, verify=verify, focus=["core", "inputs"])
    return V(name, "component", ["model_kernels"], f"--attention-backend {backend}",
             ["--attention-backend", backend], verify=verify, focus=["core", "inputs"])


def vllm_components(ctx, hw: Optional[Hardware]) -> list[Config]:
    """One knob each, relative to vllm_default."""
    mml = str(ctx.max_model_len)
    d_mnbt, d_seqs = vllm_default_batching(hw)
    mnbt_small = max(256, d_mnbt // 4)
    cc = hw.min_cc if hw else 0.0
    n_gpu = hw.n if hw else 1
    fp8_skip = None if cc == 0 or cc >= 8.0 else f"FP8 needs sm_80+ (have sm_{str(cc).replace('.', '')})"

    # speculative decoding: draft model / EAGLE-3 sharing the tokenizer
    def spec(kind: str, override: Optional[str]):
        if override:
            j = override
        else:
            d = derived(ctx.model, kind)
            j = json.dumps(d) if d else ""
        skip = None if j else ("no " + {"draft": "draft model", "eagle3": "EAGLE-3 head"}[kind] + " for this model"
                               + ("" if family(ctx.model) else f"; pass --{'spec' if kind == 'draft' else 'eagle'}-config"))
        try:
            models = [json.loads(j)["model"]] if j else []
        except (ValueError, KeyError, TypeError):
            models = []
        return j, skip, models

    draft_json, draft_skip, draft_models = spec("draft", ctx.spec_config)
    eagle_json, eagle_skip, eagle_models = spec("eagle3", ctx.eagle_config)

    # vendor pre-quantized checkpoints of the same model (different weights!)
    w4 = ctx.w4a16_model or derived(ctx.model, "w4a16")
    w4_skip = None if w4 else "no vendor 4-bit checkpoint for this model (pass --w4a16-model)"
    fp8c = derived(ctx.model, "fp8_ckpt")
    fp8c_skip = None if fp8c else "no vendor FP8 checkpoint for this model"

    # starve the KV pool to ~40% of the default KV budget (exposes preemption)
    w = ctx.weights_gib
    if w and hw and hw.per_gpu_gib:
        total = hw.per_gpu_gib * ctx.tp
        kv = total * ctx.gpu_mem - w - 1.5
        low = round((w + 1.5 + 0.4 * max(kv, 0)) / total, 2)
    else:
        low = round(ctx.gpu_mem - 0.15, 2)
    low_skip = None if low < ctx.gpu_mem - 0.02 else "default KV budget already minimal"

    v = vllm_version()
    async_new = v is None or v >= (0, 12)
    async_c = (V("no_async_scheduling", "component", ["continuous_batching"],
                 "async scheduling OFF (on by default since vLLM 0.12)", ["--no-async-scheduling"],
                 verify=[["cfg_async_scheduling", "is", "false", "async scheduling off"]])
               if async_new else
               V("async_scheduling", "component", ["continuous_batching"],
                 "async scheduling ON (overlap CPU scheduling with GPU)", ["--async-scheduling"],
                 verify=[["cfg_async_scheduling", "is", "true", "async scheduling on"]]))

    cfgs = [
        # --- CUDA graphs ----------------------------------------------------------------
        V("enforce_eager", "component", ["cuda_graphs", "kernel_fusion"], "no CUDA graphs, no torch.compile",
          ["--enforce-eager"], verify=[["cfg_enforce_eager", "is", "true", "enforce_eager=True"]]),
        V("cudagraph_none", "component", ["cuda_graphs"], "torch.compile kept, CUDA graphs off",
          _cc({"cudagraph_mode": "NONE"}), verify=[["cfg_cudagraph_mode", "is", "none", "cudagraph_mode=NONE"]]),
        V("cudagraph_piecewise", "component", ["cuda_graphs"], "CUDA graphs PIECEWISE only (attention outside)",
          _cc({"cudagraph_mode": "PIECEWISE"}), verify=[["cfg_cudagraph_mode", "is", "piecewise", "PIECEWISE"]],
          same_as=["vllm_default", ["cfg_cudagraph_mode"]]),
        V("cudagraph_full_decode_only", "component", ["cuda_graphs"],
          "full CUDA graphs for decode batches only (prefill eager)",
          _cc({"cudagraph_mode": "FULL_DECODE_ONLY"}),
          verify=[["cfg_cudagraph_mode", "is", "full_decode_only", "FULL_DECODE_ONLY"]],
          same_as=["vllm_default", ["cfg_cudagraph_mode"]]),
        # --- kernel fusion --------------------------------------------------------------
        V("no_compile", "component", ["kernel_fusion"],
          "torch.compile OFF, FULL_DECODE_ONLY CUDA graphs kept (isolates Inductor fusion)",
          _cc({"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}),
          verify=[["cfg_compile_mode", "is", "none", "compile mode NONE"],
                  ["cfg_cudagraph_mode", "has", "full", "full CUDA graphs still on"]],
          note="pair with cudagraph_full_decode_only: same graphs, only Inductor fusion differs"),
        V("compile_O1", "component", ["kernel_fusion"], "-O1: fewer optimization passes", ["-O1"]),
        V("compile_O3", "component", ["kernel_fusion"], "-O3: extra fusion passes", ["-O3"]),
        V("fp8_nofuse", "component", ["kernel_fusion", "quantization"],
          "FP8 weights with the quant-fusion passes off (norm+quant, act+quant, attn+quant)",
          ["--quantization", "fp8"] + _cc({"pass_config": {"fuse_norm_quant": False, "fuse_act_quant": False,
                                                            "fuse_attn_quant": False}}),
          skip=fp8_skip, verify=[["cfg_quantization", "has", "fp8", "quantization=fp8"],
                                 ["cfg_fuse_norm_quant", "is", "false", "norm+quant fusion off"]],
          note="only meaningful against fp8_online on sm_89+ (W8A8)"),
        # --- model-specific kernels -----------------------------------------------------
        V("custom_ops_all", "component", ["model_kernels"],
          "vLLM's hand-written CUDA ops (RMSNorm, SiLU-and-mul, RoPE, ...) instead of Inductor's",
          _cc({"custom_ops": ["all"]}), verify=[["cfg_custom_ops", "has", "all", "custom_ops=all"]],
          same_as=["vllm_default", ["cfg_custom_ops"]], focus=["core", "inputs"]),
        _attn("attn_flashinfer", "FLASHINFER"),
        _attn("attn_flash_attn", "FLASH_ATTN"),
        _attn("attn_triton", "TRITON_ATTN"),
        # --- PagedAttention (page size; paging itself can't be turned off) ---------------
        V("block_size_32", "component", ["paged_attention"], "KV block size 32 (vs 16)", ["--block-size", "32"],
          focus=["core", "inputs", "mixed"]),
        V("block_size_64", "component", ["paged_attention"], "KV block size 64 (vs 16)", ["--block-size", "64"],
          focus=["core", "inputs", "mixed"]),
        # --- continuous batching --------------------------------------------------------
        V("vllm_static_waves", "component", ["continuous_batching"],
          "static batching: client admits C requests, waits for all, then the next C",
          client="waves", focus=["core", "mixed", "inputs"],
          note="same server as vllm_default; only request admission differs"),
        async_c,
        # --- prefix caching -------------------------------------------------------------
        V("no_prefix_cache", "component", ["prefix_caching"], "automatic prefix caching off",
          ["--no-enable-prefix-caching"], focus=["core", "prefix"],
          verify=[["cfg_prefix_caching", "is", "false", "prefix caching off"]]),
        # --- chunked prefill ------------------------------------------------------------
        V("no_chunked_prefill", "component", ["chunked_prefill"],
          f"chunked prefill off (token budget = max_model_len {mml})",
          ["--no-enable-chunked-prefill", "--max-num-batched-tokens", mml], focus=["core", "mixed", "inputs"],
          verify=[["cfg_chunked_prefill", "is", "false", "chunked prefill off"]]),
        V("mnbt_max_model_len", "component", ["chunked_prefill"],
          f"max_num_batched_tokens={mml} (default here {d_mnbt}; approximates chunked prefill OFF)",
          ["--max-num-batched-tokens", mml], focus=["core", "mixed", "inputs"],
          verify=[["cfg_mnbt", "is", mml, f"max_num_batched_tokens={mml}"]],
          same_as=["vllm_default", ["cfg_mnbt"]],
          skip=None if int(mml) > d_mnbt else f"max_model_len {mml} <= the default budget {d_mnbt}"),
        V("mnbt_small", "component", ["chunked_prefill"],
          f"max_num_batched_tokens={mnbt_small} (default here {d_mnbt}): smaller prefill chunks",
          ["--max-num-batched-tokens", str(mnbt_small)], focus=["core", "mixed", "inputs"],
          verify=[["cfg_mnbt", "is", str(mnbt_small), f"max_num_batched_tokens={mnbt_small}"]],
          same_as=["vllm_default", ["cfg_mnbt"]]),
        # --- speculative decoding -------------------------------------------------------
        V("spec_ngram", "component", ["speculative_decoding"], "n-gram speculative decoding (no draft model)",
          ["--speculative-config", json.dumps({"method": "ngram", "num_speculative_tokens": 4,
                                               "prompt_lookup_max": 4, "prompt_lookup_min": 2})],
          focus=["core", "text", "sharegpt"], verify=[["cfg_speculative", "set", "", "speculative_config set"]],
          note="random-token prompts give ~0 acceptance; judge on the text workload"),
        V("spec_draft", "component", ["speculative_decoding"],
          f"draft-model speculative decoding ({', '.join(draft_models) or '-'})",
          ["--speculative-config", draft_json], skip=draft_skip, extra_models=draft_models,
          focus=["core", "text", "sharegpt"], verify=[["cfg_speculative", "set", "", "speculative_config set"]]),
        V("spec_eagle3", "component", ["speculative_decoding"],
          f"EAGLE-3 speculative decoding ({', '.join(eagle_models) or '-'})",
          ["--speculative-config", eagle_json], skip=eagle_skip, extra_models=eagle_models,
          focus=["core", "text", "sharegpt"], verify=[["cfg_speculative", "set", "", "speculative_config set"]]),
        # --- quantization ---------------------------------------------------------------
        V("fp8_online", "component", ["quantization"],
          "online FP8 weight quantization (W8A8 on sm>=8.9, Marlin weight-only on Ampere)",
          ["--quantization", "fp8"], skip=fp8_skip, verify=[["cfg_quantization", "has", "fp8", "quantization=fp8"]]),
        V("fp8_ckpt", "component", ["quantization"], f"vendor FP8 checkpoint ({fp8c or '-'})",
          model_override=fp8c, skip=fp8_skip or fp8c_skip, extra_models=[fp8c] if fp8c else [],
          verify=[["cfg_quantization", "set", "", "quantized checkpoint detected"]]),
        V("w4a16", "component", ["quantization"], f"vendor 4-bit weight-only checkpoint ({w4 or '-'}; Marlin)",
          model_override=w4, skip=w4_skip, extra_models=[w4] if w4 else [],
          verify=[["cfg_quantization", "set", "", "quantized checkpoint detected"]]),
        V("kv_fp8", "component", ["quantization"], "FP8 KV cache (2x KV capacity, decode reads half the bytes)",
          ["--kv-cache-dtype", "fp8"], skip=fp8_skip, focus=["core", "inputs", "rate"],
          verify=[["cfg_kv_cache_dtype", "has", "fp8", "kv_cache_dtype=fp8*"]]),
        # --- large-scale serving --------------------------------------------------------
        V("gpu_mem_low", "component", ["large_scale"],
          f"gpu_memory_utilization={low} (~40% of the default KV pool -> preemptions)",
          ["--gpu-memory-utilization", str(low)], skip=low_skip, focus=["core", "rate", "inputs"]),
        V("max_seqs_32", "component", ["large_scale"], f"max_num_seqs=32 (default here {d_seqs}): caps concurrency",
          ["--max-num-seqs", "32"], focus=["core", "rate"]),
        V("max_seqs_512", "component", ["large_scale"],
          f"max_num_seqs=512 (default here {d_seqs}; " + ("more CUDA-graph sizes, less KV)" if d_seqs < 512 else "lower cap)"),
          ["--max-num-seqs", "512"], skip=None if d_seqs != 512 else "512 is already the default here",
          focus=["core", "rate"]),
    ]

    # --- parallelism (multi-GPU only) ---------------------------------------------------
    base = ctx.tp
    n_pow2 = max([t for t in (1, 2, 4, 8) if t <= n_gpu], default=1)
    one_gpu = "single GPU visible" if n_gpu < 2 else None
    par = []
    if n_pow2 > base:
        par.append(V(f"par_tp{n_pow2}", "component", ["parallelism"], f"tensor parallel {n_pow2} (default TP{base})",
                     parallel={"tp": n_pow2}, focus=["core", "rate"],
                     verify=[["cfg_tp", "is", str(n_pow2), f"tensor_parallel_size={n_pow2}"]]))
        pp = n_pow2 // base
        par.append(V(f"par_pp{pp}", "component", ["parallelism"], f"pipeline parallel {pp} x TP{base}",
                     parallel={"tp": base, "pp": pp}, focus=["core", "rate"],
                     verify=[["cfg_pp", "is", str(pp), f"pipeline_parallel_size={pp}"]]))
        par.append(V(f"par_dp{pp}", "component", ["parallelism", "large_scale"],
                     f"data parallel {pp} x TP{base} (model replicated)", parallel={"tp": base, "dp": pp},
                     focus=["core", "rate"]))
    if is_moe(ctx.model) and n_pow2 > 1:
        par.append(V("par_ep", "component", ["parallelism"], f"expert parallel across {n_pow2} GPUs (MoE)",
                     ["--enable-expert-parallel"], parallel={"tp": n_pow2}, focus=["core", "rate"]))
    par.append(V("no_custom_allreduce", "component", ["parallelism"],
                 "NCCL all-reduce instead of vLLM's custom one-shot kernel", ["--disable-custom-all-reduce"],
                 skip=None if base > 1 else "TP=1 (no all-reduce)"))
    if one_gpu:
        for c in par:
            c.skip = c.skip or one_gpu
    elif not par[:-1]:
        par.insert(0, V("par_tp", "component", ["parallelism"], "more tensor parallelism",
                        skip=f"the model already uses all {n_pow2} usable GPUs (TP{base})"))
    return cfgs + par


def vllm_catalog(ctx, hw: Optional[Hardware], exclude: Optional[set[str]] = None) -> tuple[list[Config], dict]:
    """(configs in run order, ladders). `exclude` = components that failed the check."""
    exclude = exclude or set()
    comps = {c.name: c for c in vllm_components(ctx, hw)}
    async_off = "no_async_scheduling" if "no_async_scheduling" in comps else None
    chunk_off = ("no_chunked_prefill", "mnbt_max_model_len")
    fp8 = ("fp8_online", "fp8_ckpt")
    spec = ("spec_eagle3", "spec_draft", "spec_ngram")

    def C(name, tier, cats, knob, parts, focus=("all",), note=""):
        return compose(name, tier, cats, knob, parts, comps, exclude, focus, note)

    off_all = ["enforce_eager", "no_prefix_cache", chunk_off] + ([async_off] if async_off else [])
    default = V("vllm_default", "default", [], "vLLM defaults (paged KV, continuous batching, chunked prefill, "
                "prefix caching, torch.compile + CUDA graphs, async scheduling)", focus=["all"])
    anchors = [
        default,
        C("vllm_minimal", "baseline", [], "everything that can be switched off is off "
          "(eager, no prefix cache, no chunked prefill, no async scheduling)", off_all,
          note="paged KV and continuous batching are always on in vLLM"),
    ]
    ladder_steps = [
        ("vllm_minimal", "start: minimal vLLM", ""),
        ("vllm_lad_compile", "+ torch.compile (fusion)", "kernel_fusion",
         ["cudagraph_none", "no_prefix_cache", chunk_off] + ([async_off] if async_off else [])),
        ("vllm_lad_graphs", "+ CUDA graphs", "cuda_graphs", ["no_prefix_cache", chunk_off] + ([async_off] if async_off else [])),
        ("vllm_lad_prefix", "+ prefix caching", "prefix_caching", [chunk_off] + ([async_off] if async_off else [])),
    ]
    if async_off:
        ladder_steps += [("vllm_lad_chunked", "+ chunked prefill", "chunked_prefill", [async_off]),
                         ("vllm_default", "+ async scheduling (= defaults)", "continuous_batching")]
    else:
        ladder_steps += [("vllm_default", "+ chunked prefill (= defaults)", "chunked_prefill")]
    ladder_steps += [
        ("vllm_lad_fp8", "+ FP8 weights", "quantization", [fp8]),
        ("vllm_optimized", "+ FP8 KV cache (= optimized)", "quantization"),
        ("vllm_optimized_spec", "+ speculative decoding", "speculative_decoding"),
    ]
    ladder_cfgs = []
    for step in ladder_steps:
        if len(step) == 4:
            name, label, cat, parts = step
            ladder_cfgs.append(C(name, "ladder", [cat], label.lstrip("+ "), parts, focus=LADDER_FOCUS))
    optimized = [
        C("vllm_optimized", "optimized", [], "defaults + FP8 weights + FP8 KV cache", [fp8, "kv_fp8"]),
        C("vllm_optimized_spec", "optimized", [], "defaults + FP8 weights + FP8 KV cache + speculative decoding",
          [fp8, "kv_fp8", spec]),
    ]
    ordered = anchors + optimized + ladder_cfgs + list(comps.values())
    ladders = {"vllm": [dict(config=s[0], added=s[1], category=s[2]) for s in ladder_steps]}
    return ordered, ladders


# ----------------------------------------------------------------------------
# PyTorch catalog
# ----------------------------------------------------------------------------

def P(name, tier, cats, knob, focus=("all",), **pt) -> Config:
    verify = pt.pop("verify", [])
    return Config(name, "pytorch", tier, list(cats), knob, focus=list(focus), pt=pt, verify=verify)


def pytorch_catalog() -> tuple[list[Config], dict]:
    gen = dict(mode="generate", attn="sdpa", cache="dynamic", compile=None, quant=None)
    cb = dict(mode="cb", attn="sdpa", cb={})
    compiled = [["compiled", "is", "true", "torch.compile applied"]]
    cfgs = [
        P("pt_eager", "baseline", [], "HF generate(): eager attention, dynamic KV cache, static batching",
          **{**gen, "attn": "eager"}),
        P("pt_sdpa", "ladder", ["model_kernels"], "+ fused attention (SDPA / FlashAttention kernels)", focus=LADDER_FOCUS, **gen),
        P("pt_static", "ladder", ["paged_attention"], "+ preallocated static KV cache (fixed shapes)",
          focus=LADDER_FOCUS, **{**gen, "cache": "static"}),
        P("pt_compile_nograph", "ladder", ["kernel_fusion"], "+ torch.compile (Inductor fusion), no CUDA graphs",
          focus=LADDER_FOCUS, **{**gen, "cache": "static", "compile": "default"}, verify=compiled),
        P("pt_compile", "ladder", ["cuda_graphs"], "+ CUDA graphs (torch.compile mode=reduce-overhead)",
          focus=LADDER_FOCUS, **{**gen, "cache": "static", "compile": "reduce-overhead"}, verify=compiled),
        P("pt_optimized", "optimized", ["quantization"],
          "fully optimized static path: SDPA + static KV + compile + CUDA graphs + torchao FP8",
          **{**gen, "cache": "static", "compile": "reduce-overhead", "quant": "fp8"},
          verify=compiled + [["quant", "set", "", "torchao quantization applied"]]),
        P("pt_quant_int4", "component", ["quantization"], "torchao int4 weight-only (compile + CUDA graphs)",
          focus=("core", "inputs"), **{**gen, "cache": "static", "compile": "reduce-overhead", "quant": "int4"},
          verify=compiled + [["quant", "set", "", "torchao quantization applied"]]),
        P("pt_kernels", "component", ["model_kernels"], "static KV + hub kernels (from_pretrained(use_kernels=True))",
          focus=("core", "inputs"), **{**gen, "cache": "static", "kernels": True},
          verify=[["kernels", "is", "true", "hub kernels loaded"]]),
        P("pt_assisted", "component", ["speculative_decoding"],
          "assisted generation with a draft model (batch 1 only)", focus=("core", "text"), **{**gen, "assisted": True}),
        P("pt_cb", "component", ["paged_attention", "continuous_batching"],
          "transformers continuous batching: paged KV, prefix sharing, chunked prefill (eager)",
          focus=LADDER_FOCUS, **cb),
        P("pt_cb_noprefix", "component", ["prefix_caching"], "continuous batching with block sharing off",
          focus=("core", "prefix"), **{**cb, "cb": {"allow_block_sharing": False}},
          verify=[["cb_allow_block_sharing", "is", "false", "allow_block_sharing=False"]]),
        P("pt_cb_chunk_small", "component", ["chunked_prefill"], "continuous batching with a 512-token step budget",
          focus=("core", "mixed", "inputs"), **{**cb, "cb": {"max_batch_tokens": 512}},
          verify=[["cb_max_batch_tokens", "is", "512", "max_batch_tokens=512"]]),
        P("pt_cb_compile", "component", ["kernel_fusion"], "continuous batching + torch.compile (level 1: Inductor, dynamic)",
          focus=LADDER_FOCUS, **{**cb, "cb": {"default_compile_level": 1}}, verify=compiled),
        P("pt_cb_graphs", "component", ["cuda_graphs"], "continuous batching + CUDA graphs (use_cuda_graph=True)",
          focus=("core", "inputs", "mixed"), **{**cb, "cb": {"use_cuda_graph": True}},
          verify=[["cb_use_cuda_graph", "is", "true", "use_cuda_graph=True"]]),
        P("pt_serve_optimized", "optimized", [],
          "fully optimized serving path: continuous batching + compile + CUDA graphs + async batching",
          **{**cb, "cb": {"default_compile_level": 1, "use_cuda_graph": True, "use_async_batching": True}},
          verify=compiled + [["cb_use_cuda_graph", "is", "true", "use_cuda_graph=True"]]),
    ]
    ladders = {
        "pytorch_static": [
            dict(config="pt_eager", added="start: HF generate(), eager", category=""),
            dict(config="pt_sdpa", added="+ fused attention (SDPA)", category="model_kernels"),
            dict(config="pt_static", added="+ static KV cache", category="paged_attention"),
            dict(config="pt_compile_nograph", added="+ torch.compile fusion", category="kernel_fusion"),
            dict(config="pt_compile", added="+ CUDA graphs", category="cuda_graphs"),
            dict(config="pt_optimized", added="+ FP8 (= optimized)", category="quantization"),
        ],
        "pytorch_serving": [
            dict(config="pt_eager", added="start: HF generate(), eager", category=""),
            dict(config="pt_sdpa", added="+ fused attention (SDPA)", category="model_kernels"),
            dict(config="pt_static", added="+ static KV cache", category="paged_attention"),
            dict(config="pt_cb", added="+ paged KV + continuous batching", category="continuous_batching"),
            dict(config="pt_cb_compile", added="+ torch.compile", category="kernel_fusion"),
            dict(config="pt_serve_optimized", added="+ CUDA graphs + async batching (= optimized)",
                 category="cuda_graphs"),
        ],
    }
    return cfgs, ladders


# ----------------------------------------------------------------------------
# selection helpers
# ----------------------------------------------------------------------------

def select_workloads(cfg: Config, wls: list[Workload], ref: tuple[int, int, int],
                     all_workloads: bool = False) -> list[Workload]:
    """Which workloads a config runs. Anchors and ladder steps run everything; a component runs
    the `core` set (reference shape across every concurrency) plus the kinds where its
    optimization should show (`focus`)."""
    ri, ro, rc = ref
    focus = set(cfg.focus)
    out = []
    for w in wls:
        if cfg.engine == "pytorch" and w.kind == "rate":
            continue                         # open-loop arrivals need a server
        if cfg.client == "waves" and w.kind == "rate":
            continue
        if all_workloads or "all" in focus or w.kind == "reference":
            out.append(w)
            continue
        grid_like = w.kind == "grid"
        if ("core" in focus and grid_like and w.input_len == ri and w.output_len == ro) \
                or ("inputs" in focus and grid_like and w.concurrency == rc and w.output_len == ro) \
                or ("outputs" in focus and grid_like and w.concurrency == rc and w.input_len == ri) \
                or ("grid" in focus and grid_like) or w.kind in focus:
            out.append(w)
    return out


def filter_configs(cfgs: list[Config], names: Optional[str] = None, categories: Optional[str] = None,
                   tiers: Optional[str] = None, skip: Optional[str] = None) -> list[Config]:
    """--configs / --categories / --tiers / --skip. Anchors needed for comparisons are kept:
    vllm_default (vLLM) and pt_eager (PyTorch) whenever anything of that engine runs."""
    want = set(names.split(",")) if names else None
    cats = set(categories.split(",")) if categories else None
    tset = set(tiers.split(",")) if tiers else None
    drop = set(skip.split(",")) if skip else set()
    out = [c for c in cfgs if c.name not in drop
           and (want is None or c.name in want)
           and (cats is None or set(c.categories) & cats or c.tier in ("baseline", "default"))
           and (tset is None or c.tier in tset or c.tier in ("baseline", "default"))]
    return out
