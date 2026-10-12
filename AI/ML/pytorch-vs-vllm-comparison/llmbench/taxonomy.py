"""The 11 optimization categories, and which config pairs measure each one.

Every comparison is "off -> on": `off` is the config without the optimization (or the
reference), `on` the config with it (or the variant). In optimizations.csv a speedup above 1
always means `on` is better, whatever the metric's direction.

Three kinds of evidence, labelled in the `method` column:
  ablation   leave-one-out: vLLM default with one feature switched off (or on). The delta is the
             feature's marginal value with everything else still on. Deltas don't add up to the
             total gap, because features interact.
  ladder     cumulative build-up: each step adds one feature to the previous step. Steps DO add up
             to the end-to-end gap (in that order), which is what a waterfall chart needs.
  variant    an alternative implementation of the same thing (attention backend, page size,
             parallel layout). Not on/off: shows how much the choice matters.
  engine     end-to-end anchors across engines (PyTorch baseline/optimized vs vLLM).

Some features can't be switched off in vLLM (paged KV cache, continuous batching inside the
scheduler). Those comparisons use the nearest honest proxy and say so in `confounded_with` /
`note`: e.g. continuous batching is measured with a client that admits requests in static
waves, and paging is measured on the PyTorch side, where it comes bundled with continuous
batching in transformers.
"""
from __future__ import annotations

from dataclasses import dataclass

CATEGORIES: list[tuple[str, str]] = [
    ("paged_attention", "PagedAttention"),
    ("continuous_batching", "Continuous Batching"),
    ("prefix_caching", "Prefix Caching"),
    ("chunked_prefill", "Chunked Prefill"),
    ("kernel_fusion", "Kernel Fusion"),
    ("speculative_decoding", "Speculative Decoding"),
    ("parallelism", "Parallelism Strategies"),
    ("quantization", "Quantization"),
    ("cuda_graphs", "CUDA Graphs"),
    ("model_kernels", "Model-Specific Kernels"),
    ("large_scale", "Large-Scale Serving"),
]
CATEGORY_LABEL = dict(CATEGORIES)
CATEGORY_LABEL["end_to_end"] = "End-to-End"
CATEGORY_ORDER = [c for c, _ in CATEGORIES] + ["end_to_end"]

# metric -> (label, higher_is_better, unit)
METRICS: dict[str, tuple[str, bool, str]] = {
    "output_throughput": ("output tok/s", True, "tok/s"),
    "output_throughput_per_gpu": ("output tok/s per GPU", True, "tok/s/GPU"),
    "total_token_throughput": ("total tok/s", True, "tok/s"),
    "request_throughput": ("req/s", True, "req/s"),
    "request_goodput": ("goodput req/s", True, "req/s"),
    "mean_ttft_ms": ("TTFT mean", False, "ms"),
    "median_ttft_ms": ("TTFT p50", False, "ms"),
    "p90_ttft_ms": ("TTFT p90", False, "ms"),
    "p99_ttft_ms": ("TTFT p99", False, "ms"),
    "mean_tpot_ms": ("TPOT mean", False, "ms"),
    "median_tpot_ms": ("TPOT p50", False, "ms"),
    "p90_tpot_ms": ("TPOT p90", False, "ms"),
    "p99_tpot_ms": ("TPOT p99", False, "ms"),
    "median_itl_ms": ("ITL p50", False, "ms"),
    "p99_itl_ms": ("ITL p99", False, "ms"),
    "mean_e2el_ms": ("E2E mean", False, "ms"),
    "median_e2el_ms": ("E2E p50", False, "ms"),
    "p99_e2el_ms": ("E2E p99", False, "ms"),
    "peak_mem_gib": ("peak memory", False, "GiB"),
    "preemptions": ("preemptions", False, "count"),
    "prefix_hit_rate": ("prefix-cache hit rate", True, "fraction"),
    "spec_accept_rate": ("spec-decode acceptance", True, "fraction"),
    "spec_accepted_per_draft": ("accepted tokens per draft", True, "tokens"),
}
# the metrics optimizations.csv compares (the rest stay in results_long.csv)
COMPARE_METRICS = ["output_throughput", "output_throughput_per_gpu", "request_throughput", "request_goodput",
                   "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms", "median_e2el_ms",
                   "peak_mem_gib"]


@dataclass(frozen=True)
class Comparison:
    category: str
    engine: str          # vllm | pytorch | cross
    off: str             # config without the optimization (or the reference)
    on: str              # config with it (or the variant)
    what: str            # one line for the chart legend
    method: str = "ablation"
    focus: str = ""      # workload kind where this optimization should show (for summaries)
    confounded_with: str = ""
    note: str = ""


def _cmp(*a, **k) -> Comparison:
    return Comparison(*a, **k)


# Parallel / EAGLE / attention-backend comparisons are generated per run (names depend on
# GPU count and vLLM version), see dynamic_comparisons().
COMPARISONS: list[Comparison] = [
    # --- PagedAttention ---------------------------------------------------------------------
    _cmp("paged_attention", "pytorch", "pt_static", "pt_cb", "contiguous static KV -> paged KV",
         focus="mixed", confounded_with="continuous_batching",
         note="transformers ships paged KV only together with continuous batching"),
    _cmp("paged_attention", "vllm", "vllm_default", "block_size_32", "KV page size 16 -> 32 tokens",
         method="variant", note="vLLM can't run without paging; page size is the knob it has"),
    _cmp("paged_attention", "vllm", "vllm_default", "block_size_64", "KV page size 16 -> 64 tokens",
         method="variant"),
    # --- Continuous batching -----------------------------------------------------------------
    _cmp("continuous_batching", "vllm", "vllm_static_waves", "vllm_default",
         "static batches (waves of C) -> continuous admission", focus="mixed",
         note="vLLM can't disable continuous batching; the static side is the same server fed by a "
              "client that only admits a new batch when the whole previous batch has finished"),
    _cmp("continuous_batching", "pytorch", "pt_static", "pt_cb", "generate() static batches -> CB manager",
         focus="mixed", confounded_with="paged_attention"),
    _cmp("continuous_batching", "vllm", "no_async_scheduling", "vllm_default",
         "async scheduling (CPU scheduling overlapped with GPU steps)"),
    # --- Prefix caching ----------------------------------------------------------------------
    _cmp("prefix_caching", "vllm", "no_prefix_cache", "vllm_default", "automatic prefix caching", focus="prefix"),
    _cmp("prefix_caching", "pytorch", "pt_cb_noprefix", "pt_cb", "block sharing (prefix caching)", focus="prefix"),
    # --- Chunked prefill ---------------------------------------------------------------------
    _cmp("chunked_prefill", "vllm", "no_chunked_prefill", "vllm_default", "chunked prefill", focus="mixed"),
    _cmp("chunked_prefill", "vllm", "mnbt_max_model_len", "vllm_default",
         "chunked prefill (approx. off: token budget = max_model_len)", focus="mixed",
         note="fallback when --no-enable-chunked-prefill isn't accepted"),
    _cmp("chunked_prefill", "vllm", "vllm_default", "mnbt_small", "smaller prefill chunks (1/4 budget)",
         method="variant", focus="mixed"),
    _cmp("chunked_prefill", "pytorch", "pt_cb", "pt_cb_chunk_small", "smaller prefill chunks (512-token budget)",
         method="variant", focus="mixed"),
    # --- Kernel fusion -----------------------------------------------------------------------
    _cmp("kernel_fusion", "vllm", "no_compile", "cudagraph_full_decode_only",
         "torch.compile (Inductor fusion), same FULL_DECODE_ONLY graphs on both sides"),
    _cmp("kernel_fusion", "vllm", "vllm_default", "compile_O3", "-O2 (default) -> -O3 (extra fusion passes)",
         method="variant"),
    _cmp("kernel_fusion", "vllm", "compile_O1", "vllm_default", "-O1 -> -O2 (default)", method="variant"),
    _cmp("kernel_fusion", "vllm", "fp8_nofuse", "fp8_online", "quant-fusion passes (norm+quant, act+quant, attn+quant)",
         note="only does something on FP8 W8A8 (sm_89+)"),
    _cmp("kernel_fusion", "pytorch", "pt_static", "pt_compile_nograph", "torch.compile (Inductor), no CUDA graphs"),
    _cmp("kernel_fusion", "pytorch", "pt_cb", "pt_cb_compile", "torch.compile in the CB path (level 1)"),
    # --- Speculative decoding ----------------------------------------------------------------
    _cmp("speculative_decoding", "vllm", "vllm_default", "spec_ngram", "n-gram (prompt lookup) drafting", focus="text"),
    _cmp("speculative_decoding", "vllm", "vllm_default", "spec_draft", "small draft model", focus="text"),
    _cmp("speculative_decoding", "vllm", "vllm_default", "spec_eagle3", "EAGLE-3 draft head", focus="text"),
    _cmp("speculative_decoding", "pytorch", "pt_sdpa", "pt_assisted", "assisted generation (draft model, batch 1)",
         focus="text"),
    # --- Quantization ------------------------------------------------------------------------
    _cmp("quantization", "vllm", "vllm_default", "fp8_online", "FP8 weights, quantized at load"),
    _cmp("quantization", "vllm", "vllm_default", "fp8_ckpt", "vendor FP8 checkpoint", note="different weights"),
    _cmp("quantization", "vllm", "vllm_default", "w4a16", "vendor 4-bit weight-only checkpoint", note="different weights"),
    _cmp("quantization", "vllm", "vllm_default", "kv_fp8", "FP8 KV cache"),
    _cmp("quantization", "pytorch", "pt_compile", "pt_optimized", "torchao FP8 W8A8 (int8 weight-only below sm_89)"),
    _cmp("quantization", "pytorch", "pt_compile", "pt_quant_int4", "torchao int4 weight-only"),
    # --- CUDA graphs -------------------------------------------------------------------------
    _cmp("cuda_graphs", "vllm", "cudagraph_none", "vllm_default", "CUDA graphs (torch.compile kept)"),
    _cmp("cuda_graphs", "vllm", "vllm_default", "cudagraph_piecewise", "FULL_AND_PIECEWISE -> PIECEWISE",
         method="variant"),
    _cmp("cuda_graphs", "vllm", "vllm_default", "cudagraph_full_decode_only", "FULL_AND_PIECEWISE -> FULL_DECODE_ONLY",
         method="variant"),
    _cmp("cuda_graphs", "vllm", "enforce_eager", "vllm_default", "compile + CUDA graphs vs fully eager",
         confounded_with="kernel_fusion"),
    _cmp("cuda_graphs", "pytorch", "pt_compile_nograph", "pt_compile", "CUDA graphs (reduce-overhead) on decode"),
    _cmp("cuda_graphs", "pytorch", "pt_cb", "pt_cb_graphs", "CUDA graphs in the CB path"),
    # --- Model-specific kernels --------------------------------------------------------------
    _cmp("model_kernels", "vllm", "vllm_default", "custom_ops_all",
         "vLLM's hand-written CUDA ops instead of Inductor-generated ones", method="variant"),
    _cmp("model_kernels", "pytorch", "pt_eager", "pt_sdpa", "fused attention (SDPA/FlashAttention) vs eager matmuls"),
    _cmp("model_kernels", "pytorch", "pt_static", "pt_kernels", "hub kernels (use_kernels=True: fused RMSNorm/act/...)"),
    # --- Large-scale serving -----------------------------------------------------------------
    _cmp("large_scale", "vllm", "gpu_mem_low", "vllm_default", "full KV pool vs ~40% (preemption pressure)",
         focus="rate"),
    _cmp("large_scale", "vllm", "max_seqs_32", "vllm_default", "running-request cap 32 -> default", focus="rate"),
    _cmp("large_scale", "vllm", "vllm_default", "max_seqs_512", "running-request cap -> 512", method="variant",
         focus="rate"),
    _cmp("large_scale", "cross", "pt_serve_optimized", "vllm_default", "best PyTorch serving path -> vLLM default",
         method="engine", focus="grid"),
    # --- End to end --------------------------------------------------------------------------
    _cmp("end_to_end", "pytorch", "pt_eager", "pt_optimized", "PyTorch eager -> fully optimized (static batching)",
         method="engine"),
    _cmp("end_to_end", "pytorch", "pt_eager", "pt_serve_optimized", "PyTorch eager -> optimized serving (CB)",
         method="engine"),
    _cmp("end_to_end", "vllm", "vllm_minimal", "vllm_default", "vLLM minimal -> defaults", method="engine"),
    _cmp("end_to_end", "vllm", "vllm_default", "vllm_optimized", "vLLM defaults -> fully optimized", method="engine"),
    _cmp("end_to_end", "vllm", "vllm_optimized", "vllm_optimized_spec", "+ speculative decoding", method="engine"),
    _cmp("end_to_end", "cross", "pt_eager", "vllm_minimal", "PyTorch eager -> vLLM minimal", method="engine"),
    _cmp("end_to_end", "cross", "pt_optimized", "vllm_optimized", "optimized PyTorch -> optimized vLLM", method="engine"),
    _cmp("end_to_end", "cross", "pt_eager", "vllm_optimized", "PyTorch eager -> optimized vLLM", method="engine"),
]


def dynamic_comparisons(config_names: list[str], base_tp: int) -> list[Comparison]:
    """Comparisons whose config names depend on the box (GPU count) or the vLLM version."""
    out = []
    names = set(config_names)
    for n in config_names:
        if n.startswith("attn_"):
            out.append(_cmp("model_kernels", "vllm", "vllm_default", n,
                            f"attention backend auto -> {n[5:].upper()}", method="variant"))
        elif n.startswith("par_tp"):
            out.append(_cmp("parallelism", "vllm", "vllm_default", n, f"TP{base_tp} -> TP{n[6:]}",
                            method="variant", focus="grid",
                            note="more GPUs: compare output_throughput_per_gpu for efficiency"))
        elif n.startswith("par_pp"):
            out.append(_cmp("parallelism", "vllm", "vllm_default", n, f"TP{base_tp} -> pipeline parallel {n[6:]}",
                            method="variant", focus="grid"))
        elif n.startswith("par_dp"):
            out.append(_cmp("parallelism", "vllm", "vllm_default", n, f"TP{base_tp} -> data parallel {n[6:]}",
                            method="variant", focus="rate"))
            out.append(_cmp("large_scale", "vllm", "vllm_default", n, f"replicate the model x{n[6:]} (data parallel)",
                            method="variant", focus="rate"))
        elif n == "par_ep":
            tp_ref = next((c for c in config_names if c.startswith("par_tp")), "vllm_default")
            out.append(_cmp("parallelism", "vllm", tp_ref, n, "MoE experts: tensor-sliced -> expert parallel",
                            method="variant"))
        elif n == "no_custom_allreduce":
            out.append(_cmp("parallelism", "vllm", n, "vllm_default", "vLLM custom all-reduce vs NCCL"))
    if "async_scheduling" in names:     # vLLM < 0.12: async is off by default, the config turns it on
        out.append(_cmp("continuous_batching", "vllm", "vllm_default", "async_scheduling",
                        "async scheduling (CPU scheduling overlapped with GPU steps)"))
    return out


def all_comparisons(config_names: list[str], base_tp: int = 1) -> list[Comparison]:
    names = set(config_names)
    return [c for c in COMPARISONS + dynamic_comparisons(config_names, base_tp)
            if c.off in names and c.on in names]
