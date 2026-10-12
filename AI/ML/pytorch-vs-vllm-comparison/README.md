# vllm-ablation: one-knob-at-a-time vLLM sweep on any NVIDIA box

Point it at any NVIDIA machine (one GPU, several GPUs, or a unified-memory box like
DGX Spark or Jetson). It detects the hardware, picks the largest open-source model that
fits, and then runs **baseline + exactly one change** per configuration. Each
configuration gets its own fresh `vllm serve` process and runs against the same
workload grid (`vllm bench serve`). Prometheus counters are compared before and after
every run, so preemptions, prefix-cache hit rate and spec-decode acceptance show up
next to throughput and latency percentiles.

**What the deltas mean.** The vLLM side is a leave-one-out ablation. The baseline is
vLLM with all its defaults on, and each config changes one setting against it: either
turning a default off (CUDA graphs, prefix caching, async scheduling) or turning an
optional feature on (FP8 KV cache, speculative decoding, quantization). A config's delta
is therefore that feature's marginal contribution *with everything else still on*. It is
not the feature's effect in isolation. Features interact, so the deltas don't add up to
the total gap between vLLM and PyTorch. vLLM's core (paged KV cache and continuous
batching) can't be switched off at all. The PyTorch side does the opposite: it is a
cumulative build-up ladder from plain `generate()` (`pt_eager`), adding one feature per
step. The two meet in the report's PyTorch vs vLLM section.

## Quick start

```bash
./run.sh setup     # uv venv at ~/venvs/vllm (Python 3.12) + vllm, accelerate, torchao (safe to rerun)
./run.sh list      # what THIS box will run: hardware, model, TP, context, configs, grid
./run.sh smoke     # baseline only, one workload: proves the stack works
./run.sh compare   # vLLM shared-feature configs + PyTorch ladder in one report
./run.sh quick     # 7 workloads x every applicable config, 1 repeat     (a few hours)
BG=1 ./run.sh full # 26 workloads x every config, 2 repeats, detached    (overnight)
```

Run IDs are named after the hardware, e.g. `gb10_quick_20261012_1530`,
`2xh100_full_…` or `rtx3090_smoke_…`, so results from different boxes sit side by side.

## Recommended flow: check, then the full suite

```bash
./run.sh check            # ~1 h: EVERY vLLM + PyTorch config on a tiny grid -> check.md
./run.sh suite            # check, then full sweep + PyTorch ladder with only the configs that passed
BG=1 ./run.sh suite       # same, detached (unified-memory boxes: add the sudoers line it prints)
./run.sh suite --repeats 1   # args after `suite` go to the full sweep
```

`check` runs every config on three tiny workloads: concurrency 4, concurrency 1 (so the
speculative-decoding paths run), and a shared-prefix workload. It writes a checklist:

| status | meaning |
|---|---|
| PASS | started, every workload produced a row, and the engine reported the knob as applied |
| WARN | ran fine, but the knob couldn't be read back to confirm it (still included) |
| FAIL | didn't start, a workload failed, or the engine **contradicts the knob**: the flag was accepted but did nothing, so its numbers would just be baseline's under another name (excluded) |
| NOOP | ran, but on this GPU/version the engine ends up with the same settings as another config (e.g. `-O0` equals `enforce_eager` from vLLM 0.12, or a value that is already the default), so it would only re-measure that config (excluded) |
| SKIP | not applicable on this machine/model, with the reason |

Knob verification compares what each config's own engine reports (CUDA-graph mode,
enforce_eager, KV dtype, prefix caching, quantization, speculative config, attention
backend, whether torch.compile ran) against what the config is supposed to change.
`suite` feeds the PASS/WARN lists to the full run, so an unsupported config costs about
2 minutes in the check instead of a failed start in every repeat of a day-long sweep.
`suite` reuses a check only if it is under 6 hours old, its baseline passed, and it was
made by the same harness code (a fingerprint is stored in `check_passed.json`). Updating
the scripts therefore always triggers a fresh check.

## How the model is picked

The script uses the largest model from `--family` that fits, with the smallest
tensor-parallel size that fits across the visible GPUs. Context length is then sized
from the memory left for KV cache.

| family | ladder (first that fits wins) | notes |
|---|---|---|
| `qwen3` (default) | Qwen3-30B-A3B (MoE) → 14B → 8B → 4B → 1.7B → 0.6B | Apache-2.0, ungated: no HF login needed |
| `gemma4` | gemma-4-26B-A4B-it (MoE) → 12B → E4B → E2B | gated: accept the license and set `HF_TOKEN` |

What it picks on common hardware (`./run.sh list` shows the real decision):

| box | qwen3 pick | TP | max_model_len |
|---|---|---|---|
| DGX Spark (GB10, 128 GB unified) | Qwen3-30B-A3B | 1 | 32768 |
| H100 / A100 80 GB | Qwen3-30B-A3B | 1 | 16384 |
| 2× L40S 48 GB | Qwen3-30B-A3B | 2 | 32768 |
| A100 40 GB / L40S 48 GB | Qwen3-14B | 1 | 16384 |
| 2× RTX 3090 / 4090 | Qwen3-14B | 2 | 16384 |
| RTX 3090 / 4090 24 GB | Qwen3-8B | 1 | 8192 |
| RTX 5070 Ti / T4 16 GB | Qwen3-4B | 1 | 8192 |
| RTX 3060 12 GB | Qwen3-1.7B | 1 | 8192 |

You can override any of these choices:

```bash
FAMILY=gemma4 ./run.sh quick                          # Gemma 4 ladder
MODEL=meta-llama/Llama-3.1-8B-Instruct ./run.sh quick # any HF id or local path
TP=2 ./run.sh quick                                   # force tensor parallel
CUDA_VISIBLE_DEVICES=2,3 ./run.sh quick               # pick GPUs
GPU_MEM=0.8 ./run.sh quick                            # memory fraction (see below)
./run.sh quick --max-model-len 8192 --dtype float16   # anything else passes through
```

**Memory fraction (`gpu_mem`).** The defaults are 0.90 on a headless discrete GPU, and
0.75 (or 0.6 under 64 GB) on unified memory. When other processes already use the GPU,
as on a GPU that also drives your desktop, the fraction is lowered to
`(free − 1.5 GiB) / total`. vLLM checks that `gpu_mem × total` is free only after it
has created its own CUDA context (~0.6 GiB), and a desktop's usage drifts over a long
sweep. On a 16 GB card with ~1.5 GiB of desktop that comes to about 0.80. Before each
server start the harness waits for that much memory to be free. If it isn't, it prints a
`[mem]` warning naming the shortfall. `suite` reuses the check's model, TP and `gpu_mem`
so the full sweep runs exactly what the check validated.

## Config catalog

Configs that can't apply on the current box are skipped with a stated reason. Examples:
FP8 configs on pre-Ampere GPUs, `no_custom_allreduce` at TP=1, and a vendor checkpoint
that doesn't exist for the chosen model.

| group | config | knob |
|---|---|---|
| baseline | `baseline` | vLLM V1 defaults |
| graphs | `enforce_eager` | no CUDA graphs, no torch.compile |
| graphs | `cudagraph_none` | compile kept, graphs off (isolates graph replay) |
| graphs | `compile_O0` / `compile_O1` / `compile_O3` | optimization levels (`-O0` equals `enforce_eager` from vLLM 0.12; the check marks it NOOP) |
| kv | `no_prefix_cache` | automatic prefix caching off |
| kv | `kv_fp8` | FP8 KV cache |
| kv | `gpu_mem_low` | KV pool cut to ~40% of baseline, which forces preemptions |
| kv | `block_size_32` | KV block size 32 |
| scheduler | `mnbt_small` / `mnbt_max_model_len` | prefill budget at ¼ of vLLM's default for this GPU / ≈ chunked prefill off |
| scheduler | `max_seqs_32` / `max_seqs_512` | concurrency cap (vLLM's default is 256 below 70 GiB of GPU memory, 1024 above) |
| scheduler | `no_async_scheduling` | async scheduling off (it's on by default from vLLM 0.12; older versions get `async_scheduling`, which turns it on) |
| spec | `spec_ngram` | n-gram drafting |
| spec | `spec_draft` | draft model (qwen3: Qwen3-0.6B; gemma4: `<id>-assistant`) |
| quant | `fp8_online` | online FP8 weights |
| quant | `fp8_ckpt` | vendor FP8 checkpoint (qwen3: `<id>-FP8`) |
| quant | `w4a16` | vendor 4-bit checkpoint (qwen3: `<id>-AWQ`; gemma4: `<id>-qat-w4a16-ct`) |
| parallel | `no_custom_allreduce` | NCCL instead of vLLM's custom all-reduce (TP>1 only) |
| attn | `attn_flashinfer` / `attn_flash_attn` / `attn_triton` | `--attention-backend` (vLLM ≥ 0.13; `VLLM_ATTENTION_BACKEND` before) |

Before each run, the script checks that it can download every derived checkpoint. If a
checkpoint is missing or gated, only that config is dropped and the rest of the sweep runs.

## PyTorch comparison (`pytorch_bench.py`)

This runs the same model on the same workloads using only PyTorch and transformers, and
writes rows into the same run. The report then gains a **PyTorch vs vLLM** section with
three parts: a feature matrix, a head-to-head at the reference workload (× vs `pt_eager`,
% of the vLLM baseline), and throughput vs concurrency.

```bash
./run.sh pytorch            # add the PyTorch ladder to your newest vLLM run (same model + grid)
./run.sh pytorch <run_id>   # ...to a specific run
./run.sh pytorch new        # standalone PyTorch run (auto-picks the same model vLLM would)
./run.sh compare            # one shot: vLLM configs that toggle shared features + the PyTorch ladder
```

| config | what it adds | vLLM counterpart |
|---|---|---|
| `pt_eager` | HF `generate()`, eager attention, dynamic KV cache, static batching | — (naive floor) |
| `pt_sdpa` | + fused attention (SDPA / FlashAttention kernels) | attention backends |
| `pt_static` | + preallocated static KV cache | paged KV cache |
| `pt_compile` | + torch.compile (Inductor) + CUDA graphs on decode | `baseline` vs `enforce_eager` / `cudagraph_none` |
| `pt_compile_fp8` | + torchao FP8 W8A8 (int8 weight-only below sm_89) | `fp8_online` |
| `pt_assisted` | draft-model speculative decoding (batch 1) | `spec_draft` |
| `pt_cb` | transformers continuous batching: paged KV, prefix caching, chunked prefill | `baseline` |
| `pt_cb_compile` | + torch.compile (`default_compile_level=1`: Inductor, dynamic shapes) | `baseline` |

How the measurements line up:
- **Static batching** (`pt_eager` … `pt_compile_fp8`) runs concurrency C as batches of C.
  A request's TTFT is its batch's prefill and its E2E is the batch's completion.
  `--max-batches` (default 4) caps measured batches per workload; one extra warmup batch
  absorbs compile and graph capture first. Shapes that can't fit are recorded as `OOM`
  rather than skipped, because that's a result too.
- **Continuous batching** (`pt_cb*`) queues all requests up front with
  `max_requests_per_batch=C`, and measures latency from when each request was first
  scheduled. That matches `vllm bench serve --max-concurrency C`. When attached to a vLLM
  run, its KV cache is sized to the same token capacity the vLLM baseline had.
- Every config runs in its own subprocess (fresh CUDA context and compile cache), and the
  harness refuses busy GPUs the same way the vLLM side does.
- Requirements: torch and transformers in the vLLM venv (`./run.sh setup` also installs
  `accelerate` and `torchao`). `pt_cb*` need transformers ≥ 5.8 and are skipped with a
  reason otherwise. `pt_compile_fp8` needs `torchao`. Without `accelerate`, a single
  discrete GPU still works: the model loads on the CPU and is then moved to the GPU.
  Multi-GPU and unified-memory boxes need `accelerate`, and the configs are skipped
  without it.

## Workload grids

| grid | input_len | output_len | concurrency | reference | extras |
|---|---|---|---|---|---|
| quick | 256, 2048 | 256 | 1, 16, 64 | in2048/out256/c16 | shared-prefix (2048 + 256) |
| standard | 256, 1024, 4096 | 256, 1024 | 1, 8, 32, 128 | in1024/out256/c32 | shared-prefix (4096 + 256) |
| long | 1024, 4096, 16384 | 256, 1024 | 1, 8, 32, 64 | in4096/out256/c32 | shared-prefix (8192 + 512) |

Workloads longer than `max_model_len` are dropped. Add `--sharegpt-path <json>` for a
real-text workload, which is the only one where speculative decoding has anything to
accept.

## Output (`results/<run_id>/`)

| file | contents |
|---|---|
| `report.md` | hardware and auto-sizing rationale; headline Δ% vs baseline; full-grid pivots; scaling vs concurrency, input length and output length; prefix-cache table; ShareGPT/spec table; failures |
| `results.csv` / `results_all.csv` | per (config, workload) medians / per repeat (~60 columns) |
| `summary.json` | environment, hardware, auto-sizing decisions, server facts per config (startup, compile, graph capture, KV tokens, attention backend, memory) |
| `raw/`, `server_logs/`, `plots/` | untouched bench JSON, full server logs, PNGs |

## Safety rails

- **Preflight.** The script refuses to run if any of these hold on the GPUs it's about to use:
  another vLLM server or GPU process is running, the port is busy, or there isn't enough
  free memory. Any of them would skew the numbers. Pass `--allow-shared` to override; the
  run is then flagged as not clean.
- **Unified memory (Spark, Jetson).** The page cache is dropped before every server
  start, because a cached copy of the safetensors competes with CUDA allocations. This
  needs sudo for one file. Interactive runs ask once; for `BG=1` runs, add the sudoers
  line the runner prints.
- **Failure reporting.** If a server fails to start, the script prints the last startup
  step, the error lines and a likely cause (gated repo, OOM, unknown flag, unsupported
  feature), then moves on to the next config.
- **CUDA toolkit on PATH.** vLLM's `has_flashinfer()` checks `which nvcc`, while
  FlashInfer's own JIT quietly falls back to `/usr/local/cuda`. With the toolkit installed
  but not on PATH, vLLM treats FlashInfer as missing, and anything routed to it fails with
  "FlashInfer backend is not available". On sm_120 that includes the FP8 KV cache. `run.sh`
  and each server launch put the toolkit's `bin/` on PATH automatically.
- **Warm starts only.** Each config loads the model once, into one `vllm serve` process
  that runs all its workloads and repeats. Most knobs are fixed when the engine is
  built, so a server can't be reused across configs. A start that compiles fresh
  measures less free memory while profiling, and gets a noticeably smaller KV cache
  than a start that loads the compiled graph from vLLM's cache. On an RTX 5070 Ti that
  was 21.8k vs 25.9k tokens for configs with identical memory needs. For KV-limited
  workloads, that difference would look like a knob effect. So when a config's first
  start compiles, the harness restarts it once from the now-warm cache and measures
  that. `compile_s` in the facts table shows `cold / warm`, the cold log is kept as
  `server_logs/<config>.cold.log`, and `--no-warm-start` turns this off. JIT time on
  new architectures (sm_120, sm_121) is paid on the cold start too.

Pin a vLLM version for a sweep. If `smoke` flags a CLI flag as unknown, adjust it in
`build_configs()`.
