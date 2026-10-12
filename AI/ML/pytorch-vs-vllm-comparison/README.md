# vllm-ablation: one-knob-at-a-time vLLM sweep on any NVIDIA box

Point it at any NVIDIA machine (one GPU, several GPUs, or a unified-memory box like
DGX Spark or Jetson). It detects the hardware, picks the largest open-source model that
fits, and then runs **baseline + exactly one change** per configuration. Each
configuration gets its own fresh `vllm serve` process and runs against the same
workload grid (`vllm bench serve`). Prometheus counters are compared before and after
every run, so preemptions, prefix-cache hit rate and spec-decode acceptance show up
next to throughput and latency percentiles.

## Quick start

```bash
./run.sh setup     # one-time: uv venv at ~/venvs/vllm (Python 3.12) + pip install vllm
./run.sh list      # what THIS box will run: hardware, model, TP, context, configs, grid
./run.sh smoke     # baseline only, one workload: proves the stack works
./run.sh quick     # 7 workloads x every applicable config, 1 repeat     (a few hours)
BG=1 ./run.sh full # 26 workloads x every config, 2 repeats, detached    (overnight)
```

Run IDs are named after the hardware, e.g. `gb10_quick_20261012_1530`,
`2xh100_full_…` or `rtx3090_smoke_…`, so results from different boxes sit side by side.

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
GPU_MEM=0.8 ./run.sh quick                            # memory fraction
./run.sh quick --max-model-len 8192 --dtype float16   # anything else passes through
```

## Config catalog

Configs that can't apply on the current box are skipped with a stated reason. Examples:
FP8 configs on pre-Ampere GPUs, `no_custom_allreduce` at TP=1, and a vendor checkpoint
that doesn't exist for the chosen model.

| group | config | knob |
|---|---|---|
| baseline | `baseline` | vLLM V1 defaults |
| graphs | `enforce_eager` | no CUDA graphs, no torch.compile |
| graphs | `cudagraph_none` | compile kept, graphs off (isolates graph replay) |
| graphs | `compile_O0` / `compile_O1` / `compile_O3` | optimization levels |
| kv | `no_prefix_cache` | automatic prefix caching off |
| kv | `kv_fp8` | FP8 KV cache |
| kv | `gpu_mem_low` | KV pool cut to ~40% of baseline, which forces preemptions |
| kv | `block_size_32` | KV block size 32 |
| scheduler | `mnbt_2048` / `mnbt_max_model_len` | small prefill chunks / ≈ chunked prefill off |
| scheduler | `max_seqs_32` / `max_seqs_512` | concurrency cap |
| scheduler | `async_scheduling` | overlap CPU scheduling with GPU |
| spec | `spec_ngram` | n-gram drafting |
| spec | `spec_draft` | draft model (qwen3: Qwen3-0.6B; gemma4: `<id>-assistant`) |
| quant | `fp8_online` | online FP8 weights |
| quant | `fp8_ckpt` | vendor FP8 checkpoint (qwen3: `<id>-FP8`) |
| quant | `w4a16` | vendor 4-bit checkpoint (qwen3: `<id>-AWQ`; gemma4: `<id>-qat-w4a16-ct`) |
| parallel | `no_custom_allreduce` | NCCL instead of vLLM's custom all-reduce (TP>1 only) |
| attn | `attn_flashinfer` / `attn_flash_attn` / `attn_triton` | attention backend |

Before each run, the script checks that it can download every derived checkpoint. If a
checkpoint is missing or gated, only that config is dropped and the rest of the sweep runs.

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
- **Cold starts.** The first start of each config pays torch.compile time, plus JIT time
  on new architectures such as sm_120 and sm_121. `compile_s` in the facts table shows how
  much. The warmup run absorbs the rest.

Pin a vLLM version for a sweep. If `smoke` flags a CLI flag as unknown, adjust it in
`build_configs()`.
