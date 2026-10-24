# PyTorch vs vLLM: baselines, fully optimized, and every optimization in between

Point it at any NVIDIA machine (one GPU, several, or a unified-memory box like DGX Spark or
Jetson). It detects the hardware, picks the largest open model that fits, and benchmarks:

* **baselines:** plain PyTorch `generate()` (`pt_eager`) and vLLM with everything that can be
  switched off switched off (`vllm_minimal`);
* **fully optimized:** the best PyTorch static path (`pt_optimized`), the best PyTorch serving
  path (`pt_serve_optimized`), and vLLM with FP8 weights, an FP8 KV cache and speculative decoding
  (`vllm_optimized`, `vllm_optimized_spec`);
* **every component optimization** in between, organized into 11 categories, with each
  comparison isolating one thing.

Everything lands in tidy CSVs ready for graphing. After the sweep, an anomaly walk re-checks every
measurement and automatically re-runs the ones that look wrong.

## Quick start

```bash
./run.sh setup       # uv venv at ~/venvs/vllm (Python 3.12): vllm, accelerate, torchao, kernels (safe to rerun)
./run.sh list        # what THIS box will run: hardware, model, configs by tier/category, workloads
./run.sh smoke       # vllm_default + pt_eager on one workload: proves the stack works (~5 min)
./run.sh suite       # recommended: functional check (~1 h), then the standard benchmark with what passed
BG=1 ./run.sh suite  # same, detached (unified-memory boxes: add the sudoers line it prints)
```

Results go to `results/<hardware>_<preset>_<timestamp>/`, e.g. `rtx4090_standard_20261024_2100`.
Load `results_long.csv` or `optimizations.csv` into any plotting tool. `report.md` and `plots/`
summarize the same data.

## How much work: presets, adaptive repeats, right-sized runs

| preset | grid | repeats | anomaly retries | typical time |
|---|---|---|---|---|
| `check` | tiny (in256/out64, c=1,4) | 1 | none | ~1 h for every config |
| `quick` | in 256/2048 × out 256 × c 1/16/64 | 1 | 1 round | a few hours |
| `standard` | in 256/1024/4096 × out 256/1024 × c 1/2/4/8/16/32/64/128 | 2–4 adaptive | 1 round | overnight |
| `full` | standard, **every** config on **every** workload | 2–5 adaptive | 2 rounds | a day+ |
| `long` | in 1024/4096/16384 × out 256/1024 × c 1–128 (powers of 2) | 2–4 adaptive | 1 round | overnight |

Three mechanisms keep runs accurate without running forever:

* **Adaptive repeats.** Every (config, workload) pair gets `--min-repeats` runs. More are added
  only while the coefficient of variation of output throughput is above `--target-cv` (default 5%),
  up to `--max-repeats`. Quiet measurements stop at the minimum; noisy ones get the samples they
  need. Reported values are the median of the good repeats.
* **Right-sized runs.** Each run sends `max(8, 4 × concurrency)` requests (four turnovers of the
  concurrency cap, so ramp-up and drain are a small part of the window). This is capped by a 131k
  output-token budget, but never below 2 × concurrency. Long-output, high-concurrency shapes stay
  bounded.
* **Focused workloads.** Only the anchors (baselines, defaults, optimized) run the whole grid.
  Ladder steps and components run the core set (the reference shape at every concurrency) plus the
  workloads where their effect should show. For example, prefix caching runs the shared-prefix
  workload and speculative decoding runs the real-text workload. `--all-workloads` (or
  `--preset full`) turns this off.

The progress line shows a live ETA, and `--max-hours` skips retry rounds once a run has taken that
long.

## Fitting the run to the card

Before anything runs, the card is classified (memory tier, free memory, architecture). `list`
and the top of every log show the decisions:

```
[card] NVIDIA GeForce RTX 5070 Ti: small tier (15 GiB x 1, sm_120, memory fraction 0.79)
[card] estimated KV capacity ~23,413 tokens (144 KiB/token)
[tailor] spec_draft: memory fraction 0.79 -> 0.63 (draft model ~1.4 GiB + 1.0 GiB warmup)
[tailor] dropping in4096/out1024/c32: 32 x 5120 = 163,840 KV tokens in flight > 4 x ~23,413 capacity
```

* **Model, TP and context length** come from the largest model that fits (see below). On a GPU that
  also drives your desktop, the memory fraction is lowered to what is actually free.
* **Memory reservations.** vLLM sizes its KV cache from a profiling pass, and some allocations
  happen after it: the draft model's attention warmup under speculative decoding, and FlashInfer
  workspaces. On a big card the slack absorbs them; on a 16 GB card the server dies at startup.
  Configs that load extra things get a lower memory fraction, sized to the draft/EAGLE weights plus
  warmup. That smaller KV pool is a real cost of the optimization on this card, and it shows in
  `kv_cache_tokens`.
* **OOM at startup → retry.** If a server still runs out of memory while starting, it is retried
  up to twice with the memory fraction 0.04 lower each time. The fraction actually used
  (`gpu_mem_used`) and `oom_retries` are recorded in the facts.
* **Workloads the KV cache can hold.** KV capacity is estimated from the model's architecture.
  Closed-loop shapes that would keep more than `--kv-oversub` (default 4) × that many tokens in
  flight are dropped; past that point vLLM just queues and preempts, and the run would measure a
  waiting line for hours. `--kv-oversub 0` keeps everything.

## Anomaly walk and auto-retry

After the sweep, every result is checked, the suspicious ones are re-run, and the outcome is
recorded in `anomalies.csv`:

| check | what it catches | action |
|---|---|---|
| `failed` / `partial` | every / some repeats failed (not OOM) | retry; failed repeats are replaced (kept in `raw/retired/`) |
| `errors` / `incomplete` | requests errored, or fewer completed than were sent | retry |
| `impossible` | TTFT > E2E, TPOT ≤ 0, zero throughput | retry |
| `noisy` | repeats disagree (CV > `--noisy-cv`, default 10%) | retry with 2 more repeats |
| `dip` | throughput drops > `--dip` (25%) when concurrency rises, with no preemptions to explain it | retry both points |
| `prefix_reset` | the prefix cache couldn't be reset before the run | retry |
| `compiling` | torch.compile ran inside the measured window (PyTorch) | retry (the cache is warm now) |
| `ladder` | a cumulative step is > 25% slower than the step before it | retry both steps once to confirm |
| `server` | a vLLM server failed to start for a transient reason (timeout, memory) | retry the config |
| `oom`, `no_spec_stats` | deterministic / informational | reported |

Retries are capped (`--max-retry-pairs`, default 40 per round) and ordered by severity. Each
round is re-checked afterwards:

* `resolved`: the retry disagreed with the first measurement, so the first one was the outlier.
* `persisting`: the retry reproduced it, so it's likely real behavior (or a genuine problem with
  that config).

`./run.sh retry [run_id]` runs the walk and retries on a finished run, and
`./run.sh anomalies [run_id]` only reports.

A related safety net: if vLLM rejects one component's flag inside a composite config (say your
version doesn't know `--no-enable-chunked-prefill`), that composite is rebuilt without the
component and re-run on the spot. Pending composites that use it are rebuilt too. The rebuild is
noted in `summary.json` and the report.

## The 11 optimization categories

Each comparison is **off → on**, and in the CSVs a speedup above 1 always means "on" is better.
Four kinds of evidence, labelled in the `method` column:

* **ablation:** one feature switched off (or on) relative to a reference. On the vLLM side the
  reference is `vllm_default`, so the delta is that feature's marginal value with everything else
  still on.
* **ladder:** a cumulative build-up, one feature per step. The steps add up to the end-to-end gap
  (in that order), which is what a waterfall chart needs.
* **variant:** an alternative implementation of the same thing (attention backend, page size,
  parallel layout).
* **engine:** end-to-end anchors across engines.

| category | vLLM side | PyTorch side | honest caveats |
|---|---|---|---|
| **PagedAttention** | page size 16 → 32 / 64 (`block_size_*`) | static contiguous KV → paged KV (`pt_static` → `pt_cb`) | vLLM can't run without paging; transformers ships paging only together with continuous batching (marked `confounded_with`) |
| **Continuous Batching** | static waves → continuous admission (`vllm_static_waves` → `vllm_default`); async scheduling (`no_async_scheduling`) | `generate()` static batches → CB manager | vLLM can't disable it: the static side is the same server fed by a client that admits a new batch only when the whole previous one finished. Shows best on the `mixed` workload (variable output lengths) |
| **Prefix Caching** | `no_prefix_cache` → default | block sharing off → on (`pt_cb_noprefix` → `pt_cb`) | judge on the `shared_prefix` workload |
| **Chunked Prefill** | `no_chunked_prefill` (fallback `mnbt_max_model_len`) → default; smaller chunks (`mnbt_small`) | 512-token step budget (`pt_cb_chunk_small`) | |
| **Kernel Fusion** | Inductor on vs off with identical CUDA graphs (`no_compile` → `cudagraph_full_decode_only`); `-O1`/`-O3`; quant-fusion passes (`fp8_nofuse` → `fp8_online`) | `torch.compile` without graphs (`pt_static` → `pt_compile_nograph`); CB compile | quant fusion only matters on FP8 W8A8 (sm_89+) |
| **Speculative Decoding** | n-gram, draft model, EAGLE-3 | assisted generation (batch 1) | judge on the `text` workload (real English); random tokens give ~0 acceptance |
| **Parallelism Strategies** | TP vs PP vs DP vs expert parallel; custom all-reduce vs NCCL | — | multi-GPU only; compare `output_throughput_per_gpu` for efficiency |
| **Quantization** | FP8 online / vendor FP8 / vendor 4-bit / FP8 KV cache | torchao FP8 W8A8 (int8 below sm_89), int4 weight-only | vendor checkpoints are different weights |
| **CUDA Graphs** | graphs off (`cudagraph_none`); PIECEWISE / FULL_DECODE_ONLY variants; fully eager | `reduce-overhead` vs plain compile; CB `use_cuda_graph` | |
| **Model-Specific Kernels** | attention backends (FlashInfer / FlashAttention / Triton); vLLM's hand-written CUDA ops vs Inductor's (`custom_ops_all`) | SDPA vs eager attention; hub kernels (`use_kernels=True`) | |
| **Large-Scale Serving** | open-loop Poisson load at 25–125% of measured capacity (goodput vs an SLO); KV pressure (`gpu_mem_low`); concurrency caps; data parallel | best PyTorch serving vs vLLM | PyTorch has no server, so open-loop workloads are vLLM-only |

The cumulative ladders:

* **vLLM:** minimal → +torch.compile → +CUDA graphs → +prefix caching → +chunked prefill →
  +async scheduling (= defaults) → +FP8 weights → +FP8 KV (= optimized) → +speculative decoding.
* **PyTorch static:** eager → +SDPA → +static KV → +compile → +CUDA graphs → +FP8 (= optimized).
* **PyTorch serving:** … +static KV → +paged KV & continuous batching → +compile → +CUDA graphs &
  async (= optimized).

## Output files (`results/<run_id>/`)

| file | one row per | what to graph with it |
|---|---|---|
| `results_long.csv` | (config, workload, metric) | everything: filter by `engine`, `tier`, `categories`, `kind`, `concurrency`, `metric` |
| `optimizations.csv` | (comparison, workload, metric) | speedup bars per category; `speedup_x` > 1 = helps; `significant` flags changes above measured noise |
| `category_summary.csv` | comparison | one-glance chart: speedup at the focus workload + geomean over all |
| `headline.csv` | (workload, anchor) | baselines vs optimized, `x_vs_pt_eager`, `x_vs_vllm_default` |
| `ladder.csv` | (ladder, step, workload, metric) | waterfall: `step_x`, `cumulative_x`, `share_of_total_gain` |
| `kernel_breakdown.csv` | (config, kernel class) | where the GPU time goes (with `--profile`): attention / GEMM / norm / comms / …, plus `gpu_busy_pct` |
| `anomalies.csv` | finding | what was flagged, retried, resolved or persisting |
| `results.csv` / `results_all.csv` | (config, workload) median / every repeat | raw medians, `n_repeats`, `cv_output_throughput` |
| `report.md`, `plots/` | | the same, as tables and PNGs |
| `summary.json`, `raw/`, `server_logs/`, `profiles/` | | environment, engine facts, untouched bench JSON, logs, traces |

`./run.sh collect` merges every run into `results/all_*.csv` for cross-hardware charts. Every row
carries `run_id`, `hardware`, `gpu_name` and the versions.

Example:

```python
import pandas as pd, seaborn as sns
opt = pd.read_csv("results/<run_id>/category_summary.csv")
sns.barplot(opt[opt.method == "ablation"], y="category_label", x="throughput_x_focus", hue="engine")

long = pd.read_csv("results/<run_id>/results_long.csv")
d = long[(long.metric == "output_throughput") & long.tier.isin(["baseline", "default", "optimized"])
         & (long.kind.isin(["grid", "reference"]))]
sns.lineplot(d, x="concurrency", y="value", hue="config", style="input_len")
```

## Commands

```bash
./run.sh list | dry | smoke | check | quick | standard | full | long | suite
./run.sh retry [run_id] | anomalies [run_id] | report [run_id] | collect
./run.sh custom --configs vllm_default,kv_fp8 --grid standard --min-repeats 3
./run.sh quick --categories speculative_decoding,quantization     # anchors are always kept
./run.sh standard --engines vllm --profile                         # + kernel_breakdown.csv
./run.sh custom --run-id <id> --resume                             # finish an interrupted run
```

Overrides: `FAMILY=gemma4`, `MODEL=<id or path>`, `TP=2`, `GPU_MEM=0.8`,
`CUDA_VISIBLE_DEVICES=2,3`, `--max-model-len 8192`, `--sharegpt-path <json>`, `--slo "ttft:500 tpot:50"`.
`python3 -m llmbench -h` lists every flag.

## How the model is picked

The largest model from `--family` that fits is used, with the smallest tensor-parallel size that
fits across the visible GPUs. Context length is sized from what's left for the KV cache.

| family | ladder (first that fits wins) | derived checkpoints |
|---|---|---|
| `qwen3` (default, ungated) | Qwen3-30B-A3B (MoE) → 14B → 8B → 4B → 1.7B → 0.6B | draft Qwen3-0.6B; EAGLE-3 `RedHatAI/Qwen3-{8B,14B,32B}-speculator.eagle3`; `-FP8`, `-AWQ` |
| `gemma4` (gated: set `HF_TOKEN`) | gemma-4-26B-A4B-it (MoE) → 12B → E4B → E2B | draft `<id>-assistant`; 4-bit `<id>-qat-w4a16-ct` |

Derived checkpoints are verified on the Hub first; a missing or gated one only drops the configs
that need it.

## Measurement details

* **One engine process per config.** Each vLLM config gets a fresh `vllm serve` process, and each
  PyTorch config gets a fresh worker process with its own CUDA context and compile cache.
* **Warm starts only.** A vLLM start that compiled fresh gets a ~20% smaller KV cache than one
  that loads the compiled graph from cache, which would look like a knob effect on KV-limited
  workloads. The harness restarts such a server once from the warm cache and measures that. The
  cold log is kept as `server_logs/<config>.cold.log`.
* **Same prompts on both engines.** `mixed` and `text` workloads go through a built-in client that
  sends the exact token ids (and per-request output lengths) the PyTorch worker uses. Grid, prefix
  and rate workloads go through `vllm bench serve`.
* **Static batching (PyTorch).** Concurrency C means batches of C. A request's TTFT is its batch's
  first token and its E2E is the batch's end. With variable output lengths the batch runs until its
  longest member finishes, and only each request's own tokens count as output.
* **Continuous batching.** Closed loop at concurrency C, with latency measured from when each
  request was first scheduled. This matches `vllm bench serve --max-concurrency C`. The PyTorch CB
  KV cache is sized to the token capacity the vLLM default had.
* **Prefix cache isolation.** `/reset_prefix_cache` is called before every measured run.
  Prometheus counters are diffed around each run, so preemptions, prefix hit rate and spec-decode
  acceptance sit next to throughput and latency.
* **Knob verification.** The functional check compares what each engine *reports* (CUDA-graph
  mode, compile mode, KV dtype, prefix caching, chunked prefill, quantization, speculative config,
  attention backend, TP/PP size, custom ops, fusion passes) against what the config should change.
  A flag that is accepted but does nothing is a FAIL, not a silent duplicate of the default.
* **Profiling (`--profile`).** One short torch-profiler pass per config at the reference shape,
  run *after* the measured runs so profiler overhead never touches a reported number. Traces are in
  `profiles/` (open them in Perfetto) and the classified breakdown is in `kernel_breakdown.csv`.

## Safety rails

* **Preflight.** The run refuses to start if another vLLM server or GPU process is running, the
  port is busy, or there isn't enough free memory. `--allow-shared` overrides this and marks the
  run as not clean.
* **Unified memory (Spark, Jetson).** The page cache is dropped before each server start, which
  needs sudo for one file. Interactive runs ask once; `BG=1` prints the sudoers line to add.
* **CUDA toolkit on PATH.** vLLM's FlashInfer check uses `which nvcc`, so `run.sh` and every server
  launch put the toolkit on PATH. `./run.sh cuda` installs it.
* **Failure reporting.** A failed start prints the last startup step, the error lines and a likely
  cause (gated repo, OOM, unknown flag, unsupported feature), then moves on.
* **Resumable.** Every measured run is one file in `raw/`. `--resume` (or reusing a `--run-id`)
  runs only what's missing, and `report` rebuilds every CSV from those files.

Pin a vLLM version for a sweep. Flags drift between releases, which is what `check` and the
composite rebuild are for.

## Project layout

```
run.sh                  one-command runner (setup, venv/docker, background, sudo priming)
llmbench/
  cli.py                commands, presets, check -> suite flow, the post-sweep anomaly rounds
  configs.py            config catalogs: baselines, components, ladders, optimized composites
  taxonomy.py           the 11 categories and which config pairs measure each
  workloads.py          grids, run sizing, shared prompt generator (random / mixed / text)
  sampling.py           adaptive repeats + ETA
  tailor.py             card classification, memory reservations, OOM retries, workload fit
  anomalies.py          the anomaly walk and retry planning
  vllm_runner.py        vllm serve lifecycle, vllm bench serve, built-in client runs, profiling
  torch_runner.py       PyTorch orchestrator (one worker process per config, crash recovery)
  torch_worker.py       static batching, continuous batching, assisted decoding, torchao
  client.py             built-in OpenAI-API load generator (closed loop and static waves)
  metrics.py            per-request timings -> vllm-bench-compatible metrics
  results.py            aggregation and every CSV
  profiling.py          chrome trace -> GPU time by kernel class
  report.py             report.md and plots
  checklist.py          PASS/WARN/FAIL/NOOP/SKIP functional check
  hardware.py models.py util.py
tests/
  fake_vllm/vllm        stand-in vLLM CLI (server + bench) for testing without a GPU
  run_tests.py          end-to-end tests against the fake vLLM and a synthetic PyTorch worker
```

`python3 tests/run_tests.py` exercises the orchestration, client, CSVs, anomaly walk and retries
on any machine, with no GPU and no vLLM install.
