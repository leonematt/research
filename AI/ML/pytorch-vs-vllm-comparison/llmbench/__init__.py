"""llmbench: PyTorch vs vLLM serving benchmarks, broken down by optimization.

Modules
-------
util         small helpers (subprocess, HTTP, memory, formatting)
hardware     GPU detection, model auto-sizing, preflight checks
models       model families and their derived checkpoints (draft, EAGLE-3, FP8, 4-bit)
workloads    workload grids and the prompt generator shared by both engines
taxonomy     the 11 optimization categories and which configs compare what
configs      vLLM and PyTorch config catalogs (baselines, components, ladders, optimized)
metrics      per-request metrics -> the same columns `vllm bench serve` reports
client       built-in HTTP load generator (static-batch waves, token-exact prompts)
vllm_runner  `vllm serve` lifecycle, `vllm bench serve`, Prometheus deltas
torch_runner PyTorch/transformers orchestrator; torch_worker runs one config per process
profiling    torch-profiler trace -> kernel-time breakdown
results      aggregation and every CSV
report       report.md and plots
checklist    PASS/WARN/FAIL/NOOP/SKIP functional check
cli          `python -m llmbench <command>`
"""

__version__ = "2.0.0"
