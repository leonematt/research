#!/usr/bin/env bash
# PyTorch vs vLLM benchmark suite: one command on any NVIDIA box (discrete, multi-GPU, or unified
# memory like DGX Spark / Jetson). Hardware is auto-detected and the largest open model that fits
# is chosen; `./run.sh list` shows exactly what this box will run.
#
#   ./run.sh setup              one-time: uv venv (py3.12) + vllm, accelerate, torchao, kernels, pandas, matplotlib
#   ./run.sh cuda               install the CUDA toolkit (nvcc) matching torch (sudo, apt)
#   ./run.sh list               hardware, model, every config by tier/category, workloads
#   ./run.sh dry                print every server / benchmark command
#   ./run.sh smoke              vllm_default + pt_eager on one workload: proves the stack works (~5 min)
#   ./run.sh check              every config on a tiny grid -> check.md PASS/WARN/FAIL/NOOP/SKIP (~1 h)
#   ./run.sh quick              quick grid, 1 repeat, 1 anomaly-retry round              (a few hours)
#   ./run.sh standard           standard grid, 2-4 adaptive repeats, 1 retry round       (overnight)
#   ./run.sh full               every config on every workload, 2-5 repeats, 2 retry rounds
#   ./run.sh long               long-context grid
#   ./run.sh suite [args]       check (reused if < 6 h old), then PRESET (default standard) with what passed
#   ./run.sh retry [run_id]     anomaly walk + targeted retries on a finished run (default: newest)
#   ./run.sh anomalies [run_id] just the anomaly walk
#   ./run.sh report [run_id]    rebuild every CSV + report.md
#   ./run.sh collect            merge every run's CSVs into $OUT/all_*.csv
#   ./run.sh custom [args]      `python -m llmbench run` with your own flags
#
# Any extra args pass straight through, e.g.
#   ./run.sh quick --categories speculative_decoding,quantization
#   ./run.sh standard --engines vllm --profile
#   ./run.sh custom --configs vllm_default,kv_fp8 --grid standard --min-repeats 3
# `python3 -m llmbench -h` lists every flag.
#
# Env knobs:
#   FAMILY=qwen3|gemma4  model ladder for auto-selection (default qwen3: Apache-2.0, ungated)
#   MODEL=<id|path>      skip auto-selection
#   TP=<n>               tensor-parallel size (default: smallest that fits)
#   GPU_MEM=<0..1>       --gpu-memory-utilization (default 0.90 discrete, 0.75 unified)
#   PRESET=<name>        preset for `suite` (default standard)
#   CUDA_VISIBLE_DEVICES restrict which GPUs are used
#   HF_HOME=<dir>        model cache (default ~/.cache/huggingface);  HF_TOKEN for gated models
#   OUT=<dir>            results dir (default ./results);  PORT=<n> server port (default 8031)
#   BG=1                 detach with nohup (otherwise use tmux)
#   VLLM_VENV=<dir>      venv to activate if `vllm` isn't on PATH (default ~/venvs/vllm)
#   DOCKER_IMAGE=<img>   run inside a vLLM container (mounts this dir at /work, $HF_HOME for models)
set -euo pipefail
export PYTHONUNBUFFERED=1   # live progress in BG=1 logs
cd "$(dirname "$0")"

MODE="${1:-quick}"; shift || true
OUT="${OUT:-results}"
PORT="${PORT:-8031}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
VLLM_VENV="${VLLM_VENV:-$HOME/venvs/vllm}"
export HF_HOME HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

cuda_toolkit_plan() {
  local cu="$1" os ver arch repo
  . /etc/os-release 2>/dev/null || true
  os="${ID:-unknown}"; ver="${VERSION_ID//./}"
  arch="$(uname -m)"; [[ "$arch" == "aarch64" ]] && arch="sbsa"
  case "$os" in ubuntu|debian) repo="${os}${ver}/${arch}" ;; *) repo="" ;; esac
  if [[ -z "$repo" || -z "$cu" ]]; then
    echo "  install the CUDA ${cu:-13.x} toolkit for your distro: https://developer.nvidia.com/cuda-downloads"
    return 1
  fi
  echo "wget -q https://developer.download.nvidia.com/compute/cuda/repos/${repo}/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb"
  echo "sudo dpkg -i /tmp/cuda-keyring.deb && sudo apt-get update"
  echo "sudo apt-get install -y cuda-toolkit-${cu//./-}"
}
have_nvcc() { [[ -x "${CUDA_HOME:-/nonexistent}/bin/nvcc" ]] || command -v nvcc >/dev/null 2>&1 || [[ -x /usr/local/cuda/bin/nvcc ]]; }

if [[ "$MODE" == "setup" || "$MODE" == "cuda" ]]; then
  command -v uv >/dev/null 2>&1 || { echo "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
  # Python 3.12: the newest distro Pythons (e.g. 3.14 on DGX OS) lack wheels for most ML packages
  [[ -d "$VLLM_VENV" ]] || uv venv --python 3.12 "$VLLM_VENV"
  # shellcheck disable=SC1091
  source "$VLLM_VENV/bin/activate"
  if [[ "$MODE" == "setup" ]]; then
    uv pip install vllm --torch-backend=auto pandas matplotlib accelerate "$@"
    uv pip install torchao || echo "torchao optional (pt_optimized / pt_quant_int4); skipped"
    uv pip install kernels || echo "kernels optional (pt_kernels); skipped"
  fi
  python3 -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"
  CU="$(python3 -c 'import torch; print(torch.version.cuda or "")')"
  if have_nvcc; then
    command -v nvcc >/dev/null 2>&1 && echo "nvcc: $(command -v nvcc)" || echo "nvcc: found (run.sh adds it to PATH for each run)"
  elif [[ "$MODE" == "cuda" ]]; then
    PLAN="$(cuda_toolkit_plan "$CU")" || { echo "$PLAN"; exit 1; }
    echo "$PLAN"; bash -ec "$PLAN"
    have_nvcc && echo "nvcc: /usr/local/cuda/bin/nvcc" || { echo "toolkit installed but nvcc not found"; exit 1; }
  else
    echo; echo "No CUDA toolkit (nvcc). Blackwell GPUs (RTX 50xx, B200, GB10) need it: ./run.sh cuda   runs:"
    cuda_toolkit_plan "$CU" | sed 's/^/  /'
  fi
  echo "ok -> next: ./run.sh list   then   ./run.sh smoke"
  exit 0
fi

# Activate the venv when vllm isn't on PATH (activation matters: FlashInfer's JIT needs ninja on PATH)
if [[ -z "${DOCKER_IMAGE:-}" ]] && ! command -v vllm >/dev/null 2>&1 && [[ -f "$VLLM_VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VLLM_VENV/bin/activate"
fi
# vLLM's has_flashinfer() checks `which nvcc`; a toolkit in /usr/local/cuda isn't on PATH by default
if ! command -v nvcc >/dev/null 2>&1; then
  for d in "${CUDA_HOME:-}" /usr/local/cuda; do
    if [[ -n "$d" && -x "$d/bin/nvcc" ]]; then export PATH="$d/bin:$PATH" CUDA_HOME="$d"; break; fi
  done
fi

COMMON=(--port "$PORT" --out "$OUT")
[[ -n "${FAMILY:-}" ]]  && COMMON+=(--family "$FAMILY")
[[ -n "${MODEL:-}" ]]   && COMMON+=(--model "$MODEL")
[[ -n "${TP:-}" ]]      && COMMON+=(--tp "$TP")
[[ -n "${GPU_MEM:-}" ]] && COMMON+=(--gpu-mem "$GPU_MEM")

RUNS=1
case "$MODE" in
  list)      ARGS=(list); RUNS=0 ;;
  dry)       ARGS=(dry); RUNS=0 ;;
  smoke)     ARGS=(run --preset check --configs vllm_default,pt_eager --workloads reference --retry-rounds 0
                   --max-model-len 4096 --run-tag smoke) ;;
  check)     ARGS=(check) ;;
  quick|standard|full|long) ARGS=(run --preset "$MODE") ;;
  suite)     ARGS=(suite --preset "${PRESET:-standard}") ;;
  custom)    ARGS=(run) ;;
  retry|anomalies|report) ARGS=("$MODE"); [[ $# -gt 0 && "$1" != -* ]] && { ARGS+=(--run-id "$1"); shift; }
             [[ "$MODE" != retry ]] && RUNS=0 ;;
  collect)   ARGS=(collect); RUNS=0 ;;
  *) echo "unknown mode: $MODE (setup|cuda|list|dry|smoke|check|quick|standard|full|long|suite|retry|anomalies|report|collect|custom)"; exit 1 ;;
esac

PY=python3
if [[ -z "${DOCKER_IMAGE:-}" ]]; then
  if (( RUNS )) && ! command -v vllm >/dev/null 2>&1 && [[ "$*" != *"--engines pytorch"* ]]; then
    cat <<EOF
vllm CLI not on PATH and no venv at $VLLM_VENV.
  one-time install:   ./run.sh setup
  existing venv:      VLLM_VENV=/path/to/venv ./run.sh $MODE
  container instead:  DOCKER_IMAGE=<vllm image> ./run.sh $MODE
EOF
    exit 1
  fi
  $PY -c "import pandas" >/dev/null 2>&1 || pip install --quiet pandas matplotlib
fi

PY_CMD=($PY -m llmbench "${ARGS[@]}" "${COMMON[@]}" "$@")
if [[ -n "${DOCKER_IMAGE:-}" ]]; then
  mkdir -p "$HF_HOME"
  GPUS=all; [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && GPUS="\"device=${CUDA_VISIBLE_DEVICES}\""
  CMD=(docker run --rm --gpus "$GPUS" --ipc=host --network host -v "$PWD":/work -w /work
       -v "$HF_HOME":/root/.cache/huggingface -e HF_TOKEN="${HF_TOKEN:-}" -e HF_XET_HIGH_PERFORMANCE
       "$DOCKER_IMAGE" bash -c "pip install -q pandas matplotlib >/dev/null 2>&1 || true; exec ${PY_CMD[*]@Q}")
else
  CMD=("${PY_CMD[@]}")
fi

# Unified memory: the page cache is dropped before each server start (a cached copy of the
# safetensors competes with CUDA allocations). That needs root for one file.
UNIFIED=0
(( RUNS )) && UNIFIED="$($PY -m llmbench probe 2>/dev/null | awk '{print $2}' || echo 0)"
if (( RUNS )) && [[ "$UNIFIED" == "1" && -z "${DOCKER_IMAGE:-}" ]] \
   && ! { sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }; then
  RULE="$USER ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches"
  if [[ "${BG:-0}" == "1" ]]; then
    echo "note: detached runs can't prompt for sudo, so page-cache drops will be skipped. One-time fix:"
    echo "  echo '$RULE' | sudo tee /etc/sudoers.d/drop-caches"
  else
    echo "unified memory: sudo is used only to drop the page cache before each server start (or add: $RULE)"
    if sudo -v; then
      ( while true; do sudo -n true 2>/dev/null; sleep 50; done ) &
      SUDO_KEEPALIVE=$!
      trap 'kill $SUDO_KEEPALIVE 2>/dev/null' EXIT
    fi
  fi
fi

echo "+ ${CMD[*]}"
if [[ "${BG:-0}" == "1" ]] && (( RUNS )); then
  mkdir -p "$OUT"; LOG="$OUT/${MODE}_$(date +%Y%m%d_%H%M%S).log"
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
  echo "detached: pid $!"
  echo "follow:   tail -f $LOG"
  echo "resume:   ./run.sh custom --run-id <run_id> --resume    (the run id is printed at the top of the log)"
else
  "${CMD[@]}"
fi
