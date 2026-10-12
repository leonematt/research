#!/usr/bin/env bash
# One-command runner for vllm_ablation.py on any NVIDIA box (discrete GPUs, multi-GPU,
# or unified memory like DGX Spark / Jetson). Hardware is auto-detected and the largest
# open-source model that fits is chosen; `./run.sh list` shows what this box will run.
#
#   ./run.sh setup                  # one-time: uv venv (py3.12) + pip install vllm
#   ./run.sh list                   # detected hardware, auto choices, configs, grid
#   ./run.sh dry                    # print every vllm serve / bench command
#   ./run.sh smoke                  # baseline only, 1 workload: proves the stack works
#   ./run.sh quick                  # quick grid (7 workloads) x all configs, 1 repeat
#   ./run.sh core                   # quick grid, baseline + graphs/kv/scheduler groups
#   ./run.sh full                   # standard grid (26 workloads) x all configs, 2 repeats
#   ./run.sh custom --configs baseline,kv_fp8,no_prefix_cache --repeats 3
#   ./run.sh report <run_id>        # rebuild report.md/plots for an existing run
#
# Extra args pass straight through to vllm_ablation.py (see `python3 vllm_ablation.py -h`).
#
# Env knobs:
#   FAMILY=qwen3|gemma4  model ladder for auto-selection (default qwen3: Apache-2.0, ungated)
#   MODEL=<id|path>      skip auto-selection (any HF id or local path)
#   TP=<n>               tensor-parallel size (default: smallest that fits)
#   GPU_MEM=<0..1>       --gpu-memory-utilization (default 0.90 discrete, 0.75 unified)
#   CUDA_VISIBLE_DEVICES restrict which GPUs are used/considered
#   HF_HOME=<dir>        model cache on a big disk (default ~/.cache/huggingface)
#   HF_TOKEN=<token>     only needed for gated models (e.g. FAMILY=gemma4)
#   OUT=<dir>            results dir (default ./results)
#   PORT=<n>             vllm serve port (default 8031)
#   BG=1                 detach with nohup (otherwise use tmux)
#   VLLM_VENV=<dir>      venv to activate if `vllm` isn't on PATH (default ~/venvs/vllm)
#   DOCKER_IMAGE=<img>   run inside a vLLM container instead (e.g. vllm/vllm-openai:<tag>,
#                        or the NGC image on DGX Spark). Mounts this dir at /work and
#                        $HF_HOME at /root/.cache/huggingface.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-quick}"; shift || true
OUT="${OUT:-results}"
PORT="${PORT:-8031}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
VLLM_VENV="${VLLM_VENV:-$HOME/venvs/vllm}"
export HF_HOME HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

if [[ "$MODE" == "setup" ]]; then
  command -v uv >/dev/null 2>&1 || { echo "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
  # Python 3.12: newest distro Pythons (e.g. 3.14 on DGX OS) lack wheels for most ML packages
  [[ -d "$VLLM_VENV" ]] || uv venv --python 3.12 "$VLLM_VENV"
  # shellcheck disable=SC1091
  source "$VLLM_VENV/bin/activate"
  uv pip install vllm --torch-backend=auto pandas matplotlib "$@"
  python3 -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"
  echo "ok -> next: ./run.sh list   then   ./run.sh smoke"
  exit 0
fi

# Auto-activate the venv when vllm isn't on PATH. Activation (not calling the venv's
# python by absolute path) matters: FlashInfer's JIT needs `ninja` on PATH.
if [[ -z "${DOCKER_IMAGE:-}" ]] && ! command -v vllm >/dev/null 2>&1 && [[ -f "$VLLM_VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VLLM_VENV/bin/activate"
fi

COMMON=(--port "$PORT" --out "$OUT" --plots)
[[ -n "${FAMILY:-}" ]]  && COMMON+=(--family "$FAMILY")
[[ -n "${MODEL:-}" ]]   && COMMON+=(--model "$MODEL")
[[ -n "${TP:-}" ]]      && COMMON+=(--tp "$TP")
[[ -n "${GPU_MEM:-}" ]] && COMMON+=(--gpu-mem "$GPU_MEM")

RUNS=0
case "$MODE" in
  list)   ARGS=(--list) ;;
  dry)    ARGS=(--quick --dry-run) ;;
  smoke)  ARGS=(--configs baseline --grid quick --workloads reference --max-model-len 4096
                --warmup-prompts 2); RUNS=1 ;;
  quick)  ARGS=(--quick); RUNS=1 ;;
  core)   ARGS=(--quick --groups graphs,kv,scheduler); RUNS=1 ;;
  full)   ARGS=(--grid standard --repeats 2); RUNS=1 ;;
  custom) ARGS=(); RUNS=1 ;;
  report) RID="${1:?usage: run.sh report <run_id>}"; shift
          ARGS=(--report-only --run-id "$RID") ;;
  *) echo "unknown mode: $MODE (setup|list|dry|smoke|quick|core|full|custom|report)"; exit 1 ;;
esac

PY=python3
if [[ -z "${DOCKER_IMAGE:-}" ]]; then
  command -v vllm >/dev/null 2>&1 || {
    cat <<EOF
vllm CLI not on PATH and no venv at $VLLM_VENV.
  one-time install:   ./run.sh setup        (uv venv py3.12 + pip install vllm, ~1 min)
  existing venv:      VLLM_VENV=/path/to/venv ./run.sh $MODE
  container instead:  DOCKER_IMAGE=<vllm image> ./run.sh $MODE
EOF
    exit 1; }
  $PY -c "import pandas" >/dev/null 2>&1 || pip install --quiet pandas matplotlib
fi

# Name the run after the hardware (e.g. gb10_quick_..., 2xh100_full_...) and learn
# whether this is a unified-memory box (page-cache drops need sudo there).
UNIFIED=0
if (( RUNS )); then
  if read -r RUN_ID UNIFIED < <($PY vllm_ablation.py "${COMMON[@]}" --probe --run-tag "$MODE" 2>/dev/null); then :; else
    RUN_ID="nvidia_${MODE}_$(date +%Y%m%d_%H%M%S)"; UNIFIED=0
  fi
  ARGS+=(--run-id "$RUN_ID")
fi
mkdir -p "$OUT"

PY_CMD=($PY vllm_ablation.py "${COMMON[@]}" "${ARGS[@]}" "$@")

if [[ -n "${DOCKER_IMAGE:-}" ]]; then
  mkdir -p "$HF_HOME"
  GPUS=all
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && GPUS="\"device=${CUDA_VISIBLE_DEVICES}\""
  CMD=(docker run --rm --gpus "$GPUS" --ipc=host --network host
       -v "$PWD":/work -w /work
       -v "$HF_HOME":/root/.cache/huggingface
       -e HF_TOKEN="${HF_TOKEN:-}" -e HF_XET_HIGH_PERFORMANCE
       "$DOCKER_IMAGE"
       bash -c "pip install -q pandas matplotlib >/dev/null 2>&1 || true; exec ${PY_CMD[*]@Q}")
else
  CMD=("${PY_CMD[@]}")
fi

# Unified memory: the harness drops the page cache before each `vllm serve` (a cached copy
# of the safetensors competes with CUDA allocations). That needs root for one file.
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
  LOG="$OUT/$RUN_ID.log"
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
  echo "detached: pid $!"
  echo "follow:   tail -f $LOG"
  echo "report:   $OUT/$RUN_ID/report.md   (rebuild: ./run.sh report $RUN_ID)"
  echo "resume:   ./run.sh $MODE --resume --run-id $RUN_ID"
else
  "${CMD[@]}"
fi
