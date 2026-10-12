#!/usr/bin/env bash
# One-command runner for the DGX Spark (GB10, aarch64).
#
#   ./run_spark.sh setup                  # one-time: uv venv (py3.12) + pip install vllm
#   ./run_spark.sh list                   # show configs + workload grid, run nothing
#   ./run_spark.sh dry                    # print every vllm serve / bench command
#   ./run_spark.sh smoke                  # baseline only, 1 tiny workload — proves the stack works
#   ./run_spark.sh quick                  # quick grid (7 workloads) x all configs, 1 repeat
#   ./run_spark.sh core                   # quick grid, baseline + graphs/kv/scheduler groups
#   ./run_spark.sh full                   # standard grid (25 workloads) x all configs, 2 repeats
#   ./run_spark.sh custom --configs baseline,kv_fp8,no_prefix_cache --repeats 3
#   ./run_spark.sh report <run_id>        # rebuild report.md/plots for an existing run
#
# Extra args pass straight through to gemma4_vllm_ablation.py.
#
# Env knobs:
#   MODEL=<hf id>        override the preset model (default google/gemma-4-26B-A4B-it)
#   HF_HOME=<dir>        model cache on the big disk (default ~/.cache/huggingface)
#   HF_TOKEN=<token>     needed the first time Gemma weights are pulled
#   OUT=<dir>            results dir (default ./results)
#   PORT=<n>             vllm serve port (default 8031)
#   BG=1                 detach with nohup and tail the log (otherwise run in tmux)
#   VLLM_VENV=<dir>      venv to activate if `vllm` isn't already on PATH (default ~/venvs/vllm);
#                        `./run_spark.sh setup` creates it with uv (Python 3.12 — the Spark's
#                        default 3.14 has no wheels for most ML packages)
#   DOCKER_IMAGE=<img>   run everything inside a vLLM container instead of the host env,
#                        e.g. the NGC vLLM image for DGX Spark. Mounts this dir at /work and
#                        $HF_HOME at /root/.cache/huggingface.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-quick}"; shift || true
OUT="${OUT:-results}"
PORT="${PORT:-8031}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
VLLM_VENV="${VLLM_VENV:-$HOME/venvs/vllm}"
export HF_HOME HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

if [[ "$MODE" == "setup" ]]; then
  command -v uv >/dev/null 2>&1 || { echo "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
  [[ -d "$VLLM_VENV" ]] || uv venv --python 3.12 "$VLLM_VENV"
  # shellcheck disable=SC1091
  source "$VLLM_VENV/bin/activate"
  uv pip install vllm --torch-backend=auto pandas matplotlib "$@"
  python3 -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"
  echo "ok -> next: source $VLLM_VENV/bin/activate && ./run_spark.sh smoke"
  exit 0
fi

# Auto-activate a venv when vllm isn't on PATH. Activation (not calling the venv's
# python by absolute path) matters: FlashInfer's JIT needs `ninja` on PATH.
if [[ -z "${DOCKER_IMAGE:-}" ]] && ! command -v vllm >/dev/null 2>&1 && [[ -f "$VLLM_VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VLLM_VENV/bin/activate"
fi

COMMON=(--preset spark --port "$PORT" --out "$OUT" --plots)
[[ -n "${MODEL:-}" ]] && COMMON+=(--model "$MODEL")
[[ -n "${GPU_MEM:-}" ]] && COMMON+=(--gpu-mem "$GPU_MEM")

case "$MODE" in
  list)   ARGS=(--list) ;;
  dry)    ARGS=(--quick --dry-run) ;;
  smoke)  ARGS=(--configs baseline --grid quick --workloads reference
                --max-model-len 4096 --warmup-prompts 2) ;;
  quick)  ARGS=(--quick) ;;
  core)   ARGS=(--quick --groups graphs,kv,scheduler) ;;
  full)   ARGS=(--grid standard --repeats 2) ;;
  custom) ARGS=() ;;
  report) RID="${1:?usage: run_spark.sh report <run_id>}"; shift
          ARGS=(--report-only --run-id "$RID") ;;
  *) echo "unknown mode: $MODE (setup|list|dry|smoke|quick|core|full|custom|report)"; exit 1 ;;
esac

if [[ "$MODE" != "list" && "$MODE" != "dry" && "$MODE" != "report" ]]; then
  RUN_ID="spark_${MODE}_$(date +%Y%m%d_%H%M%S)"
  ARGS+=(--run-id "$RUN_ID")
fi
mkdir -p "$OUT"

PY_CMD=(python3 gemma4_vllm_ablation.py "${COMMON[@]}" "${ARGS[@]}" "$@")

if [[ -n "${DOCKER_IMAGE:-}" ]]; then
  mkdir -p "$HF_HOME"
  CMD=(docker run --rm --gpus all --ipc=host --network host
       -v "$PWD":/work -w /work
       -v "$HF_HOME":/root/.cache/huggingface
       -e HF_TOKEN="${HF_TOKEN:-}" -e HF_HUB_ENABLE_HF_TRANSFER
       "$DOCKER_IMAGE"
       bash -c "pip install -q pandas matplotlib >/dev/null 2>&1 || true; exec ${PY_CMD[*]@Q}")
else
  command -v vllm >/dev/null 2>&1 || {
    cat <<EOF
vllm CLI not on PATH and no venv at $VLLM_VENV.
  one-time install:   ./run_spark.sh setup        (uv venv py3.12 + pip install vllm, ~1 min)
  existing venv:      VLLM_VENV=/path/to/venv ./run_spark.sh $MODE
  container instead:  DOCKER_IMAGE=<vllm image> ./run_spark.sh $MODE
EOF
    exit 1; }
  python3 -c "import pandas" >/dev/null 2>&1 || pip install --quiet pandas matplotlib
  CMD=("${PY_CMD[@]}")
fi

# Preflight: can we actually pull the model? (gated Gemma + no token is the #1 first-run failure)
if [[ "$MODE" != "list" && "$MODE" != "dry" && "$MODE" != "report" && "${HF_HUB_OFFLINE:-0}" != "1" \
      && -z "${DOCKER_IMAGE:-}" ]]; then
  python3 - "${MODEL:-google/gemma-4-26B-A4B-it}" <<'PYEOF' || exit 1
import os, sys
repo = sys.argv[1]
if os.path.exists(repo):
    sys.exit(0)  # local path
try:
    from huggingface_hub import auth_check
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
except Exception:
    sys.exit(0)  # old hub; skip the check
try:
    auth_check(repo)
    print(f"preflight: HF access to {repo} ok")
except GatedRepoError:
    print(f"preflight: {repo} is gated and your HF credentials don't have access.\n"
          f"  1) accept the license at https://huggingface.co/{repo}\n"
          f"  2) export HF_TOKEN=hf_...   (or: hf auth login)")
    sys.exit(1)
except RepositoryNotFoundError:
    print(f"preflight: {repo} not found on the Hub (or private). Fix with MODEL=<id or local path>.")
    sys.exit(1)
except Exception as e:
    print(f"preflight: couldn't verify HF access ({type(e).__name__}: {e}); continuing")
PYEOF
fi

# The harness drops the page cache before every `vllm serve` (unified memory: a cached copy
# of the safetensors competes with CUDA allocations). That needs root for one file.
if [[ "$MODE" =~ ^(smoke|quick|core|full|custom)$ && -z "${DOCKER_IMAGE:-}" ]] \
   && ! { sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }; then
  RULE="$USER ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches"
  if [[ "${BG:-0}" == "1" ]]; then
    echo "note: detached runs can't prompt for sudo, so page-cache drops will be skipped. One-time fix:"
    echo "  echo '$RULE' | sudo tee /etc/sudoers.d/drop-caches"
  else
    echo "sudo is used only to drop the page cache before each server start (or add: $RULE)"
    if sudo -v; then
      ( while true; do sudo -n true 2>/dev/null; sleep 50; done ) &
      SUDO_KEEPALIVE=$!
      trap 'kill $SUDO_KEEPALIVE 2>/dev/null' EXIT
    fi
  fi
fi

echo "+ ${CMD[*]}"
if [[ "${BG:-0}" == "1" && "$MODE" != "list" && "$MODE" != "dry" && "$MODE" != "report" ]]; then
  LOG="$OUT/$RUN_ID.log"
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
  echo "detached: pid $!"
  echo "follow:   tail -f $LOG"
  echo "report:   $OUT/$RUN_ID/report.md   (rebuild: ./run_spark.sh report $RUN_ID)"
  echo "resume:   ./run_spark.sh $MODE --resume --run-id $RUN_ID"
else
  "${CMD[@]}"
fi
