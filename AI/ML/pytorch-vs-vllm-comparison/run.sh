#!/usr/bin/env bash
# One-command runner for vllm_ablation.py on any NVIDIA box (discrete GPUs, multi-GPU,
# or unified memory like DGX Spark / Jetson). Hardware is auto-detected and the largest
# open-source model that fits is chosen; `./run.sh list` shows what this box will run.
#
#   ./run.sh setup                  # one-time: uv venv (py3.12) + pip install vllm
#   ./run.sh cuda                   # install the CUDA toolkit (nvcc) matching torch (sudo, apt)
#   ./run.sh list                   # detected hardware, auto choices, configs, grid
#   ./run.sh dry                    # print every vllm serve / bench command
#   ./run.sh smoke                  # baseline only, 1 workload: proves the stack works
#   ./run.sh quick                  # quick grid (7 workloads) x all configs, 1 repeat
#   ./run.sh core                   # quick grid, baseline + graphs/kv/scheduler groups
#   ./run.sh full                   # standard grid (26 workloads) x all configs, 2 repeats
#   ./run.sh custom --configs baseline,kv_fp8,no_prefix_cache --repeats 3
#   ./run.sh report <run_id>        # rebuild report.md/plots for an existing run
#   ./run.sh pytorch [run_id]       # add the PyTorch ladder (same model + workloads) to a vLLM run:
#                                   #   default = newest run in $OUT; `new` = standalone PyTorch run
#   ./run.sh compare                # quick grid: vLLM shared-feature configs + PyTorch ladder, one report
#   ./run.sh check                  # functional test: EVERY vLLM + PyTorch config on a tiny grid (~1 h),
#                                   #   writes a PASS/WARN/FAIL/SKIP checklist (check.md)
#   ./run.sh suite [args]           # check, then the full sweep + PyTorch ladder with only the configs
#                                   #   that passed; [args] go to the full sweep (e.g. --repeats 1)
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
export PYTHONUNBUFFERED=1   # live progress in BG=1 logs: Python block-buffers stdout when it goes to a file
cd "$(dirname "$0")"

MODE="${1:-quick}"; shift || true
OUT="${OUT:-results}"
PORT="${PORT:-8031}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
VLLM_VENV="${VLLM_VENV:-$HOME/venvs/vllm}"
export HF_HOME HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

# CUDA toolkit (nvcc): FlashInfer JIT-compiles Blackwell kernels at server start, and the
# driver alone doesn't ship a compiler. Prints (or with `cuda` mode, runs) the apt install
# of the toolkit matching torch's CUDA, from NVIDIA's repo.
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
  # Python 3.12: newest distro Pythons (e.g. 3.14 on DGX OS) lack wheels for most ML packages
  [[ -d "$VLLM_VENV" ]] || uv venv --python 3.12 "$VLLM_VENV"
  # shellcheck disable=SC1091
  source "$VLLM_VENV/bin/activate"
  if [[ "$MODE" == "setup" ]]; then
    uv pip install vllm --torch-backend=auto pandas matplotlib accelerate "$@"
    uv pip install torchao || echo "torchao optional (enables pt_compile_fp8); skipped"
  fi
  python3 -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"
  CU="$(python3 -c 'import torch; print(torch.version.cuda or "")')"
  if have_nvcc; then
    if command -v nvcc >/dev/null 2>&1; then echo "nvcc: $(command -v nvcc)"
    else echo "nvcc: found but not on PATH (run.sh adds it for each run; for your shell:"
         echo "      echo 'export PATH=/usr/local/cuda/bin:\$PATH' >> ~/.bashrc)"; fi
  elif [[ "$MODE" == "cuda" ]]; then
    echo "installing CUDA toolkit $CU (sudo):"
    PLAN="$(cuda_toolkit_plan "$CU")" || { echo "$PLAN"; exit 1; }
    echo "$PLAN"; bash -ec "$PLAN"
    have_nvcc && echo "nvcc: /usr/local/cuda/bin/nvcc" || { echo "toolkit installed but nvcc not found"; exit 1; }
  else
    echo
    echo "No CUDA toolkit (nvcc) found. Blackwell GPUs (RTX 50xx, B200, GB10) need it: FlashInfer"
    echo "compiles some kernels at server start. Install it with:   ./run.sh cuda   which runs:"
    cuda_toolkit_plan "$CU" | sed 's/^/  /'
  fi
  echo "ok -> next: ./run.sh list   then   ./run.sh smoke"
  exit 0
fi

# Auto-activate the venv when vllm isn't on PATH. Activation (not calling the venv's
# python by absolute path) matters: FlashInfer's JIT needs `ninja` on PATH.
if [[ -z "${DOCKER_IMAGE:-}" ]] && ! command -v vllm >/dev/null 2>&1 && [[ -f "$VLLM_VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VLLM_VENV/bin/activate"
fi

# A toolkit installed from NVIDIA's repo lands in /usr/local/cuda without touching PATH.
# FlashInfer's JIT finds it anyway, but vLLM's has_flashinfer() checks `which nvcc` and
# otherwise treats FlashInfer as missing (FP8 KV on sm_120 then fails to start).
if ! command -v nvcc >/dev/null 2>&1; then
  for d in "${CUDA_HOME:-}" /usr/local/cuda; do
    if [[ -n "$d" && -x "$d/bin/nvcc" ]]; then export PATH="$d/bin:$PATH" CUDA_HOME="$d"; break; fi
  done
fi

COMMON=(--port "$PORT" --out "$OUT" --plots)
[[ -n "${FAMILY:-}" ]]  && COMMON+=(--family "$FAMILY")
[[ -n "${MODEL:-}" ]]   && COMMON+=(--model "$MODEL")
[[ -n "${TP:-}" ]]      && COMMON+=(--tp "$TP")
[[ -n "${GPU_MEM:-}" ]] && COMMON+=(--gpu-mem "$GPU_MEM")

# vLLM configs that toggle a feature PyTorch also has (the PyTorch ladder's counterparts)
CMP_VLLM="baseline,enforce_eager,cudagraph_none,no_prefix_cache,mnbt_max_model_len,fp8_online,spec_draft"

RUNS=0; PT_ONLY=0; COMPARE=0; CHECK=0; RID=""
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
  pytorch) if [[ $# -gt 0 && "$1" != -* ]]; then RID="$1"; shift; fi
          if [[ -z "$RID" ]]; then     # newest run that has a summary.json
            RID="$( (ls -td "$OUT"/*/ 2>/dev/null || true) | while read -r d; do
                     [[ -f "${d}summary.json" ]] && { basename "$d"; break; }; done || true)"
          fi
          [[ "$RID" == "new" ]] && RID=""
          [[ -n "$RID" ]] && echo "attaching PyTorch runs to $OUT/$RID"
          PT_ONLY=1; RUNS=1; ARGS=() ;;
  compare) ARGS=(--quick --configs "$CMP_VLLM"); RUNS=1; COMPARE=1 ;;
  check)  ARGS=(--grid check --repeats 1 --warmup-prompts 2 --settle-s 5)   # same max_model_len as the full run: same engine, and its compile cache carries over
          RUNS=1; CHECK=1 ;;
  suite)  ;;   # handled below, after the environment checks
  *) echo "unknown mode: $MODE (setup|cuda|list|dry|smoke|quick|core|full|custom|report|pytorch|compare|check|suite)"; exit 1 ;;
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

if [[ "$MODE" == "suite" ]]; then
  if [[ "${BG:-0}" == "1" ]]; then
    mkdir -p "$OUT"; LOG="$OUT/suite_$(date +%Y%m%d_%H%M%S).log"
    BG=0 nohup "$0" suite "$@" > "$LOG" 2>&1 &
    echo "detached: pid $!   follow: tail -f $LOG"; exit 0
  fi
  baseline_ok() { [[ -f "${1}check_passed.json" ]] && $PY -c "import json,sys
sys.exit(0 if 'baseline' in json.load(open(sys.argv[1]))['vllm'] else 1)" "${1}check_passed.json"; }
  # a check made by different harness code tested different config definitions: never reuse it
  same_harness() { $PY -c "import json,sys; import vllm_ablation as V
sys.exit(0 if json.load(open(sys.argv[1])).get('harness') == V.harness_fingerprint() else 1)" "${1}check_passed.json"; }
  CHECK_DIR="$( (ls -td "$OUT"/*_check_*/ 2>/dev/null || true) | head -1)"
  if [[ -z "${FRESH_CHECK:-}" && -n "$CHECK_DIR" ]] && baseline_ok "$CHECK_DIR" && same_harness "$CHECK_DIR" \
     && (( $(date +%s) - $(stat -c %Y "${CHECK_DIR}check_passed.json") < 6 * 3600 )); then
    echo "=== suite 1/3: reusing the check from ${CHECK_DIR} (same harness, under 6 h old; FRESH_CHECK=1 to redo it)"
  else
    echo "=== suite 1/3: functional check (every config, tiny grid)"
    "$0" check || true
    CHECK_DIR="$( (ls -td "$OUT"/*_check_*/ 2>/dev/null || true) | head -1)"
  fi
  [[ -f "${CHECK_DIR}check_passed.json" ]] || { echo "suite: no checklist from the check run; stopping"; exit 1; }
  baseline_ok "$CHECK_DIR" || { echo "suite: baseline failed the check, so every delta would be meaningless;"
                                echo "       fix that first (see ${CHECK_DIR}check.md), then rerun"; exit 1; }
  read -r VL_PASS PT_PASS CHK_MEM CHK_MODEL CHK_TP < <($PY -c "import json,sys; d=json.load(open(sys.argv[1]))
s=json.load(open(sys.argv[2]))
print(','.join(d['vllm']) or '-', ','.join(d['pytorch']) or '-', s.get('gpu_mem') or '-',
      s.get('model') or '-', s.get('tp') or '-')" "${CHECK_DIR}check_passed.json" "${CHECK_DIR}summary.json")
  # Run the sweep with the model / TP / memory fraction the check validated, so a desktop that
  # happens to use more memory today can't silently change the setup (explicit env vars win).
  if [[ -z "${GPU_MEM:-}" && "$CHK_MEM" != "-" ]]; then export GPU_MEM="$CHK_MEM"; fi
  if [[ -z "${MODEL:-}" && "$CHK_MODEL" != "-" ]]; then export MODEL="$CHK_MODEL"; fi
  if [[ -z "${TP:-}" && "$CHK_TP" != "-" ]]; then export TP="$CHK_TP"; fi
  echo "=== suite 2/3: full vLLM sweep ($MODEL, TP=$TP, gpu_mem $GPU_MEM) with the configs that passed: $VL_PASS"
  "$0" full --configs "$VL_PASS" "$@" || { echo "suite: full sweep failed"; exit 1; }
  if [[ "$PT_PASS" != "-" ]]; then
    echo "=== suite 3/3: PyTorch ladder on the same run: $PT_PASS"
    "$0" pytorch --pt-configs "$PT_PASS"
  fi
  FULL_DIR="$( (ls -td "$OUT"/*_full_*/ 2>/dev/null || true) | head -1)"
  echo "=== suite done: ${FULL_DIR}report.md   (checklist: ${CHECK_DIR}check.md)"
  exit 0
fi

# Name the run after the hardware (e.g. gb10_quick_..., 2xh100_full_...) and learn
# whether this is a unified-memory box (page-cache drops need sudo there).
UNIFIED=0
if (( RUNS )); then
  if read -r RUN_ID UNIFIED < <($PY vllm_ablation.py "${COMMON[@]}" --probe --run-tag "$MODE" \
                                  ${RID:+--run-id "$RID"} 2>/dev/null); then :; else
    RUN_ID="nvidia_${MODE}_$(date +%Y%m%d_%H%M%S)"; UNIFIED=0
  fi
  (( PT_ONLY )) || ARGS+=(--run-id "$RUN_ID")
fi
mkdir -p "$OUT"

PT_COMMON=(--out "$OUT" --plots)
if (( PT_ONLY )); then
  if [[ -n "$RID" ]]; then
    PY_CMD=($PY pytorch_bench.py "${PT_COMMON[@]}" --attach "$RUN_ID" "$@")
  else
    [[ -n "${FAMILY:-}" ]]  && PT_COMMON+=(--family "$FAMILY")
    [[ -n "${MODEL:-}" ]]   && PT_COMMON+=(--model "$MODEL")
    [[ -n "${TP:-}" ]]      && PT_COMMON+=(--tp "$TP")
    [[ -n "${GPU_MEM:-}" ]] && PT_COMMON+=(--gpu-mem "$GPU_MEM")
    PY_CMD=($PY pytorch_bench.py "${PT_COMMON[@]}" --quick --run-id "$RUN_ID" "$@")
  fi
elif (( CHECK )); then
  VL_CMD=($PY vllm_ablation.py "${COMMON[@]}" "${ARGS[@]}" "$@")
  PT_CMD=($PY pytorch_bench.py "${PT_COMMON[@]}" --attach "$RUN_ID" --max-batches 1)
  CK_CMD=($PY vllm_ablation.py --out "$OUT" --checklist --run-id "$RUN_ID")
  # if vLLM's preflight refused to start (busy GPU, no nvcc, gated model) there is no run to check
  PY_CMD=(bash -c "${VL_CMD[*]@Q}; if [[ -f ${OUT@Q}/${RUN_ID@Q}/summary.json ]]; then ${PT_CMD[*]@Q}; ${CK_CMD[*]@Q};
    else echo 'check: stopped before any config ran (see the preflight message above)'; exit 1; fi")
elif (( COMPARE )); then
  VL_CMD=($PY vllm_ablation.py "${COMMON[@]}" "${ARGS[@]}" "$@")
  PT_CMD=($PY pytorch_bench.py "${PT_COMMON[@]}" --attach "$RUN_ID")
  PY_CMD=(bash -c "${VL_CMD[*]@Q} && ${PT_CMD[*]@Q}")
else
  PY_CMD=($PY vllm_ablation.py "${COMMON[@]}" "${ARGS[@]}" "$@")
fi

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
  (( PT_ONLY )) || echo "resume:   ./run.sh $MODE --resume --run-id $RUN_ID"
else
  "${CMD[@]}"
fi
