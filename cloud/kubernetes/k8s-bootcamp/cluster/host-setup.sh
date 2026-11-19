#!/usr/bin/env bash
# One-time host setup for the boot camp. Idempotent: safe to re-run.
# Checks: Linux, NVIDIA driver, native Docker Engine (not snap / Docker Desktop), cgroup v2.
# Configures: NVIDIA Container Toolkit as Docker's default runtime, volume-mount device
# selection (needed by nvkind), persistent inotify limits for multi-node kind.
# Installs via Go if missing: kind, nvkind. Reports any other missing CLIs.
set -euo pipefail

KIND_MIN="0.31.0"

log()  { printf '\n== %s\n' "$*"; }
ok()   { printf '   ok: %s\n' "$*"; }
warn() { printf '   WARN: %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
ver_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" == "$2" ]]; }

[[ "$(uname -s)" == "Linux" ]] || die "Linux host required."
[[ $EUID -ne 0 ]] || die "Run as your normal user; the script calls sudo where needed."

# ---------------------------------------------------------------------------
log "NVIDIA driver"
have nvidia-smi || die "nvidia-smi not found. Install the NVIDIA driver first, then re-run."
nvidia-smi -L

# ---------------------------------------------------------------------------
log "Docker daemon"
have docker || die "docker not found. Install Docker Engine (https://docs.docker.com/engine/install/)."
if have snap && snap list docker >/dev/null 2>&1; then
  die "Docker is installed as a snap; it ignores /etc/docker/daemon.json. Remove it and install Docker Engine."
fi
ctx="$(docker context show)"
[[ "$ctx" == "default" ]] || die "Docker context is '$ctx' (Docker Desktop's VM can't see the host GPU). Run: docker context use default"
docker info >/dev/null 2>&1 || die "Can't talk to the Docker daemon. Is your user in the 'docker' group?"
[[ "$(docker info --format '{{.CgroupVersion}}')" == "2" ]] || die "cgroup v2 required (Kubernetes 1.35+ dropped cgroup v1)."
ok "native Docker Engine, cgroup v2"

# ---------------------------------------------------------------------------
log "NVIDIA Container Toolkit"
if ! have nvidia-ctk; then
  have apt-get || die "nvidia-ctk missing and this isn't an apt-based distro. Install nvidia-container-toolkit per NVIDIA's docs, then re-run."
  echo "   installing nvidia-container-toolkit via apt"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq nvidia-container-toolkit
fi
ok "$(nvidia-ctk --version | head -1)"

# Only restart Docker if a config file actually changed: a restart stops running containers,
# including an existing kind cluster.
cfg_hash() { sudo cat /etc/docker/daemon.json /etc/nvidia-container-runtime/config.toml 2>/dev/null | sha256sum; }
before="$(cfg_hash)"
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default --cdi.enabled >/dev/null
sudo nvidia-ctk config --set accept-nvidia-visible-devices-as-volume-mounts=true --in-place
after="$(cfg_hash)"
if [[ "$before" != "$after" ]]; then
  echo "   config changed; restarting Docker (this stops running containers)"
  sudo systemctl restart docker
else
  ok "runtime config already in place; no Docker restart"
fi
[[ "$(docker info --format '{{.DefaultRuntime}}')" == "nvidia" ]] \
  || die "Docker's default runtime is still '$(docker info --format '{{.DefaultRuntime}}')'. Check /etc/docker/daemon.json."
ok "default runtime: nvidia"

# ---------------------------------------------------------------------------
log "Kernel limits for multi-node kind"
printf 'fs.inotify.max_user_watches=524288\nfs.inotify.max_user_instances=512\n' \
  | sudo tee /etc/sysctl.d/99-kind.conf >/dev/null
sudo sysctl --system >/dev/null
ok "inotify limits persisted in /etc/sysctl.d/99-kind.conf"

# ---------------------------------------------------------------------------
log "CLIs"
missing=()
path_fix=()
BIN="$HOME/.local/bin"            # user-level installs, no sudo
mkdir -p "$BIN"
case "$(uname -m)" in x86_64) ARCH=amd64;; aarch64|arm64) ARCH=arm64;; *) die "unsupported arch $(uname -m)";; esac
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

for d in "$BIN" "$(have go && go env GOPATH || echo "$HOME/go")/bin"; do
  case ":$PATH:" in *":$d:"*) ;; *) export PATH="$d:$PATH"; path_fix+=("$d");; esac
done

if ! have kubectl; then
  echo "   installing kubectl (1.35.x, matches the cluster)"
  kv="$(curl -fsSL https://dl.k8s.io/release/stable-1.35.txt)"
  curl -fsSLo "$tmp/kubectl" "https://dl.k8s.io/release/$kv/bin/linux/$ARCH/kubectl"
  curl -fsSLo "$tmp/kubectl.sha256" "https://dl.k8s.io/release/$kv/bin/linux/$ARCH/kubectl.sha256"
  (cd "$tmp" && echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check --quiet)
  install -m 0755 "$tmp/kubectl" "$BIN/kubectl"
fi

if ! have helm; then
  echo "   installing helm"
  curl -fsSLo "$tmp/get-helm.sh" https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
  HELM_INSTALL_DIR="$BIN" USE_SUDO=false bash "$tmp/get-helm.sh" >/dev/null
fi

if ! have cilium; then
  echo "   installing cilium CLI"
  cv="$(curl -fsSL https://raw.githubusercontent.com/cilium/cilium-cli/main/stable.txt)"
  base="https://github.com/cilium/cilium-cli/releases/download/$cv/cilium-linux-$ARCH.tar.gz"
  curl -fsSLo "$tmp/cilium.tgz" "$base"
  curl -fsSLo "$tmp/cilium.tgz.sha256sum" "$base.sha256sum"
  (cd "$tmp" && sed "s#cilium-linux-$ARCH.tar.gz#cilium.tgz#" cilium.tgz.sha256sum | sha256sum --check --quiet)
  tar -xzf "$tmp/cilium.tgz" -C "$BIN" cilium
fi

if have go; then
  if ! have kind || ! ver_ge "$(kind version | awk '{print $2}' | tr -d v)" "$KIND_MIN"; then
    echo "   installing kind v$KIND_MIN"
    go install "sigs.k8s.io/kind@v$KIND_MIN"
  fi
  have nvkind || { echo "   installing nvkind"; go install github.com/NVIDIA/nvkind/cmd/nvkind@latest; }
else
  have kind   || missing+=("kind (needs Go, or a release binary from github.com/kubernetes-sigs/kind)")
  have nvkind || missing+=("nvkind (needs Go: go install github.com/NVIDIA/nvkind/cmd/nvkind@latest)")
fi
for t in kind nvkind kubectl helm cilium; do
  have "$t" && ok "$t: $(command -v "$t")" || missing+=("$t")
done
if (( ${#missing[@]} )); then
  printf '\n   Missing:\n'; printf '     - %s\n' "${missing[@]}"
  die "Install the tools above, then re-run."
fi
if (( ${#path_fix[@]} )); then
  warn "not on your shell PATH: ${path_fix[*]}"
  printf '   add to ~/.zshrc (or ~/.bashrc):\n     export PATH="%s:$PATH"\n' "$(IFS=:; echo "${path_fix[*]}")"
fi

# ---------------------------------------------------------------------------
log "GPU in containers"
echo "   via --runtime=nvidia:"
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all ubuntu:22.04 nvidia-smi -L
echo "   via /dev/null volume mount (the path kind/nvkind uses):"
docker run --rm -v /dev/null:/var/run/nvidia-container-devices/all ubuntu:22.04 nvidia-smi -L

log "Host ready. Next: ./cluster/up.sh"
