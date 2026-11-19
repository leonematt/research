#!/usr/bin/env bash
# Bring up the bootcamp cluster with GPU passthrough and Cilium.
# Run ./host-setup.sh once first.
set -euo pipefail
cd "$(dirname "$0")"
# Tools installed by host-setup.sh, even if your shell rc hasn't been updated yet
export PATH="$HOME/.local/bin:${GOPATH:-$HOME/go}/bin:$PATH"

echo "== preflight"
if [[ "$(docker info --format '{{.DefaultRuntime}}')" != "nvidia" ]] \
   || ! docker run --rm -v /dev/null:/var/run/nvidia-container-devices/all ubuntu:22.04 nvidia-smi -L; then
  echo "Host isn't set up for GPU containers. Run ./cluster/host-setup.sh first." >&2
  exit 1
fi

echo "== create cluster"
nvkind cluster create --config-template=kind-bootcamp.yaml
kubectl config use-context kind-bootcamp

echo "== install Cilium"
cilium install --wait
cilium status --wait

echo "== verify"
kubectl get nodes -o wide -L topology.kubernetes.io/zone,node.bootcamp.io/pool
docker exec bootcamp-worker2 nvidia-smi -L
