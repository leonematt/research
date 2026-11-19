# Production Kubernetes Boot Camp

A hands-on 6-hour session on a local kind cluster, with real GPU scheduling on the host's RTX 3090.

| Block | Time | Focus | Failure drill |
|---|---|---|---|
| 0 | 15m | Cluster bring-up: kind + nvkind, Cilium (eBPF), zone-labeled workers | — |
| 1 | 60m | Workload resilience: topology spread, PDBs, probes, graceful termination | Drain deadlock, node loss, taint-based eviction timing |
| 2 | 60m | Resources: QoS, CFS throttling, PriorityClass/preemption, quotas, CPU Manager on hybrid cores | OOMKill vs. eviction vs. throttling triage |
| 3 | 60m | Traffic: Gateway API (weighted canary, header routing), NetworkPolicy, Hubble | Blackholed traffic, bad canary, DNS egress denied |
| 4 | 60m | Security: PSA restricted, least-privilege RBAC, ValidatingAdmissionPolicy (CEL), audit logs | Compromised pod escalation, root-caused from audit logs |
| 5 | 60m | GPUs: GPU Operator, time-slicing the 3090, affinity/taints, DRA | Pending GPU pods, preemption, affinity mismatch |
| 6 | 45m | Control plane: static pods, etcd snapshot/restore, kubelet loss | Broken apiserver manifest, etcd restore |

## Layout

```
cluster/             kind config, up.sh, down.sh
block1-resilience/   manifests and verify scripts for Block 1
notes/drill-log.md   prediction -> symptom -> root cause, per drill
```

Later blocks get their own `blockN-*/` folder as we go.

## One-time host setup (Linux, NVIDIA driver installed)

Tools: Docker (cgroup v2), kind >= v0.31, kubectl, helm, cilium CLI, Go, and the NVIDIA Container Toolkit.

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default --cdi.enabled
sudo nvidia-ctk config --set accept-nvidia-visible-devices-as-volume-mounts=true --in-place
sudo systemctl restart docker
go install github.com/NVIDIA/nvkind/cmd/nvkind@latest

# Both should list the 3090:
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all ubuntu:22.04 nvidia-smi -L
docker run --rm -v /dev/null:/var/run/nvidia-container-devices/all ubuntu:22.04 nvidia-smi -L
```

## Run

```bash
./cluster/up.sh
kubectl apply -f block1-resilience/01-api.yaml
kubectl -n bootcamp rollout status deploy/api --timeout=120s
./block1-resilience/verify-step1.sh
```

Tear down with `./cluster/down.sh`.
