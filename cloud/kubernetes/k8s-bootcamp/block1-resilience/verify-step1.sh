#!/usr/bin/env bash
# Block 1, Step 1 verification. Expect: 2 pods per worker, PDB allows 1 disruption,
# 4 ready endpoints, curl responses rotating across pods.
set -euo pipefail

echo "== placement"
kubectl -n bootcamp get pods -l app=api \
  -o custom-columns=POD:.metadata.name,NODE:.spec.nodeName,QOS:.status.qosClass,READY:.status.containerStatuses[0].ready
kubectl -n bootcamp get pods -l app=api -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' | sort | uniq -c

echo "== disruption budget"
kubectl -n bootcamp get pdb api

echo "== endpoints"
kubectl -n bootcamp get endpointslices -l kubernetes.io/service-name=api -o wide

echo "== traffic (ignore the PSA 'restricted' warning; that's Block 4)"
kubectl -n bootcamp run curl --rm -i --restart=Never --image=curlimages/curl -- \
  sh -c 'for i in $(seq 1 8); do curl -s api/hostname; echo; done'

echo "== cgroup path of one api container on bootcamp-worker"
docker exec bootcamp-worker sh -c \
  'crictl ps --name api -q | head -1 | xargs crictl inspect | grep -m1 cgroupsPath'
