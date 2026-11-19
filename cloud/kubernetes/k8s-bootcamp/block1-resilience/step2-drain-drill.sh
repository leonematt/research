#!/usr/bin/env bash
# Block 1, Step 2 drill: drain zone-a (bootcamp-worker) during "maintenance".
# Write your prediction in notes/drill-log.md BEFORE running this.
# Tip: in a second terminal, watch live:  watch -n1 ./block1-resilience/zones.sh
set -uo pipefail   # no -e: the drain is expected to be interesting

NODE=bootcamp-worker

echo "== before"
kubectl -n bootcamp get pods -l app=api -o wide
kubectl -n bootcamp get pdb api

echo
echo "== draining $NODE (90s timeout)"
start=$(date +%s)
kubectl drain "$NODE" --ignore-daemonsets --delete-emptydir-data --timeout=90s
rc=$?
echo "drain exit code: $rc after $(( $(date +%s) - start ))s"

echo
echo "== after"
kubectl get nodes
kubectl -n bootcamp get pods -l app=api -o wide
kubectl -n bootcamp get pdb api

echo
echo "Node is left cordoned on purpose. Don't uncordon until we debrief."
