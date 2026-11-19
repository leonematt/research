#!/usr/bin/env bash
# Block 1, Step 2 fix: set nodeTaintsPolicy=Honor on the zone spread constraint,
# let the rollout run, then finish the drain. Prints a zones.sh snapshot every time
# placement changes, so you can see the progression.
set -uo pipefail
cd "$(dirname "$0")"
NODE=bootcamp-worker

snap() { echo; echo "---- $(date +%T)  $1"; ./zones.sh; }

# Run a command in the background; print zones.sh whenever its output changes.
watch_while() {
  local label=$1; shift
  echo; echo "==== $label"
  "$@" &
  local pid=$! last="" now
  while kill -0 "$pid" 2>/dev/null; do
    now="$(./zones.sh 2>/dev/null)"
    if [[ "$now" != "$last" ]]; then
      echo; echo "---- $(date +%T)"; echo "$now"; last="$now"
    fi
    sleep 1
  done
  wait "$pid"; local rc=$?
  echo "($label exit code: $rc)"
  return $rc
}

snap "BEFORE: stuck state"
kubectl -n bootcamp get pdb api -o jsonpath='{"PDB: currentHealthy="}{.status.currentHealthy}{" desiredHealthy="}{.status.desiredHealthy}{" disruptionsAllowed="}{.status.disruptionsAllowed}{"\n"}'

policy="$(kubectl -n bootcamp get deploy api -o jsonpath='{.spec.template.spec.topologySpreadConstraints[0].nodeTaintsPolicy}')"
echo; echo "zone constraint nodeTaintsPolicy: ${policy:-<unset, defaults to Ignore>}"
if [[ "$policy" != "Honor" ]]; then
  echo "==== patching nodeTaintsPolicy=Honor (changes the pod template, so this starts a rollout)"
  kubectl -n bootcamp patch deploy api --type=json \
    -p='[{"op":"add","path":"/spec/template/spec/topologySpreadConstraints/0/nodeTaintsPolicy","value":"Honor"}]'
fi

watch_while "rollout" kubectl -n bootcamp rollout status deploy/api --timeout=180s
watch_while "drain $NODE" kubectl drain "$NODE" --ignore-daemonsets --delete-emptydir-data --timeout=90s

snap "AFTER"
kubectl -n bootcamp get pdb api -o jsonpath='{"PDB: currentHealthy="}{.status.currentHealthy}{" desiredHealthy="}{.status.desiredHealthy}{" disruptionsAllowed="}{.status.disruptionsAllowed}{"\n"}'
kubectl get nodes
echo
echo "$NODE stays cordoned. Rebalancing after uncordon is part of the debrief."
