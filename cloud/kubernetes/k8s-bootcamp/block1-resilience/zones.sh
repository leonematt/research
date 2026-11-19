#!/usr/bin/env bash
# Show pods grouped by zone, including Pending (unscheduled) pods.
# Usage: ./zones.sh [label-selector]      e.g. ./zones.sh app=api
#        watch -n1 ./zones.sh             live view during drills
set -euo pipefail
SEL="${1:-app=api}"
NS="${NS:-bootcamp}"

nodes="$(kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"|"}{.metadata.labels.topology\.kubernetes\.io/zone}{"|"}{.spec.unschedulable}{"\n"}{end}')"
pods="$(kubectl -n "$NS" get pods -l "$SEL" -o jsonpath='{range .items[*]}{.metadata.name}{"|"}{.spec.nodeName}{"|"}{.status.phase}{"\n"}{end}')"

awk -F'|' '
  BEGIN { printf "%-12s %-32s %-28s %s\n", "ZONE", "POD", "NODE", "PHASE" }
  NR==FNR { zone[$1] = ($2 == "" ? "none" : $2); cord[$1] = ($3 == "true" ? " (cordoned)" : ""); next }
  NF {
    node = $2
    z = (node == "" ? "UNSCHEDULED" : zone[node])
    n = (node == "" ? "-" : node cord[node])
    printf "%-12s %-32s %-28s %s\n", z, $1, n, $3
    count[z]++
  }
  END {
    print ""
    for (z in count) printf "%-12s %d pod(s)\n", z, count[z]
  }
' <(printf '%s\n' "$nodes") <(printf '%s\n' "$pods")
