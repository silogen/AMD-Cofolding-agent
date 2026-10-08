#!/usr/bin/env bash
# What the supervisor fleet is doing, from both sides.
#
#   k8s/scripts/campaign-status.sh --namespace demo-lux [--campaign ID] [--watch [SECONDS]]
#
# Capacity and activity are different questions, so both are printed:
#
#   * `kubectl get deploy` says how many supervisors *exist*. That is the
#     ceiling on parallelism, and it is what `kubectl scale` changes.
#   * the ledger says how many are actually *holding a task*. An idle
#     supervisor is a running pod with nothing claimed, which looks identical
#     to a busy one in `kubectl get pods`.
#
# A healthy campaign has those two numbers equal. `claimed` below the pod count
# means supervisors are idle -- usually the queue is empty or the conductor is
# holding dispatch back. `queued` sitting above zero with every pod busy means
# the campaign wants more concurrency than the pool has pods, which is what
# happens when a campaign's `concurrency` and `kubectl scale --replicas` were
# set to different numbers.
set -euo pipefail

PROJECT=""
CAMPAIGN=""
WATCH=0
INTERVAL=30
SUPERVISOR=dd-agentic-supervisor
CONDUCTOR=deployment/dd-agentic-conductor

while [[ $# -gt 0 ]]; do
    case "$1" in
        --namespace)
            PROJECT="$2"
            shift 2
            ;;
        --campaign)
            CAMPAIGN="$2"
            shift 2
            ;;
        --watch)
            WATCH=1
            shift
            if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
                INTERVAL="$1"
                shift
            fi
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${PROJECT}" ]]; then
    echo "usage: $0 --namespace <project-namespace> [--campaign ID] [--watch [SECONDS]]" >&2
    exit 1
fi

report() {
    echo "--- pool capacity (what kubectl scale controls)"
    kubectl --namespace "${PROJECT}" get deployment "${SUPERVISOR}" \
        --output custom-columns='DESIRED:.spec.replicas,READY:.status.readyReplicas,AVAILABLE:.status.availableReplicas' \
        --no-headers 2>/dev/null |
        awk '{printf "  desired=%s ready=%s available=%s\n", $1, $2, $3}'

    kubectl --namespace "${PROJECT}" get pods \
        --selector "app.kubernetes.io/name=${SUPERVISOR}" \
        --output custom-columns='NAME:.metadata.name,STATUS:.status.phase,AGE:.metadata.creationTimestamp' \
        --no-headers 2>/dev/null | sed 's/^/  /'

    echo
    echo "--- ledger (who is actually working)"
    # Run inside the conductor, which already has the client and the store URL.
    # -i so the heredoc reaches `python -` on stdin.
    kubectl --namespace "${PROJECT}" exec -i "${CONDUCTOR}" -- python - "${CAMPAIGN}" <<'PY'
import sys

from reinvent4_agent.campaign_client import CampaignStoreClient
from reinvent4_agent.conductor import format_status

store = CampaignStoreClient()
wanted = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None

if wanted:
    ids = [wanted]
else:
    # Everything still live. A finalized campaign has nothing left to report.
    ids = [
        c["campaign_id"]
        for c in store.campaigns()
        if c["state"] in ("dispatching", "draining")
    ]

if not ids:
    print("  no live campaigns")
else:
    for campaign_id in ids:
        print(format_status(store.status(campaign_id)))
        print()
PY
}

if [[ "${WATCH}" -eq 1 ]]; then
    while true; do
        printf '\033[2J\033[H'
        echo "=== $(date -u '+%Y-%m-%d %H:%M:%SZ')  namespace ${PROJECT}"
        echo
        report
        sleep "${INTERVAL}"
    done
else
    report
fi
