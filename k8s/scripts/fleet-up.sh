#!/usr/bin/env bash
# Bring the dd-agentic fleet up (or down) in dependency order.
#
#   k8s/scripts/fleet-up.sh --namespace demo-lux [--supervisors N]
#   k8s/scripts/fleet-up.sh --namespace demo-lux --down
#
# Kubernetes Deployments are independent: scaling the supervisor pool does not
# start the scorers it needs, and `dd-agent` only starts a process in a pod
# that already exists. Nothing in this stack has a dependency graph, so this
# script is it.
#
# Why the order matters, and why the waits are not optional:
#
#   * A supervisor that starts before the runner is ready claims a task, fails
#     to reach the runner, and burns one of its two attempts. So supervisors
#     come last, after everything they depend on reports ready.
#   * A missing *scorer* is worse than a missing runner, because it is quiet.
#     The scoring component turns a connection error into NaN for every
#     molecule -- correct behaviour for a flaky service mid-campaign, but with
#     the service down for the whole run you get a completed job full of NaN
#     and a geometric mean built on them. That failure produces results rather
#     than an error, which is the one worth preventing.
#
# Every CPU scorer is started regardless of which objectives the campaign
# actually uses. They hold no GPU, so the cost of starting one that is not
# needed is nothing, while the cost of *not* starting one the TTP references is
# a silently NaN-scored campaign.
#
# The waits use `kubectl rollout status`, which is gated on each Deployment's
# readiness probe -- and those probes are the services' own /health endpoints,
# so "ready" here means the service answered, not merely that a pod exists.
#
# NB `--down` takes the runner with it, and RL jobs live in the runner's pod:
# anything still running is killed, not checkpointed. Stop jobs first if the
# work matters.
set -euo pipefail

PROJECT=""
SUPERVISORS=2
DOWN=0

# Boltz runs two replicas because the whole point of the least-busy proxy in
# front of it is having somewhere idle to place a batch; at one replica the
# proxy is just an extra hop.
BOLTZ_REPLICAS=2

# No dependencies of their own. The runner is here rather than later because
# it takes the longest to become ready (an 11 GiB image and a GPU), so starting
# it first means its warm-up overlaps everything else's.
TIER_DATA=(
    dd-agentic-reinvent-runner
    dd-agentic-campaign-store
    dd-agentic-boltz
    dd-agentic-boltz-proxy
    dd-agentic-admet
    dd-agentic-posebusters
    dd-agentic-interaction-profiler
    dd-agentic-retrodmta
)

# Reads the proxy, and its readiness probe deliberately fails while no Boltz
# replica is healthy -- so it has to come after them or the wait below trips.
TIER_GATEWAY=(dd-agentic-boltz-gateway)

# Clients of everything above.
TIER_CONTROL=(
    dd-agentic-conductor
    dd-agentic
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --namespace)
            PROJECT="$2"
            shift 2
            ;;
        --supervisors)
            SUPERVISORS="$2"
            shift 2
            ;;
        --down)
            DOWN=1
            shift
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${PROJECT}" ]]; then
    echo "usage: $0 --namespace <project-namespace> [--supervisors N] [--down]" >&2
    exit 1
fi

scale() {
    local deployment="$1" replicas="$2"
    if ! kubectl --namespace "${PROJECT}" get deployment "${deployment}" \
        --output name >/dev/null 2>&1; then
        echo "  ${deployment}: not deployed, skipping"
        return
    fi
    kubectl --namespace "${PROJECT}" scale deployment "${deployment}" \
        --replicas="${replicas}" >/dev/null
    echo "  ${deployment} -> ${replicas}"
}

wait_ready() {
    for deployment in "$@"; do
        kubectl --namespace "${PROJECT}" get deployment "${deployment}" \
            --output name >/dev/null 2>&1 || continue
        echo "  waiting for ${deployment}"
        kubectl --namespace "${PROJECT}" rollout status "deployment/${deployment}" \
            --timeout=15m
    done
}

if [[ "${DOWN}" -eq 1 ]]; then
    echo "stopping the fleet in ${PROJECT}"
    # Workers first, so none of them claims a task while its dependencies are
    # being taken away.
    scale dd-agentic-supervisor 0
    for deployment in "${TIER_CONTROL[@]}" "${TIER_GATEWAY[@]}" "${TIER_DATA[@]}"; do
        scale "${deployment}" 0
    done
    echo
    echo "AIM is left running: it is reconciled by the dd-agentic-aim AIMService,"
    echo "so scaling its Deployment would be undone. Delete the AIMService to"
    echo "free its GPU."
    exit 0
fi

echo "starting the fleet in ${PROJECT} with ${SUPERVISORS} supervisor(s)"
echo
echo "scorers, store and runner:"
for deployment in "${TIER_DATA[@]}"; do
    if [[ "${deployment}" == "dd-agentic-boltz" ]]; then
        scale "${deployment}" "${BOLTZ_REPLICAS}"
    else
        scale "${deployment}" 1
    fi
done
wait_ready "${TIER_DATA[@]}"

echo
echo "boltz gateway:"
for deployment in "${TIER_GATEWAY[@]}"; do
    scale "${deployment}" 1
done
wait_ready "${TIER_GATEWAY[@]}"

echo
echo "conductor and interactive agent:"
for deployment in "${TIER_CONTROL[@]}"; do
    scale "${deployment}" 1
done
wait_ready "${TIER_CONTROL[@]}"

echo
echo "supervisors:"
scale dd-agentic-supervisor "${SUPERVISORS}"
wait_ready dd-agentic-supervisor

echo
echo "fleet up. Next:"
echo "  k8s/scripts/campaign-status.sh --namespace ${PROJECT}"
echo "  kubectl --namespace ${PROJECT} exec -it deployment/dd-agentic-conductor -- \\"
echo "    python -m reinvent4_agent.conductor --start --goal '...' \\"
echo "      --base-ttp brd4-bd1-lead-opt --concurrency ${SUPERVISORS} --no-scale"
