#!/usr/bin/env bash
# Apply the k8s manifests into an existing project namespace and pin every
# Deployment to one GPU node (the one that already has the selected images).
# Does not pin AIMService/dd-agentic-aim — that scheduler places the predictor
# where acceleratorCount GPUs are free. Re-run with a different --node to move
# the workloads.
set -euo pipefail

# The label kustomization.yaml stamps on every resource; used here to restart
# and wait for the whole set without naming each Deployment.
PART_OF="app.kubernetes.io/part-of=dd-agentic"

PROJECT=""
NODE=""
TAG="local"
K8S_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFESTS_DIR="${K8S_DIR}/manifests"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --namespace)
            PROJECT="$2"
            shift 2
            ;;
        --node)
            NODE="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${PROJECT}" || -z "${NODE}" ]]; then
    echo "usage: $0 --namespace <project-namespace> --node <dd-agentic-gpu-node> [--tag local]" >&2
    echo "  --node pins every dd-agentic Deployment (not AIM). See k8s/README.md." >&2
    exit 1
fi

if [[ ! "${PROJECT}" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
    echo "invalid Kubernetes namespace: ${PROJECT}" >&2
    exit 1
fi

if [[ ! "${NODE}" =~ ^[A-Za-z0-9]([-A-Za-z0-9.]*[A-Za-z0-9])?$ ]]; then
    echo "invalid Kubernetes node name: ${NODE}" >&2
    exit 1
fi

if [[ ! "${TAG}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
    echo "invalid image tag: ${TAG}" >&2
    exit 1
fi

if ! kubectl get namespace "${PROJECT}" >/dev/null; then
    echo "namespace ${PROJECT} does not exist; pass --namespace with an existing Kubernetes namespace" >&2
    exit 1
fi

if ! kubectl get node "${NODE}" >/dev/null; then
    echo "node ${NODE} not found" >&2
    exit 1
fi

OVERLAY="$(mktemp --directory --tmpdir dd-agentic-overlay.XXXXXX)"
cleanup() { rm -rf "${OVERLAY}"; }
trap cleanup EXIT

# Keep every resource beneath the temporary kustomization root so the default
# load restrictor remains enabled.
cp --archive "${MANIFESTS_DIR}/." "${OVERLAY}/"
sed --in-place "s/newTag: local/newTag: ${TAG}/g" "${OVERLAY}/kustomization.yaml"

# Untargeted by name, so it covers every Deployment the overlay carries. Two
# reasons, both node-local: the *:local images live in one node's kubelet
# store, and the dd-agentic-data claim is ReadWriteOnce on a single-node
# cluster — readable and writable by any number of pods, but only from the
# node that mounts it. So boltz, posebusters, and the interaction profiler
# have to sit beside the agent to resolve the structure_paths they hand each
# other. Kyverno mutates the claim to RWX on multi-node clusters, where those
# three could spread; the image store still keeps them here.
cat >> "${OVERLAY}/kustomization.yaml" <<EOF
namespace: ${PROJECT}
patches:
  - target:
      kind: Deployment
    patch: |-
      - op: add
        path: /spec/template/spec/nodeSelector
        value:
          kubernetes.io/hostname: "${NODE}"
EOF

echo "applying dd-agentic:${TAG} into namespace ${PROJECT}; deployments pinned to ${NODE} (AIM unpinned)"
kubectl kustomize "${OVERLAY}" |
    kubectl apply --server-side --force-conflicts --filename -

echo "restarting the dd-agentic deployments to load tag ${TAG}"
kubectl --namespace "${PROJECT}" rollout restart deployment --selector "${PART_OF}"

DEPLOYMENTS="$(
    kubectl --namespace "${PROJECT}" get deployments \
        --selector "${PART_OF}" --output name
)"
if [[ -z "${DEPLOYMENTS}" ]]; then
    echo "no deployments matched ${PART_OF} in ${PROJECT}; nothing was applied" >&2
    exit 1
fi

# Names cannot contain whitespace, so splitting on newlines is safe.
while read -r deployment; do
    echo "waiting for ${deployment}"
    kubectl --namespace "${PROJECT}" rollout status "${deployment}" --timeout=20m
done <<< "${DEPLOYMENTS}"
echo "watch AIM with: kubectl --namespace ${PROJECT} get aimservice dd-agentic-aim --watch"
