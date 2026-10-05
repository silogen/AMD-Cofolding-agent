#!/usr/bin/env bash
# Apply the k8s manifests into an existing project namespace and pin
# deployment/dd-agentic to a GPU node (the one that already has the selected
# images). Does not pin AIMService/dd-agentic-aim — that scheduler places the
# predictor where acceleratorCount GPUs are free. Re-run with a different
# --node to move the agent pod (strategic-merge nodeSelector).
set -euo pipefail

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
    echo "  --node pins deployment/dd-agentic only (not AIM). See k8s/README.md." >&2
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

cat >> "${OVERLAY}/kustomization.yaml" <<EOF
namespace: ${PROJECT}
patches:
  - target:
      kind: Deployment
      name: dd-agentic
    patch: |-
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: dd-agentic
      spec:
        template:
          spec:
            nodeSelector:
              kubernetes.io/hostname: "${NODE}"
EOF

echo "applying dd-agentic:${TAG} into namespace ${PROJECT}; deployment pinned to ${NODE} (AIM unpinned)"
kubectl kustomize "${OVERLAY}" |
    kubectl apply --server-side --force-conflicts --filename -

echo "restarting deployment/dd-agentic to load tag ${TAG}"
kubectl --namespace "${PROJECT}" rollout restart deployment/dd-agentic
echo "waiting for deployment/dd-agentic"
kubectl --namespace "${PROJECT}" rollout status deployment/dd-agentic --timeout=20m
echo "watch AIM with: kubectl --namespace ${PROJECT} get aimservice dd-agentic-aim --watch"
