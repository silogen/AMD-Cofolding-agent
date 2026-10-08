#!/usr/bin/env bash
# Import locally built images into the kubelet containerd store on this node.
# Docker and kubelet do not share an image store, so a docker build on the
# node is not enough — kubelet will not see those tags until they are imported.
#
# Must run ON the target node (same host as kubelet). Defaults to the
# k3s/RKE2 kubelet socket; override with CONTAINERD_SOCK. Do not use
# Docker's /run/containerd/containerd.sock.
#
# Usage:
#   ./load-images.sh --node <kubernetes-node-name>
set -euo pipefail

TAG=local
NODE=""
NAMESPACE=""
IMAGES=(
    dd-agent
    boltz2-wrapper
    posebusters
    admet-predictor
    retrodmta-predictor
    interaction-profiler
    boltz-gateway
)
CONTAINERD_SOCK="${CONTAINERD_SOCK:-/run/k3s/containerd/containerd.sock}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --node)
            NODE="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${NODE}" ]]; then
    echo "usage: $0 --node <kubernetes-node-name> [--namespace <ns>] [--tag local]" >&2
    exit 1
fi

if [[ "$(hostname)" != "${NODE}" ]]; then
    echo "must run on ${NODE}; this host is $(hostname)" >&2
    exit 1
fi

if [[ ! -S "${CONTAINERD_SOCK}" ]]; then
    echo "missing kubelet containerd socket ${CONTAINERD_SOCK}" >&2
    echo "set CONTAINERD_SOCK if this node is not k3s/RKE2" >&2
    exit 1
fi

# --namespace is accepted for compatibility with apply.sh; imports are
# node-local into containerd namespace k8s.io, not a Kubernetes namespace.
if [[ -n "${NAMESPACE}" ]]; then
    echo "note: --namespace ${NAMESPACE} is unused for image import (node-local containerd)"
fi

CTR=(sudo ctr --address "${CONTAINERD_SOCK}" --namespace k8s.io)
# docker save tars are huge; keep them off a small root /tmp.
IMPORT_TMP="${IMPORT_TMP:-${HOME}/dd-agentic-import}"
mkdir --parents "${IMPORT_TMP}"
WORKDIR="$(mktemp --directory "${IMPORT_TMP}/dd-agentic-images.XXXXXX")"
cleanup() {
    rm -rf "${WORKDIR}"
}
trap cleanup EXIT

for name in "${IMAGES[@]}"; do
    if ! docker image inspect "${name}:${TAG}" >/dev/null 2>&1; then
        echo "missing local image ${name}:${TAG}; run build-images.sh first" >&2
        exit 1
    fi
done

echo "importing images into kubelet containerd (${CONTAINERD_SOCK}) on ${NODE}"
for name in "${IMAGES[@]}"; do
    ref="docker.io/library/${name}:${TAG}"
    tar_name="${name}.tar"
    echo "  saving ${name}:${TAG}"
    docker save --output "${WORKDIR}/${tar_name}" "${name}:${TAG}"
    # ctr import updates an existing image record, so mutable tags do not leave
    # stale content behind and running containers retain their current lease.
    echo "  importing ${ref} (replacing the tag if it exists)"
    "${CTR[@]}" images import "${WORKDIR}/${tar_name}"
    rm --force "${WORKDIR}/${tar_name}"
done

echo "loaded ${#IMAGES[@]} images onto ${NODE} (kubelet/containerd k8s.io)"
