#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ──────────────────────────────────────────────────────────────────────
# Build the dd_agentic Apptainer/Singularity images (no Docker daemon).
#
# Usage:
#   bash services/singularity/build.sh <amd|nvidia> [service ...] [--dry-run]
#
# Examples:
#   bash services/singularity/build.sh nvidia            # build all images
#   bash services/singularity/build.sh amd boltz         # just boltz (AMD)
#   bash services/singularity/build.sh amd --dry-run     # print commands only
#
# Services:
#   GPU / vendor-specific : dd-agent boltz admet-predictor
#   CPU / vendor-neutral  : posebusters retrodmta-predictor interaction-profiler
#
# Vendor-specific images are tagged   <service>-<vendor>.sif
# Vendor-neutral images are tagged     <service>.sif
#
# Must be run from anywhere; the build CONTEXT is forced to the dd_agentic/
# root so the .def %files sections resolve (REINVENT4/, models/, packages, ...).
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"   # dd_agentic/

# shellcheck source=vendors.sh
source "${SCRIPT_DIR}/vendors.sh"

GPU_SERVICES="dd-agent boltz admet-predictor"
CPU_SERVICES="posebusters retrodmta-predictor interaction-profiler"
ALL_SERVICES="${GPU_SERVICES} ${CPU_SERVICES}"

usage() {
    sed -n '5,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

# --- Parse args -------------------------------------------------------
VENDOR=""
DRY_RUN=0
SERVICES=""
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage 0 ;;
        amd|nvidia)
            if [ -z "${VENDOR}" ]; then VENDOR="${arg}"; else SERVICES="${SERVICES} ${arg}"; fi ;;
        *) SERVICES="${SERVICES} ${arg}" ;;
    esac
done

if [ -z "${VENDOR}" ]; then
    echo "error: first argument must be a vendor (amd | nvidia)" >&2
    usage 1
fi

vendor_matrix "${VENDOR}"

# Default to all services when none specified (trim surrounding whitespace).
SERVICES="$(printf '%s' "${SERVICES}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
if [ -z "${SERVICES}" ]; then
    SERVICES="${ALL_SERVICES}"
fi

# --- Resolve the container runtime (apptainer or singularity) ---------
# HPC clusters often expose it as a module (`module load apptainer`) or as the
# legacy `singularity` command. Override explicitly with APPTAINER_BIN=/path.
APPTAINER="${APPTAINER_BIN:-}"
if [ -z "${APPTAINER}" ]; then
    if command -v apptainer >/dev/null 2>&1; then
        APPTAINER=apptainer
    elif command -v singularity >/dev/null 2>&1; then
        APPTAINER=singularity
    else
        APPTAINER=apptainer   # keep --dry-run output readable
        if [ "${DRY_RUN}" -eq 0 ]; then
            echo "error: neither 'apptainer' nor 'singularity' found on PATH." >&2
            echo "  On HPC try:  module load apptainer   (or: module load singularity)" >&2
            echo "  Or set:      APPTAINER_BIN=/path/to/apptainer" >&2
            exit 127
        fi
    fi
fi

# --- Helpers ----------------------------------------------------------
is_in() { case " $1 " in *" $2 "*) return 0 ;; *) return 1 ;; esac; }

run() {
    echo "+ $*"
    if [ "${DRY_RUN}" -eq 0 ]; then
        "$@"
    fi
}

build_one() {
    local svc="$1"
    local def="${SCRIPT_DIR}/${svc}.def"
    local out args=()

    if [ ! -f "${def}" ]; then
        echo "error: no def file for service '${svc}' (${def})" >&2
        return 1
    fi

    case "${svc}" in
        dd-agent|boltz)
            out="${SCRIPT_DIR}/${svc}-${VENDOR}.sif"
            args=(--build-arg "BASE_IMAGE=${BASE_IMAGE}") ;;
        admet-predictor)
            out="${SCRIPT_DIR}/${svc}-${VENDOR}.sif"
            args=(--build-arg "TORCH_INDEX_URL=${TORCH_INDEX_URL}" \
                  --build-arg "PRED_DEVICE=${PRED_DEVICE}") ;;
        *)
            # vendor-neutral CPU services
            out="${SCRIPT_DIR}/${svc}.sif"
            args=() ;;
    esac

    echo "=== ${svc} (${VENDOR}) -> ${out} ==="
    # Build context = dd_agentic/ root so %files paths resolve.
    # ${args[@]+...} guards empty-array expansion under `set -u` (bash 3.2).
    ( cd "${ROOT_DIR}" && run "${APPTAINER}" build --fakeroot ${args[@]+"${args[@]}"} "${out}" "${def}" )
}

echo "Vendor      : ${VENDOR}"
echo "Base image  : ${BASE_IMAGE}"
echo "Torch index : ${TORCH_INDEX_URL} (admet)"
echo "Pred device : ${PRED_DEVICE} (admet)"
echo "Services    : ${SERVICES}"
[ "${DRY_RUN}" -eq 1 ] && echo "(dry-run: printing commands only)"
echo ""

for svc in ${SERVICES}; do
    if ! is_in "${ALL_SERVICES}" "${svc}"; then
        echo "error: unknown service '${svc}' (valid: ${ALL_SERVICES})" >&2
        exit 1
    fi
    build_one "${svc}"
done

echo ""
echo "Done."
