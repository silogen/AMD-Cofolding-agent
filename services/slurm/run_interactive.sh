#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Launch the complete Cofolding Agent stack on one Slurm GPU allocation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: no SLURM allocation detected." >&2
    echo "Launch this script through srun --pty on a GPU allocation." >&2
    exit 1
fi

ENV_FILE="${PROJECT_DIR}/services/env.local.sh"
if [ ! -f "${ENV_FILE}" ]; then
    ENV_FILE="${PROJECT_DIR}/services/env.sh"
fi
source "${ENV_FILE}"

echo "Agent configuration:"
echo "  env file: ${ENV_FILE}"
echo "  backend:  ${REINVENT4_AGENT_BACKEND}"
echo "  model:    ${REINVENT4_AGENT_MODEL}"
echo "  base URL: ${REINVENT4_AGENT_BASE_URL}"

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
export DATA_DIR
mkdir -p \
    "${DATA_DIR}/shared" \
    "${DATA_DIR}/jobs" \
    "${DATA_DIR}/logs" \
    "${DATA_DIR}/tb_logs" \
    "${DATA_DIR}/agent_logs" \
    "${DATA_DIR}/synthesis_cache"

restore_pinned_checkout() {
    local checkout="$1"
    local repository="$2"
    local ref_file="$3"
    local required_path="$4"
    local ref
    ref="$(tr -d '[:space:]' < "${ref_file}")"

    if [ ! -e "${checkout}/${required_path}" ]; then
        if [ -e "${checkout}" ]; then
            echo "ERROR: ${checkout} exists but is not a valid checkout." >&2
            exit 1
        fi
        echo "Restoring ${repository##*/} revision ${ref}..."
        git init -q "${checkout}"
        git -C "${checkout}" remote add origin "${repository}"
        git -C "${checkout}" fetch -q --depth 1 origin "${ref}"
        git -C "${checkout}" checkout -q --detach FETCH_HEAD
    fi

    local current
    current="$(git -C "${checkout}" rev-parse HEAD)"
    if [ "${current}" != "${ref}" ]; then
        echo "ERROR: ${checkout} is ${current}, expected ${ref}." >&2
        exit 1
    fi
}

restore_pinned_checkout \
    "${PROJECT_DIR}/REINVENT4" \
    "https://github.com/MolecularAI/REINVENT4.git" \
    "${PROJECT_DIR}/services/reinvent-runner/REINVENT4_COMMIT" \
    "reinvent"

if ! git -C "${PROJECT_DIR}/REINVENT4" diff --quiet -- \
    reinvent/runmodes/samplers/libinvent.py; then
    echo "ERROR: REINVENT4 checkout is locally modified." >&2
    exit 1
fi

restore_pinned_checkout \
    "${PROJECT_DIR}/boltz2" \
    "https://github.com/jwohlwend/boltz.git" \
    "${PROJECT_DIR}/services/boltz2/UPSTREAM_COMMIT" \
    "src/boltz"

COMPOSE_PROJECT_NAME="cofolding-${SLURM_JOB_ID}"
COMPOSE_ARGS=(
    -p "${COMPOSE_PROJECT_NAME}"
    -f "${PROJECT_DIR}/docker-compose.yml"
)

cleanup() {
    docker compose "${COMPOSE_ARGS[@]}" down --remove-orphans \
        >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "Building complete Cofolding Agent stack..."
docker compose "${COMPOSE_ARGS[@]}" build

echo "Starting all sidecars and interactive agent..."
docker compose "${COMPOSE_ARGS[@]}" run --rm reinvent4-mcp "$@"
