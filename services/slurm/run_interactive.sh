#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ──────────────────────────────────────────────────────────────────────
# Run the agent interactively via Docker on a Slurm compute node.
#
# Usage from login node:
#   srun --partition=batch --gres=gpu:2 --cpus-per-task=4 --mem=16G --time=4:00:00 \
#        --pty bash services/slurm/run_interactive.sh
#
# Or if you already have an allocation (salloc / interactive shell):
#   bash services/slurm/run_interactive.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source environment (use env.local.sh if it exists, otherwise env.sh)
ENV_FILE="${SCRIPT_DIR}/env.local.sh"
if [ ! -f "${ENV_FILE}" ]; then
    ENV_FILE="${SCRIPT_DIR}/env.sh"
fi
echo "Loading env from: ${ENV_FILE}"
source "${ENV_FILE}"

IMAGE="${REINVENT4_AGENT_IMAGE:-dd-agent}"
LOG_DIR="${REINVENT4_AGENT_LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

# Build the image on this node if it doesn't exist

docker build -f "${PROJECT_DIR}/services/dd-agent/Dockerfile" \
    -t "${IMAGE}" "${PROJECT_DIR}"
echo ""

echo "=== Drug Discovery Agent ==="
echo "Node     : $(hostname)"
echo "Image    : ${IMAGE}"
echo "Backend  : ${REINVENT4_AGENT_BACKEND}"
echo "Model    : ${REINVENT4_AGENT_MODEL}"
echo "Endpoint : ${REINVENT4_AGENT_BASE_URL}"
echo "Thinking : ${REINVENT4_AGENT_THINKING_BUDGET:-off}"
echo "Logs     : ${LOG_DIR}"
echo ""

exec docker run -it --rm \
    --device=/dev/kfd --device=/dev/dri \
    --network host --ipc host --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --shm-size 128G \
    -e LLM_GATEWAY_KEY="${LLM_GATEWAY_KEY}" \
    -v "${LOG_DIR}":/logs \
    "${IMAGE}" "$@"
