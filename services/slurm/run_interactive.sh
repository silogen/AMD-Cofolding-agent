#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Run the agent interactively via Docker on a Slurm compute node.
#
# Usage from login node:
#   srun --reservation=MDL --partition=AIG_Models -w tw046 \
#        --gres=gpu:1 --cpus-per-task=8 --mem=50G --time=2:00:00 \
#        --pty bash services/slurm/run_interactive.sh
#
# Or if you already have an allocation (salloc / interactive shell):
#   bash services/slurm/run_interactive.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: no SLURM allocation detected." >&2
    echo "Start this script with the one-GPU srun command shown above." >&2
    exit 1
fi

# Source environment (use env.local.sh if it exists, otherwise env.sh)
ENV_FILE="${PROJECT_DIR}/services/env.local.sh"
if [ ! -f "${ENV_FILE}" ]; then
    ENV_FILE="${PROJECT_DIR}/services/env.sh"
fi
echo "Loading env from: ${ENV_FILE}"
source "${ENV_FILE}"

IMAGE="${REINVENT4_AGENT_IMAGE:-dd-agent}"
LOG_DIR="${REINVENT4_AGENT_LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

# The Dockerfile installs the official REINVENT4 checkout into the ROCm image.
# Restore the checkout once when this source bundle does not include it.
if [ ! -d "${PROJECT_DIR}/REINVENT4/reinvent" ]; then
    echo "Restoring official REINVENT4 source checkout..."
    git clone --depth 1 \
        https://github.com/MolecularAI/REINVENT4.git \
        "${PROJECT_DIR}/REINVENT4"
fi

# Build on the allocated GPU node, never on the login/head node.
docker build -f "${PROJECT_DIR}/services/dd-agent/Dockerfile" \
    -t "${IMAGE}" "${PROJECT_DIR}"
echo ""

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
TTP_DIR="${TTP_DIR:-${HOME}/.reinvent4_mcp/ttps}"
mkdir -p \
    "${DATA_DIR}/shared" \
    "${DATA_DIR}/jobs" \
    "${DATA_DIR}/logs" \
    "${DATA_DIR}/tb_logs" \
    "${DATA_DIR}/agent_logs" \
    "${TTP_DIR}"

echo "=== Drug Discovery Agent ==="
echo "Node     : $(hostname)"
echo "Image    : ${IMAGE}"
echo "Backend  : ${REINVENT4_AGENT_BACKEND}"
echo "Model    : ${REINVENT4_AGENT_MODEL}"
echo "Endpoint : ${REINVENT4_AGENT_BASE_URL}"
echo "Thinking : ${REINVENT4_AGENT_THINKING_BUDGET:-off}"
echo "Logs     : ${LOG_DIR}"
echo "GPU      : ${REINVENT_GPU:-0}"
echo ""

GATEWAY_HOST_ARGS=()
if [[ "${REINVENT4_AGENT_BASE_URL}" =~ llm-api\.amd\.com:[0-9]+ ]]; then
    # With --network host the container can reach the SSH reverse tunnel
    # directly on host loopback; preserve the TLS hostname via /etc/hosts.
    GATEWAY_HOST_ARGS=(--add-host llm-api.amd.com:127.0.0.1)
fi

# Fail before opening the agent UI unless ROCm PyTorch can see the allocated GPU.
docker run --rm \
    "${GATEWAY_HOST_ARGS[@]}" \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined \
    -e AMD_VISIBLE_DEVICES="${REINVENT_GPU:-0}" \
    -e HIP_VISIBLE_DEVICES="${REINVENT_GPU:-0}" \
    --entrypoint python \
    "${IMAGE}" \
    -c 'import torch; assert torch.cuda.is_available(), "ROCm GPU unavailable"; print("REINVENT GPU OK:", torch.cuda.get_device_name(0))'

exec docker run -it --rm \
    "${GATEWAY_HOST_ARGS[@]}" \
    --device=/dev/kfd --device=/dev/dri \
    --network host --ipc host --group-add video --group-add render \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --shm-size 32G \
    -e LLM_GATEWAY_KEY="${LLM_GATEWAY_KEY}" \
    -e REINVENT4_AGENT_BACKEND="${REINVENT4_AGENT_BACKEND}" \
    -e REINVENT4_AGENT_MODEL="${REINVENT4_AGENT_MODEL}" \
    -e REINVENT4_AGENT_BASE_URL="${REINVENT4_AGENT_BASE_URL}" \
    -e REINVENT4_AGENT_NO_VERIFY_SSL="${REINVENT4_AGENT_NO_VERIFY_SSL}" \
    -e REINVENT4_AGENT_MULTI_AGENT="${REINVENT4_AGENT_MULTI_AGENT:-true}" \
    -e AMD_VISIBLE_DEVICES="${REINVENT_GPU:-0}" \
    -e HIP_VISIBLE_DEVICES="${REINVENT_GPU:-0}" \
    -e SYNTHESIS_SERVICE_URL="${SYNTHESIS_SERVICE_URL:-http://127.0.0.1:8086}" \
    -e INTERACTION_PROFILER_URL="${INTERACTION_PROFILER_URL:-http://127.0.0.1:8083}" \
    -e SP_SERVICE_URL="${SP_SERVICE_URL:-http://127.0.0.1:8087}" \
    -e BOLTZ_SERVICE_URL="${BOLTZ_SERVICE_URL:-http://127.0.0.1:8091}" \
    -e BOLTZ_SHARED_DIR=/shared \
    -e IP_SHARED_DIR=/shared \
    -v "${DATA_DIR}/shared":/shared \
    -v "${TTP_DIR}":/root/.reinvent4_mcp/ttps \
    -v "${DATA_DIR}/jobs":/data/reinvent4_mcp/jobs \
    -v "${DATA_DIR}/logs":/data/reinvent4_mcp/logs \
    -v "${DATA_DIR}/tb_logs":/data/reinvent4_mcp/tb_logs \
    -v "${DATA_DIR}/agent_logs":/logs \
    "${IMAGE}" "$@"
