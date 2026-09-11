#!/usr/bin/env bash
#SBATCH --job-name=dd-agent
#SBATCH --output=dd-agent-%j.out
#SBATCH --error=dd-agent-%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
# ──────────────────────────────────────────────────────────────────────
# Batch-mode agent job.  Feed input from a file or heredoc.
#
# Usage:
#   sbatch services/slurm/agent.sbatch
#
# To provide prompts non-interactively, pipe input:
#   echo "Score aspirin (CC(=O)OC1=CC=CC=C1C(=O)O) for drug-likeness" \
#     | sbatch services/slurm/agent.sbatch
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source environment
ENV_FILE="${SCRIPT_DIR}/env.local.sh"
if [ ! -f "${ENV_FILE}" ]; then
    ENV_FILE="${SCRIPT_DIR}/env.sh"
fi
source "${ENV_FILE}"

# Activate virtualenv
VENV_DIR="${REINVENT4_VENV:-${PROJECT_DIR}/.venv}"
source "${VENV_DIR}/bin/activate"

export REINVENT4_ROOT="${REINVENT4_ROOT:-${PROJECT_DIR}/REINVENT4}"
export REINVENT4_MCP_CONFIG="${REINVENT4_MCP_CONFIG:-${PROJECT_DIR}/reinvent4_mcp_config.yaml}"

echo "=== Drug Discovery Agent (batch) ==="
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : $(hostname)"
echo "Backend  : ${REINVENT4_AGENT_BACKEND}"
echo "Model    : ${REINVENT4_AGENT_MODEL}"
echo ""

dd-agent "$@"
