#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# One-time setup: create a virtualenv and install REINVENT4 (with ROCm
# PyTorch) + dd-mcp + dd-agent.
#
# Usage:
#   cd /path/to/dd_agentic
#   bash services/slurm/setup.sh
#
# REINVENT4's pyproject.toml pins CUDA torch, so we install ROCm torch
# separately and then install REINVENT4 with --no-deps to avoid
# pulling the wrong torch.  This mirrors the approach in
# reinvent4_amd_helpers/install_reinvent.sh.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV_DIR="${REINVENT4_VENV:-${PROJECT_DIR}/.venv}"
ROCM_TORCH_INDEX="https://download.pytorch.org/whl/rocm5.7"

echo "=== Drug Discovery Agent Slurm Setup ==="
echo "Project dir : ${PROJECT_DIR}"
echo "Venv dir    : ${VENV_DIR}"
echo ""

# Create virtualenv if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtualenv..."
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

# 1. Install ROCm PyTorch (before REINVENT4 so it doesn't pull CUDA torch)
echo "[1/4] Installing ROCm PyTorch..."
pip install -q \
    torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url "${ROCM_TORCH_INDEX}"

# 2. Install REINVENT4 deps then REINVENT4 itself (--no-deps to keep ROCm torch)
echo "[2/4] Installing REINVENT4..."
if [ -d "${PROJECT_DIR}/REINVENT4" ]; then
    if [ -f "${PROJECT_DIR}/REINVENT4/requirements-linux-64.lock" ]; then
        pip install -q -r "${PROJECT_DIR}/REINVENT4/requirements-linux-64.lock"
    fi
    pip install -q --no-deps "${PROJECT_DIR}/REINVENT4"
else
    echo "WARNING: REINVENT4/ directory not found at ${PROJECT_DIR}/REINVENT4"
    echo "  Scoring tools will not work until REINVENT4 is installed."
fi

# 3. Install MCP server (the agent spawns it as a subprocess)
echo "[3/4] Installing reinvent4_mcp..."
pip install -q "${PROJECT_DIR}/reinvent4_mcp"

# 4. Install the agent CLI
echo "[4/4] Installing reinvent4_agent..."
pip install -q "${PROJECT_DIR}/reinvent4_agent"

echo ""
echo "=== Setup complete ==="
echo "Installed commands:"
which dd-mcp
which dd-agent
echo ""
echo "PyTorch: $(python3 -c 'import torch; print(f"{torch.__version__} (ROCm: {torch.version.hip})")' 2>/dev/null || echo 'not found')"
echo ""
echo "Next steps:"
echo "  1. Copy services/env.local.sh.example to services/env.local.sh and fill in LLM_GATEWAY_KEY"
echo "  2. Interactive: srun --pty bash services/slurm/run_interactive.sh"
echo "  3. Batch:       sbatch services/slurm/agent.sbatch"
