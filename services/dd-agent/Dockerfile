# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The agent and the MCP server. CPU only, no REINVENT4, no priors.
#
# This image used to extend the REINVENT4 ROCm image and bake in REINVENT4 plus
# the Zenodo priors: ~11 GiB that every agent replica had to pull, and a GPU it
# had to pin whether or not it was running a job. All of that now lives in
# reinvent-runner, and the agent reaches it over HTTP (runner.enabled in the
# process config), so agent count and RL capacity scale independently.
#
# Neither reinvent4_mcp nor reinvent4_agent depends on torch or REINVENT4 --
# they need mcp, pydantic, numpy, pandas, requests and the LLM clients -- so the
# base here is plain python.
#
# Build from the repo root:
#   docker build -f services/dd-agent/Dockerfile -t dd-agent .
#
# Run interactively:
#   docker run -it -e LLM_GATEWAY_KEY="$LLM_GATEWAY_KEY" -v $(pwd)/logs:/logs dd-agent

FROM python:3.12-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /app
COPY reinvent4_mcp/ /app/reinvent4_mcp/
COPY reinvent4_agent/ /app/reinvent4_agent/
COPY reinvent4_mcp_config.yaml /app/reinvent4_mcp_config.yaml

# rdkit is not a declared dependency of either package: it is used in one
# guarded place, to validate Mol2Mol seed SMILES before a job is dispatched.
# Worth ~120 MB here against the 11 GiB removed, since without it that check
# degrades to a warning and a bad seed only fails later, inside the RL run.
RUN pip install --no-cache-dir /app/reinvent4_mcp /app/reinvent4_agent "rdkit>=2024.3.1"

# The inverse of the check this image used to make. REINVENT4 being *absent* is
# the point of the split, so a torch or REINVENT4 dependency creeping back in
# should fail the build rather than quietly restoring an 11 GiB image.
RUN python -c "import importlib.util as u; \
assert u.find_spec('reinvent') is None, 'REINVENT4 is back in the agent image; it belongs in reinvent-runner'; \
assert u.find_spec('torch') is None, 'torch is back in the agent image'; \
import reinvent4_mcp.server, reinvent4_agent.cli, rdkit; \
from reinvent4_mcp.config import ProcessConfig; ProcessConfig(); \
print('agent image OK: MCP server and agent import, REINVENT4 absent')"

# --- Environment ---
ENV REINVENT4_MCP_CONFIG=/app/reinvent4_mcp_config.yaml
# No REINVENT4_ROOT and no reinvent_plugins on the path: both exist for
# REINVENT's own scorer, which runs in the RL subprocess on the runner.

# Agent defaults. No endpoint is baked in: pass REINVENT4_AGENT_BASE_URL at
# runtime (-e / compose) to target a gateway, or omit it for the SDK default.
ENV REINVENT4_AGENT_BACKEND=anthropic
ENV REINVENT4_AGENT_MODEL=claude-sonnet-4.6
ENV REINVENT4_AGENT_NO_VERIFY_SSL=true
ENV REINVENT4_AGENT_THINKING_BUDGET=1024
ENV REINVENT4_AGENT_LOG_DIR=/logs
# LLM_GATEWAY_KEY must be provided at runtime via -e

# Entrypoint wrapper timestamps captured logs (file / docker logs) while leaving
# interactive TTY sessions untouched.
COPY services/dd-agent/entrypoint.sh /usr/local/bin/dd-agent-entrypoint.sh
RUN chmod +x /usr/local/bin/dd-agent-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/dd-agent-entrypoint.sh"]
CMD ["dd-agent"]
