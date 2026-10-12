# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Extends the REINVENT4 ROCm image with the MCP server + agent CLI.
#
# Build from the repo root:
#   docker build -f services/dd-agent/Dockerfile -t dd-agent .
#
# Run interactively on a GPU node:
#   docker run -it \
#     --device=/dev/kfd --device=/dev/dri/renderD<ID> \
#     --network host --ipc host --group-add video \
#     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
#     --shm-size 128G \
#     -e LLM_GATEWAY_KEY="$LLM_GATEWAY_KEY" \
#     -v $(pwd)/logs:/logs \
#     dd-agent

# REINVENT4 (this checkout) requires Python >=3.11, so we build on the py3.12
# ROCm base (same family the boltz service uses). Building on py3.10 makes
# `pip install .` silently skip via requires-python, and install.py's sp.run()
# has no check=True, so the failure is swallowed and the image ends up WITHOUT
# rdkit/xxhash/numpy/etc. -> endless "No module named X" at runtime.
FROM rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# --- Install REINVENT4 (torch comes from the base image; strip the pins) ---
COPY REINVENT4/ /app/REINVENT4/
WORKDIR /app/REINVENT4

# `-d none` = core deps only (rdkit, xxhash, numpy, pandas, pydantic, ...),
# skipping openeye (needs license/index) and isim (git). The final import check
# turns any silent install failure into a hard build failure.
RUN sed -i.bak '/torch==/d' pyproject.toml && \
    sed -i.bak '/torchvision/d' pyproject.toml && \
    python install.py cpu -d none && \
    pip install --no-cache-dir --upgrade "numpy>=2,<3" "scipy>=1.13" && \
    python -c "import numpy, scipy; from scipy.stats import gaussian_kde; import rdkit, xxhash, tomllib, pandas, pydantic; print('REINVENT4 deps OK; numpy', numpy.__version__, 'scipy', scipy.__version__)"

# --- Download prior models from Zenodo ---
# Fail the build (not silently at runtime) if a prior is missing, an HTTP
# error page, or truncated. Checksums are the MD5s published by Zenodo for
# record 15641297; verifying them also catches partial/corrupt downloads.
RUN mkdir -p /app/REINVENT4/priors && \
    wget --tries=3 -q -O /app/REINVENT4/priors/reinvent.prior \
      "https://zenodo.org/records/15641297/files/reinvent.prior?download=1" && \
    wget --tries=3 -q -O /app/REINVENT4/priors/mol2mol_medium_similarity.prior \
      "https://zenodo.org/records/15641297/files/mol2mol_medium_similarity.prior?download=1" && \
    echo "f268eb072f4fca69ca9434768d3cd461  /app/REINVENT4/priors/reinvent.prior" | md5sum -c - && \
    echo "78b75e478bfd7dcccee8ece010bc0b07  /app/REINVENT4/priors/mol2mol_medium_similarity.prior" | md5sum -c -

# --- Install MCP server + agent on top ---
WORKDIR /app
COPY reinvent4_mcp/ /app/reinvent4_mcp/
COPY reinvent4_agent/ /app/reinvent4_agent/
COPY reinvent4_mcp_config.yaml /app/reinvent4_mcp_config.yaml

# Install the MCP server + agent, then pin numpy/scipy LAST so their resolution
# can't be clobbered by a transitive dependency (e.g. an older numpy<2 pin).
# Verify the exact import REINVENT4 needs at RL startup so any ABI mismatch
# fails the build, not the run.
RUN pip install --no-cache-dir /app/reinvent4_mcp /app/reinvent4_agent && \
    pip install --no-cache-dir --upgrade "numpy>=2,<3" "scipy>=1.13" && \
    python -c "import numpy, scipy; from scipy.stats import gaussian_kde; import reinvent; \
import reinvent4_mcp.utils as u; u.ensure_reinvent4_importable(); \
from reinvent.scoring.importer import get_registry; r = get_registry(); \
assert 'externalrestscorer' in r, sorted(r); \
print('FINAL env OK; externalrestscorer registered; numpy', numpy.__version__, 'scipy', scipy.__version__)"

# --- Environment ---
ENV REINVENT4_ROOT=/app/REINVENT4
ENV REINVENT4_MCP_CONFIG=/app/reinvent4_mcp_config.yaml
# Custom REINVENT4 plugins (reinvent_plugins/components namespace layout) on the
# path so ExternalRESTScorer is discoverable natively in the server AND the RL
# worker subprocess. Points at the source copy (always present); the installed
# copy under site-packages/reinvent4_mcp/_plugins is added by the code too.
ENV PYTHONPATH=/app/reinvent4_mcp/_plugins:${PYTHONPATH}

# Agent defaults. No endpoint is baked in: pass REINVENT4_AGENT_BASE_URL at
# runtime (-e / compose) to target a gateway, or omit it for the SDK default.
ENV REINVENT4_AGENT_BACKEND=anthropic
ENV REINVENT4_AGENT_MODEL=claude-sonnet-4.6
ENV REINVENT4_AGENT_NO_VERIFY_SSL=true
ENV REINVENT4_AGENT_THINKING_BUDGET=1024
ENV REINVENT4_AGENT_LOG_DIR=/logs
# LLM_GATEWAY_KEY must be provided at runtime via -e

# Entrypoint wrapper timestamps captured logs (file / docker logs) while leaving
# interactive TTY sessions untouched. Copied late so it doesn't bust the cache
# of the heavy install layers above.
COPY services/dd-agent/entrypoint.sh /usr/local/bin/dd-agent-entrypoint.sh
RUN chmod +x /usr/local/bin/dd-agent-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/dd-agent-entrypoint.sh"]
CMD ["dd-agent"]
