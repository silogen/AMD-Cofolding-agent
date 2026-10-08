# reinvent-runner: REINVENT4 job lifecycle and scoring behind REST.
#
# This is now the image that owns REINVENT4 and the Zenodo priors -- the heavy
# half that used to be baked into dd-agent, where every agent replica paid for
# it and pinned a GPU it mostly did not use. The agent is CPU-only and reaches
# this service over HTTP.
#
# It installs reinvent4_mcp (for the JobManager, the in-process Scorer and the
# registries) but *not* reinvent4_agent: nothing here talks to an LLM.
#
# Build from the repo root, which is where REINVENT4/ and the packages live:
#   docker build -f services/reinvent-runner/Dockerfile -t reinvent-runner:local .

# REINVENT4 requires Python >=3.11, so build on the py3.12 ROCm base (the same
# family the boltz service uses). Building on py3.10 makes `pip install .`
# silently skip via requires-python, and install.py's sp.run() has no
# check=True, so the failure is swallowed and the image ends up WITHOUT
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

# --- Install the MCP server package + the REST wrapper ---
WORKDIR /app
COPY reinvent4_mcp/ /app/reinvent4_mcp/
COPY reinvent4_mcp_config.yaml /app/reinvent4_mcp_config.yaml

# Pin numpy/scipy LAST so their resolution can't be clobbered by a transitive
# dependency (e.g. an older numpy<2 pin). Verify the exact import REINVENT4
# needs at RL startup, and that the custom scorer is registered, so any ABI
# mismatch or plugin-path mistake fails the build rather than the first job.
RUN pip install --no-cache-dir /app/reinvent4_mcp "fastapi>=0.110" "uvicorn[standard]>=0.29" && \
    pip install --no-cache-dir --upgrade "numpy>=2,<3" "scipy>=1.13" && \
    python -c "import numpy, scipy; from scipy.stats import gaussian_kde; import reinvent; \
import reinvent4_mcp.utils as u; u.ensure_reinvent4_importable(); \
from reinvent.scoring.importer import get_registry; r = get_registry(); \
assert 'externalrestscorer' in r, sorted(r); \
print('runner image OK; externalrestscorer registered; numpy', numpy.__version__, 'scipy', scipy.__version__)"

COPY services/reinvent-runner/app.py /app/runner/app.py
WORKDIR /app/runner

# --- Environment ---
ENV REINVENT4_ROOT=/app/REINVENT4
ENV REINVENT4_MCP_CONFIG=/app/reinvent4_mcp_config.yaml
# Custom REINVENT4 plugins (reinvent_plugins/components namespace layout) on the
# path so ExternalRESTScorer is discoverable natively in this process AND in the
# RL worker subprocess it spawns.
ENV PYTHONPATH=/app/reinvent4_mcp/_plugins:${PYTHONPATH}
# This process *is* the runner. It reads the same config file as the agent, so
# without this it would honour runner.enabled and forward every job to itself.
ENV REINVENT_RUNNER_SELF=1

EXPOSE 8086

# One worker, always. The JobManager holds the job table and the concurrency
# slots in process memory, so a second worker would keep its own copy and
# double the effective max_concurrent_jobs.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8086", "--workers", "1"]
