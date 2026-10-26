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

# Apply the one audited compatibility patch to a pinned, otherwise-pristine
# REINVENT4 checkout. The source itself is restored by the launcher and is not
# vendored in this repository.
RUN apt-get update && apt-get install -y --no-install-recommends patch && \
    rm -rf /var/lib/apt/lists/*

COPY REINVENT4/ /app/REINVENT4/
COPY services/reinvent-runner/REINVENT4_COMMIT /app/REINVENT4_COMMIT
COPY services/reinvent-runner/patches/reinvent4-libinvent-transformer-rgroups.patch \
    /tmp/reinvent4-libinvent-transformer-rgroups.patch
WORKDIR /app/REINVENT4
RUN patch --batch --forward --dry-run -p1 \
      < /tmp/reinvent4-libinvent-transformer-rgroups.patch && \
    patch --batch --forward -p1 \
      < /tmp/reinvent4-libinvent-transformer-rgroups.patch && \
    rm /tmp/reinvent4-libinvent-transformer-rgroups.patch

# --- Install REINVENT4 (torch comes from the base image; strip the pins) ---
# Chemprop v2 is the open scoring plugin supported on Python 3.12. Chemprop v1
# cannot coexist with v2 and remains intentionally unavailable; OpenEye still
# requires a licensed index. The final import check turns any silent install
# failure into a hard build failure.
RUN sed -i.bak '/torch==/d' pyproject.toml && \
    sed -i.bak '/torchvision/d' pyproject.toml && \
    python install.py cpu -d chemprop2 && \
    pip install --no-cache-dir --upgrade "numpy>=2,<3" "scipy>=1.13" && \
    python -c "import numpy, scipy, chemprop; from scipy.stats import gaussian_kde; import rdkit, xxhash, tomllib, pandas, pydantic; assert chemprop.__version__.split('.')[0] == '2'; print('REINVENT4 deps OK; chemprop', chemprop.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"

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
    echo "78b75e478bfd7dcccee8ece010bc0b07  /app/REINVENT4/priors/mol2mol_medium_similarity.prior" | md5sum -c - && \
    wget --tries=3 -q -O /app/REINVENT4/priors/libinvent.prior \
      "https://zenodo.org/records/15641297/files/libinvent.prior?download=1" && \
    wget --tries=3 -q -O /app/REINVENT4/priors/linkinvent.prior \
      "https://zenodo.org/records/15641297/files/linkinvent.prior?download=1" && \
    echo "fea30c293909216fe183bb0d1b7ca432  /app/REINVENT4/priors/libinvent.prior" | md5sum -c - && \
    echo "65af13963aa2bf80fc941c0400288c8e  /app/REINVENT4/priors/linkinvent.prior" | md5sum -c - && \
    echo "All 4 RNN priors downloaded and verified OK" && \
    wget --tries=3 -q -O /app/REINVENT4/priors/libinvent_transformer_pubchem.prior \
      "https://zenodo.org/records/15641297/files/libinvent_transformer_pubchem.prior?download=1" && \
    wget --tries=3 -q -O /app/REINVENT4/priors/linkinvent_transformer_pubchem.prior \
      "https://zenodo.org/records/15641297/files/linkinvent_transformer_pubchem.prior?download=1" && \
    echo "Transformer priors downloaded OK"
# SHA-256 (for reference, computed from Zenodo downloads):
#   reinvent.prior:                  b6513ec6dbc54c87ea45cdbf9b4aaefadd7652548b74175366b27f12ec5732fe
#   libinvent.prior:                 03e6cbe8a53e59a4ac3aa6728d041f1957bdd07b5eefdf2cfc5c8591036075af
#   linkinvent.prior:                3aa629ab769e7c29f747f0a1ad3e9d90b33d7be9e04ec47b7db416026fc568ed
#   mol2mol_medium_similarity.prior: a0f40a2fb4baedaf51ae199bb38a61a8c8d26f03062977eaf06e34079e9f22ca

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
