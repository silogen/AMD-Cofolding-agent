# Structure Preparator service
# Tools: gemmi, pdb2pqr3+propka, pdbfixer, openmm, rdkit,
#        openff-toolkit, openff-nagl, fastapi, uvicorn
# All commercially free (MPL-2.0, BSD, MIT)

FROM mambaorg/micromamba:1.5-jammy

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

USER mambauser

# Conda-forge scientific stack
RUN micromamba install -y -n base -c conda-forge \
    python=3.11.9 \
    gemmi=0.6.5 \
    rdkit=2024.03.5 \
    openmm=8.1.2 \
    pdbfixer=1.9.0 \
    openff-toolkit=0.16.6 \
    openff-nagl=0.4.0 \
    openff-nagl-models=0.4.1 \
    openff-forcefields=2024.07.0 \
    setuptools=80.9.0 \
    numpy=1.26.4 \
    scipy=1.13.1 \
    && micromamba clean -afy

# pip packages (commercially free)
RUN micromamba run -n base pip install --no-cache-dir \
    "pdb2pqr==3.6.1" \
    "propka==3.5.0" \
    "fastapi==0.114.0" \
    "uvicorn[standard]==0.30.6" \
    "pydantic>=2.9,<3"

# Build-time sanity check
RUN micromamba run -n base python -c "\
import gemmi; import pdbfixer; import openmm; \
import openff.toolkit; import openff.nagl; import openff.nagl_models; \
import rdkit; import pdb2pqr; \
import fastapi; import pydantic; \
print('All imports OK')"

WORKDIR /app
COPY services/structure-prep/ /app/

ENV SP_SHARED_DIR=/shared \
    SP_HOST=0.0.0.0 \
    SP_PORT=8080 \
    SP_MAX_CONCURRENCY=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD micromamba run -n base python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["micromamba", "run", "-n", "base", "python", "-m", "uvicorn", \
     "app:app", "--host", "0.0.0.0", "--port", "8080"]
