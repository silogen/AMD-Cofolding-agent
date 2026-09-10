# interaction-profiler REST service.
#
# Follows the services/<name>/ convention: this dir holds only the HTTP
# wrapper (app.py) + this Dockerfile. The library lives in the sibling
# package dd_agentic/interaction-profiler, installed below.
#
# Build with the dd_agentic/ directory as the build context so both the
# package and this wrapper are visible:
#
#   docker build -f services/interaction-profiler/Dockerfile \
#       -t interaction-profiler:latest .
#   # (run the above from dd_agentic/)
#
# Run (mount the artifact dir the caller writes structures into):
#   docker run --rm -p 8080:8080 -v /path/to/shared:/shared interaction-profiler:latest
#
# The scientific stack (RDKit/OpenMM/MDAnalysis) ships conda-forge binaries,
# so we base on micromamba.
FROM mambaorg/micromamba:1.5.8

USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

RUN micromamba install -y -n base -c conda-forge \
        python=3.11 rdkit openmm pdbfixer mdanalysis numpy scipy pandas \
    && micromamba clean --all --yes

WORKDIR /app
# Build context = dd_agentic/ : copy the package + this wrapper.
COPY --chown=$MAMBA_USER:$MAMBA_USER interaction-profiler /app/interaction-profiler
COPY --chown=$MAMBA_USER:$MAMBA_USER services/interaction-profiler/app.py /app/app.py

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN python -m pip install --no-cache-dir \
        "accfg>=0.0.9" "dimorphite-dl>=2.0" "typer>=0.9" \
        "fastapi>=0.110" "uvicorn[standard]>=0.29" \
    && python -m pip install --no-cache-dir --no-deps /app/interaction-profiler

ENV IP_SHARED_DIR=/shared \
    IP_CACHE_DIR=/tmp/interaction_profiler_receptors
EXPOSE 8080
VOLUME ["/shared"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
