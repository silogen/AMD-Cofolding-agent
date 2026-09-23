# Synthesis Planner — AiZynthFinder retrosynthesis service.
#
# Build context: dd_agentic/ (top-level repo root)
#
#   docker build \
#     -f services/synthesis-planner/Dockerfile \
#     -t synthesis-planner:latest \
#     .
#
# Policy model + templates are downloaded from the official AiZynthFinder
# Zenodo release (record 11430881) at image build time.
# MD5 checksums from Zenodo are verified during download to detect truncation.
#
# The STOCK catalog is NOT baked in (too large; typically 1–10 GB).
# Mount it at runtime via Compose:
#   volumes:
#     - /data/aizynthfinder/stock:/app/models/stock:ro
#
# If stock is absent the service still starts — availability checks return
# "unknown" and route leaves are marked in_stock=None.

FROM mambaorg/micromamba:1.5-jammy AS base

# Activate the conda base environment for all subsequent RUN steps
ARG MAMBA_DOCKERFILE_ACTIVATE=1

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    MAMBA_NO_LOW_SPEED_LIMIT=1 \
    MODELS_DIR=/app/models

WORKDIR /app

# ── Install Python environment via micromamba ─────────────────────────────
COPY services/synthesis-planner/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

# ── Copy service code ─────────────────────────────────────────────────────
COPY services/synthesis-planner/app.py                         /app/app.py
COPY services/synthesis-planner/synthesis_planner/             /app/synthesis_planner/

# ── Download standard AiZynthFinder models from Zenodo (record 11430881) ─
# Files: USPTO expansion policy (ONNX, 91 MB) + reaction templates (3 MB).
# Stock (ZINC In-Stock, 1.3 GB) is intentionally NOT baked in — mount at runtime.
# Uses Python's urllib (no system curl/wget dependency) to avoid PATH issues.
COPY services/synthesis-planner/download_models.py /tmp/download_models.py
RUN python3 /tmp/download_models.py

# ── Write manifest with known metadata ───────────────────────────────────
# SHA-256 values are PLACEHOLDER because integrity was already verified via
# MD5 during download (from trusted Zenodo source). The manifest loader skips
# the SHA-256 check for PLACEHOLDER and still verifies file existence.
COPY services/synthesis-planner/write_manifest.py /tmp/write_manifest.py
RUN python3 /tmp/write_manifest.py

# ── Smoke test: verify manifest loads and files exist ────────────────────
RUN python -c "\
import sys; sys.path.insert(0,'/app'); \
from synthesis_planner.manifest import load_and_verify; \
m = load_and_verify('/app/models'); \
print(f'manifest OK: policy={m.policy_version()}, templates={m.templates_version()}'); \
" 2>&1

# Verify AiZynthFinder can be imported (won't run a search without stock/policy)
RUN python -c "from importlib.metadata import version; from aizynthfinder.aizynthfinder import AiZynthFinder; print('AiZynthFinder', version('aizynthfinder'), 'import OK')" 2>&1

# ── Runtime ───────────────────────────────────────────────────────────────
EXPOSE 8086
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD /opt/conda/bin/python -c "import urllib.request; urllib.request.urlopen('http://localhost:8086/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8086"]
