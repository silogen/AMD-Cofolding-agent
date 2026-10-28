# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# admet-predictor REST service.
#
# Follows the services/<name>/ convention: this dir holds only the HTTP wrapper (app.py) +
# this Dockerfile. The library lives in the sibling package dd_agentic/admet-predictor.
# Model weights are NOT in git: they live centrally in dd_agentic/models/ (gitignored),
# restored from the model weights archive, and baked in here from dd_agentic/models/admet.
#
# Build with dd_agentic/ as the build context (package + wrapper + models must be present):
#   docker build -f services/admet-predictor/Dockerfile -t admet-predictor:latest .
# Run:
#   docker run --rm -p 8084:8084 admet-predictor:latest
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    MODELS_DIR=/app/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 libgomp1 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

# build context = dd_agentic/ : copy the package + wrapper; bake models from the central dir
COPY admet-predictor /app/admet-predictor
COPY services/admet-predictor/app.py /app/app.py
COPY models/admet /app/models
RUN pip install --no-cache-dir /app/admet-predictor

# Self-test both independently selectable model stacks.
RUN python -c "from admet_predictor import core as P; \
d=P.predict(['CC(=O)Oc1ccccc1C(=O)O'], model_set='drug'); \
a=P.predict(['CC(=O)Oc1ccccc1C(=O)O'], model_set='agro'); \
assert d[0]['predictions']['logp']['mean'] is not None; \
assert a[0]['predictions']['bcf']['mean'] is not None; \
assert 'Chemprop-MTL' in d[0]['predictions']['ppb']['per_model']; \
assert 'Chemprop-MTL' in a[0]['predictions']['boiling_point']['per_model']; \
assert d[0]['predictions']['ppb']['response_domain']['available']; \
assert a[0]['predictions']['boiling_point']['response_domain']['available']"

EXPOSE 8084
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8084/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8084"]
