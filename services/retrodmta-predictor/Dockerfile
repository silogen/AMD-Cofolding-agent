# retrodmta-predictor REST service.
#
# Follows the services/<name>/ convention: this dir holds only the HTTP wrapper (app.py) +
# this Dockerfile. The library lives in the sibling package dd_agentic/retrodmta-predictor.
# Model weights are NOT in git: they live centrally in dd_agentic/models/ (gitignored),
# restored from the model weights archive, and baked in here from dd_agentic/models/retrodmta.
#
# Build with dd_agentic/ as the build context (package + wrapper + models must be present):
#   docker build -f services/retrodmta-predictor/Dockerfile -t retrodmta-predictor:latest .
# Run:
#   docker run --rm -p 8085:8085 retrodmta-predictor:latest
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
COPY retrodmta-predictor /app/retrodmta-predictor
COPY services/retrodmta-predictor/app.py /app/app.py
COPY models/retrodmta /app/models
RUN pip install --no-cache-dir /app/retrodmta-predictor

# self-test: load a target/window + predict
RUN python -c "from retrodmta_predictor import core as P; d=P.predict('FXA', ['CC(=O)Oc1ccccc1C(=O)O'], window='latest', method='ensemble'); assert d[0]['predictions']['FXA_pKi']['mean'] is not None"

EXPOSE 8085
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8085/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8085"]
