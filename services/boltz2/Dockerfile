# Self-contained Boltz2 scoring wrapper.
#
# Installs Boltz2 from the local source tree (boltz2/) and adds the
# FastAPI wrapper on top.  Build context must be the project root.
#
# Build via compose (recommended):
#   docker compose build boltz
#
# Or standalone:
#   docker build -f services/boltz2/Dockerfile -t boltz2-wrapper .

FROM rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# --- Install Boltz2 from source ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install DockQ

COPY boltz2/ /app/boltz
WORKDIR /app/boltz

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .

# --- Install FastAPI wrapper ---
COPY services/boltz2/app.py services/boltz2/requirements.txt /app/wrapper/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /app/wrapper/requirements.txt

WORKDIR /app/wrapper
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
