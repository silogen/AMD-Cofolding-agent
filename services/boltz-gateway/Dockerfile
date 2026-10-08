# Boltz gateway: result cache + batch coalescing in front of the replicas.
#
# CPU only and small -- the GPU work stays in boltz2-wrapper. Build context is
# this directory:
#
#   docker build -f services/boltz-gateway/Dockerfile \
#     -t boltz-gateway:local services/boltz-gateway

FROM python:3.11-slim

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY app.py /app/
WORKDIR /app

EXPOSE 8091
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8091"]
