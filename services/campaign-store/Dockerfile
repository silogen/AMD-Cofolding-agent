# Campaign store: the conductor/supervisor control-plane ledger.
#
# CPU only, and tiny -- it holds no model and does no chemistry. Build context
# is this directory:
#
#   docker build -f services/campaign-store/Dockerfile \
#     -t campaign-store:local services/campaign-store

FROM python:3.12-slim

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY app.py /app/
WORKDIR /app

EXPOSE 8087
# One worker. The ledger's atomic claim relies on SQLite's write lock, which
# works across processes, but a single process keeps the lock contention and
# the WAL to one writer -- and the service is nowhere near CPU-bound.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8087", "--workers", "1"]
