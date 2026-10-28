# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

FROM python:3.11-slim
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY app.py /app/
WORKDIR /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
