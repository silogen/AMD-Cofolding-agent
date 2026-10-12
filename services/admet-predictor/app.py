#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ADMET inference REST service (single FastAPI app, boltz2/posebusters convention).

Models (XGBoost + Chemprop-MTL + CheMeleon-MTL) and the applicability-domain reference set
load ONCE at startup and stay resident, so each request is only compute. The implementation
lives in the ``admet_predictor`` package (installed into the image by the Dockerfile); this
file is the only FastAPI app.

Run:  uvicorn app:app --host 0.0.0.0 --port 8084
Endpoints:
  GET  /health                          -> liveness + endpoint list + device
  POST /score    {"smiles":[...], "params":{"endpoint": "logp"}}  -> REINVENT scorer contract
  POST /predict  {"smiles":[...], "format": "records|compact|table"}  -> full per-endpoint records
  POST /predict_csv  (multipart CSV upload)  -> JSON or CSV
"""
from __future__ import annotations

import io
from contextlib import asynccontextmanager
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from admet_predictor import core as P


class PredictRequest(BaseModel):
    smiles: list[str]
    batch_size: Optional[int] = None
    format: str = "records"          # records | compact | table


class ScoreParams(BaseModel):
    endpoint: str = "logp"           # one of core.ENDPOINTS
    batch_size: Optional[int] = None


class ScoreRequest(BaseModel):
    smiles: list[str]
    params: ScoreParams = Field(default_factory=ScoreParams)


class ScoreResponse(BaseModel):
    scores: list[Optional[float]]
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # warm the resident models + AD matrix so the first real request is already fast
    P._gnn("cp", "chemprop_mtl")
    P._gnn("ce", "chemeleon_mtl")
    if P._AD["bits"] is not None:
        P._ad_trainf()
    P.predict(["CCO"])                      # trip the featurizers / lazy imports once
    yield


app = FastAPI(title="ADMET predictor", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "endpoints": P.ENDPOINTS,
            "batch_size": P.BATCH_SIZE, "device": P._device()}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """REINVENT scorer contract: one scalar per input SMILES (the ensemble mean of the
    selected endpoint), None on failure, strict index alignment. REINVENT applies the
    transform/direction, so we return the raw predicted value (no baked objective)."""
    ep = req.params.endpoint
    if ep not in P.ENDPOINTS:
        raise HTTPException(status_code=400, detail=f"unknown endpoint {ep!r}; have {P.ENDPOINTS}")
    recs = P.predict(req.smiles, batch_size=req.params.batch_size or P.BATCH_SIZE)  # one record per input, in order
    scores = [(r["predictions"].get(ep, {}).get("mean") if r["ok"] else None) for r in recs]
    metadata = {
        "endpoint": ep,
        "unit": P.ENDPOINT_META.get(ep, {}).get("unit"),
        "std": [(r["predictions"].get(ep, {}).get("std") if r["ok"] else None) for r in recs],
        "ad_pass": [r.get("applicability_domain", {}).get("AD Pass") for r in recs],
    }
    return ScoreResponse(scores=scores, metadata=metadata)


@app.post("/predict")
def predict(req: PredictRequest):
    return P.predict(req.smiles, batch_size=req.batch_size or P.BATCH_SIZE, fmt=req.format)


def _flatten_admet(records):
    rows = []
    for r in records:
        row = {"input_smiles": r["input_smiles"], "standardized_smiles": r.get("standardized_smiles"),
               "ok": r["ok"]}
        for ep, v in r.get("predictions", {}).items():
            row[f"{ep}_mean"] = v.get("mean")
            row[f"{ep}_std"] = v.get("std")
        ad = r.get("applicability_domain", {})
        if ad:
            row["AD_pass"] = ad.get("AD Pass")
        rows.append(row)
    return pd.DataFrame(rows)


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), smiles_col: str = Form("smiles"),
                      batch_size: int = Form(0), format: str = Form("json")):
    """Upload a CSV -> predictions. format=json (records) or csv (flattened mean/std per endpoint)."""
    raw = await file.read()
    df = pd.read_csv(io.BytesIO(raw))
    if smiles_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"smiles_col {smiles_col!r} not in CSV columns {list(df.columns)}")
    smiles = df[smiles_col].astype(str).tolist()
    recs = P.predict(smiles, batch_size=batch_size or P.BATCH_SIZE)
    if format == "csv":
        out = _flatten_admet(recs).to_csv(index=False)
        return Response(content=out, media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=admet_preds.csv"})
    return recs


def create_app():
    return app


def main() -> None:
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8084")))


if __name__ == "__main__":
    main()
