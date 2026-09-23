#!/usr/bin/env python3
"""ADMET inference REST service (single FastAPI app, boltz2/posebusters convention).

Models (XGBoost + Chemprop-MTL), response-domain statistics, and the
applicability-domain reference set load ONCE at startup and stay resident, so
each request is only compute. The implementation lives in the
``admet_predictor`` package (installed into the image by the Dockerfile); this
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
    model_set: str = "drug"          # drug | agro (pharma accepted as alias)
    batch_size: Optional[int] = None
    format: str = "records"          # records | compact | table


class ScoreParams(BaseModel):
    endpoint: str = "logp"           # one of core.ENDPOINTS
    model_set: str = "drug"
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
    # Warm the default stack; the alternative stack remains lazy.
    P.predict(["CCO"], model_set=P.DEFAULT_MODEL_SET)
    yield


app = FastAPI(title="ADMET predictor", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    model_sets = {}
    for model_set in P.available_model_sets():
        manifest = P.get_manifest(model_set)
        model_sets[model_set] = {
            "display_name": manifest.get("display_name", model_set),
            "endpoints": manifest["endpoint_order"],
            "model_members": list(manifest.get("models", {})),
            "response_domain": manifest.get("response_domain"),
            "physical_member_validation": True,
        }
    return {
        "status": "ok",
        "default_model_set": P.DEFAULT_MODEL_SET,
        "model_sets": model_sets,
        # Backward-compatible endpoint list for the default drug stack.
        "endpoints": P.get_endpoints(P.DEFAULT_MODEL_SET),
        "batch_size": P.BATCH_SIZE,
        "device": P._device(),
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """REINVENT scorer contract: one scalar per input SMILES (the ensemble mean of the
    selected endpoint), None on failure, strict index alignment. REINVENT applies the
    transform/direction, so we return the raw predicted value (no baked objective)."""
    try:
        model_set = P.normalize_model_set(req.params.model_set)
        endpoints = P.get_endpoints(model_set)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    ep = req.params.endpoint
    if ep not in endpoints:
        raise HTTPException(
            status_code=400,
            detail=f"unknown endpoint {ep!r} for {model_set}; have {endpoints}",
        )
    recs = P.predict(
        req.smiles,
        model_set=model_set,
        batch_size=req.params.batch_size or P.BATCH_SIZE,
    )
    scores = [(r["predictions"].get(ep, {}).get("mean") if r["ok"] else None) for r in recs]
    metadata = {
        "model_set": model_set,
        "endpoint": ep,
        "unit": P.get_endpoint_meta(model_set).get(ep, {}).get("unit"),
        "std": [(r["predictions"].get(ep, {}).get("std") if r["ok"] else None) for r in recs],
        "ad_pass": [r.get("applicability_domain", {}).get("AD Pass") for r in recs],
        "response_ad_pass": [
            (
                r["predictions"].get(ep, {}).get("response_domain", {}).get("pass")
                if r["ok"]
                else None
            )
            for r in recs
        ],
        "out_of_training_range": [
            (
                r["predictions"].get(ep, {}).get("out_of_training_range")
                if r["ok"]
                else None
            )
            for r in recs
        ],
        "excluded_models": [
            (
                r["predictions"].get(ep, {}).get("excluded_models", {})
                if r["ok"]
                else {}
            )
            for r in recs
        ],
    }
    return ScoreResponse(scores=scores, metadata=metadata)


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return P.predict(
            req.smiles,
            model_set=req.model_set,
            batch_size=req.batch_size or P.BATCH_SIZE,
            fmt=req.format,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _flatten_admet(records):
    rows = []
    for r in records:
        row = {"input_smiles": r["input_smiles"], "standardized_smiles": r.get("standardized_smiles"),
               "ok": r["ok"]}
        for ep, v in r.get("predictions", {}).items():
            row[f"{ep}_mean"] = v.get("mean")
            row[f"{ep}_std"] = v.get("std")
            row[f"{ep}_response_ad_pass"] = v.get("response_domain", {}).get("pass")
            row[f"{ep}_out_of_training_range"] = v.get("out_of_training_range")
            row[f"{ep}_n_models"] = v.get("n_models")
        ad = r.get("applicability_domain", {})
        if ad:
            row["AD_pass"] = ad.get("AD Pass")
        rows.append(row)
    return pd.DataFrame(rows)


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), smiles_col: str = Form("smiles"),
                      model_set: str = Form("drug"),
                      batch_size: int = Form(0), format: str = Form("json")):
    """Upload a CSV -> predictions. format=json (records) or csv (flattened mean/std per endpoint)."""
    raw = await file.read()
    df = pd.read_csv(io.BytesIO(raw))
    if smiles_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"smiles_col {smiles_col!r} not in CSV columns {list(df.columns)}")
    smiles = df[smiles_col].astype(str).tolist()
    try:
        recs = P.predict(
            smiles,
            model_set=model_set,
            batch_size=batch_size or P.BATCH_SIZE,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
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
