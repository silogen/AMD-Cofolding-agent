#!/usr/bin/env python3
"""RetroDMTA project-endpoint REST service (single FastAPI app, boltz2/posebusters convention).

Per-target, per-temporal-window models. An agent fixes a (target, window) for a campaign; the
model loads once and stays resident (LRU cache). The implementation lives in the
``retrodmta_predictor`` package (installed into the image by the Dockerfile); this file is the
only FastAPI app.

Run:  uvicorn app:app --host 0.0.0.0 --port 8085
Endpoints:
  GET  /health
  GET  /targets                                             -> targets, endpoints, windows
  POST /score    {"smiles":[...], "params":{"target":"FXA","endpoint":"FXA_pKi"}}  -> REINVENT scorer
  POST /predict  {"target":"FXA","window":4,"method":"ensemble","smiles":["..."]}
  POST /predict_csv  (multipart CSV upload)
"""
from __future__ import annotations

import io
from typing import Any, Optional, Union

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from retrodmta_predictor import core as P


class PredictRequest(BaseModel):
    target: str
    smiles: list[str]
    window: Union[int, str] = "latest"
    method: str = "ensemble"          # ensemble | xgboost | chemprop | chemeleon
    batch_size: Optional[int] = None
    format: str = "records"           # records | compact | table


class ScoreParams(BaseModel):
    target: str
    window: Union[int, str] = "latest"
    method: str = "ensemble"
    endpoint: Optional[str] = None    # default: the target's sole endpoint; required if it has >1
    batch_size: Optional[int] = None


class ScoreRequest(BaseModel):
    smiles: list[str]
    params: ScoreParams


class ScoreResponse(BaseModel):
    scores: list[Optional[float]]
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="RetroDMTA project-endpoint predictor", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok", "device": P._device(),
            "targets": list(P.MANIFEST["targets"]), "cache_max": P.CACHE_MAX}


@app.get("/targets")
def targets():
    out = {}
    for tgt, t in P.MANIFEST["targets"].items():
        out[tgt] = {"endpoints": t["endpoints"], "latest_window": t["latest_window"],
                    "windows": {w: {"cut_date": info.get("cut_date"),
                                    "methods": info.get("methods"), "n_train": info.get("n_train")}
                                for w, info in t["windows"].items()}}
    return out


def _resolve_endpoint(target: str, endpoint: Optional[str]) -> str:
    if target not in P.MANIFEST["targets"]:
        raise HTTPException(status_code=400, detail=f"unknown target {target!r}; have {list(P.MANIFEST['targets'])}")
    eps = P.MANIFEST["targets"][target]["endpoints"]
    # Default to the target's PRIMARY endpoint (first listed -- the activity
    # endpoint, e.g. FXA_pKi) when the caller omits it, so the scorer contract
    # works for RL without a fiddly per-target endpoint lookup. The chosen
    # endpoint is echoed in the response metadata; pass params.endpoint to override.
    ep = endpoint or eps[0]
    if ep not in eps:
        raise HTTPException(status_code=400, detail=f"endpoint {ep!r} not in {eps} for target {target!r}")
    return ep


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """REINVENT scorer contract: one scalar per input SMILES (ensemble mean of the selected
    target endpoint), None on failure, strict index alignment. Raw value -- REINVENT owns the
    transform/direction."""
    p = req.params
    ep = _resolve_endpoint(p.target, p.endpoint)
    try:
        recs = P.predict(p.target, req.smiles, window=p.window, method=p.method,
                         batch_size=p.batch_size or P.BATCH_SIZE)  # one record per input, in order
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    scores = [(r["predictions"].get(ep, {}).get("mean") if r["ok"] else None) for r in recs]
    metadata = {
        "target": p.target, "endpoint": ep,
        "window": recs[0].get("window") if recs else None,
        "cut_date": recs[0].get("cut_date") if recs else None,
        "std": [(r["predictions"].get(ep, {}).get("std") if r["ok"] else None) for r in recs],
    }
    return ScoreResponse(scores=scores, metadata=metadata)


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return P.predict(req.target, req.smiles, window=req.window, method=req.method,
                         batch_size=req.batch_size or P.BATCH_SIZE, fmt=req.format)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _flatten_project(records):
    rows = []
    for r in records:
        row = {"input_smiles": r["input_smiles"], "standardized_smiles": r.get("standardized_smiles"),
               "ok": r["ok"], "target": r.get("target"), "window": r.get("window")}
        for ep, v in r.get("predictions", {}).items():
            row[f"{ep}_mean"] = v.get("mean")
            row[f"{ep}_std"] = v.get("std")
        rows.append(row)
    return pd.DataFrame(rows)


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), target: str = Form(...),
                      window: str = Form("latest"), method: str = Form("ensemble"),
                      smiles_col: str = Form("smiles"), batch_size: int = Form(0),
                      format: str = Form("json")):
    """Upload a CSV -> predictions for a target/window. format=json or csv."""
    raw = await file.read()
    df = pd.read_csv(io.BytesIO(raw))
    if smiles_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"smiles_col {smiles_col!r} not in CSV columns {list(df.columns)}")
    smiles = df[smiles_col].astype(str).tolist()
    try:
        recs = P.predict(target, smiles, window=window, method=method,
                         batch_size=batch_size or P.BATCH_SIZE)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if format == "csv":
        out = _flatten_project(recs).to_csv(index=False)
        return Response(content=out, media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={target}_preds.csv"})
    return recs


def create_app():
    return app


def main() -> None:
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8085")))


if __name__ == "__main__":
    main()
