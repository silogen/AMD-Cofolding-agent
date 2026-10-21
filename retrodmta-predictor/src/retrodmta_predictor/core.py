#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Project-endpoint inference: per-target, per-temporal-window models (RetroDMTA).

Pick a target + window (+ method), predict that target's endpoints for input SMILES.
Each (target, window) snapshot bundles:
  - Chemprop-MTL and CheMeleon-MTL (multitask: 7 general ADME tasks + the target's endpoints)
  - one single-task XGBoost per target endpoint
`method` = xgboost | chemprop | chemeleon | ensemble (mean +/- std of the available methods).

Models load ONCE on first use and stay resident (LRU cache), so an agent that fixes a
(target, window) for a campaign pays the load cost only once. Same standardization, frozen
in-process models, fixed batching, parallel featurization, and CPU/GPU toggle as the general
ADMET service. No applicability domain (per request).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Avalon import pyAvalonTools

from retrodmta_predictor.standardize import standardize_one

RDLogger.DisableLog("rdApp.*")

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
# molecules per frozen pass; default 1000, raise via PRED_BATCH / --batch-size / API batch_size.
BATCH_SIZE = int(os.environ.get("PRED_BATCH", "1000"))
N_WORKERS = int(os.environ.get("PRED_WORKERS", str(min(32, (os.cpu_count() or 4)))))
LOADER_WORKERS = int(os.environ.get("PRED_LOADER_WORKERS", "8"))
PRED_DEVICE = os.environ.get("PRED_DEVICE", "auto").lower()
CACHE_MAX = int(os.environ.get("PRED_CACHE_MAX", "6"))       # resident (target,window) snapshots
AVALON_NBITS = 2048
_PMIN = 128

MANIFEST = json.load(open(MODELS_DIR / "manifest.json"))
GENERAL_ENDPOINTS = MANIFEST["general_endpoints"]
ENDPOINT_META = MANIFEST["endpoint_meta"]

_DESC_NAMES = [n for n, _ in Descriptors._descList]
_CALC = MolecularDescriptorCalculator(_DESC_NAMES)
_POOL = None
_DEV = None
_FEATURIZER = None
_CACHE = OrderedDict()   # (target, window, component) -> model object


# ---------------- parallel featurization ----------------
def _pool():
    global _POOL
    if _POOL is None and N_WORKERS > 1:
        _POOL = ProcessPoolExecutor(max_workers=N_WORKERS)
    return _POOL


def _pmap(fn, items):
    items = list(items)
    if len(items) < _PMIN or N_WORKERS <= 1 or _pool() is None:
        return [fn(x) for x in items]
    chunk = max(1, len(items) // (N_WORKERS * 4))
    return list(_pool().map(fn, items, chunksize=chunk))


def _feat_one(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    phys = np.asarray(_CALC.CalcDescriptors(m), dtype=np.float32)
    arr = np.zeros((AVALON_NBITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(pyAvalonTools.GetAvalonFP(m, AVALON_NBITS), arr)
    return np.concatenate([phys, arr])


def _featurize(smiles):
    n = len(smiles)
    X = np.full((n, len(_DESC_NAMES) + AVALON_NBITS), np.nan, dtype=np.float32)
    for i, r in enumerate(_pmap(_feat_one, smiles)):
        if r is not None:
            X[i] = r
    X[~np.isfinite(X)] = np.nan
    return X


def _featurize_contract(smiles, avalon_bits: int, desc_cols: list[str]):
    """Feature matrix for newly exported project XGBoost models.

    Their training pipeline used Avalon bits followed by zero-filled descriptor
    columns; preserve that exact representation.
    """
    X = np.zeros((len(smiles), avalon_bits + len(desc_cols)), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=avalon_bits)
        arr = np.zeros(avalon_bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i, :avalon_bits] = arr
    return X


# ---------------- device + GNN plumbing ----------------
def _device():
    global _DEV
    if _DEV is None:
        import torch
        want = PRED_DEVICE in ("auto", "cuda", "gpu")
        _DEV = "cuda" if (want and torch.cuda.is_available()) else "cpu"
    return _DEV


def _featurizer():
    global _FEATURIZER
    if _FEATURIZER is None:
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
        _FEATURIZER = SimpleMoleculeMolGraphFeaturizer()
    return _FEATURIZER


def _make_loader(smiles):
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
    dps = [MoleculeDatapoint.from_smi(str(s)) for s in smiles]
    ds = MoleculeDataset(dps, featurizer=_featurizer())
    return build_dataloader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=LOADER_WORKERS)


def _to_dev(batch):
    dev = _device()
    if dev == "cpu":
        return batch.bmg, batch.V_d, batch.X_d
    batch.bmg.to(dev)
    vd = batch.V_d.to(dev) if batch.V_d is not None else None
    xd = batch.X_d.to(dev) if batch.X_d is not None else None
    return batch.bmg, vd, xd


# ---------------- resident model cache ----------------
def _cache_get(key, loader):
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    obj = loader()
    _CACHE[key] = obj
    _CACHE.move_to_end(key)
    while len(_CACHE) > CACHE_MAX:
        _CACHE.popitem(last=False)
    return obj


def _load_gnn(path):
    from chemprop import models
    return models.MPNN.load_from_file(str(path), map_location="cpu").eval().to(_device())


def _get_gnn(target, w, component):
    pt = MODELS_DIR / target / f"w{w:02d}" / f"{component}.pt"
    if not pt.exists():
        return None
    return _cache_get((target, w, component), lambda: _load_gnn(pt))


def _get_xgb(target, w, task):
    p = MODELS_DIR / target / f"w{w:02d}" / "xgb" / f"{task}.pkl"
    if not p.exists():
        return None
    def load():
        artifact = pickle.load(open(p, "rb"))
        if isinstance(artifact, dict) and "model" in artifact:
            return {
                "model": artifact["model"],
                "avalon_bits": int(artifact.get("avalon_bits", AVALON_NBITS)),
                "desc_cols": list(artifact.get("desc_cols", [])),
            }
        return {"model": artifact, "avalon_bits": None, "desc_cols": None}

    return _cache_get((target, w, f"xgb:{task}"), load)


# ---------------- prediction ----------------
def _resolve_target(target):
    """Canonical manifest key for a caller-supplied target name.

    The returned string is the manifest's own key, never the caller's string,
    so the `MODELS_DIR / target / ...` paths built by `_get_gnn`/`_get_xgb`
    can never contain a caller-controlled path segment.
    """
    for key in MANIFEST["targets"]:
        if key == target:
            return key
    raise ValueError(f"unknown target {target}; have {list(MANIFEST['targets'])}")


def _resolve_window(target, window):
    t = MANIFEST["targets"][target]
    if window in (None, "latest"):
        return t["latest_window"]
    w = int(window)
    if str(w) not in t["windows"]:
        raise ValueError(f"window {w} not available for {target}; have {sorted(int(k) for k in t['windows'])}")
    return w


def _gnn_predict(model, smiles, endpoints):
    """Return dict endpoint -> np.array(n) for the target endpoints (sliced from the MTL head)."""
    import torch
    n = len(smiles)
    padded = list(smiles)
    # chemprop's BatchNorm fails on a batch of a single molecule, so pad the input
    # to at least 2 rows and avoid a trailing batch of size 1 (len % BATCH_SIZE == 1)
    # by repeating the last SMILES. The extra rows are trimmed off via [:n] below.
    while len(padded) < 2 or len(padded) % BATCH_SIZE == 1:
        padded.append(padded[-1])
    loader = _make_loader(padded)
    outs = []
    with torch.inference_mode():
        for b in loader:
            bmg, vd, xd = _to_dev(b)
            outs.append(model(bmg, vd, xd).cpu().numpy())
    P = np.vstack(outs)[:n]
    base = len(GENERAL_ENDPOINTS)
    return {ep: P[:, base + j] for j, ep in enumerate(endpoints)}


def condense(records, target, endpoints, window, cut_date, fmt="records"):
    """records (default) | compact (metadata once + arrays) | table (columns+rows)."""
    if fmt == "records":
        return records
    units = {ep: ENDPOINT_META.get(ep, {}).get("unit") for ep in endpoints}
    head = {"target": target, "window": window, "cut_date": cut_date,
            "endpoints": endpoints, "units": units}
    if fmt == "compact":
        mols = []
        for r in records:
            m = {"input_smiles": r["input_smiles"], "standardized_smiles": r.get("standardized_smiles"),
                 "ok": r["ok"]}
            if r.get("ok"):
                p = r["predictions"]
                m["mean"] = [p[ep]["mean"] for ep in endpoints]
                m["std"] = [p[ep]["std"] for ep in endpoints]
            else:
                m["error"] = r.get("error")
            mols.append(m)
        return {**head, "molecules": mols}
    if fmt == "table":
        cols = (["input_smiles", "standardized_smiles", "ok"]
                + [f"{ep}_mean" for ep in endpoints] + [f"{ep}_std" for ep in endpoints])
        rows = []
        for r in records:
            p = r.get("predictions", {})
            rows.append([r["input_smiles"], r.get("standardized_smiles"), r["ok"]]
                        + [p.get(ep, {}).get("mean") for ep in endpoints]
                        + [p.get(ep, {}).get("std") for ep in endpoints])
        return {**head, "columns": cols, "rows": rows}
    raise ValueError(f"unknown format '{fmt}' (use records|compact|table)")


def predict(target, smiles_list, window="latest", method="ensemble", batch_size=BATCH_SIZE,
            fmt="records"):
    target = _resolve_target(target)
    w = _resolve_window(target, window)
    endpoints = MANIFEST["targets"][target]["endpoints"]
    cut_date = MANIFEST["targets"][target]["windows"][str(w)].get("cut_date")
    smiles_list = list(smiles_list)
    results = []
    for c0 in range(0, len(smiles_list), batch_size):
        results.extend(_predict_batch(target, w, endpoints, cut_date,
                                      smiles_list[c0:c0 + batch_size], method))
    return condense(results, target, endpoints, w, cut_date, fmt)


def _predict_batch(target, w, endpoints, cut_date, smiles_list, method):
    std = _pmap(standardize_one, smiles_list)
    std_smiles = [r["std_smiles"] if r["ok"] and r["std_smiles"] else None for r in std]
    valid = [s if s else "C" for s in std_smiles]
    want = ("xgboost", "chemprop", "chemeleon") if method == "ensemble" else (method,)

    per_method = {}   # method -> {endpoint: array}
    if "xgboost" in want:
        xg = {}
        feature_cache = {}
        for ep in endpoints:
            artifact = _get_xgb(target, w, ep)
            if artifact is None:
                continue
            if artifact["avalon_bits"] is None:
                key = ("legacy",)
                if key not in feature_cache:
                    feature_cache[key] = _featurize(valid)
            else:
                key = (
                    artifact["avalon_bits"],
                    tuple(artifact["desc_cols"]),
                )
                if key not in feature_cache:
                    feature_cache[key] = _featurize_contract(
                        valid,
                        artifact["avalon_bits"],
                        artifact["desc_cols"],
                    )
            xg[ep] = artifact["model"].predict(feature_cache[key])
        if xg:
            per_method["XGBoost"] = xg
    for comp, label in (("chemprop", "Chemprop-MTL"), ("chemeleon", "CheMeleon-MTL")):
        if comp in want:
            model = _get_gnn(target, w, comp)
            if model is not None:
                per_method[label] = _gnn_predict(model, valid, endpoints)

    out = []
    for i, smi in enumerate(smiles_list):
        rec = {"input_smiles": smi, "standardized_smiles": std_smiles[i],
               "ok": std_smiles[i] is not None, "target": target, "window": w,
               "cut_date": cut_date, "predictions": {}}
        if std_smiles[i] is None:
            rec["error"] = "standardization/parse failed"
            out.append(rec)
            continue
        for ep in endpoints:
            vals = {m: float(d[ep][i]) for m, d in per_method.items() if ep in d}
            arr = np.array(list(vals.values()), dtype=float)
            rec["predictions"][ep] = {
                "mean": round(float(arr.mean()), 4) if len(arr) else None,
                # std is a consensus signal only with >= 2 models; a single model
                # yields 0.0 which would misread as perfect agreement. Report None.
                "std": round(float(arr.std(ddof=0)), 4) if len(arr) >= 2 else None,
                "unit": ENDPOINT_META.get(ep, {}).get("unit"),
                "n_models": int(len(arr)),
                "per_model": {k: round(v, 4) for k, v in vals.items()},
            }
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--window", default="latest")
    ap.add_argument("--method", default="ensemble",
                    choices=["ensemble", "xgboost", "chemprop", "chemeleon"])
    ap.add_argument("--smiles", nargs="*", default=[])
    ap.add_argument("--csv")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--out")
    ap.add_argument("--format", default="records", choices=["records", "compact", "table"],
                    help="output shape: records (verbose) | compact (metadata once + arrays) | table")
    a = ap.parse_args()
    smiles = list(a.smiles)
    if a.csv:
        smiles += pd.read_csv(a.csv)[a.smiles_col].astype(str).tolist()
    if not smiles:
        ap.error("provide --smiles or --csv")
    res = predict(a.target, smiles, window=a.window, method=a.method, fmt=a.format)
    txt = json.dumps(res, indent=2)
    if a.out:
        Path(a.out).write_text(txt)
        print(f"wrote {a.out}")
    else:
        print(txt)


if __name__ == "__main__":
    main()
