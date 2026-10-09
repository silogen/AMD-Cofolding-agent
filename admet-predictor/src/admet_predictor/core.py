#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Standardized ADMET inference: SMILES in -> per-endpoint mean +/- std across the three
deployment models (XGBoost, Chemprop-MTL, CheMeleon-MTL), all trained on the full ADME data.

Input SMILES are standardized with the SAME pipeline used for training
(desalt -> stereo-preserving re-aromatization -> Normalizer -> canonical tautomer ->
dimorphite pH 7.4 -> N-charge check).

Usage:
  python predict_admet.py --smiles "CC(=O)Oc1ccccc1C(=O)O" "CCN(CC)CC"
  python predict_admet.py --csv input.csv --smiles-col smiles --out preds.json
Env:
  MODELS_DIR   directory holding the models (default: ./models)
  CHEMPROP_BIN chemprop executable (default: chemprop on PATH)
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Avalon import pyAvalonTools

from admet_predictor.standardize import standardize_one

RDLogger.DisableLog("rdApp.*")

ENDPOINTS = ["logp", "logd", "aqueous_solubility", "kinetic_solubility",
             "caco2_permeability", "efflux_mdck", "efflux_caco2"]

# canonical unit + assay condition per endpoint (from the training annotations / aggregate_config)
ENDPOINT_META = {
    "logp":               {"unit": "logP (dimensionless)",
                           "condition": "octanol/water partition coefficient, neutral species"},
    "logd":               {"unit": "logD (dimensionless)",
                           "condition": "octanol/water distribution coefficient at pH 7.4"},
    "aqueous_solubility": {"unit": "log10(mol/L)",
                           "condition": "thermodynamic aqueous solubility"},
    "kinetic_solubility": {"unit": "log10(mol/L)",
                           "condition": "kinetic solubility, DMSO stock spiked into aqueous buffer"},
    "caco2_permeability": {"unit": "log10(cm/s)",
                           "condition": "Caco-2 monolayer apparent permeability, apical->basolateral (A->B)"},
    "efflux_mdck":        {"unit": "log10(efflux ratio)",
                           "condition": "MDR1-MDCK monolayer, efflux ratio Papp(B-A)/Papp(A-B)"},
    "efflux_caco2":       {"unit": "log10(efflux ratio)",
                           "condition": "Caco-2 monolayer, efflux ratio Papp(B-A)/Papp(A-B)"},
}
AVALON_NBITS = 2048
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
# frozen models, fixed molecules-per-pass: bounds memory regardless of total input size.
# default 1000; raise via PRED_BATCH / --batch-size / API batch_size if you have headroom,
# lower (e.g. 128) if memory is tight.
BATCH_SIZE = int(os.environ.get("PRED_BATCH", "1000"))
# CPU workers for RDKit standardization + featurization (single largest remaining cost).
N_WORKERS = int(os.environ.get("PRED_WORKERS", str(min(32, (os.cpu_count() or 4)))))
# chemprop graph-featurization workers (dataloader). 0 = in the main process.
LOADER_WORKERS = int(os.environ.get("PRED_LOADER_WORKERS", "8"))
_PMIN = 128  # only fan out to processes above this many molecules (pool overhead otherwise)
# device for the GNN forward passes: auto | cpu | cuda  (falls back to cpu if no GPU present)
PRED_DEVICE = os.environ.get("PRED_DEVICE", "auto").lower()
_DEV = None

_DESC_NAMES = [n for n, _ in Descriptors._descList]
_CALC = MolecularDescriptorCalculator(_DESC_NAMES)
_POOL = None


def _device():
    global _DEV
    if _DEV is None:
        import torch
        want_gpu = PRED_DEVICE in ("auto", "cuda", "gpu")
        _DEV = "cuda" if (want_gpu and torch.cuda.is_available()) else "cpu"
    return _DEV


def _to_dev(batch):
    """Move a chemprop batch's graph + descriptor tensors to the active device."""
    dev = _device()
    if dev == "cpu":
        return batch.bmg, batch.V_d, batch.X_d
    batch.bmg.to(dev)                                   # BatchMolGraph.to is in-place (returns None)
    vd = batch.V_d.to(dev) if batch.V_d is not None else None
    xd = batch.X_d.to(dev) if batch.X_d is not None else None
    return batch.bmg, vd, xd


def _pool():
    global _POOL
    if _POOL is None and N_WORKERS > 1:
        _POOL = ProcessPoolExecutor(max_workers=N_WORKERS)
    return _POOL


def _pmap(fn, items):
    """Parallel map over a process pool for large inputs; serial for small ones."""
    items = list(items)
    if len(items) < _PMIN or N_WORKERS <= 1:
        return [fn(x) for x in items]
    pool = _pool()
    if pool is None:
        return [fn(x) for x in items]
    chunk = max(1, len(items) // (N_WORKERS * 4))
    return list(pool.map(fn, items, chunksize=chunk))

# ---------------- GNN models: loaded ONCE and kept resident (no per-batch reloads) ----------------
_GNN = {"cp": None, "ce": None, "featurizer": None, "loaded": {"cp": False, "ce": False}}
_XGB = {}  # endpoint -> [loaded seed models]


def _featurizer():
    if _GNN["featurizer"] is None:
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
        _GNN["featurizer"] = SimpleMoleculeMolGraphFeaturizer()
    return _GNN["featurizer"]


def _gnn(key, subdir):
    """Lazy-load and cache an MPNN checkpoint (frozen, eval), placed on the active device."""
    if not _GNN["loaded"][key]:
        _GNN["loaded"][key] = True
        pt = MODELS_DIR / subdir / "best.pt"
        if pt.exists():
            from chemprop import models
            _GNN[key] = models.MPNN.load_from_file(str(pt), map_location="cpu").eval().to(_device())
    return _GNN[key]


def _make_loader(smiles):
    """Build a chemprop dataloader over the standardized SMILES (reused for both models + encoding)."""
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
    dps = [MoleculeDatapoint.from_smi(str(s)) for s in smiles]
    ds = MoleculeDataset(dps, featurizer=_featurizer())
    return build_dataloader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=LOADER_WORKERS)

# ---------------- applicability domain (double: structural Tanimoto + latent cosine) ----------------
_AD_DIR = MODELS_DIR / "ad"
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
_AD = {"bits": None, "bitsf": None, "card": None, "emb": None, "thr": None}
if (_AD_DIR / "train_avalon_bits.npy").exists():
    _AD["bits"] = np.load(_AD_DIR / "train_avalon_bits.npy")          # packed uint8 (N, 256)
    _AD["card"] = _POP[_AD["bits"]].sum(1).astype(np.float32)         # per-train popcount (N,)
    _AD["emb"] = np.load(_AD_DIR / "train_embeddings.npy")            # (N, D) L2-normalized
    _AD["thr"] = json.load(open(_AD_DIR / "thresholds.json"))


def _ad_trainf():
    """Unpack the training bits to a float 0/1 matrix (N, 2048) once -> enables BLAS matmul."""
    if _AD["bitsf"] is None:
        _AD["bitsf"] = np.unpackbits(_AD["bits"], axis=1).astype(np.float32)
    return _AD["bitsf"]


def _ad_structural(Q, ok):
    """max + top-5 mean Tanimoto (Avalon) to the training set, from a precomputed Avalon matrix.
    Vectorized: intersection = Q . T^T (BLAS, multi-threaded); Tanimoto = inter / (|q|+|t|-inter)."""
    n = Q.shape[0]
    if _AD["bits"] is None:
        return [(None, None)] * n
    Tf = _ad_trainf()
    tcard = _AD["card"]
    qcard = Q.sum(1)
    out = [(None, None)] * n
    SUB = 256   # bound the (SUB, N_train) intermediate regardless of the request size
    for c0 in range(0, n, SUB):
        c1 = min(c0 + SUB, n)
        inter = Q[c0:c1] @ Tf.T                                       # (b, N_train)
        union = qcard[c0:c1, None] + tcard[None, :] - inter
        sims = inter / np.maximum(union, 1.0)
        mx = sims.max(1)
        top5 = np.partition(sims, -5, axis=1)[:, -5:].mean(1)
        for i in range(c1 - c0):
            gi = c0 + i
            if ok[gi]:
                out[gi] = (float(mx[i]), float(top5[i]))
    return out


def _ad_latent(loader, n):
    """max + top-5 mean cosine of the (in-process) Chemprop encoding to the training set."""
    model = _gnn("cp", "chemprop_mtl")
    if _AD["emb"] is None or model is None:
        return [(None, None)] * n
    import torch
    encs = []
    with torch.inference_mode():
        for b in loader:
            bmg, vd, xd = _to_dev(b)
            encs.append(model.encoding(bmg, vd, xd, i=0).cpu().numpy())
    E = np.vstack(encs)[:n].astype(np.float32)          # drop any padding rows
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    res = []
    for i in range(E.shape[0]):
        cos = _AD["emb"] @ E[i]
        top = np.sort(cos)[-5:]
        res.append((float(cos.max()), float(top.mean())))
    return res


# ---------------- featurization (matches training: physchem + Avalon) ----------------
def _feat_one(smi):
    """Physchem descriptors + Avalon bits for one SMILES (top-level -> picklable for the pool)."""
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    phys = np.asarray(_CALC.CalcDescriptors(m), dtype=np.float32)
    arr = np.zeros((AVALON_NBITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(pyAvalonTools.GetAvalonFP(m, AVALON_NBITS), arr)
    return phys, arr


def _featurize(smiles):
    """Parallel physchem+Avalon. Returns (X for XGB, Avalon float matrix for AD, ok mask)."""
    n = len(smiles)
    nphys = len(_DESC_NAMES)
    X = np.full((n, nphys + AVALON_NBITS), np.nan, dtype=np.float32)
    Q = np.zeros((n, AVALON_NBITS), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    for i, r in enumerate(_pmap(_feat_one, smiles)):
        if r is None:
            continue
        phys, arr = r
        X[i, :nphys] = phys
        X[i, nphys:] = arr
        Q[i] = arr
        ok[i] = True
    X[~np.isfinite(X)] = np.nan
    return X, Q, ok


# ---------------- per-method prediction ----------------
def _predict_xgb(X):
    """Mean over the XGBoost seed models per endpoint (models cached in memory). (n, n_endpoints)."""
    preds = np.full((X.shape[0], len(ENDPOINTS)), np.nan)
    for j, ep in enumerate(ENDPOINTS):
        if ep not in _XGB:
            _XGB[ep] = [pickle.load(open(s, "rb"))
                        for s in sorted((MODELS_DIR / "xgb" / ep).glob("seed*.pkl"))]
        if not _XGB[ep]:
            continue
        preds[:, j] = np.mean([mdl.predict(X) for mdl in _XGB[ep]], axis=0)
    return preds


def _predict_gnn(model, loader, n):
    """In-process MPNN prediction; output columns are already in ENDPOINTS order."""
    if model is None:
        return None
    import torch
    outs = []
    with torch.inference_mode():
        for b in loader:
            bmg, vd, xd = _to_dev(b)
            outs.append(model(bmg, vd, xd).cpu().numpy())
    P = np.vstack(outs)[:n]                             # drop any padding rows
    out = np.full((n, len(ENDPOINTS)), np.nan)
    out[:, :min(P.shape[1], len(ENDPOINTS))] = P[:, :len(ENDPOINTS)]
    return out


_MODEL_NAMES = ["XGBoost", "Chemprop-MTL", "CheMeleon-MTL"]


def condense(records, fmt="records"):
    """Reshape verbose per-molecule records into a smaller payload.
    fmt: 'records' (default, unchanged) | 'compact' (metadata once + arrays) | 'table' (columns+rows)."""
    if fmt == "records":
        return records
    eps = ENDPOINTS
    units = {ep: ENDPOINT_META[ep]["unit"] for ep in eps}
    conditions = {ep: ENDPOINT_META[ep]["condition"] for ep in eps}
    if fmt == "compact":
        mols = []
        for r in records:
            m = {"input_smiles": r["input_smiles"], "standardized_smiles": r.get("standardized_smiles"),
                 "ok": r["ok"]}
            if r.get("ok"):
                p = r["predictions"]
                m["mean"] = [p[ep]["mean"] for ep in eps]
                m["std"] = [p[ep]["std"] for ep in eps]
                ad = r.get("applicability_domain")
                if ad:
                    m["ad_pass"] = ad.get("AD Pass")
            else:
                m["error"] = r.get("error")
            mols.append(m)
        return {"endpoints": eps, "units": units, "conditions": conditions,
                "models": _MODEL_NAMES, "molecules": mols}
    if fmt == "table":
        cols = (["input_smiles", "standardized_smiles", "ok"]
                + [f"{ep}_mean" for ep in eps] + [f"{ep}_std" for ep in eps] + ["ad_pass"])
        rows = []
        for r in records:
            p = r.get("predictions", {})
            rows.append([r["input_smiles"], r.get("standardized_smiles"), r["ok"]]
                        + [p.get(ep, {}).get("mean") for ep in eps]
                        + [p.get(ep, {}).get("std") for ep in eps]
                        + [r.get("applicability_domain", {}).get("AD Pass")])
        return {"columns": cols, "rows": rows, "endpoints": eps, "units": units}
    raise ValueError(f"unknown format '{fmt}' (use records|compact|table)")


# ---------------- main ----------------
def predict(smiles_list, batch_size=BATCH_SIZE, fmt="records"):
    """Frozen-model inference over the input in fixed-size passes (bounds memory).
    fmt controls the output shape (records|compact|table); see condense()."""
    smiles_list = list(smiles_list)
    if batch_size is None or batch_size <= 0:
        batch_size = len(smiles_list) or 1
    results = []
    for c0 in range(0, len(smiles_list), batch_size):
        results.extend(_predict_batch(smiles_list[c0:c0 + batch_size]))
    return condense(results, fmt)


def _predict_batch(smiles_list):
    std = _pmap(standardize_one, smiles_list)
    std_smiles = [r["std_smiles"] if r["ok"] and r["std_smiles"] else None for r in std]
    valid = [s if s else "C" for s in std_smiles]   # placeholder for failed parses

    n = len(valid)
    # chemprop's build_dataloader drops a trailing batch of size 1 (batch-norm guard),
    # which would misalign outputs when n % BATCH_SIZE == 1; pad then slice back to n.
    padded = list(valid)
    while len(padded) < 2 or len(padded) % BATCH_SIZE == 1:
        padded.append(padded[-1])
    loader = _make_loader(padded)                      # featurize once, reuse for both models + encoding
    X, Q, ok = _featurize(valid)                       # one parallel pass -> XGB + AD structural
    xgb = _predict_xgb(X)
    cp = _predict_gnn(_gnn("cp", "chemprop_mtl"), loader, n)
    ce = _predict_gnn(_gnn("ce", "chemeleon_mtl"), loader, n)
    ad_s = _ad_structural(Q, ok)
    ad_l = _ad_latent(loader, n)

    results = []
    for i, smi in enumerate(smiles_list):
        rec = {"input_smiles": smi, "standardized_smiles": std_smiles[i],
               "ok": std_smiles[i] is not None, "predictions": {}}
        if std_smiles[i] is None:
            rec["error"] = "standardization/parse failed"
            results.append(rec)
            continue
        # applicability domain (two orthogonal signals; pass = within domain on that axis)
        thr = _AD["thr"]
        s_nn, s_top = ad_s[i]
        l_nn, l_top = ad_l[i]
        if thr and (s_nn is not None or l_nn is not None):
            ad = {}
            passes = []
            if s_nn is not None:
                p = bool(s_nn >= thr["tanimoto"]["pass"])
                passes.append(p)
                ad["AD Tanimoto"] = {"score": round(s_nn, 4), "top5_mean": round(s_top, 4),
                                     "threshold": thr["tanimoto"]["pass"], "pass": p}
            if l_nn is not None:
                p = bool(l_nn >= thr["cosine"]["pass"])
                passes.append(p)
                ad["AD Property profile FFN"] = {"score": round(l_nn, 4), "top5_mean": round(l_top, 4),
                                                 "threshold": thr["cosine"]["pass"], "pass": p}
            ad["AD Pass"] = bool(all(passes)) if passes else None
            rec["applicability_domain"] = ad
        for j, ep in enumerate(ENDPOINTS):
            vals = {}
            if not np.isnan(xgb[i, j]):
                vals["XGBoost"] = float(xgb[i, j])
            if cp is not None and not np.isnan(cp[i, j]):
                vals["Chemprop-MTL"] = float(cp[i, j])
            if ce is not None and not np.isnan(ce[i, j]):
                vals["CheMeleon-MTL"] = float(ce[i, j])
            arr = np.array(list(vals.values()), dtype=float)
            rec["predictions"][ep] = {
                "mean": round(float(arr.mean()), 4) if len(arr) else None,
                "std": round(float(arr.std(ddof=0)), 4) if len(arr) else None,
                "unit": ENDPOINT_META[ep]["unit"],
                "condition": ENDPOINT_META[ep]["condition"],
                "n_models": int(len(arr)),
                "per_model": {k: round(v, 4) for k, v in vals.items()},
            }
        results.append(rec)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", nargs="*", default=[])
    ap.add_argument("--csv")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--out")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="molecules per frozen pass (default env PRED_BATCH or 1000)")
    ap.add_argument("--format", default="records", choices=["records", "compact", "table"],
                    help="output shape: records (verbose) | compact (metadata once + arrays) | table")
    a = ap.parse_args()
    smiles = list(a.smiles)
    if a.csv:
        smiles += pd.read_csv(a.csv)[a.smiles_col].astype(str).tolist()
    if not smiles:
        ap.error("provide --smiles or --csv")
    res = predict(smiles, batch_size=a.batch_size, fmt=a.format)
    txt = json.dumps(res, indent=2)
    if a.out:
        Path(a.out).write_text(txt)
        print(f"wrote {a.out}")
    else:
        print(txt)


if __name__ == "__main__":
    main()
