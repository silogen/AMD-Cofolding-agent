#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Manifest-driven dual-stack ADMET inference.

The runtime bundle contains ``models/admet/{drug,agro}/manifest.json``.  Both
stacks preserve the historical per-molecule record shape while exposing their
own endpoint list and model members.  ``pharma`` is accepted as an input alias
for ``drug`` but is never emitted.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Avalon import pyAvalonTools

from admet_predictor.standardize import standardize_one

RDLogger.DisableLog("rdApp.*")

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
BATCH_SIZE = int(os.environ.get("PRED_BATCH", "1000"))
LOADER_WORKERS = int(os.environ.get("PRED_LOADER_WORKERS", "8"))
PRED_DEVICE = os.environ.get("PRED_DEVICE", "auto").lower()
DEFAULT_MODEL_SET = os.environ.get("ADMET_MODEL_SET", "drug").lower()
ALIASES = {"pharma": "drug", "pharmaceutical": "drug", "agrochemical": "agro"}

# Dense-matrix placeholder for rows whose standardization failed: only used to
# keep the batch feature matrix/loader aligned; these rows are dropped (ok=False)
# before any prediction is reported, so the placeholder's values never surface.
_PLACEHOLDER_SMILES = "C"

# Absolute plausibility limits. A member outside its endpoint's limits is
# excluded from the consensus rather than clipped to the boundary.
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "logp": (-8.0, 14.0),
    "logd": (-12.0, 12.0),
    "aqueous_solubility": (-15.0, 2.0),
    "kinetic_solubility": (-15.0, -2.0),
    "caco2_permeability": (-12.0, -2.0),
    "efflux_mdck": (-1.5, 2.5),
    "efflux_caco2": (-1.5, 2.5),
    "bcf": (-1.0, 8.0),
    "boiling_point": (-200.0, 800.0),
    "cyp1a2": (2.0, 12.0),
    "cyp2c9": (2.0, 12.0),
    "cyp2d6": (2.0, 12.0),
    "cyp3a4": (2.0, 12.0),
    "herg": (2.0, 12.0),
    "igc50_daphnia": (-2.0, 12.0),
    "ld50_bee": (-3.0, 9.0),
    "ld50_mouse": (-2.0, 8.0),
    "ld50_rat": (-2.0, 8.0),
    "logkoc": (-1.0, 9.0),
    "hlm_t12": (-4.0, 3.0),
    "plasma_t12_human": (-4.0, 3.0),
    "rlm_t12": (-4.0, 3.0),
    "plasma_t12_rat": (-4.0, 3.0),
    "ppb": (0.0, 100.0),
    "vapor_pressure": (-30.0, 2.0),
}

_STATES: dict[str, dict[str, Any]] = {}
_DEVICE: str | None = None


def normalize_model_set(model_set: str | None) -> str:
    value = (model_set or DEFAULT_MODEL_SET).strip().lower()
    value = ALIASES.get(value, value)
    if value not in available_model_sets():
        raise ValueError(
            f"unknown ADMET model_set {model_set!r}; "
            f"available: {available_model_sets()}"
        )
    return value


def available_model_sets() -> list[str]:
    if not MODELS_DIR.is_dir():
        return []
    return sorted(
        path.name
        for path in MODELS_DIR.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    )


def _device() -> str:
    global _DEVICE
    if _DEVICE is None:
        import torch

        want_gpu = PRED_DEVICE in ("auto", "cuda", "gpu")
        _DEVICE = "cuda" if want_gpu and torch.cuda.is_available() else "cpu"
    return _DEVICE


def _threshold_value(data: dict[str, Any], axis: str) -> float | None:
    direct = data.get(f"{axis}_thr")
    if direct is not None:
        return float(direct)
    nested = data.get("tanimoto" if axis == "avalon" else axis)
    if isinstance(nested, dict) and nested.get("pass") is not None:
        return float(nested["pass"])
    return None


def _load_state(model_set: str | None = None) -> dict[str, Any]:
    key = normalize_model_set(model_set)
    if key in _STATES:
        return _STATES[key]

    root = MODELS_DIR / key
    manifest = json.loads((root / "manifest.json").read_text())
    endpoints = list(manifest["endpoint_order"])
    metadata = manifest["endpoints"]

    xgb_models: dict[str, Any] = {}
    pattern = manifest["models"]["xgboost"]["path_pattern"]
    pickle_key = manifest["models"]["xgboost"].get("pickle_key")
    feature_contract: tuple[int, tuple[str, ...]] | None = None
    for endpoint in endpoints:
        path = root / pattern.format(endpoint=endpoint)
        if not path.is_file():
            continue
        with open(path, "rb") as handle:
            artifact = pickle.load(handle)
        model = artifact.get(pickle_key) if pickle_key and isinstance(artifact, dict) else artifact
        if model is None or not hasattr(model, "predict"):
            continue
        xgb_models[endpoint] = model
        if isinstance(artifact, dict):
            avalon_bits = int(artifact.get("avalon_bits", 2048))
            n_features = int(getattr(model, "n_features_in_", avalon_bits))
            if n_features < avalon_bits:
                raise ValueError(
                    f"{key}/{endpoint} expects {n_features} features, "
                    f"fewer than its {avalon_bits}-bit Avalon fingerprint"
                )
            # Some historical artifacts recorded the intended descriptor list
            # even though the trained estimator used Avalon bits only. Trust
            # the serialized estimator's feature count.
            n_descriptors = n_features - avalon_bits
            contract = (
                avalon_bits,
                tuple(artifact.get("desc_cols", [])[:n_descriptors]),
            )
            if feature_contract is None:
                feature_contract = contract
            elif feature_contract != contract:
                raise ValueError(f"inconsistent XGBoost feature contract in {key}")

    chemprop_model = None
    chemprop_scalers = None
    chemprop = manifest["models"].get("chemprop")
    if chemprop:
        model_path = root / chemprop["path"]
        scaler_path = root / chemprop["scalers"]
        if model_path.is_file() and scaler_path.is_file():
            try:
                from chemprop.models import MPNN

                chemprop_model = MPNN.load_from_checkpoint(
                    str(model_path), map_location="cpu"
                ).eval().to(_device())
                chemprop_scalers = json.loads(scaler_path.read_text())
                if list(chemprop_scalers["endpoints"]) != endpoints:
                    raise ValueError(
                        f"Chemprop task order does not match {key} manifest"
                    )
            except Exception as exc:  # keep XGBoost fallback available
                print(f"[admet] {key} Chemprop unavailable: {exc}", flush=True)
                chemprop_model = None
                chemprop_scalers = None

    thresholds: dict[str, Any] = {}
    train_fps: list[Any] = []
    ad = manifest.get("applicability_domain") or {}
    threshold_path = root / ad.get("thresholds", "")
    fps_path = root / ad.get("training_fingerprints", "")
    if threshold_path.is_file():
        thresholds = json.loads(threshold_path.read_text())
    if fps_path.is_file():
        with open(fps_path, "rb") as handle:
            train_data = pickle.load(handle)
        train_fps = [fp for fp in train_data.get("fps", []) if fp is not None]

    training_ranges: dict[str, dict[str, float]] = {}
    response_domain = manifest.get("response_domain") or {}
    ranges_name = response_domain.get("training_ranges")
    if ranges_name:
        ranges_path = root / ranges_name
        if ranges_path.is_file():
            loaded_ranges = json.loads(ranges_path.read_text())
            training_ranges = {
                endpoint: stats
                for endpoint, stats in loaded_ranges.items()
                if endpoint in endpoints and isinstance(stats, dict)
            }

    state = {
        "name": key,
        "root": root,
        "manifest": manifest,
        "endpoints": endpoints,
        "metadata": metadata,
        "xgb_models": xgb_models,
        "feature_contract": feature_contract or (2048, ()),
        "chemprop_model": chemprop_model,
        "chemprop_scalers": chemprop_scalers,
        "thresholds": thresholds,
        "train_fps": train_fps,
        "training_ranges": training_ranges,
    }
    _STATES[key] = state
    return state


def get_manifest(model_set: str | None = None) -> dict[str, Any]:
    return _load_state(model_set)["manifest"]


def get_endpoints(model_set: str | None = None) -> list[str]:
    return list(_load_state(model_set)["endpoints"])


def get_endpoint_meta(model_set: str | None = None) -> dict[str, dict[str, Any]]:
    return dict(_load_state(model_set)["metadata"])


def _avalon_matrix(smiles: list[str], n_bits: int, n_descriptors: int) -> np.ndarray:
    """Reproduce the deployed XGBoost training representation.

    The current training/export pipeline used Avalon bits followed by zero-filled
    descriptor columns.  Keep that exact contract until the models are retrained
    with calculated descriptors.
    """
    matrix = np.zeros((len(smiles), n_bits + n_descriptors), dtype=np.float32)
    for index, smiles_value in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smiles_value))
        if mol is None:
            continue
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        matrix[index, :n_bits] = arr
    return matrix


def _make_loader(smiles: list[str], batch_size: int):
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

    points = [MoleculeDatapoint.from_smi(str(value)) for value in smiles]
    dataset = MoleculeDataset(points, featurizer=SimpleMoleculeMolGraphFeaturizer())
    return build_dataloader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=LOADER_WORKERS
    )


def _predict_chemprop(state: dict[str, Any], smiles: list[str], batch_size: int):
    model = state["chemprop_model"]
    scalers = state["chemprop_scalers"]
    if model is None or scalers is None:
        return None

    import torch

    padded = list(smiles)
    while len(padded) < 2 or len(padded) % batch_size == 1:
        padded.append(padded[-1])
    outputs = []
    with torch.inference_mode():
        for batch in _make_loader(padded, batch_size):
            batch.bmg.to(_device())
            vertex = batch.V_d.to(_device()) if batch.V_d is not None else None
            extra = batch.X_d.to(_device()) if batch.X_d is not None else None
            outputs.append(model(batch.bmg, vertex, extra).detach().cpu().numpy())
    predictions = np.concatenate(outputs, axis=0)[: len(smiles)]
    expected_shape = (len(smiles), len(state["endpoints"]))
    if predictions.shape != expected_shape:
        raise ValueError(
            f"Chemprop returned shape {predictions.shape}, expected {expected_shape}"
        )
    # Chemprop v2 regression checkpoints contain an UnscaleTransform. In eval
    # mode model(...) therefore already returns values in endpoint units.
    # The standalone scaler file remains useful for task-order validation only.
    return predictions


def _endpoint_bounds(
    state: dict[str, Any], endpoint: str
) -> tuple[float, float] | None:
    configured = state["metadata"].get(endpoint, {}).get("physical_bounds")
    if isinstance(configured, dict):
        lower = configured.get("lower")
        upper = configured.get("upper")
        if lower is not None and upper is not None:
            return float(lower), float(upper)
    if isinstance(configured, (list, tuple)) and len(configured) == 2:
        return float(configured[0]), float(configured[1])
    return PHYSICAL_BOUNDS.get(endpoint)


def _summarize_endpoint(
    state: dict[str, Any],
    endpoint: str,
    raw_members: dict[str, float],
) -> dict[str, Any]:
    bounds = _endpoint_bounds(state, endpoint)
    accepted: dict[str, float] = {}
    excluded: dict[str, dict[str, Any]] = {}
    bounds_payload = (
        {"lower": bounds[0], "upper": bounds[1]} if bounds is not None else None
    )

    for name, raw_value in raw_members.items():
        value = float(raw_value)
        if not np.isfinite(value):
            excluded[name] = {"value": None, "reason": "non_finite"}
        elif bounds is not None and not bounds[0] <= value <= bounds[1]:
            excluded[name] = {
                "value": round(value, 4),
                "reason": "outside_physical_bounds",
                "bounds": bounds_payload,
            }
        else:
            accepted[name] = value

    values = np.asarray(list(accepted.values()), dtype=float)
    mean = float(values.mean()) if len(values) else None
    # std is only a consensus signal with >= 2 accepted members. With a single
    # member numpy returns 0.0, which reads as "models agreed perfectly" when it
    # actually means "one model, no consensus" — report None instead so callers
    # (and n_models) aren't misled.
    std = float(values.std(ddof=0)) if len(values) >= 2 else None

    range_info = state.get("training_ranges", {}).get(endpoint)
    response_domain: dict[str, Any] = {
        "method": "training_p01_p99",
        "available": False,
        "pass": None,
    }
    out_of_training_range: bool | None = None
    if range_info and range_info.get("p01") is not None and range_info.get("p99") is not None:
        lower = float(range_info["p01"])
        upper = float(range_info["p99"])
        response_pass = bool(mean is not None and lower <= mean <= upper)
        response_domain = {
            "method": "training_p01_p99",
            "available": True,
            "lower": round(lower, 4),
            "upper": round(upper, 4),
            "training_min": round(float(range_info["min"]), 4),
            "training_max": round(float(range_info["max"]), 4),
            "n_train": int(range_info["n"]),
            "pass": response_pass if mean is not None else None,
        }
        out_of_training_range = not response_pass if mean is not None else None

    metadata = state["metadata"].get(endpoint, {})
    return {
        "mean": round(mean, 4) if mean is not None else None,
        "std": round(std, 4) if std is not None else None,
        "unit": metadata.get("unit"),
        "condition": metadata.get("condition"),
        "n_models": len(accepted),
        "n_models_total": len(raw_members),
        "per_model": {name: round(value, 4) for name, value in accepted.items()},
        "excluded_models": excluded,
        "physical_bounds": bounds_payload,
        "response_domain": response_domain,
        "out_of_training_range": out_of_training_range,
    }


def _ad_scores(state: dict[str, Any], smiles: list[str]):
    threshold = _threshold_value(state["thresholds"], "avalon")
    train_fps = state["train_fps"]
    if threshold is None or not train_fps:
        return [(None, None)] * len(smiles)
    if len(train_fps) > 10000:
        rng = np.random.default_rng(42)
        indices = sorted(rng.choice(len(train_fps), size=10000, replace=False))
        train_fps = [train_fps[index] for index in indices]
    results = []
    n_bits = int(state["thresholds"].get("avalon_bits", 2048))
    for value in smiles:
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            results.append((None, False))
            continue
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
        similarities = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        score = float(max(similarities)) if similarities else 0.0
        results.append((score, bool(score >= threshold)))
    return results


def condense(records: list[dict[str, Any]], model_set: str, fmt: str = "records"):
    if fmt == "records":
        return records
    state = _load_state(model_set)
    endpoints = state["endpoints"]
    units = {ep: state["metadata"].get(ep, {}).get("unit") for ep in endpoints}
    conditions = {
        ep: state["metadata"].get(ep, {}).get("condition") for ep in endpoints
    }
    if fmt == "compact":
        molecules = []
        for record in records:
            item = {
                "input_smiles": record["input_smiles"],
                "standardized_smiles": record.get("standardized_smiles"),
                "ok": record["ok"],
            }
            if record.get("ok"):
                item["mean"] = [record["predictions"][ep]["mean"] for ep in endpoints]
                item["std"] = [record["predictions"][ep]["std"] for ep in endpoints]
                item["response_ad_pass"] = [
                    record["predictions"][ep]["response_domain"]["pass"]
                    for ep in endpoints
                ]
                item["ad_pass"] = record.get("applicability_domain", {}).get("AD Pass")
            else:
                item["error"] = record.get("error")
            molecules.append(item)
        return {
            "model_set": model_set,
            "endpoints": endpoints,
            "units": units,
            "conditions": conditions,
            "molecules": molecules,
        }
    if fmt == "table":
        columns = ["input_smiles", "standardized_smiles", "ok"]
        columns += [f"{ep}_mean" for ep in endpoints]
        columns += [f"{ep}_std" for ep in endpoints]
        columns += [f"{ep}_response_ad_pass" for ep in endpoints]
        columns += ["ad_pass"]
        rows = []
        for record in records:
            predictions = record.get("predictions", {})
            rows.append(
                [record["input_smiles"], record.get("standardized_smiles"), record["ok"]]
                + [predictions.get(ep, {}).get("mean") for ep in endpoints]
                + [predictions.get(ep, {}).get("std") for ep in endpoints]
                + [
                    predictions.get(ep, {}).get("response_domain", {}).get("pass")
                    for ep in endpoints
                ]
                + [record.get("applicability_domain", {}).get("AD Pass")]
            )
        return {
            "model_set": model_set,
            "columns": columns,
            "rows": rows,
            "endpoints": endpoints,
            "units": units,
        }
    raise ValueError(f"unknown format {fmt!r} (use records|compact|table)")


def predict(
    smiles_list,
    model_set: str = "drug",
    batch_size: int = BATCH_SIZE,
    fmt: str = "records",
):
    model_set = normalize_model_set(model_set)
    state = _load_state(model_set)
    values = list(smiles_list)
    if not values:
        return condense([], model_set, fmt)
    batch_size = batch_size if batch_size and batch_size > 0 else len(values)
    records = []
    for start in range(0, len(values), batch_size):
        records.extend(_predict_batch(state, values[start : start + batch_size], batch_size))
    return condense(records, model_set, fmt)


def _predict_batch(state: dict[str, Any], smiles_list: list[str], batch_size: int):
    standardized = [standardize_one(value) for value in smiles_list]
    # A row is usable only if standardization succeeded AND produced a non-empty
    # SMILES. An ok-but-empty std_smiles would otherwise slip through (``"" or "C"``
    # → methane) and be reported with methane's ADMET values as if they were the
    # molecule's. Mark such rows failed instead.
    ok_flags = [
        bool(item.get("ok")) and bool((item.get("std_smiles") or "").strip())
        for item in standardized
    ]
    std_smiles = [
        (item["std_smiles"] if ok else None)
        for item, ok in zip(standardized, ok_flags)
    ]
    # Placeholder only keeps the batch feature matrix dense for failed rows; those
    # rows are dropped (ok=False) before any prediction is reported.
    valid = [smi if smi else _PLACEHOLDER_SMILES for smi in std_smiles]
    n_bits, desc_cols = state["feature_contract"]
    features = _avalon_matrix(valid, n_bits, len(desc_cols))

    xgb_predictions: dict[str, np.ndarray] = {}
    for endpoint, model in state["xgb_models"].items():
        xgb_predictions[endpoint] = model.predict(features)
    chemprop_predictions = _predict_chemprop(state, valid, batch_size)
    ad = _ad_scores(state, valid)

    records = []
    for row_index, input_smiles in enumerate(smiles_list):
        record: dict[str, Any] = {
            "input_smiles": input_smiles,
            "standardized_smiles": std_smiles[row_index],
            "ok": ok_flags[row_index],
            "model_set": state["name"],
            "predictions": {},
        }
        if not record["ok"]:
            record["error"] = standardized[row_index].get("error", "standardization failed")
            records.append(record)
            continue

        ad_score, ad_pass = ad[row_index]
        record["applicability_domain"] = {
            "AD Tanimoto": {
                "score": round(ad_score, 4) if ad_score is not None else None,
                "threshold": _threshold_value(state["thresholds"], "avalon"),
                "pass": ad_pass,
            },
            "AD Pass": ad_pass,
            "axes_available": ["tanimoto"],
        }

        for endpoint_index, endpoint in enumerate(state["endpoints"]):
            raw_members: dict[str, float] = {}
            if endpoint in xgb_predictions:
                raw_members["XGBoost"] = float(xgb_predictions[endpoint][row_index])
            if chemprop_predictions is not None:
                raw_members["Chemprop-MTL"] = float(
                    chemprop_predictions[row_index, endpoint_index]
                )
            record["predictions"][endpoint] = _summarize_endpoint(
                state, endpoint, raw_members
            )
        records.append(record)
    return records


# Backward-compatible aliases for callers that inspect the default stack.
try:
    ENDPOINTS = get_endpoints(DEFAULT_MODEL_SET)
    ENDPOINT_META = get_endpoint_meta(DEFAULT_MODEL_SET)
except Exception:
    ENDPOINTS = []
    ENDPOINT_META = {}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Predict ADMET properties")
    parser.add_argument("--smiles", nargs="+", required=True)
    parser.add_argument("--model-set", default="drug", choices=["drug", "agro", "pharma"])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--format", default="records", choices=["records", "compact", "table"])
    arguments = parser.parse_args()
    print(json.dumps(predict(arguments.smiles, arguments.model_set, arguments.batch_size, arguments.format), indent=2))


if __name__ == "__main__":
    main()
