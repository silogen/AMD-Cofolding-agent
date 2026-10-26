#!/usr/bin/env python3
"""Generate response-domain statistics for an existing ADMET model bundle.

The source datasets are the exact post-filtering ``data.csv`` files used for
the endpoint models. Only rows with ``modeling_set == True`` contribute.
The generated p01/p99 interval is a warning domain, not a clipping interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _modeled_values(path: Path) -> pd.Series:
    frame = pd.read_csv(path, usecols=["value", "modeling_set"])
    selected = frame["modeling_set"].fillna(False).map(
        lambda value: value is True or str(value).strip().lower() == "true"
    )
    return pd.to_numeric(frame.loc[selected, "value"], errors="coerce").dropna()


def generate_for_set(dataset_root: Path, bundle_root: Path) -> dict[str, dict]:
    manifest_path = bundle_root / "manifest.json"
    training_manifest_path = bundle_root / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    training_manifest = json.loads(training_manifest_path.read_text())
    expected_counts = {
        entry["endpoint"]: int(entry["n_train"])
        for entry in training_manifest["xgb_models"]
    }

    ranges: dict[str, dict] = {}
    for endpoint in manifest["endpoint_order"]:
        values = _modeled_values(dataset_root / endpoint / "data.csv")
        expected = expected_counts.get(endpoint)
        if expected is not None and len(values) != expected:
            raise ValueError(
                f"{manifest['model_set']}/{endpoint}: found {len(values)} modeled "
                f"values, expected {expected}"
            )
        ranges[endpoint] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "p01": float(values.quantile(0.01)),
            "p99": float(values.quantile(0.99)),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "n": int(len(values)),
        }

    ranges_name = "training_ranges.json"
    (bundle_root / ranges_name).write_text(json.dumps(ranges, indent=2) + "\n")
    manifest["response_domain"] = {
        "training_ranges": ranges_name,
        "method": "training_response_quantiles",
        "lower_quantile": 0.01,
        "upper_quantile": 0.99,
        "behavior": "flag_only",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug-data", type=Path, required=True)
    parser.add_argument("--agro-data", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    args = parser.parse_args()

    for model_set, dataset_root in (
        ("drug", args.drug_data),
        ("agro", args.agro_data),
    ):
        ranges = generate_for_set(
            dataset_root.resolve(),
            (args.models_root / model_set).resolve(),
        )
        print(f"{model_set}: wrote {len(ranges)} endpoint ranges")


if __name__ == "__main__":
    main()
