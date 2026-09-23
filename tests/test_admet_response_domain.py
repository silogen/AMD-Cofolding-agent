"""Regression tests for ADMET member validation and response-domain flags."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "admet-predictor" / "src"))

from admet_predictor import core  # noqa: E402


class _DeviceValue:
    def to(self, _device):
        return self


class _PredictionValue:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def test_chemprop_checkpoint_output_is_not_inverse_scaled_twice(monkeypatch) -> None:
    class Model:
        def __call__(self, _bmg, _vertex, _extra):
            return _PredictionValue([[283.0], [283.0]])

    batch = SimpleNamespace(bmg=_DeviceValue(), V_d=None, X_d=None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(inference_mode=lambda: nullcontext()),
    )
    monkeypatch.setattr(core, "_make_loader", lambda _smiles, _batch_size: [batch])
    monkeypatch.setattr(core, "_device", lambda: "cpu")

    state = {
        "chemprop_model": Model(),
        "chemprop_scalers": {
            "endpoints": ["boiling_point"],
            "means": [197.1],
            "stds": [72.47],
        },
        "endpoints": ["boiling_point"],
    }
    result = core._predict_chemprop(state, ["CCO"], batch_size=1000)

    assert result.tolist() == [[283.0]]


def _state() -> dict:
    return {
        "metadata": {
            "ppb": {
                "unit": "%",
                "condition": "human plasma",
                "physical_bounds": {"lower": 0.0, "upper": 100.0},
            }
        },
        "training_ranges": {
            "ppb": {
                "min": 0.0,
                "max": 100.0,
                "p01": 10.0,
                "p99": 95.0,
                "mean": 70.0,
                "std": 20.0,
                "n": 7338,
            }
        },
    }


def test_physically_impossible_member_is_excluded_from_consensus() -> None:
    result = core._summarize_endpoint(
        _state(),
        "ppb",
        {"XGBoost": 80.0, "Chemprop-MTL": 2230.7},
    )

    assert result["mean"] == 80.0
    assert result["n_models"] == 1
    assert result["n_models_total"] == 2
    assert result["per_model"] == {"XGBoost": 80.0}
    assert result["excluded_models"]["Chemprop-MTL"]["reason"] == (
        "outside_physical_bounds"
    )
    assert result["response_domain"]["pass"] is True
    assert result["out_of_training_range"] is False


def test_consensus_outside_training_p01_p99_is_flagged_not_discarded() -> None:
    result = core._summarize_endpoint(
        _state(),
        "ppb",
        {"XGBoost": 98.0, "Chemprop-MTL": 96.0},
    )

    assert result["mean"] == 97.0
    assert result["n_models"] == 2
    assert result["response_domain"] == {
        "method": "training_p01_p99",
        "available": True,
        "lower": 10.0,
        "upper": 95.0,
        "training_min": 0.0,
        "training_max": 100.0,
        "n_train": 7338,
        "pass": False,
    }
    assert result["out_of_training_range"] is True
