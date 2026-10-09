# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RetroDMTA project-endpoint predictor: per-target, per-temporal-window models (XGBoost +
Chemprop-MTL + CheMeleon-MTL) with ensemble/single-method selection."""
from retrodmta_predictor import core

__all__ = ["core"]
__version__ = "1.0.0"
