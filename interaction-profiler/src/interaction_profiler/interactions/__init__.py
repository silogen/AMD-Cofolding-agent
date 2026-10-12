# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Custom geometric interaction detection (v2).

The v1 ProLIF backend is gone: ProLIF only reliably yielded
``vdw_contact`` here because it needs a fully hydrogenated, bond-order-
correct ligand and does not expose backbone/sidechain, metal, or
water-bridge annotations. The geometric detector in
:mod:`interaction_profiler.interactions.detector` computes all of those
directly.
"""

from __future__ import annotations

from interaction_profiler.interactions.detector import detect_interactions

__all__ = ["detect_interactions"]
