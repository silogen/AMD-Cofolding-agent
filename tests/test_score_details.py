# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-molecule service details in ``score_molecules`` results."""

from __future__ import annotations

import math

import pytest

scoring = pytest.importorskip("reinvent4_mcp.tools.scoring")


def test_boltz_confidence_affinity_and_structure_are_kept():
    md = {
        "affinity_pred_value": [-1.2, 0.4],
        "affinity_probability_binary": [0.8, 0.3],
        "confidence_score": [0.91, 0.42],
        "ligand_iptm": [0.77, 0.31],
        "structure_path": ["/shared/a.pdb", "/shared/b.pdb"],
        "msa_note": ["MSA-free", "MSA-free"],
    }
    assert scoring._component_details(md, 1) == {
        "affinity_pred_value": 0.4,
        "affinity_probability_binary": 0.3,
        "confidence_score": 0.42,
        "ligand_iptm": 0.31,
        "structure_path": "/shared/b.pdb",
    }


def test_posebusters_names_the_failed_checks():
    md = {
        "n_checks_passed": [20, 18],
        "n_checks_total": [20, 20],
        "bond_lengths": [True, False],
        "internal_steric_clash": [True, False],
        "tetrahedral_chirality": [True, None],
    }
    assert scoring._component_details(md, 0) == {"n_checks_passed": 20, "n_checks_total": 20}
    assert scoring._component_details(md, 1)["failed_checks"] == ["bond_lengths", "internal_steric_clash"]


def test_missing_and_nan_values_are_left_out():
    md = {"std": [math.nan], "ad_pass": [None], "confidence_score": []}
    assert scoring._component_details(md, 0) == {}
