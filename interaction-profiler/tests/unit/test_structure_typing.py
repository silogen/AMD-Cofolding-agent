# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for residue-kind / backbone classification (RNA vs DNA)."""

from __future__ import annotations

from interaction_profiler.io.structure import is_backbone, residue_kind


def test_residue_kind_protein():
    assert residue_kind("ALA") == "protein"
    assert residue_kind("HIP") == "protein"


def test_residue_kind_rna_vs_dna():
    assert residue_kind("A") == "rna"
    assert residue_kind("U") == "rna"
    assert residue_kind("G") == "rna"
    assert residue_kind("DA") == "dna"
    assert residue_kind("DT") == "dna"
    assert residue_kind("DG") == "dna"


def test_residue_kind_other():
    assert residue_kind("ZN") == "ion"
    assert residue_kind("HOH") == "water"
    assert residue_kind("LIG") == "other"


def test_backbone_names():
    assert is_backbone("protein", "CA")
    assert is_backbone("protein", "N")
    assert not is_backbone("protein", "CB")
    # nucleic backbone (phosphate/sugar) for both rna and dna
    assert is_backbone("rna", "P")
    assert is_backbone("dna", "O5'")
    assert is_backbone("rna", "O2'")   # ribose 2'-OH
    assert not is_backbone("dna", "N1")  # base atom is sidechain
