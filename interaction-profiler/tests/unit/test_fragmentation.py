"""Unit tests for hierarchical fragmentation (requires RDKit + AccFG)."""

from __future__ import annotations

import pytest

rdkit = pytest.importorskip("rdkit")
pytest.importorskip("accfg")

from rdkit import Chem  # noqa: E402

from interaction_profiler.fragmentation import build_composition  # noqa: E402


def test_aspirin_composition():
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    comp = build_composition(mol)
    names = {f.canonical_name for f in comp.fragments}
    levels = {f.level for f in comp.fragments}
    # should find an aromatic ring system and an ester/carboxylic group
    assert "ring_system" in levels
    assert any("benzene" in n or "ring" in n for n in names)
    assert comp.coverage > 0.5
    assert comp.n_heavy_atoms == mol.GetNumAtoms()


def test_no_duplicate_fragments():
    mol = Chem.MolFromSmiles("c1ccccc1")
    comp = build_composition(mol)
    keys = [(f.level, f.canonical_name, tuple(sorted(f.atom_indices))) for f in comp.fragments]
    assert len(keys) == len(set(keys))


def test_every_fragment_has_smiles():
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    comp = build_composition(mol)
    assert comp.fragments
    for f in comp.fragments:
        assert f.smiles, f"fragment {f.canonical_name} missing smiles"
