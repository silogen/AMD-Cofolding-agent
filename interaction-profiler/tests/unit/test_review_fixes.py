"""Regression tests for the PR #4 (02-structural-intelligence) review fixes.

Each test pins a specific bug called out in review so it cannot silently return:

* strain.py      — MMFF94 energies must use the same atom set (relaxed pose → ~0).
* resolver.py    — caller-supplied chains validated (no phantom entities), case
                   normalised, buffer/hetero chains not mistyped as protein.
* enricher.py    — annotation lookups are chain-aware (no cross-chain collision).
* interface.py   — fixed-column PDB overflow is surfaced, not silent.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# strain.py — same atom-set energies
# ---------------------------------------------------------------------------

rdkit = pytest.importorskip("rdkit")


def _embedded(smiles: str, seed: int = 0xC0FFEE):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def test_relaxed_pose_has_near_zero_strain():
    """A fully-relaxed conformer must have ~0 strain.

    Under the old code (bound energy WITH hydrogens minus optimised energy
    WITHOUT hydrogens) a relaxed pose showed a large spurious offset dominated
    by the hydrogen count.  With a consistent atom set it must be ~0.
    """
    from interaction_profiler.metrics.strain import compute_strain_from_mol

    mol = _embedded("CCCCCc1ccccc1OC(=O)C")
    res = compute_strain_from_mol(mol)

    assert res.error is None
    assert abs(res.strain_kcal_mol) < 1.0  # ≈ 0; old bug gave tens of kcal/mol


def test_strained_pose_exceeds_relaxed_pose():
    from rdkit.Chem import rdMolTransforms

    from interaction_profiler.metrics.strain import compute_strain_from_mol

    mol = _embedded("CCCCCc1ccccc1OC(=O)C")
    relaxed = compute_strain_from_mol(mol).strain_kcal_mol

    rdMolTransforms.SetDihedralDeg(mol.GetConformer(), 0, 1, 2, 3, 20.0)
    strained = compute_strain_from_mol(mol).strain_kcal_mol

    assert strained > relaxed


def test_strain_adds_missing_hydrogens():
    """Input without explicit H must still work (H added internally)."""
    from rdkit import Chem

    from interaction_profiler.metrics.strain import compute_strain_from_mol

    mol_h = _embedded("c1ccccc1")
    mol_no_h = Chem.RemoveHs(mol_h)
    res = compute_strain_from_mol(mol_no_h)

    assert res.error is None
    assert res.strain_kcal_mol is not None


# ---------------------------------------------------------------------------
# resolver.py — chain validation / typing
# ---------------------------------------------------------------------------

class _FakeStruct:
    def __init__(self, ct, ligands=None):
        self._ct = ct
        self._ligands = ligands or []

    def chain_types(self):
        return dict(self._ct)

    def detect_ligands(self, resname=None):
        return self._ligands


def test_bogus_target_chain_does_not_create_phantom_entity():
    from interaction_profiler.entity.resolver import resolve_entities

    struct = _FakeStruct({"A": "protein", "B": "protein"})
    entities, warnings = resolve_entities(
        struct, binder_chains=["A"], target_chains=["Z"],
        ligand_comp_id=None, mode="binder_target",
    )

    assert all("Z" not in e.chain_ids for e in entities)
    assert any("Z" in w and "not found" in w for w in warnings)


def test_chain_type_case_is_normalised():
    from interaction_profiler.entity.resolver import resolve_entities

    struct = _FakeStruct({"A": "PROTEIN", "L": "LIGAND"})
    entities, _ = resolve_entities(
        struct, binder_chains=["L"], target_chains=["A"],
        ligand_comp_id=None, mode="binder_target",
    )

    roles = {e.role.value for e in entities}
    assert "small_molecule" in roles
    assert "protein" in roles


def test_buffer_target_is_not_typed_as_protein():
    from interaction_profiler.entity.resolver import resolve_entities

    struct = _FakeStruct({"A": "protein", "X": "buffer"})
    entities, _ = resolve_entities(
        struct, binder_chains=["A"], target_chains=["X"],
        ligand_comp_id=None, mode="binder_target",
    )

    target = next(e for e in entities if e.label == "target")
    assert target.role.value == "small_molecule"


def test_unknown_chain_type_warns_instead_of_silent_default():
    from interaction_profiler.entity.resolver import _infer_entity_role_from_chain_type

    warnings: list[str] = []
    role = _infer_entity_role_from_chain_type("quark", warnings=warnings, context="target")
    assert role.value == "protein"
    assert any("quark" in w for w in warnings)


# ---------------------------------------------------------------------------
# enricher.py — chain-aware annotation lookup
# ---------------------------------------------------------------------------

def test_annotation_lookup_prefers_chain_aware_key():
    from interaction_profiler.interactions.enricher import _lookup_annotation

    amap = {("A", 50): {"feature_type": "A50"}, ("B", 50): {"feature_type": "B50"}}
    assert _lookup_annotation(amap, "A", 50)["feature_type"] == "A50"
    assert _lookup_annotation(amap, "B", 50)["feature_type"] == "B50"


def test_annotation_lookup_falls_back_to_resnum_only():
    from interaction_profiler.interactions.enricher import _lookup_annotation

    amap = {50: {"feature_type": "R50"}}
    assert _lookup_annotation(amap, "A", 50)["feature_type"] == "R50"
    assert _lookup_annotation(amap, "A", None) is None
    assert _lookup_annotation(None, "A", 50) is None


# ---------------------------------------------------------------------------
# interface.py — fixed-column PDB overflow is surfaced
# ---------------------------------------------------------------------------

class _FakeAtoms:
    def __init__(self, chain_ids):
        self.chainIDs = chain_ids


class _FakeResidues:
    def __init__(self, resids):
        self.resids = resids


class _FakeAtomGroup:
    def __init__(self, n_atoms, resids, chain_ids):
        self._n = n_atoms
        self.residues = _FakeResidues(resids)
        self.atoms = _FakeAtoms(chain_ids)

    def __len__(self):
        return self._n


def test_detect_pdb_overflow_flags_multichar_chains():
    from interaction_profiler.metrics.interface import _detect_pdb_overflow

    ag = _FakeAtomGroup(100, [1, 2, 3], ["AAA", "B"])
    warns = _detect_pdb_overflow(ag, "binder")
    assert any("multi-character chain" in w for w in warns)


def test_detect_pdb_overflow_flags_large_counts():
    from interaction_profiler.metrics.interface import _detect_pdb_overflow

    ag = _FakeAtomGroup(100_001, [1, 20_000], ["A"])
    warns = _detect_pdb_overflow(ag, "target")
    assert any("99,999" in w for w in warns)
    assert any("4-column" in w for w in warns)


def test_detect_pdb_overflow_clean_selection_has_no_warnings():
    from interaction_profiler.metrics.interface import _detect_pdb_overflow

    ag = _FakeAtomGroup(2000, [10, 20, 30], ["A", "B"])
    assert _detect_pdb_overflow(ag, "binder") == []
