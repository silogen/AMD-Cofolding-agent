"""Unit tests for input-mode classification and SMILES↔ligand matching helpers."""

from __future__ import annotations

from interaction_profiler.io.batch_input import MoleculeInput
from interaction_profiler.prep.ligand import (
    formula_of_smiles,
    inchikey_of_smiles,
    inchikey_skeleton,
    smiles_advisories,
    validate_smiles,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRINATE = "CC(=O)Oc1ccccc1C(=O)[O-]"


def test_formula_of_smiles() -> None:
    # aspirin is C9H8O4 -> heavy-atom formula C9 O4
    assert formula_of_smiles(ASPIRIN) == {"C": 9, "O": 4}
    assert formula_of_smiles("bad!!smiles") is None
    assert formula_of_smiles(None) is None


def test_inchikey_skeleton_ignores_protonation() -> None:
    k_acid = inchikey_of_smiles(ASPIRIN)
    k_anion = inchikey_of_smiles(ASPIRINATE)
    assert k_acid and k_anion
    # Full keys differ (charge), but the connectivity skeleton (first block) matches.
    assert inchikey_skeleton(k_acid) == inchikey_skeleton(k_anion)
    assert inchikey_skeleton(k_acid) != inchikey_skeleton(inchikey_of_smiles("c1ccccc1"))


def test_inchikey_of_smiles_bad_input() -> None:
    assert inchikey_of_smiles(None) is None
    assert inchikey_of_smiles("") is None


def test_validate_smiles() -> None:
    ok, key = validate_smiles(ASPIRIN)
    assert ok and key and len(key.split("-")[0]) == 14
    # unparseable / empty
    assert validate_smiles("C1CC")[0] is False       # unclosed ring
    assert validate_smiles("not_a_smiles")[0] is False
    assert validate_smiles(None) == (False, None)


def test_smiles_advisories() -> None:
    # clean single organic ligand -> no advisories
    assert smiles_advisories(ASPIRIN) == []
    # multi-component salt
    assert any("multi-component" in a for a in smiles_advisories("CC(=O)[O-].[Na+]"))
    # wildcard / dummy atom
    assert any("wildcard" in a for a in smiles_advisories("*c1ccccc1"))
    # inorganic (no carbon)
    assert any("no carbon" in a for a in smiles_advisories("O"))
    # invalid / empty -> no advisories (handled elsewhere)
    assert smiles_advisories("C1CC") == []
    assert smiles_advisories(None) == []


def test_input_mode_classification() -> None:
    from interaction_profiler.core import _input_mode

    assert _input_mode(MoleculeInput(name="a", smiles=ASPIRIN)) == "smiles_only"
    # A non-existent structure path -> has_structure is False -> smiles_only.
    assert _input_mode(MoleculeInput(name="b", smiles=ASPIRIN, structure="/no/such.pdb")) == "smiles_only"


def test_strict_match_ignores_smiles_only() -> None:
    # strict_ligand_match must not error a taxonomy-only input even when the
    # prepared form's InChIKey skeleton differs from the raw SMILES (e.g. a salt).
    from interaction_profiler.core import profile_molecule_all

    r = profile_molecule_all(
        MoleculeInput(name="salt", smiles="CC(=O)[O-].[Na+]"), strict_ligand_match=True
    )[0]
    assert r.metadata.input_mode == "smiles_only"
    assert r.error is None
    # advisory about the multi-component input is still surfaced
    assert any("multi-component" in w for w in r.preparation.warnings)
    # ...but no misleading "wrong ligand / template" warning in taxonomy mode
    assert not any("does not match the SMILES template" in w for w in r.preparation.warnings)
