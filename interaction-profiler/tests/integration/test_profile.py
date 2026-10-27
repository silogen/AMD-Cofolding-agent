# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Integration test: full pipeline on the bundled Boltz-2 example.

Uses the single in-repo structure ``examples/1OIU_boltz.pdb`` (CDK2 +
sulfonamide inhibitor, predicted by Boltz-2, ligand residue ``LIG``).
Auto-skipped when the heavy stack or the example file is missing, so a
fresh checkout still passes the unit tests without the full stack.
"""

from __future__ import annotations

from collections import Counter

import pytest

from interaction_profiler.core import profile_molecule
from interaction_profiler.io.batch_input import MoleculeInput

pytestmark = pytest.mark.integration

N76_SMILES = "c1cc(cc(c1)S(=O)(=O)N)Nc2nc3c(c(n2)OCC4CCCCC4)nc[nH]3"


def _profile(example_pdb, *, smiles=N76_SMILES, drop_uncertain=False, **kw):
    inp = MoleculeInput(name="1OIU_boltz", structure=str(example_pdb),
                        ligand_resname="LIG", smiles=smiles, source="cofold")
    return profile_molecule(inp, ph_default=7.4, radius=8.0,
                            max_interactions=30, drop_uncertain=drop_uncertain, **kw)


def test_structure_pipeline_typed_interactions(example_pdb, requires_full_stack):
    res = _profile(example_pdb)
    assert res.error is None
    assert res.metadata.input_mode == "structure+smiles"
    assert res.metadata.receptor == "protein"
    types = Counter(i.interaction_type for i in res.interactions)
    # goes beyond vdW: both H-bonds and hydrophobic contacts are present
    assert types.get("hbond", 0) >= 1
    assert types.get("hydrophobic", 0) >= 1
    # every triple is fully annotated
    for t in res.triples:
        assert t.fragment and t.residue_name
        assert t.moiety in (None, "sidechain", "backbone", "ion", "water")


def test_template_applied_and_inchikey_matches(example_pdb, requires_full_stack):
    # The SMILES template transfers onto the predicted coordinates and the
    # modeled ligand matches it by InChIKey skeleton.
    res = _profile(example_pdb)
    prep = res.preparation
    assert prep.bond_assignment == "template"
    assert prep.inchikey_match is True
    # aromatic system round-trips (purine present)
    assert "c" in (res.metadata.ligand_smiles or "")


def test_strict_ligand_match_rejects_wrong_smiles(example_pdb, requires_full_stack):
    # A wrong template under strict matching is rejected with a clear error.
    res = _profile(example_pdb, smiles="CC(=O)Oc1ccccc1C(=O)O", strict_ligand_match=True)
    assert res.error and "does not match" in res.error
    assert res.metadata.n_interactions == 0


def test_interaction_cap_respected(example_pdb, requires_full_stack):
    res = _profile(example_pdb)
    assert res.metadata.n_interactions <= 30
