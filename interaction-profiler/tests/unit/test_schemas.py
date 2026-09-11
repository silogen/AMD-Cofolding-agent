"""Unit tests for the v2 schemas and flattening."""

from __future__ import annotations

from interaction_profiler.schemas import (
    BatchResult,
    Composition,
    Fragment,
    Interaction,
    MoleculeMetadata,
    MoleculeResult,
)


def test_interaction_residue_label():
    it = Interaction(
        interaction_type="hbond", ligand_atom_idx=3, receptor_chain="A",
        receptor_resname="ILE", receptor_resnum=50,
    )
    assert it.residue_label == "ILE50.A"


def test_batch_result_json_roundtrip_and_rows():
    mol = MoleculeResult(
        metadata=MoleculeMetadata(name="X", receptor="protein", chains=["A"]),
        composition=Composition(
            fragments=[Fragment(name="benzene", canonical_name="benzene",
                                 level="ring_system", atom_indices=[0, 1, 2, 3, 4, 5])],
            coverage=1.0, n_heavy_atoms=6,
        ),
        interactions=[
            Interaction(interaction_type="hbond", subtype="hbond_donor", ligand_atom_idx=1,
                        receptor_chain="A", receptor_resname="ASP", receptor_resnum=25,
                        distance=3.0, angle=160.0, direction="ligand_donor",
                        fragment="hydroxyl", confidence="confident"),
        ],
    )
    batch = BatchResult(batch_id="b", n_molecules=1, molecules=[mol])
    js = batch.model_dump_json()
    reparsed = BatchResult.model_validate_json(js)
    assert reparsed == batch

    rows = batch.to_rows()
    assert len(rows) == 1
    assert rows[0]["interaction_type"] == "hbond"
    assert rows[0]["residue"] == "ASP25.A"
    assert rows[0]["fragment"] == "hydroxyl"


def test_empty_interactions_yield_placeholder_row():
    mol = MoleculeResult(metadata=MoleculeMetadata(name="taxonomy_only"))
    batch = BatchResult(batch_id="b", n_molecules=1, molecules=[mol])
    rows = batch.to_rows()
    assert rows and rows[0]["interaction_type"] is None
