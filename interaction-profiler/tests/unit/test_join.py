# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for fragment attachment + grouped triple building."""

from __future__ import annotations

from interaction_profiler.join import attach_fragments, build_triples
from interaction_profiler.schemas import Composition, Fragment, Interaction


def _composition():
    return Composition(
        fragments=[
            Fragment(name="urea", canonical_name="urea", level="functional_group",
                     atom_indices=[10, 11, 12]),
            Fragment(name="ring", canonical_name="ring_system", level="ring_system",
                     atom_indices=[0, 1, 2, 3, 4, 5]),
        ],
        coverage=1.0, n_heavy_atoms=9,
    )


def test_attach_prefers_functional_group():
    comp = _composition()
    inters = [
        Interaction(interaction_type="hbond", ligand_atom_idx=11, receptor_chain="A",
                    receptor_resname="ILE", receptor_resnum=50),
        Interaction(interaction_type="hydrophobic", ligand_atom_idx=2, receptor_chain="A",
                    receptor_resname="VAL", receptor_resnum=32),
    ]
    attach_fragments(inters, comp)
    assert inters[0].fragment == "urea"
    assert inters[1].fragment == "ring_system"


def test_build_triples_groups_and_ranges():
    comp = _composition()
    inters = [
        Interaction(interaction_type="hbond", subtype="hbond_acceptor", ligand_atom_idx=11,
                    receptor_chain="A", receptor_resname="ILE", receptor_resnum=50,
                    receptor_moiety="backbone", direction="ligand_acceptor",
                    distance=3.2, angle=161.0, confidence="confident"),
        Interaction(interaction_type="hbond", subtype="hbond_acceptor", ligand_atom_idx=12,
                    receptor_chain="A", receptor_resname="ILE", receptor_resnum=50,
                    receptor_moiety="backbone", direction="ligand_acceptor",
                    distance=3.4, angle=150.0, confidence="uncertain"),
    ]
    attach_fragments(inters, comp)
    triples = build_triples(inters)
    assert len(triples) == 1
    t = triples[0]
    assert t.count == 2
    assert t.residue_label == "ILE50.A"
    assert t.distance_min == 3.2 and t.distance_max == 3.4
    assert t.angle_min == 150.0 and t.angle_max == 161.0
    assert t.n_confident == 1 and t.n_uncertain == 1


def test_interactions_without_fragment_are_not_triples():
    comp = _composition()
    inters = [Interaction(interaction_type="vdw_contact", ligand_atom_idx=99,
                          receptor_chain="A", receptor_resname="GLY", receptor_resnum=1)]
    attach_fragments(inters, comp)
    assert inters[0].fragment is None
    assert build_triples(inters) == []
