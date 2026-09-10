"""Unit tests for interaction ranking + capping."""

from __future__ import annotations

from interaction_profiler.interactions.ranking import rank_and_cap, rank_interactions
from interaction_profiler.schemas import Interaction


def _mk(itype, dist, conf="confident"):
    return Interaction(interaction_type=itype, ligand_atom_idx=0, receptor_chain="A",
                       receptor_resname="ALA", receptor_resnum=1, distance=dist, confidence=conf)


def test_priority_order():
    inters = [
        _mk("vdw_contact", 3.0),
        _mk("hydrophobic", 3.0),
        _mk("hbond", 3.5),
        _mk("metal", 2.2),
        _mk("ionic", 4.0),
    ]
    ranked = rank_interactions(inters)
    assert [i.interaction_type for i in ranked] == [
        "metal", "ionic", "hbond", "hydrophobic", "vdw_contact",
    ]


def test_within_tier_confidence_then_distance():
    inters = [
        _mk("hbond", 3.4, "uncertain"),
        _mk("hbond", 3.3, "confident"),
        _mk("hbond", 3.1, "confident"),
    ]
    ranked = rank_interactions(inters)
    # confident ones first, shortest distance first
    assert ranked[0].distance == 3.1 and ranked[0].confidence == "confident"
    assert ranked[1].distance == 3.3 and ranked[1].confidence == "confident"
    assert ranked[2].confidence == "uncertain"


def test_cap_keeps_top_n():
    inters = [_mk("hydrophobic", d) for d in (4.0, 3.9, 3.8)] + [_mk("metal", 2.2), _mk("hbond", 3.0)]
    capped = rank_and_cap(inters, 2)
    assert len(capped) == 2
    assert [i.interaction_type for i in capped] == ["metal", "hbond"]


def test_no_cap_when_zero_or_none():
    inters = [_mk("hbond", 3.0), _mk("vdw_contact", 3.0)]
    assert len(rank_and_cap(inters, 0)) == 2
    assert len(rank_and_cap(inters, None)) == 2
