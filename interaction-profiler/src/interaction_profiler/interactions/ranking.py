# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rank interactions by importance and cap to a maximum count.

Priority follows approximate interaction energetics (rank 0 = kept
first): covalent > metal coordination > ionic/salt-bridge > H-bond >
halogen bond > pi-cation / cation-pi > pi-stacking > water-bridge >
hydrophobic > vdW. Within a priority tier, confident contacts come
before uncertain ones, then shorter distance first.
"""

from __future__ import annotations

from interaction_profiler.schemas import Interaction

INTERACTION_PRIORITY: dict[str, int] = {
    "covalent": 0,
    "metal": 1,
    "ionic": 2,
    "hbond": 3,
    "halogen_bond": 4,
    "chalcogen_bond": 4,
    "pi_cation": 5,
    "cation_pi": 5,
    "pi_stacking": 6,
    "amide_pi": 6,
    "chalcogen_pi": 6,
    "water_bridge": 7,
    "hydrophobic": 8,
    "ch_pi": 8,
    "vdw_contact": 9,
}

_UNKNOWN_PRIORITY = 50


def interaction_sort_key(inter: Interaction) -> tuple[int, int, float]:
    prio = INTERACTION_PRIORITY.get(inter.interaction_type, _UNKNOWN_PRIORITY)
    conf = 0 if inter.confidence == "confident" else 1
    dist = inter.distance if inter.distance is not None else float("inf")
    return (prio, conf, dist)


def rank_interactions(interactions: list[Interaction]) -> list[Interaction]:
    """Return interactions sorted by (priority, confidence, distance)."""
    return sorted(interactions, key=interaction_sort_key)


def rank_and_cap(interactions: list[Interaction], max_n: int | None) -> list[Interaction]:
    """Rank interactions and keep at most ``max_n`` (None/<=0 = no cap)."""
    ranked = rank_interactions(interactions)
    if max_n is None or max_n <= 0:
        return ranked
    return ranked[:max_n]


__all__ = ["INTERACTION_PRIORITY", "interaction_sort_key", "rank_interactions", "rank_and_cap"]
