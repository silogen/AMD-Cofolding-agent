# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Thin canonical vocabulary layer.

v1 is near-identity: fragment names are lowercased and underscore-joined;
interaction types are normalised to a short canonical set covering the
ProLIF default interaction list plus halogen / metal / covalent.

The seam is here so later versions can map SmartChemist or PLIP outputs
through the same module without invalidating earlier outputs that were
keyed on a canonical label.
"""

from __future__ import annotations

# ProLIF interaction class name -> canonical type.
# Tool-native names appear on the left; aliases keep the canonical form
# stable across ProLIF versions (which occasionally rename classes).
_INTERACTION_ALIASES: dict[str, str] = {
    "HBDonor": "hbond_donor",
    "HBAcceptor": "hbond_acceptor",
    "PiStacking": "pi_stacking",
    "FaceToFace": "pi_stacking",
    "EdgeToFace": "pi_stacking",
    "PiCation": "pi_cation",
    "CationPi": "cation_pi",
    "Anionic": "ionic_neg",
    "Cationic": "ionic_pos",
    "Hydrophobic": "hydrophobic",
    "MetalDonor": "metal",
    "MetalAcceptor": "metal",
    "XBDonor": "halogen_bond",
    "XBAcceptor": "halogen_bond",
    "VdWContact": "vdw_contact",
    "VanDerWaals": "vdw_contact",
    # already-canonical pass-throughs
    "hbond_donor": "hbond_donor",
    "hbond_acceptor": "hbond_acceptor",
    "pi_stack": "pi_stacking",
    "pi_cation": "pi_cation",
    "cation_pi": "cation_pi",
    "ionic_pos": "ionic_pos",
    "ionic_neg": "ionic_neg",
    "hydrophobic": "hydrophobic",
    "metal": "metal",
    "halogen_bond": "halogen_bond",
    "vdw_contact": "vdw_contact",
    "covalent": "covalent",
}


def canonical_interaction(name: str) -> str:
    """Return the canonical interaction-type label for a tool-native name.

    Unknown names fall through to a snake-cased version so we never lose
    signal even when a new ProLIF interaction class lands.
    """
    if name in _INTERACTION_ALIASES:
        return _INTERACTION_ALIASES[name]
    return name.lower().replace(" ", "_").replace("-", "_")


def canonical_fragment(name: str) -> str:
    """Return the canonical fragment label for a tool-native name.

    AccFG already gives human-readable names (e.g. ``"Primary amide"``,
    ``"Triazene"``, ``"imidazole"``) so v1 just lowercases and joins with
    underscores. This is the seam where SmartChemist / EFG names will be
    normalised when those backends are added.
    """
    return name.strip().lower().replace(" ", "_").replace("-", "_")


__all__ = ["canonical_interaction", "canonical_fragment"]
