"""Attach fragments to interactions and build grouped triples.

Each interaction whose ligand atom belongs to a functional-group (or, as
a fallback, ring-system) fragment is annotated with that fragment name.
Grouped triples collapse equivalent contacts (same fragment, residue,
moiety, interaction subtype, direction) into one row with a count and
distance / angle ranges, matching the ``HBondDonor x2  d=3.0-3.3A`` style.
"""

from __future__ import annotations

from collections import defaultdict

from interaction_profiler.schemas import Composition, Interaction, Triple


def _atom_to_fragment(composition: Composition) -> dict[int, str]:
    """Map ligand atom index -> representative fragment canonical name.

    Prefer the smallest functional-group fragment covering the atom
    (most specific); fall back to ring_system, then scaffold.
    """
    fg: dict[int, tuple[int, str]] = {}
    ring: dict[int, str] = {}
    scaf: dict[int, str] = {}
    for frag in composition.fragments:
        size = len(frag.atom_indices)
        for idx in frag.atom_indices:
            if frag.level == "functional_group":
                cur = fg.get(idx)
                if cur is None or size < cur[0]:
                    fg[idx] = (size, frag.canonical_name)
            elif frag.level == "ring_system":
                ring.setdefault(idx, frag.canonical_name)
            elif frag.level == "scaffold":
                scaf.setdefault(idx, frag.canonical_name)

    out: dict[int, str] = {}
    all_idx = set(fg) | set(ring) | set(scaf)
    for idx in all_idx:
        if idx in fg:
            out[idx] = fg[idx][1]
        elif idx in ring:
            out[idx] = ring[idx]
        else:
            out[idx] = scaf[idx]
    return out


def attach_fragments(interactions: list[Interaction], composition: Composition) -> None:
    """Set ``interaction.fragment`` in place from the composition."""
    mapping = _atom_to_fragment(composition)
    for inter in interactions:
        if inter.ligand_atom_idx is not None and inter.ligand_atom_idx >= 0:
            inter.fragment = mapping.get(inter.ligand_atom_idx)


def build_triples(interactions: list[Interaction]) -> list[Triple]:
    """Group interactions attributable to a fragment into triples."""
    groups: dict[tuple, list[Interaction]] = defaultdict(list)
    for inter in interactions:
        if not inter.fragment:
            continue
        key = (
            inter.fragment,
            inter.receptor_resname,
            inter.receptor_resnum,
            inter.receptor_chain,
            inter.receptor_moiety,
            inter.interaction_type,
            inter.subtype,
            inter.direction,
        )
        groups[key].append(inter)

    triples: list[Triple] = []
    for key, members in groups.items():
        (frag, resname, resnum, chain, moiety, itype, subtype, direction) = key
        dists = [m.distance for m in members if m.distance is not None]
        angles = [m.angle for m in members if m.angle is not None]
        n_conf = sum(1 for m in members if m.confidence == "confident")
        label = f"{resname}{'' if resnum is None else resnum}"
        if chain:
            label = f"{label}.{chain}"
        triples.append(
            Triple(
                fragment=frag,
                residue_label=label,
                residue_name=resname,
                residue_number=resnum,
                chain=chain,
                moiety=moiety,
                interaction_type=itype,
                subtype=subtype,
                direction=direction,
                count=len(members),
                distance_min=(round(min(dists), 2) if dists else None),
                distance_max=(round(max(dists), 2) if dists else None),
                angle_min=(round(min(angles), 1) if angles else None),
                angle_max=(round(max(angles), 1) if angles else None),
                n_confident=n_conf,
                n_uncertain=len(members) - n_conf,
            )
        )

    triples.sort(key=lambda t: (t.n_confident == 0, -(t.count), t.residue_label))
    return triples


__all__ = ["attach_fragments", "build_triples"]
