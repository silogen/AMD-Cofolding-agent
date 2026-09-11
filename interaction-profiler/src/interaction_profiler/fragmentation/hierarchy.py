"""Build a hierarchical medchem composition for one ligand.

Combines three levels, all preserving atom indices into the ligand Mol:

- ``functional_group``: AccFG hierarchical taxonomy + curated SMARTS.
- ``ring_system``: each fused-ring system (SSSR grouped by shared atoms).
- ``scaffold``: the Bemis-Murcko scaffold atoms.

Also reports ``coverage`` -- the fraction of heavy atoms that belong to
at least one functional-group or ring-system fragment -- so a downstream
consumer can tell whether the taxonomy tiled the molecule.
"""

from __future__ import annotations

import logging
from typing import Any

from interaction_profiler.fragmentation.accfg_adapter import AccFGAdapter
from interaction_profiler.fragmentation.medchem_library import compiled_patterns
from interaction_profiler.schemas import Composition, Fragment
from interaction_profiler.vocabulary import canonical_fragment

logger = logging.getLogger(__name__)

_ACCFG_SINGLETON: AccFGAdapter | None = None


def _accfg() -> AccFGAdapter | None:
    global _ACCFG_SINGLETON
    if _ACCFG_SINGLETON is None:
        try:
            _ACCFG_SINGLETON = AccFGAdapter()
        except Exception as exc:
            logger.warning("AccFG unavailable, falling back to SMARTS only: %s", exc)
            _ACCFG_SINGLETON = None
    return _ACCFG_SINGLETON


def _fragment_smiles(mol: Any, atom_indices: list[int]) -> str | None:
    try:
        from rdkit import Chem

        return Chem.MolFragmentToSmiles(mol, atomsToUse=list(atom_indices))
    except Exception:
        return None


def _heavy_atom_indices(mol: Any) -> list[int]:
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def _smarts_fragments(mol: Any) -> list[Fragment]:
    out: list[Fragment] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for name, smarts, patt in compiled_patterns():
        try:
            matches = mol.GetSubstructMatches(patt, uniquify=True)
        except Exception:
            continue
        for match in matches:
            key = (name, tuple(sorted(match)))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                Fragment(
                    name=name,
                    canonical_name=canonical_fragment(name),
                    level="functional_group",
                    atom_indices=[int(i) for i in match],
                    smarts=smarts,
                )
            )
    return out


def _ring_systems(mol: Any) -> list[Fragment]:
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    # Merge rings that share atoms (fused systems).
    merged: list[set[int]] = []
    for ring in rings:
        placed = False
        for m in merged:
            if m & ring:
                m |= ring
                placed = True
                break
        if not placed:
            merged.append(set(ring))
    # Second pass to catch transitive fusions.
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if merged[i] & merged[j]:
                    merged[i] |= merged[j]
                    merged.pop(j)
                    changed = True
                    break
            if changed:
                break

    out: list[Fragment] = []
    for system in merged:
        idxs = sorted(int(i) for i in system)
        out.append(
            Fragment(
                name="ring_system",
                canonical_name="ring_system",
                level="ring_system",
                atom_indices=idxs,
                smiles=_fragment_smiles(mol, idxs),
            )
        )
    return out


def _scaffold(mol: Any) -> Fragment | None:
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return None
        match = mol.GetSubstructMatch(scaf)
        if not match:
            return None
        idxs = sorted(int(i) for i in match)
        return Fragment(
            name="murcko_scaffold",
            canonical_name="murcko_scaffold",
            level="scaffold",
            atom_indices=idxs,
            smiles=_fragment_smiles(mol, idxs),
        )
    except Exception as exc:
        logger.debug("scaffold perception failed: %s", exc)
        return None


def build_composition(mol: Any) -> Composition:
    """Return the hierarchical composition for a ligand RDKit Mol."""
    fragments: list[Fragment] = []

    accfg = _accfg()
    if accfg is not None:
        try:
            fragments.extend(accfg.fragment(mol))
        except Exception as exc:
            logger.debug("AccFG fragmentation failed: %s", exc)

    fragments.extend(_smarts_fragments(mol))
    ring_frags = _ring_systems(mol)
    fragments.extend(ring_frags)
    scaf = _scaffold(mol)
    if scaf is not None:
        fragments.append(scaf)

    fragments = _dedupe(fragments)

    # Ensure every fragment carries its isolated SMILES (AccFG functional
    # groups come with neither SMARTS nor SMILES otherwise), so the JSON is
    # self-verifiable.
    for f in fragments:
        if not f.smiles:
            f.smiles = _fragment_smiles(mol, f.atom_indices)

    heavy = set(_heavy_atom_indices(mol))
    covered: set[int] = set()
    for f in fragments:
        if f.level in ("functional_group", "ring_system"):
            covered.update(i for i in f.atom_indices if i in heavy)
    coverage = (len(covered) / len(heavy)) if heavy else 0.0

    return Composition(
        fragments=fragments,
        coverage=round(coverage, 4),
        n_heavy_atoms=len(heavy),
    )


def _dedupe(fragments: list[Fragment]) -> list[Fragment]:
    """Drop duplicate fragments (same canonical name + same atom set)."""
    seen: set[tuple[str, str, tuple[int, ...]]] = set()
    out: list[Fragment] = []
    for f in fragments:
        key = (f.level, f.canonical_name, tuple(sorted(f.atom_indices)))
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


__all__ = ["build_composition"]
