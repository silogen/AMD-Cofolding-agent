"""Additive registry loader and classification helpers.

Loads additives_registry.json at service startup (or from a custom path).
Exposes fast set-based lookups used by Stage 0 (HETATM initial classification)
and Stage 3 (retention policy enforcement).

Also provides the COFACTORS and METALS sets that parallel the interaction-
profiler's own vocabulary so both tools agree on entity roles.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in registry path (relative to this file — overrideable at runtime)
# ---------------------------------------------------------------------------

# Keep the registry inside the service tree so copied/packaged source remains
# self-contained.  The previous four-parent traversal escaped the block and
# silently produced an empty registry after integration.
_DEFAULT_REGISTRY = Path(__file__).parent.parent / "additives_registry.json"

# ---------------------------------------------------------------------------
# Known metal elements (single-atom HETATM)
# ---------------------------------------------------------------------------

METAL_ELEMENTS: frozenset[str] = frozenset({
    "ZN", "MG", "CA", "FE", "FE2", "MN", "CU", "CO", "NI", "MO", "W",
    "V", "CR", "PT", "AU", "AG", "PB", "HG", "CD", "IR", "SE",
})

# Heavy-atom phasing ions (also single-atom, removed by default)
PHASING_ELEMENTS: frozenset[str] = frozenset({
    "XE", "KR", "BR", "IOD", "CS", "PB", "HG", "AU", "PT", "IR", "AG",
    "BA", "SR", "TL", "SM", "GD",
})

# Crystallographic counter-ions (usually removed)
BUFFER_ION_COMP_IDS: frozenset[str] = frozenset({
    "NA", "K", "LI", "RB", "CL", "FLC", "NO3", "NH4",
})

# ---------------------------------------------------------------------------
# Built-in cofactor set (mirrors interaction-profiler COFACTORS + extras)
# ---------------------------------------------------------------------------

COFACTORS: frozenset[str] = frozenset({
    # nucleotides
    "GDP", "GTP", "GNP", "GSP", "ADP", "ATP", "ANP", "AGS", "ACP", "APC",
    "AMP", "CMP", "UMP", "UDP", "UTP", "TTP", "GCP", "AMPPNP",
    # redox / carriers
    "NAD", "NAI", "NAP", "NDP", "FAD", "FMN", "SAM", "SAH", "COA", "ACO",
    "TPP", "PLP", "BTN", "B12", "HEM", "HEC", "HEA", "DHE",
    # iron-sulfur
    "SF4", "FES", "FS3",
    # phospholipids + sugars in glycan detection (keep=false by default handled separately)
    "IHP", "IPD", "I3P",
    # common saccharides (handled by Stage 2 glycan detection)
    "NAG", "BMA", "MAN", "GAL", "GLC", "FUC", "SIA",
})

# ---------------------------------------------------------------------------
# Water residue names
# ---------------------------------------------------------------------------

WATER_RESNAMES: frozenset[str] = frozenset({
    "HOH", "WAT", "SOL", "TIP3", "TIP4", "TIP5", "T3P", "T4P", "DOD", "H2O",
})

# ---------------------------------------------------------------------------
# Standard amino acid residue names (including protonation state variants)
# ---------------------------------------------------------------------------

STANDARD_AA: frozenset[str] = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # protonation state variants (after Stage 6)
    "HID", "HIE", "HIP", "CYX", "CYM", "ASH", "GLH", "LYN", "ARN",
    # nonstandard but commonly mapped
    "MSE", "SEC", "PYL",
})

# Standard nucleotide residue names
DNA_RESNAMES: frozenset[str] = frozenset({
    "DA", "DT", "DG", "DC", "DU", "DI",
    "DA3", "DA5", "DT3", "DT5", "DG3", "DG5", "DC3", "DC5",
})
RNA_RESNAMES: frozenset[str] = frozenset({
    "A", "U", "G", "C", "I", "RA", "RU", "RG", "RC", "RI",
    "A3", "A5", "U3", "U5", "G3", "G5", "C3", "C5",
})
NUCLEIC_RESNAMES: frozenset[str] = DNA_RESNAMES | RNA_RESNAMES

# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------

class AdditivesRegistry:
    """In-memory additive registry loaded from JSON."""

    def __init__(self, registry_path: str | Path | None = None) -> None:
        path = Path(registry_path) if registry_path else _DEFAULT_REGISTRY
        self._entries: dict[str, dict[str, Any]] = {}
        self._remove_by_default: frozenset[str] = frozenset()
        self._load(path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Additive registry not found at %s; using empty registry", path)
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
            entries = data.get("entries", [])
            for e in entries:
                comp_id = e.get("comp_id", "").strip().upper()
                if comp_id:
                    self._entries[comp_id] = e
            self._remove_by_default = frozenset(
                cid for cid, e in self._entries.items()
                if e.get("remove_by_default", True)
            )
            logger.info("Loaded %d additive registry entries from %s", len(self._entries), path)
        except Exception:
            logger.exception("Failed to load additive registry from %s", path)

    def is_additive(self, comp_id: str) -> bool:
        return comp_id.strip().upper() in self._entries

    def should_remove(self, comp_id: str) -> bool:
        return comp_id.strip().upper() in self._remove_by_default

    def get_category(self, comp_id: str) -> str | None:
        entry = self._entries.get(comp_id.strip().upper())
        return entry.get("category") if entry else None

    def get_entry(self, comp_id: str) -> dict[str, Any] | None:
        return self._entries.get(comp_id.strip().upper())

    def all_comp_ids(self) -> frozenset[str]:
        return frozenset(self._entries)

    def remove_by_default_set(self) -> frozenset[str]:
        return self._remove_by_default


# ---------------------------------------------------------------------------
# Singleton (loaded once at service startup)
# ---------------------------------------------------------------------------

_registry: AdditivesRegistry | None = None


def get_registry(registry_path: str | Path | None = None) -> AdditivesRegistry:
    """Return the singleton registry, loading it on first call."""
    global _registry
    if _registry is None or registry_path is not None:
        _registry = AdditivesRegistry(registry_path)
    return _registry


# ---------------------------------------------------------------------------
# Convenience classification helpers
# ---------------------------------------------------------------------------

def classify_hetatm(
    comp_id: str,
    element: str = "",
    n_atoms_in_residue: int = 1,
    registry: AdditivesRegistry | None = None,
) -> str:
    """Return initial classification for a HETATM comp_id.

    Returns one of:
        water / metal / phasing_agent / buffer_ion /
        cofactor / additive / unknown
    """
    cid = comp_id.strip().upper()
    el = element.strip().upper()

    if cid in WATER_RESNAMES:
        return "water"

    # Single-atom residues
    if n_atoms_in_residue == 1:
        if el in METAL_ELEMENTS or cid in METAL_ELEMENTS:
            return "metal"
        if cid in PHASING_ELEMENTS or el in PHASING_ELEMENTS:
            return "phasing_agent"
        if cid in BUFFER_ION_COMP_IDS:
            return "buffer_ion"

    if cid in COFACTORS:
        return "cofactor"

    reg = registry or get_registry()
    if reg.is_additive(cid):
        return "additive"

    return "unknown"


__all__ = [
    "AdditivesRegistry",
    "get_registry",
    "classify_hetatm",
    "METAL_ELEMENTS",
    "COFACTORS",
    "WATER_RESNAMES",
    "STANDARD_AA",
    "NUCLEIC_RESNAMES",
    "DNA_RESNAMES",
    "RNA_RESNAMES",
]
