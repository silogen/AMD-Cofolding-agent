"""SMILES standardization for retrosynthesis planning.

Unlike the ADMET standardizer (which re-protonates at pH 7.4), this module
preserves stereochemistry and formal charge exactly as provided — retrosynthesis
depends on the exact molecular graph.  The goal is:

1. Validate that the input can be parsed by RDKit.
2. Reject clearly unsupported chemistry before touching the MCTS search.
3. Canonicalize to detect duplicates.
4. Return the canonical SMILES and any error/warning message.

All operations are pure-Python / RDKit; no network calls are made here.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chemistry that AiZynthFinder cannot handle reliably.  Flagging these early
# prevents wasted MCTS iterations that would produce an opaque "no route".
# ---------------------------------------------------------------------------

# Atomic numbers of metals that are broadly unsupported by template-based
# retrosynthesis models (the model was trained on C/N/O/S/P/halogens-focused
# chemistry).  This list is advisory — borderline cases still reach the search.
_PROBLEMATIC_ATOMIC_NUMS: frozenset[int] = frozenset({
    # Lanthanides 57-71, Actinides 89-103
    *range(57, 72),
    *range(89, 104),
    # Heavy transition metals not in typical synthesis datasets
    76, 77, 78, 79, 80,  # Os, Ir, Pt, Au, Hg
})

# Max heavy-atom count; very large molecules time out reliably.
_MAX_HEAVY_ATOMS = 100


class StandardizeResult(NamedTuple):
    """Result of standardizing one SMILES input."""
    canonical_smiles: str | None
    """Canonical SMILES, or None on hard failure."""
    error: str | None
    """Error string for invalid/unsupported inputs; None when OK."""
    warning: str | None
    """Non-fatal warning (e.g. large molecule, unusual element)."""


def standardize_smiles(smiles: str) -> StandardizeResult:
    """Parse, validate, and canonicalize a SMILES string for retrosynthesis.

    Parameters
    ----------
    smiles:
        Input SMILES (may be non-canonical or have explicit Hs).

    Returns
    -------
    StandardizeResult
        ``canonical_smiles`` is set on success; ``error`` is set on hard
        failure; ``warning`` is set for soft issues (molecule still passed).
    """
    try:
        from rdkit import Chem  # noqa: PLC0415
    except ImportError:
        return StandardizeResult(
            canonical_smiles=None,
            error="RDKit is not installed — cannot validate SMILES.",
            warning=None,
        )

    if not smiles or not smiles.strip():
        return StandardizeResult(
            canonical_smiles=None,
            error="Empty SMILES.",
            warning=None,
        )

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return StandardizeResult(
            canonical_smiles=None,
            error=f"RDKit could not parse SMILES: {smiles!r}.",
            warning=None,
        )

    # ── Hard failures ────────────────────────────────────────────────────────

    # Query atoms (e.g. [#6]) are meaningless for retrosynthesis
    has_query = any(
        atom.HasQuery() for atom in mol.GetAtoms()
    )
    if has_query:
        return StandardizeResult(
            canonical_smiles=None,
            error="SMILES contains query atoms — unsupported for retrosynthesis.",
            warning=None,
        )

    # Radicals (unpaired electrons)
    has_radical = any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms())
    if has_radical:
        return StandardizeResult(
            canonical_smiles=None,
            error="SMILES contains radical atoms — unsupported for retrosynthesis.",
            warning=None,
        )

    # ── Soft warnings ────────────────────────────────────────────────────────
    warning: str | None = None

    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy > _MAX_HEAVY_ATOMS:
        # Still pass through — the search will time out gracefully — but warn.
        warning = (
            f"Molecule has {n_heavy} heavy atoms (> {_MAX_HEAVY_ATOMS}). "
            "The route search may time out."
        )

    # Check for problematic metals
    metal_atoms = [
        atom for atom in mol.GetAtoms()
        if atom.GetAtomicNum() in _PROBLEMATIC_ATOMIC_NUMS
    ]
    if metal_atoms:
        symbols = sorted({atom.GetSymbol() for atom in metal_atoms})
        metal_warning = (
            f"Molecule contains elements ({', '.join(symbols)}) that may not "
            "be well-covered by the retrosynthesis model."
        )
        warning = f"{warning}  {metal_warning}" if warning else metal_warning

    # Canonicalize — preserves stereo by default in RDKit
    try:
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception as exc:
        return StandardizeResult(
            canonical_smiles=None,
            error=f"Failed to canonicalize SMILES: {exc}",
            warning=None,
        )

    return StandardizeResult(
        canonical_smiles=canonical,
        error=None,
        warning=warning,
    )


def deduplicate(
    smiles_list: list[str],
) -> tuple[list[str], dict[str, list[int]]]:
    """Remove duplicate canonical SMILES while preserving the first occurrence.

    Parameters
    ----------
    smiles_list:
        Raw input SMILES (non-canonical, may contain duplicates).

    Returns
    -------
    (unique_smiles, duplicate_map)
        ``unique_smiles`` — deduplicated canonical SMILES, order-preserving.
        ``duplicate_map`` — maps each canonical SMILES to the list of original
        indices that collapsed to it (including the first occurrence).
    """
    seen: dict[str, int] = {}        # canonical SMILES -> first-seen index in output
    unique: list[str] = []
    dup_map: dict[str, list[int]] = {}

    for orig_idx, smi in enumerate(smiles_list):
        result = standardize_smiles(smi)
        key = result.canonical_smiles or smi  # fall back to raw for invalid

        if key not in seen:
            seen[key] = len(unique)
            unique.append(key)
            dup_map[key] = [orig_idx]
        else:
            dup_map[key].append(orig_idx)

    return unique, dup_map
