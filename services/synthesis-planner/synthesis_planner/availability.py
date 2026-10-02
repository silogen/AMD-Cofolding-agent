"""Building-block availability check against the configured stock catalog.

Availability is assessed independently of route planning — a molecule may be
purchasable even if no retrosynthetic route was found, and vice-versa.

Per the PLAN requirement:
- ``available``   — SMILES found in the configured stock index.
- ``unavailable`` — SMILES NOT found in a configured and loaded stock index.
- ``unknown``     — no stock catalog is configured or loaded.

Do not claim ``unavailable`` when no catalog is configured; ``unknown`` is the
honest answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BuildingBlockStatus:
    """Availability result for one SMILES."""
    smiles: str
    canonical_smiles: str | None
    availability: str               # "available" | "unavailable" | "unknown"
    in_stock: bool | None           # True/False/None (None = unknown)
    stock_source: str | None        # stock catalog key that matched, or None
    warning: str | None = None      # non-fatal warnings from standardization


def check_availability(
    smiles_list: list[str],
    stock: Any | None,
    stock_key: str = "default",
) -> list[BuildingBlockStatus]:
    """Check a list of SMILES against the given stock object.

    Parameters
    ----------
    smiles_list:
        Raw input SMILES to check.
    stock:
        AiZynthFinder ``Stock`` object (already loaded), or ``None`` if no
        stock is configured.
    stock_key:
        Logical name of the stock being queried (for provenance).

    Returns
    -------
    list[BuildingBlockStatus]
        One entry per input SMILES, in the original order.
    """
    from synthesis_planner.standardize import standardize_smiles  # noqa: PLC0415

    results: list[BuildingBlockStatus] = []

    for raw_smiles in smiles_list:
        std = standardize_smiles(raw_smiles)

        if std.error:
            results.append(BuildingBlockStatus(
                smiles=raw_smiles,
                canonical_smiles=None,
                availability="unknown",
                in_stock=None,
                stock_source=None,
                warning=std.error,
            ))
            continue

        canonical = std.canonical_smiles

        if stock is None:
            # No stock configured — honest "unknown"
            results.append(BuildingBlockStatus(
                smiles=raw_smiles,
                canonical_smiles=canonical,
                availability="unknown",
                in_stock=None,
                stock_source=None,
                warning="No stock catalog is configured for this service.",
            ))
            continue

        # Perform the lookup
        try:
            # AiZynthFinder Stock.mol_in_stock expects a Molecule object
            # We use the lower-level SMILES-based lookup when available,
            # or fall back to creating a Molecule from the SMILES.
            in_stock = _check_in_stock(canonical, stock)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stock lookup failed for %s: %s", canonical, exc)
            results.append(BuildingBlockStatus(
                smiles=raw_smiles,
                canonical_smiles=canonical,
                availability="unknown",
                in_stock=None,
                stock_source=None,
                warning=f"Stock lookup error: {exc}",
            ))
            continue

        results.append(BuildingBlockStatus(
            smiles=raw_smiles,
            canonical_smiles=canonical,
            availability="available" if in_stock else "unavailable",
            in_stock=in_stock,
            stock_source=stock_key if in_stock else None,
            warning=std.warning,
        ))

    return results


def _check_in_stock(canonical_smiles: str, stock: Any) -> bool:
    """Perform the actual stock lookup against an AiZynthFinder Stock object.

    AiZynthFinder's Stock API changed across versions.  This function tries
    multiple access patterns in order:

    1. ``stock[canonical_smiles]`` — dict-like access (some wrappers).
    2. ``stock.mol_in_stock(molecule)`` — standard API.
    3. Fallback via SMILES fingerprint lookup.
    """
    # Pattern 1: dict-like direct SMILES access (thin wrappers)
    try:
        return bool(stock[canonical_smiles])
    except (KeyError, TypeError):
        pass

    # Pattern 2: standard mol_in_stock API
    try:
        from aizynthfinder.chem import Molecule  # noqa: PLC0415
        mol = Molecule(smiles=canonical_smiles)
        return bool(stock.mol_in_stock(mol))
    except (ImportError, Exception):
        pass

    # Pattern 3: is_available (newer AiZynthFinder versions)
    try:
        from aizynthfinder.chem import Molecule  # noqa: PLC0415
        mol = Molecule(smiles=canonical_smiles)
        return bool(stock.is_available(mol))
    except (ImportError, Exception):
        pass

    raise RuntimeError(
        f"Could not query stock for SMILES {canonical_smiles!r}: "
        "no compatible AiZynthFinder stock API found."
    )
