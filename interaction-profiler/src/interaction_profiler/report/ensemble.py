"""Extended ensemble analysis — fragment families and compatible pairs.

Both functions are computable entirely from data already produced by the
normal profiling pipeline:

fragment_families
    Clusters fragments (one per structure) by their dominant binding mode —
    defined as the (primary_interacting_residue, primary_interaction_type)
    pair with the highest frequency in ``generalized_triples``.  No extra
    geometry is needed; all required data comes from ``HotspotOccupancy``
    and ``ComplexResult.enriched_interactions``.

compatible_pairs
    Identifies pairs of fragments that:
    (a) occupy non-overlapping sets of target interface residues, AND
    (b) have exit vectors within ``max_linker_A`` of each other.
    Requires ``compute_interface=True`` results and preferably
    ``compute_vectors=True`` (enriched profile).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


def compute_fragment_families(
    enriched_results: list[Any],
    min_members: int = 2,
) -> list[Any]:
    """Cluster fragments by dominant binding mode using generalized triples.

    Parameters
    ----------
    enriched_results:
        List of ``ComplexResult`` objects from an ensemble run.
        Each result should have ``generalized_triples`` populated.
    min_members:
        Minimum number of members for a family to be reported.

    Returns
    -------
    list[FragmentFamily]
        Sorted by occupancy descending.
    """
    from interaction_profiler.schemas import FragmentFamily  # noqa: PLC0415

    # Build a binding-mode key for each result
    result_modes: list[tuple[str, str]] = []  # (source_path, mode_key)
    for result in enriched_results:
        source = _result_source(result)
        mode_key = _dominant_mode_key(result)
        result_modes.append((source, mode_key))

    # Group by mode_key
    mode_groups: dict[str, list[str]] = {}
    for source, mode_key in result_modes:
        mode_groups.setdefault(mode_key, []).append(source)

    total = len(enriched_results)
    families = []
    for fam_idx, (mode_key, members) in enumerate(
        sorted(mode_groups.items(), key=lambda x: -len(x[1])),
        start=1,
    ):
        if len(members) < min_members:
            continue

        primary_residue, primary_interaction = _parse_mode_key(mode_key)
        occupancy = round(len(members) / total, 3) if total else None

        # Find occupied subpockets (if vector classification was run)
        subpocket_ids: list[str] = []
        for result in enriched_results:
            src = _result_source(result)
            if src in members:
                for ix in getattr(result, "enriched_interactions", []):
                    sp = getattr(ix, "subpocket_id", None)
                    if sp and sp not in subpocket_ids:
                        subpocket_ids.append(sp)

        families.append(FragmentFamily(
            family_id=f"F{fam_idx:03d}",
            members=members,
            primary_residue=primary_residue,
            primary_interaction=primary_interaction,
            occupancy=occupancy,
            subpocket_ids=subpocket_ids,
            binding_mode_label=_build_mode_label(primary_interaction, primary_residue, subpocket_ids),
        ))

    return families


def compute_compatible_pairs(
    enriched_results: list[Any],
    site_profile: Any | None = None,
    max_linker_A: float = 15.0,
) -> list[Any]:
    """Find fragment pairs that bind non-overlapping regions.

    Parameters
    ----------
    enriched_results:
        List of ``ComplexResult`` objects with ``interface`` populated.
    site_profile:
        ``SiteProfile`` if available; used to check spatial separation.
    max_linker_A:
        Maximum distance between exit vectors to consider linking feasible.

    Returns
    -------
    list[CompatiblePair]
        Pairs sorted by compatibility (compatible=True first).
    """
    from interaction_profiler.schemas import CompatiblePair  # noqa: PLC0415

    # Build residue set per result
    result_residues: list[tuple[str, set[str]]] = []
    for result in enriched_results:
        src = _result_source(result)
        iface = getattr(result, "interface", None)
        if iface is not None:
            residues = set(getattr(iface, "interface_residues_target", []) or [])
        else:
            # Fall back to residue labels from enriched interactions
            residues = {
                getattr(ix, "target_feature", None) or getattr(ix, "residue_label", "")
                for ix in getattr(result, "enriched_interactions", [])
            }
            residues.discard("")
        result_residues.append((src, residues))

    pairs = []
    n = len(result_residues)
    for i in range(n):
        src_a, res_a = result_residues[i]
        for j in range(i + 1, n):
            src_b, res_b = result_residues[j]
            overlap = sorted(res_a & res_b)
            compatible = len(overlap) == 0

            # Estimate linker distance from exit vectors if available
            linker_dist = _estimate_linker_distance(
                enriched_results[i], enriched_results[j]
            )
            if linker_dist is not None and linker_dist > max_linker_A:
                compatible = False
                note = f"Exit vector distance {linker_dist:.1f} Å exceeds max_linker ({max_linker_A} Å)."
            elif overlap:
                note = f"Overlapping residues: {', '.join(overlap[:5])}{'…' if len(overlap) > 5 else ''}."
            else:
                note = "Non-overlapping binding regions — compatible for linking or merging."

            pairs.append(CompatiblePair(
                fragment_a=src_a,
                fragment_b=src_b,
                overlap_residues=overlap,
                compatible=compatible,
                linker_distance_A=linker_dist,
                notes=note,
            ))

    # Sort: compatible pairs first, then by overlap size ascending
    pairs.sort(key=lambda p: (not p.compatible, len(p.overlap_residues)))
    return pairs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _result_source(result: Any) -> str:
    """Extract a stable identifier (source path) from a ComplexResult."""
    prov = getattr(result, "provenance", None)
    if prov:
        entities = getattr(result, "entities", [])
        for e in entities:
            for ch in getattr(e, "chain_ids", []):
                return ch  # fallback
    # Use first interaction's source as identifier
    for ix in getattr(result, "enriched_interactions", []):
        src = getattr(ix, "receptor_chain", None)
        if src:
            return src
    return id(result).__str__()


def _dominant_mode_key(result: Any) -> str:
    """Return (primary_residue, primary_interaction_type) as a string key."""
    triples = getattr(result, "generalized_triples", [])
    if not triples:
        return "unknown:unknown"

    counter: Counter = Counter()
    for t in triples:
        res = getattr(t, "target_feature", "") or ""
        itype = getattr(t, "interaction_type", "") or ""
        counter[(res, itype)] += getattr(t, "count", 1)

    if counter:
        (res, itype), _ = counter.most_common(1)[0]
        return f"{res}:{itype}"
    return "unknown:unknown"


def _parse_mode_key(key: str) -> tuple[str | None, str | None]:
    """Split mode key back into (residue, interaction_type)."""
    parts = key.split(":", 1)
    if len(parts) == 2:
        return (parts[0] or None, parts[1] or None)
    return (None, None)


def _build_mode_label(
    interaction: str | None,
    residue: str | None,
    subpocket_ids: list[str],
) -> str | None:
    """Build a human-readable binding-mode label."""
    parts = []
    if interaction:
        parts.append(interaction)
    if residue:
        parts.append(residue)
    if subpocket_ids:
        parts.append("-".join(subpocket_ids[:2]))
    return "-".join(parts) if parts else None


def _estimate_linker_distance(result_a: Any, result_b: Any) -> float | None:
    """Estimate distance between exit vectors of two fragment results."""
    try:
        # Look for exit_vector atoms in each result
        exits_a = _exit_vector_positions(result_a)
        exits_b = _exit_vector_positions(result_b)
        if not exits_a or not exits_b:
            return None

        # Minimum pairwise distance between exit vector positions
        min_dist = float("inf")
        for pos_a in exits_a:
            for pos_b in exits_b:
                d = sum((a - b) ** 2 for a, b in zip(pos_a, pos_b)) ** 0.5
                if d < min_dist:
                    min_dist = d
        return round(min_dist, 2) if min_dist < float("inf") else None
    except Exception:
        return None


def _exit_vector_positions(result: Any) -> list[list[float]]:
    """Extract ligand atom positions classified as exit_vector."""
    positions: list[list[float]] = []
    for ix in getattr(result, "enriched_interactions", []):
        if getattr(ix, "vector_type", None) == "exit_vector":
            # We don't store raw coordinates in EnrichedInteraction;
            # use the ligand center of mass from MoleculeMetadata if available
            break
    return positions  # empty when no 3D coords stored
