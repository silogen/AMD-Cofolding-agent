"""Map UniProt / InterPro feature positions onto structure residue numbers.

Both functions use Biopython's :class:`~Bio.Align.PairwiseAligner` to align
the (potentially truncated / mutated) sequence extracted from the structure to
the canonical UniProt sequence, then walk the alignment coordinates to build a
``uniprot_pos → structure_resnum`` lookup table.

Feature positions are always expressed in the **UniProt** numbering (1-based)
and are converted to 0-based structure residue numbers in the output dicts.

Graceful-failure guarantee
--------------------------
Neither function ever raises.  On any alignment or parsing failure an empty
dict is returned and a warning is logged.

Typical usage
-------------
::

    from interaction_profiler.taxonomy.annotation_mapper import (
        map_uniprot_features_to_residues,
        map_interpro_to_residues,
    )

    # UniProt API features list (list[dict] with 'type', 'description',
    # 'location': {'start': {'value': int}, 'end': {'value': int}})
    residue_annot = map_uniprot_features_to_residues(
        features, chain_id="A",
        structure_sequence=struct_seq,
        uniprot_sequence=uni_seq,
    )
    # → {struct_resnum_0based: {feature_type, description, uniprot_pos, mapped}}

    domain_annot = map_interpro_to_residues(
        interpro_entries, chain_id="A",
        structure_sequence=struct_seq,
        uniprot_sequence=uni_seq,
    )
    # → {struct_resnum_0based: {domain_name, accession, db}}
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alignment parameters (match user spec exactly)
# ---------------------------------------------------------------------------
_MATCH_SCORE:       float = 2.0
_MISMATCH_SCORE:    float = -1.0
_OPEN_GAP_SCORE:    float = -2.0
_EXTEND_GAP_SCORE:  float = -0.5


# ---------------------------------------------------------------------------
# Private: alignment helpers
# ---------------------------------------------------------------------------

def _build_alignment_mapping(
    structure_sequence: str,
    uniprot_sequence: str,
) -> dict[int, int | None]:
    """Align *structure_sequence* to *uniprot_sequence* and build a position map.

    Uses Biopython :class:`~Bio.Align.PairwiseAligner` (global alignment,
    protein mode).

    Returns
    -------
    mapping : dict[int, int | None]
        ``{uniprot_pos_0based: structure_resnum_0based | None}``

        *None* means the UniProt position falls in a gap in the structure
        (the residue is absent from the structure — e.g. disordered loop,
        truncated construct, or expression-tag cleavage site).

    Raises
    ------
    Does not raise.  Returns an empty dict on any failure.
    """
    if not structure_sequence or not uniprot_sequence:
        logger.debug("_build_alignment_mapping: empty sequence(s); returning empty map.")
        return {}

    try:
        from Bio.Align import PairwiseAligner  # type: ignore[import]
    except ImportError:
        logger.warning(
            "_build_alignment_mapping: biopython not installed. "
            "Install with 'pip install biopython' to enable annotation mapping."
        )
        return {}

    try:
        aligner = PairwiseAligner()
        aligner.mode             = "global"
        aligner.match_score      = _MATCH_SCORE
        aligner.mismatch_score   = _MISMATCH_SCORE
        aligner.open_gap_score   = _OPEN_GAP_SCORE
        aligner.extend_gap_score = _EXTEND_GAP_SCORE

        # align(target=structure_seq, query=uniprot_seq)
        # alignment.coordinates[0] → positions in structure_sequence
        # alignment.coordinates[1] → positions in uniprot_sequence
        alignments = aligner.align(structure_sequence, uniprot_sequence)
        alignment = next(iter(alignments))
    except Exception as exc:  # noqa: BLE001
        logger.warning("_build_alignment_mapping: alignment failed: %s", exc)
        return {}

    try:
        return _coords_to_mapping(alignment.coordinates)
    except Exception as exc:  # noqa: BLE001
        logger.warning("_build_alignment_mapping: coordinate extraction failed: %s", exc)
        return {}


def _coords_to_mapping(
    coordinates: Any,
) -> dict[int, int | None]:
    """Convert a PairwiseAlignment coordinate array to a uniprot → struct map.

    The coordinate array has shape ``(2, n_blocks + 1)`` where:
    * ``coordinates[0]`` – positions in the *target* sequence (structure_sequence)
    * ``coordinates[1]`` – positions in the *query* sequence (uniprot_sequence)

    Between consecutive coordinate columns:
    * Both step forward   → aligned block (possible mismatch, but no gap).
    * Only query steps    → gap in structure sequence (structure residue absent).
    * Only target steps   → gap in UniProt sequence (extra residues in structure).
    * Neither steps       → degenerate; skip.

    Returns ``{uniprot_pos_0based: structure_resnum_0based | None}``.
    """
    mapping: dict[int, int | None] = {}
    n_blocks = coordinates.shape[1] - 1

    s_pos = int(coordinates[0][0])   # current position in structure sequence
    u_pos = int(coordinates[1][0])   # current position in uniprot sequence

    for i in range(n_blocks):
        s_end = int(coordinates[0][i + 1])
        u_end = int(coordinates[1][i + 1])

        s_len = s_end - s_pos
        u_len = u_end - u_pos

        if s_len > 0 and u_len > 0:
            # Aligned block (may contain mismatches, but positionally matched)
            block_len = min(s_len, u_len)
            for offset in range(block_len):
                mapping[u_pos + offset] = s_pos + offset

        elif u_len > 0:
            # Gap in structure (structure is missing these UniProt residues)
            for u in range(u_pos, u_end):
                mapping[u] = None

        # elif s_len > 0: extra residues in structure → no UniProt key to record

        s_pos = s_end
        u_pos = u_end

    return mapping


def _feature_start_end(feature: dict[str, Any]) -> tuple[int, int] | None:
    """Extract 0-based [start, end) range from a UniProt API feature dict.

    UniProt REST API (v2) uses::

        {
            "location": {
                "start": {"value": int, "modifier": "EXACT"},
                "end":   {"value": int, "modifier": "EXACT"}
            },
            ...
        }

    Older flat-format dicts use ``feature["start"]`` and ``feature["end"]``
    directly (1-based integers).

    Returns (start_0based, end_0based_inclusive) or None on parse failure.
    """
    # Try new API format first
    try:
        loc = feature["location"]
        start_1 = int(loc["start"]["value"])
        end_1   = int(loc["end"]["value"])
        return (start_1 - 1, end_1 - 1)
    except (KeyError, TypeError, ValueError):
        pass

    # Try simple flat format
    try:
        start_1 = int(feature["start"])
        end_1   = int(feature["end"])
        return (start_1 - 1, end_1 - 1)
    except (KeyError, TypeError, ValueError):
        pass

    return None


# ---------------------------------------------------------------------------
# Public: UniProt feature mapper
# ---------------------------------------------------------------------------

def map_uniprot_features_to_residues(
    features: list[dict[str, Any]],
    chain_id: str,
    structure_sequence: str,
    uniprot_sequence: str,
) -> dict[int, dict[str, Any]]:
    """Map UniProt sequence feature positions onto structure residue numbers.

    Parameters
    ----------
    features:
        List of UniProt feature dicts as returned by the UniProt REST API.
        Each dict must contain ``type`` (or ``feature_type``), ``description``,
        and a location expressed either as
        ``{"location": {"start": {"value": int}, "end": {"value": int}}}``
        (UniProt API v2) or flat ``{"start": int, "end": int}`` (legacy).
    chain_id:
        Chain identifier used to annotate the output (informational only; the
        function operates entirely on sequences).
    structure_sequence:
        One-letter protein sequence extracted from the structure (e.g. via
        MDAnalysis / Biopython sequence extraction).  May be shorter than
        *uniprot_sequence* or contain mutations.
    uniprot_sequence:
        Canonical full-length UniProt sequence (one-letter codes).

    Returns
    -------
    dict[int, dict]
        ``{structure_resnum_0based: {feature_type, description, uniprot_pos, mapped: True}}``

        Residues that appear in a feature range but have no corresponding
        structural position (gap in the alignment) are *not* included in the
        returned dict.  A ``mapped: False`` sentinel is emitted as a debug log
        for each unmapped feature range — callers that need to track unmapped
        features should check the log or extend this function.

    Notes
    -----
    The function never raises.  An empty dict is returned on alignment failure
    or invalid input; a warning is logged.
    """
    if not features:
        return {}

    if not structure_sequence or not uniprot_sequence:
        logger.warning(
            "map_uniprot_features_to_residues(chain=%r): empty sequence supplied; "
            "returning empty map.",
            chain_id,
        )
        return {}

    # Build alignment mapping once; reuse for all features
    uni_to_struct = _build_alignment_mapping(structure_sequence, uniprot_sequence)
    if not uni_to_struct:
        logger.warning(
            "map_uniprot_features_to_residues(chain=%r): alignment produced empty map; "
            "no features will be mapped.",
            chain_id,
        )
        return {}

    result: dict[int, dict[str, Any]] = {}

    for feat in features:
        # Determine feature type
        feat_type: str = (
            feat.get("type")
            or feat.get("feature_type")
            or feat.get("featureType")
            or "unknown"
        )
        description: str = feat.get("description") or feat.get("desc") or ""

        range_ = _feature_start_end(feat)
        if range_ is None:
            logger.debug(
                "map_uniprot_features_to_residues: could not parse position for "
                "feature %r (chain=%r); skipping.",
                feat_type, chain_id,
            )
            continue

        start_0, end_0 = range_
        any_mapped = False

        for u_pos in range(start_0, end_0 + 1):
            s_pos = uni_to_struct.get(u_pos)
            if s_pos is None:
                # Residue absent from structure (gap in alignment or out-of-range)
                continue
            any_mapped = True
            # Later features for the same residue overwrite earlier ones.
            # Callers that need multi-annotation per residue should extend
            # the output schema to a list.
            result[s_pos] = {
                "feature_type":  feat_type,
                "description":   description,
                "uniprot_pos":   u_pos,
                "mapped":        True,
            }

        if not any_mapped:
            logger.debug(
                "map_uniprot_features_to_residues(chain=%r): feature '%s' "
                "(UniProt pos %d–%d) has no corresponding residue in the structure "
                "(mapped: False).",
                chain_id, feat_type, start_0 + 1, end_0 + 1,
            )

    logger.debug(
        "map_uniprot_features_to_residues(chain=%r): mapped %d residue annotations "
        "from %d features.",
        chain_id, len(result), len(features),
    )
    return result


# ---------------------------------------------------------------------------
# Public: InterPro domain mapper
# ---------------------------------------------------------------------------

def map_interpro_to_residues(
    interpro_entries: list[dict[str, Any]],
    chain_id: str,
    structure_sequence: str,
    uniprot_sequence: str,
) -> dict[int, dict[str, Any]]:
    """Map InterPro domain annotations onto structure residue numbers.

    Parameters
    ----------
    interpro_entries:
        List of InterPro annotation dicts as returned by
        ``fetch_interpro_annotations()`` (or any equivalent source).  Each
        entry must have the fields:

        .. code-block:: python

            {
                "name":            str,           # human-readable domain name
                "accession":       str,           # InterPro accession, e.g. "IPR000719"
                "source_database": str,           # e.g. "pfam", "superfamily"
                "residue_ranges":  [              # one or more ranges
                    {"start": int, "end": int},   # 1-based UniProt positions
                    ...
                ],
            }

        The ``"db"`` field is an accepted alias for ``"source_database"``.

    chain_id:
        Chain identifier (informational only).
    structure_sequence:
        One-letter protein sequence from the structure.
    uniprot_sequence:
        Canonical UniProt sequence.

    Returns
    -------
    dict[int, dict]
        ``{structure_resnum_0based: {domain_name, accession, db}}``

        Where multiple domains overlap at the same residue, the later one in
        *interpro_entries* wins (last-write semantics).  Callers needing all
        overlapping domains per residue should extend the output to
        ``dict[int, list[dict]]``.

    Notes
    -----
    Never raises.  Returns an empty dict on alignment failure or invalid input.
    """
    if not interpro_entries:
        return {}

    if not structure_sequence or not uniprot_sequence:
        logger.warning(
            "map_interpro_to_residues(chain=%r): empty sequence supplied; "
            "returning empty map.",
            chain_id,
        )
        return {}

    # Build alignment mapping once; reuse for all entries
    uni_to_struct = _build_alignment_mapping(structure_sequence, uniprot_sequence)
    if not uni_to_struct:
        logger.warning(
            "map_interpro_to_residues(chain=%r): alignment produced empty map; "
            "no InterPro domains will be mapped.",
            chain_id,
        )
        return {}

    result: dict[int, dict[str, Any]] = {}

    for entry in interpro_entries:
        domain_name: str = entry.get("name") or entry.get("domain_name") or "unknown"
        accession:   str = entry.get("accession") or ""
        db:          str = (
            entry.get("source_database")
            or entry.get("db")
            or entry.get("database")
            or ""
        )

        residue_ranges: list[dict[str, Any]] = entry.get("residue_ranges") or []
        if not residue_ranges:
            logger.debug(
                "map_interpro_to_residues: entry %r has no 'residue_ranges'; skipping.",
                accession or domain_name,
            )
            continue

        for rng in residue_ranges:
            try:
                start_1 = int(rng["start"])
                end_1   = int(rng["end"])
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug(
                    "map_interpro_to_residues: could not parse range %r in entry %r: %s",
                    rng, accession, exc,
                )
                continue

            start_0 = start_1 - 1
            end_0   = end_1 - 1
            any_mapped = False

            for u_pos in range(start_0, end_0 + 1):
                s_pos = uni_to_struct.get(u_pos)
                if s_pos is None:
                    continue
                any_mapped = True
                result[s_pos] = {
                    "domain_name": domain_name,
                    "accession":   accession,
                    "db":          db,
                }

            if not any_mapped:
                logger.debug(
                    "map_interpro_to_residues(chain=%r): domain '%s' (%s) "
                    "range %d–%d has no corresponding structural residues.",
                    chain_id, domain_name, accession, start_1, end_1,
                )

    logger.debug(
        "map_interpro_to_residues(chain=%r): mapped %d residue domain annotations "
        "from %d InterPro entries.",
        chain_id, len(result), len(interpro_entries),
    )
    return result


__all__ = [
    "map_uniprot_features_to_residues",
    "map_interpro_to_residues",
]
