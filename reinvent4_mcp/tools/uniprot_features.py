"""Extended UniProt annotation fetcher — features, active sites, domains.

Complements the existing fetch_target_info() in uniprot.py by parsing
the full features[] array from the UniProt API response.

Provides one MCP tool:
  fetch_uniprot_features(accession, feature_types?) -> JSON str

Returns active sites, binding sites, domains, motifs, natural variants,
modified residues, disulfide bonds, and cross-references to InterPro,
Pfam, and PANTHER for any UniProt accession.  Network failures return
``{"error": "..."}``; data is never fabricated.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BASE = "https://rest.uniprot.org/uniprotkb"
_TIMEOUT = 30

# Annotation service (block 10). When set, UniProt feature fetches go through
# the service which returns features[] + sequence in a single upstream hit and
# caches results on disk.  Falls back to direct API when absent.
_ANN_URL: str = os.environ.get("ANNOTATION_SERVICE_URL", "").rstrip("/")

# All feature types we understand; None = include everything.
_ALL_FEATURE_TYPES: frozenset[str] = frozenset(
    {
        "Active site",
        "Binding site",
        "Domain",
        "Region",
        "Motif",
        "Site",
        "Natural variant",
        "Modified residue",
        "Disulfide bond",
    }
)


def _parse_features(raw_features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the raw UniProt ``features`` array into a clean list."""
    results = []
    for feat in raw_features or []:
        ftype = feat.get("type", "")
        loc = feat.get("location", {}) or {}
        start_obj = loc.get("start", {}) or {}
        end_obj = loc.get("end", {}) or {}
        start = start_obj.get("value")
        end = end_obj.get("value")
        description = feat.get("description", "")
        # Some features have evidences but no description; keep only non-empty.
        entry: dict[str, Any] = {"type": ftype}
        if start is not None:
            entry["start"] = start
        if end is not None:
            entry["end"] = end
        if description:
            entry["description"] = description
        results.append(entry)
    return results


def _parse_cross_references(
    cross_refs: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Extract InterPro, Pfam, and PANTHER cross-references."""
    interpro: list[dict[str, Any]] = []
    pfam: list[dict[str, Any]] = []
    panther: list[dict[str, Any]] = []

    for ref in cross_refs or []:
        db = (ref.get("database") or "").lower()
        ref_id = ref.get("id") or ""
        if not ref_id:
            continue

        if db == "interpro":
            # The first property with key "EntryName" is the human-readable name.
            name = ""
            for prop in ref.get("properties", []) or []:
                if prop.get("key") == "EntryName":
                    name = prop.get("value", "")
                    break
            entry: dict[str, Any] = {"id": ref_id}
            if name:
                entry["name"] = name
            interpro.append(entry)

        elif db == "pfam":
            pfam.append({"id": ref_id})

        elif db == "panther":
            panther.append({"id": ref_id})

    return {"interpro": interpro, "pfam": pfam, "panther": panther}


def _fetch_features(
    accession: str,
    feature_types: list[str] | None,
) -> dict[str, Any]:
    """Core implementation — separated for testability.

    Route priority
    --------------
    1. ``ANNOTATION_SERVICE_URL`` set → ``POST /uniprot/features`` on the
       annotation service (single upstream hit, disk-cached by the service).
    2. Direct UniProt API call (legacy path, no caching).
    """
    accession = accession.strip().upper()

    # ── 1. Annotation service ────────────────────────────────────────────────
    if _ANN_URL:
        try:
            resp = requests.post(
                f"{_ANN_URL}/uniprot/features",
                json={"accession": accession, "feature_types": feature_types},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning(
                "Annotation service call failed for %s: %s. Falling back to direct API.",
                accession, exc,
            )
            # fall through to direct API

    # ── 2. Direct UniProt API (legacy) ───────────────────────────────────────
    url = f"{_BASE}/{accession}.json"
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        logger.warning("UniProt request failed for %s: %s", accession, exc)
        return {
            "error": (
                f"Could not reach UniProt for accession {accession!r}: {exc}. "
                "Data not fabricated."
            )
        }

    if resp.status_code == 404:
        return {"error": f"UniProt accession {accession!r} not found (HTTP 404)."}
    if resp.status_code != 200:
        return {
            "error": (
                f"UniProt returned HTTP {resp.status_code} for accession {accession!r}."
            )
        }

    try:
        data = resp.json()
    except ValueError as exc:
        return {"error": f"UniProt returned non-JSON response: {exc}"}

    # --- Protein name -------------------------------------------------------
    protein_desc = data.get("proteinDescription", {}) or {}
    rec_name = (
        protein_desc.get("recommendedName", {}).get("fullName", {}).get("value")
    )
    if not rec_name:
        submitted = protein_desc.get("submissionNames") or []
        if submitted:
            rec_name = submitted[0].get("fullName", {}).get("value", "")

    # --- Sequence length ----------------------------------------------------
    seq = data.get("sequence", {}) or {}
    seq_length = seq.get("length")

    # --- Features -----------------------------------------------------------
    raw_features = data.get("features") or []
    all_features = _parse_features(raw_features)

    # Apply optional type filter.
    if feature_types is not None:
        requested = {ft.lower() for ft in feature_types}
        all_features = [
            f for f in all_features if f.get("type", "").lower() in requested
        ]

    # --- Cross-references ---------------------------------------------------
    cross_refs_raw = data.get("uniProtKBCrossReferences") or []
    cross_refs = _parse_cross_references(cross_refs_raw)

    return {
        "accession": accession,
        "protein_name": rec_name or "",
        "sequence_length": seq_length,
        "features": all_features,
        "cross_references": cross_refs,
    }


def register_tools(mcp) -> None:
    """Register the ``fetch_uniprot_features`` tool with the MCP server."""

    @mcp.tool()
    def fetch_uniprot_features(
        accession: str,
        feature_types: list[str] | None = None,
    ) -> str:
        """Fetch feature annotations for a UniProt accession.

        Returns active sites, binding sites, domains, motifs, natural
        variants, modified residues, and disulfide bonds, together with
        cross-references to InterPro, Pfam, and PANTHER.

        Args:
            accession:     UniProt accession, e.g. ``"P24941"`` for CDK2.
            feature_types: Optional subset of feature types to return.
                           Recognised values: "Active site", "Binding site",
                           "Domain", "Region", "Motif", "Site",
                           "Natural variant", "Modified residue",
                           "Disulfide bond".
                           ``None`` returns every type.

        Returns:
            JSON string with keys:
            ``accession``, ``protein_name``, ``sequence_length``,
            ``features`` (list of {type, start, end, description}),
            ``cross_references`` ({interpro, pfam, panther}).
            On failure: ``{"error": "..."}``.
        """
        result = _fetch_features(accession, feature_types)
        return json.dumps(result, indent=2)
