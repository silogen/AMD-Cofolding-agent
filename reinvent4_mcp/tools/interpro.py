"""InterPro domain classification fetcher + RCSB nucleic acid chain annotator.

No equivalent exists in the existing codebase.

Provides two MCP tools:

  fetch_interpro_annotations(uniprot_accession, entry_types?) -> JSON str
      Fetches domain/family/site annotations from the EBI InterPro REST
      API for a given UniProt accession.  Results are optionally cached
      to ``/shared/annotations/interpro/{accession}.json``.

  fetch_nucleic_annotation(pdb_id, chain_id) -> JSON str
      Fetches nucleic-acid chain metadata from the RCSB REST API,
      returning sequence, type (dna/rna), and an RNAcentral cross-ref
      when available.

Network failures return ``{"error": "..."}``. Data is never fabricated.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

_INTERPRO_BASE = "https://www.ebi.ac.uk/interpro/api"
_RCSB_BASE = "https://data.rcsb.org/rest/v1/core"
_TIMEOUT = 30

# Default entry types included when caller passes None.
_DEFAULT_ENTRY_TYPES: tuple[str, ...] = ("Domain", "Family", "Site")

# Annotation service (block 10).  When set, InterPro fetches go through the
# service (which handles caching + 204 normalisation).  Falls back to the
# direct-API + local-file-cache path when absent.
_ANN_URL: str = os.environ.get("ANNOTATION_SERVICE_URL", "").rstrip("/")


# ---------------------------------------------------------------------------
# InterPro helpers
# ---------------------------------------------------------------------------

def _interpro_cache_path(accession: str) -> Path | None:
    """Return the cache file path if the cache root exists, else None."""
    cache_root = Path("/shared/annotations")
    if cache_root.is_dir():
        subdir = cache_root / "interpro"
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{accession.upper()}.json"
    return None


def _load_cached_interpro(accession: str) -> dict[str, Any] | None:
    cache = _interpro_cache_path(accession)
    if cache is None or not cache.is_file():
        return None
    try:
        return json.loads(cache.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Failed to read InterPro cache %s: %s", cache, exc)
        return None


def _save_interpro_cache(accession: str, data: dict[str, Any]) -> None:
    cache = _interpro_cache_path(accession)
    if cache is None:
        return
    try:
        cache.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("Cached InterPro annotations to %s", cache)
    except Exception as exc:
        logger.debug("Failed to write InterPro cache %s: %s", cache, exc)


def _fetch_interpro(
    accession: str,
    entry_types: list[str] | None,
) -> dict[str, Any]:
    """Core InterPro fetch — separated for testability.

    Route priority
    --------------
    1. ``ANNOTATION_SERVICE_URL`` set → ``POST /interpro/domains`` on the
       annotation service (handles caching + HTTP 204 normalisation).
    2. Local file cache at ``/shared/annotations/interpro/{ACC}.json``.
    3. Direct EBI InterPro API call.
    """
    accession = accession.strip().upper()

    # ── 1. Annotation service ────────────────────────────────────────────────
    if _ANN_URL:
        try:
            resp = requests.post(
                f"{_ANN_URL}/interpro/domains",
                json={"uniprot_accession": accession, "entry_types": entry_types},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning(
                "Annotation service InterPro call failed for %s: %s. Falling back.",
                accession, exc,
            )
            # fall through to local cache / direct API

    # ── 2. Local file cache (legacy path) ────────────────────────────────────
    # Try cache first (before network call).
    cached = _load_cached_interpro(accession)
    if cached is not None:
        # Still apply the entry_types filter if requested.
        if entry_types is not None and "interpro_entries" in cached:
            requested = {t.lower() for t in entry_types}
            cached = dict(cached)
            cached["interpro_entries"] = [
                e for e in cached["interpro_entries"]
                if e.get("type", "").lower() in requested
            ]
        return cached

    url = (
        f"{_INTERPRO_BASE}/entry/interpro/protein/uniprot"
        f"/{accession}/?format=json&page_size=100"
    )
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        logger.warning("InterPro request failed for %s: %s", accession, exc)
        return {
            "error": (
                f"Could not reach InterPro for accession {accession!r}: {exc}. "
                "Data not fabricated."
            )
        }

    if resp.status_code == 404:
        return {
            "error": (
                f"No InterPro entries found for UniProt accession {accession!r} (HTTP 404)."
            )
        }
    if resp.status_code != 200:
        return {
            "error": (
                f"InterPro API returned HTTP {resp.status_code} for "
                f"accession {accession!r}."
            )
        }

    try:
        payload = resp.json()
    except ValueError as exc:
        return {"error": f"InterPro returned non-JSON response: {exc}"}

    raw_results = payload.get("results") or []
    entries: list[dict[str, Any]] = []

    for item in raw_results:
        metadata = item.get("metadata") or {}
        ipr_acc = metadata.get("accession", "")
        ipr_name = metadata.get("name", "")
        ipr_type = metadata.get("type", "")

        # Location fragments for this protein.
        proteins = item.get("proteins") or []
        residue_ranges: list[dict[str, int]] = []
        for prot in proteins:
            if (prot.get("accession") or "").upper() != accession:
                continue
            for entry_loc in prot.get("entry_protein_locations") or []:
                for fragment in entry_loc.get("fragments") or []:
                    start = fragment.get("start")
                    end = fragment.get("end")
                    if start is not None and end is not None:
                        residue_ranges.append({"start": start, "end": end})

        # Source databases (Pfam, PANTHER, etc.) that contributed this entry.
        source_databases: list[dict[str, str]] = []
        for db_name, db_entries in (item.get("entry_integrated") or {}).items():
            for db_entry in db_entries or []:
                db_acc = db_entry.get("accession", "")
                if db_acc:
                    source_databases.append({"db": db_name, "accession": db_acc})
        # Also extract from "cross_references" when present.
        for xref in (item.get("cross_references") or []):
            db_name = xref.get("db", "")
            db_acc = xref.get("accession", "")
            if db_name and db_acc:
                source_databases.append({"db": db_name, "accession": db_acc})

        entries.append(
            {
                "accession": ipr_acc,
                "name": ipr_name,
                "type": ipr_type,
                "residue_ranges": residue_ranges,
                "source_databases": source_databases,
            }
        )

    full_result: dict[str, Any] = {
        "uniprot_accession": accession,
        "interpro_entries": entries,
    }

    # Persist to cache (unfiltered) before applying the type filter.
    _save_interpro_cache(accession, full_result)

    if entry_types is not None:
        requested = {t.lower() for t in entry_types}
        full_result["interpro_entries"] = [
            e for e in entries if e.get("type", "").lower() in requested
        ]

    return full_result


# ---------------------------------------------------------------------------
# RCSB nucleic acid helpers
# ---------------------------------------------------------------------------

def _fetch_nucleic(pdb_id: str, chain_id: str) -> dict[str, Any]:
    """Core RCSB nucleic-acid fetch — separated for testability."""
    pdb_id = pdb_id.strip().upper()
    chain_id = chain_id.strip()

    # The RCSB polymer-entity endpoint is keyed by PDB ID + entity index.
    # We first fetch the entry-level info to identify which entity corresponds
    # to the requested chain, then pull per-entity sequence data.
    entry_url = f"{_RCSB_BASE}/entry/{pdb_id}"
    try:
        entry_resp = requests.get(entry_url, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        logger.warning("RCSB entry request failed for %s: %s", pdb_id, exc)
        return {
            "error": (
                f"Could not reach RCSB for PDB entry {pdb_id!r}: {exc}. "
                "Data not fabricated."
            )
        }

    if entry_resp.status_code == 404:
        return {"error": f"PDB entry {pdb_id!r} not found (HTTP 404)."}
    if entry_resp.status_code != 200:
        return {
            "error": (
                f"RCSB returned HTTP {entry_resp.status_code} for PDB entry {pdb_id!r}."
            )
        }

    try:
        entry_data = entry_resp.json()
    except ValueError as exc:
        return {"error": f"RCSB returned non-JSON response for entry {pdb_id!r}: {exc}"}

    # Walk through polymer entities to find the one whose auth_asym_ids list
    # contains chain_id.
    polymer_entities = (
        entry_data.get("rcsb_entry_container_identifiers", {}) or {}
    ).get("polymer_entity_ids") or []

    target_entity_id: str | None = None
    for entity_id in polymer_entities:
        entity_url = f"{_RCSB_BASE}/polymer_entity/{pdb_id}/{entity_id}"
        try:
            e_resp = requests.get(entity_url, timeout=_TIMEOUT)
            if e_resp.status_code != 200:
                continue
            e_data = e_resp.json()
        except Exception:
            continue

        # Check if our chain_id is in the auth asymmetric IDs for this entity.
        chain_ids = (
            (e_data.get("rcsb_polymer_entity_container_identifiers") or {})
            .get("auth_asym_ids") or []
        )
        if chain_id in chain_ids:
            target_entity_id = entity_id
            # Determine nucleic acid type.
            entity_type = (
                (e_data.get("entity") or {}).get("type", "").lower()
            )
            poly_type = (
                (e_data.get("rcsb_polymer_entity") or {})
                .get("type", "").lower()
            )
            combined_type = entity_type + " " + poly_type

            if "rna" in combined_type:
                nuc_type = "rna"
            elif "dna" in combined_type:
                nuc_type = "dna"
            else:
                # Fall back to checking the actual sequence letters.
                nuc_type = "unknown"

            # Extract the canonical sequence.
            sequence = (
                (e_data.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can")
                or (e_data.get("entity_poly") or {}).get("pdbx_seq_one_letter_code")
                or ""
            )
            sequence = sequence.replace("\n", "").replace(" ", "").upper()

            if nuc_type == "unknown" and sequence:
                # Heuristic: RNA contains U; DNA is typically all A/T/C/G.
                if "U" in sequence:
                    nuc_type = "rna"
                elif set(sequence).issubset({"A", "T", "C", "G", "N", "-"}):
                    nuc_type = "dna"

            base_identity = list(sequence) if sequence else []

            # RNAcentral cross-reference (RNA only).
            rnacentral_id: str | None = None
            if nuc_type == "rna":
                for xref in (e_data.get("rcsb_external_references") or []):
                    if (xref.get("type") or "").upper() == "RNACENTRAL":
                        rnacentral_id = xref.get("reference_id")
                        break

            return {
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "type": nuc_type,
                "sequence": sequence,
                "length": len(sequence),
                "rnacentral_id": rnacentral_id,
                "base_identity": base_identity,
            }

    return {
        "error": (
            f"Chain {chain_id!r} not found in polymer entities of PDB entry {pdb_id!r}. "
            "Verify the chain ID and that the chain is a nucleic acid."
        )
    }


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register_tools(mcp) -> None:
    """Register ``fetch_interpro_annotations`` and ``fetch_nucleic_annotation``."""

    @mcp.tool()
    def fetch_interpro_annotations(
        uniprot_accession: str,
        entry_types: list[str] | None = None,
    ) -> str:
        """Fetch InterPro domain annotations for a UniProt accession.

        Queries the EBI InterPro REST API and returns all domain, family,
        and site entries that cover the protein, together with their
        residue ranges and contributing source databases (Pfam, PANTHER, …).

        Results are cached to ``/shared/annotations/interpro/{accession}.json``
        when the ``/shared/annotations/`` directory exists, so repeated calls
        for the same accession are served instantly from disk.

        Args:
            uniprot_accession: UniProt accession, e.g. ``"P24941"`` for CDK2.
            entry_types:       Optional filter list; recognised values include
                               ``"Domain"``, ``"Family"``, ``"Site"``,
                               ``"Homologous_superfamily"``, ``"Repeat"``.
                               ``None`` defaults to ``["Domain", "Family", "Site"]``.

        Returns:
            JSON string with keys:
            ``uniprot_accession`` and ``interpro_entries`` (list of objects
            with ``accession``, ``name``, ``type``, ``residue_ranges``,
            ``source_databases``).
            On failure: ``{"error": "..."}``.
        """
        if entry_types is None:
            entry_types = list(_DEFAULT_ENTRY_TYPES)
        result = _fetch_interpro(uniprot_accession, entry_types)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def fetch_nucleic_annotation(
        pdb_id: str,
        chain_id: str,
    ) -> str:
        """Fetch nucleic-acid chain annotations from RCSB for a PDB structure.

        Identifies whether the specified chain is DNA or RNA, returns the
        canonical sequence, and provides an RNAcentral cross-reference when
        available.

        Args:
            pdb_id:   Four-letter PDB identifier, e.g. ``"1BNA"``.
            chain_id: Author asymmetric chain ID, e.g. ``"A"``.

        Returns:
            JSON string with keys:
            ``pdb_id``, ``chain_id``, ``type`` (``"dna"`` or ``"rna"``),
            ``sequence``, ``length``, ``rnacentral_id`` (null if not
            RNA or not cross-referenced), ``base_identity`` (list of
            single-character bases).
            On failure: ``{"error": "..."}``.
        """
        result = _fetch_nucleic(pdb_id, chain_id)
        return json.dumps(result, indent=2)
