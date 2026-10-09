# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""UniProt target-lookup helper.

Fetches canonical protein information from the UniProt REST API so the agent
can ground a TTP target in a real sequence and candidate PDB structures
instead of reciting a sequence from memory.

The single entry point ``fetch_target_info`` accepts either a UniProt
accession (e.g. ``O60885``) or a free-text query (e.g. ``"BRD4 human"``) and
returns the primary accession, names, organism, canonical amino-acid
sequence, a short function summary, and cross-referenced PDB IDs. Network
failures return a clear error dict -- the caller never fabricates data.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_BASE = "https://rest.uniprot.org/uniprotkb"
_TIMEOUT = 30
_MAX_PDB = 25

# Common organism words -> NCBI taxonomy IDs, so "BRD4 human" scopes the
# search to Homo sapiens instead of matching an unrelated reviewed entry.
_ORGANISM_IDS = {
    "human": 9606,
    "homo sapiens": 9606,
    "mouse": 10090,
    "mus musculus": 10090,
    "rat": 10116,
    "yeast": 559292,
    "zebrafish": 7955,
    "drosophila": 7227,
    "ecoli": 83333,
    "e. coli": 83333,
    "e.coli": 83333,
}

# UniProtKB accession pattern (see UniProt accession-number regex).
_ACCESSION_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
)


def _looks_like_accession(query: str) -> bool:
    return bool(_ACCESSION_RE.match(query.strip().upper()))


def _extract_organism(query: str) -> tuple[str, Optional[int]]:
    """Split a free-text query into (remaining_terms, organism_id).

    Recognizes common organism words (e.g. "human") and removes them so the
    remaining text can be used as a gene/name search scoped by organism.
    """
    low = query.lower()
    for word in sorted(_ORGANISM_IDS, key=len, reverse=True):
        if word in low:
            low = low.replace(word, " ")
            remaining = " ".join(low.split()).strip()
            return remaining, _ORGANISM_IDS[word]
    return query.strip(), None


def _parse_entry(d: dict[str, Any]) -> dict[str, Any]:
    """Extract the useful fields from a UniProtKB JSON entry."""
    protein_desc = d.get("proteinDescription", {}) or {}
    rec_name = (
        protein_desc.get("recommendedName", {}).get("fullName", {}).get("value")
    )
    if not rec_name:
        submitted = protein_desc.get("submissionNames") or []
        if submitted:
            rec_name = submitted[0].get("fullName", {}).get("value")

    genes = d.get("genes") or []
    gene_name = genes[0].get("geneName", {}).get("value") if genes else None

    function = None
    for comment in d.get("comments", []) or []:
        if comment.get("commentType") == "FUNCTION":
            texts = comment.get("texts") or []
            if texts:
                function = texts[0].get("value")
            break

    pdb_ids = [
        x.get("id")
        for x in d.get("uniProtKBCrossReferences", []) or []
        if x.get("database") == "PDB" and x.get("id")
    ]

    sequence = d.get("sequence", {}) or {}

    return {
        "accession": d.get("primaryAccession"),
        "entry_name": d.get("uniProtkbId"),
        "protein_name": rec_name,
        "gene": gene_name,
        "organism": d.get("organism", {}).get("scientificName"),
        "reviewed": d.get("entryType", "").lower().startswith("uniprotkb reviewed"),
        "sequence": sequence.get("value"),
        "sequence_length": sequence.get("length"),
        "function": function,
        "pdb_ids": pdb_ids[:_MAX_PDB],
        "pdb_count": len(pdb_ids),
    }


def fetch_target_info(query: str) -> dict[str, Any]:
    """Look up a protein target on UniProt by accession or free-text query.

    Args:
        query: A UniProt accession (e.g. ``"O60885"``) or free text such as a
            gene name plus organism (e.g. ``"BRD4 human"``).

    Returns:
        A dict with accession, names, organism, canonical ``sequence`` (usable
        directly as a TTP ``target.protein_sequence``), a short ``function``
        summary, and candidate ``pdb_ids``. On failure returns ``{"error": ...}``.
    """
    query = (query or "").strip()
    if not query:
        return {"error": "Empty query. Provide a UniProt accession or search text."}

    try:
        if _looks_like_accession(query):
            resp = requests.get(
                f"{_BASE}/{query.upper()}.json", timeout=_TIMEOUT
            )
            if resp.status_code == 200:
                return _parse_entry(resp.json())
            if resp.status_code != 404:
                return {
                    "error": (
                        f"UniProt returned HTTP {resp.status_code} for accession "
                        f"{query!r}."
                    )
                }
            # 404 -> fall through and try a text search instead.

        # Free-text (or accession miss): pull out an organism word, then try a
        # gene-scoped search first (so "BRD4 human" -> gene BRD4 in H. sapiens)
        # before falling back to a broad full-text search.
        remaining, organism_id = _extract_organism(query)
        org_clause = f" AND organism_id:{organism_id}" if organism_id else ""

        candidate_queries = []
        if remaining:
            candidate_queries.append(
                f"(gene:{remaining} OR gene_exact:{remaining}) AND reviewed:true{org_clause}"
            )
        candidate_queries.append(f"({remaining or query}) AND reviewed:true{org_clause}")
        candidate_queries.append(f"({remaining or query}){org_clause}")

        results: list[dict[str, Any]] = []
        for q in candidate_queries:
            params = {
                "query": q,
                "format": "json",
                "size": "1",
                "fields": (
                    "accession,id,protein_name,gene_names,organism_name,"
                    "sequence,cc_function,xref_pdb"
                ),
            }
            resp = requests.get(f"{_BASE}/search", params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            results = resp.json().get("results") or []
            if results:
                break

        if not results:
            return {"error": f"No UniProt entry found for query {query!r}."}
        info = _parse_entry(results[0])
        info["query"] = query
        return info

    except requests.RequestException as exc:
        logger.warning("UniProt request failed: %s", exc)
        return {
            "error": (
                f"Could not reach UniProt ({exc}). The target sequence was NOT "
                "retrieved; do not fabricate one."
            )
        }
