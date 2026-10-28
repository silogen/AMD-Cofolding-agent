# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""UniProt target-lookup helper.

Fetches canonical protein information from the UniProt REST API so the agent
can ground a TTP target in a real sequence and candidate PDB structures
instead of reciting a sequence from memory.

The single entry point ``fetch_target_info`` accepts either a UniProt
accession (e.g. ``O60885``) or a free-text query (e.g. ``"BRD4 human"``) and
returns the primary accession, names, organism, the full-length canonical
sequence, a short function summary, the cross-referenced PDB entries ranked
best first, and, from the RCSB data API, the construct actually crystallised
in the top entries (usually one domain). Network failures return a clear
error dict -- the caller never fabricates data.
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
_MAX_STRUCTURES = 10
_MAX_CONSTRUCTS = 3
_RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
# (connect, read) seconds; one request for all entries.
_RCSB_TIMEOUT = (5, 15)
# Entries covering fewer UniProt residues are peptide complexes, not a domain.
_MIN_CONSTRUCT = 50
_METHOD_RANK = {"x-ray": 0, "em": 1, "electron microscopy": 1, "nmr": 2}

_CONSTRUCTS_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    rcsb_entry_info { resolution_combined experimental_method }
    polymer_entities {
      rcsb_polymer_entity { pdbx_description }
      entity_poly { pdbx_seq_one_letter_code_can rcsb_sample_sequence_length type }
      rcsb_polymer_entity_container_identifiers { auth_asym_ids uniprot_ids }
      rcsb_polymer_entity_align {
        reference_database_name reference_database_accession
        aligned_regions { entity_beg_seq_id ref_beg_seq_id length }
      }
    }
    nonpolymer_entities { nonpolymer_comp { chem_comp { id name } } }
  }
}
"""

_COFOLD_ADVICE = (
    "`sequence` is the FULL-LENGTH UniProt protein ({full} aa), not the construct in the "
    "PDB structures. To cofold, use the sequence of the construct in the structure you "
    "work with (`constructs[].sequence`, {example}) unless the user asks for full length: "
    "Boltz time grows with the square to cube of the length, so a ~300-residue domain folds "
    "10-50x faster than a 1,000+-residue protein, and its MSA must match the same sequence."
)

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

    structures = _rank_structures([
        _pdb_xref(x)
        for x in d.get("uniProtKBCrossReferences", []) or []
        if x.get("database") == "PDB" and x.get("id")
    ])

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
        "sequence_kind": "full_length",
        "function": function,
        "pdb_ids": [s["pdb_id"] for s in structures[:_MAX_PDB]],
        "pdb_count": len(structures),
        "pdb_structures": [
            {k: s[k] for k in ("pdb_id", "method", "resolution", "uniprot_range")}
            for s in structures[:_MAX_STRUCTURES]
        ],
    }


def _pdb_xref(xref: dict[str, Any]) -> dict[str, Any]:
    """One UniProt PDB cross-reference: method, resolution and covered range.

    UniProt writes the chains as ``"A/B=695-1022"`` or ``"A=1-90, B=100-200"``;
    the longest segment is the construct's range on the UniProt sequence.
    """
    props = {p.get("key"): p.get("value") for p in xref.get("properties", []) or []}
    resolution = None
    match = re.match(r"\s*([\d.]+)", props.get("Resolution") or "")
    if match:
        try:
            resolution = float(match.group(1))
        except ValueError:
            resolution = None
    best: Optional[tuple[int, int]] = None
    for beg, end in re.findall(r"(\d+)\s*-\s*(\d+)", props.get("Chains") or ""):
        span = (int(beg), int(end))
        if best is None or span[1] - span[0] > best[1] - best[0]:
            best = span
    return {
        "pdb_id": xref.get("id"),
        "method": props.get("Method"),
        "resolution": resolution,
        "uniprot_range": list(best) if best else None,
        "covered": best[1] - best[0] + 1 if best else None,
    }


def _rank_structures(structures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Best first: a domain-sized construct, X-ray before EM before NMR, then resolution."""
    def key(s: dict[str, Any]) -> tuple:
        small = (s.get("covered") or 0) < _MIN_CONSTRUCT
        method = _METHOD_RANK.get((s.get("method") or "").lower(), 3)
        resolution = s.get("resolution")
        return (small, method, resolution if resolution is not None else 99.0, s.get("pdb_id") or "")
    return sorted(structures, key=key)


def _construct(entry: dict[str, Any], accession: Optional[str]) -> Optional[dict[str, Any]]:
    """The polymer entity of *entry* that is the target protein, as crystallised."""
    proteins = [
        e for e in entry.get("polymer_entities") or []
        if "polypeptide" in ((e.get("entity_poly") or {}).get("type") or "polypeptide")
    ]
    if not proteins:
        return None

    def uniprots(e: dict[str, Any]) -> list[str]:
        return (e.get("rcsb_polymer_entity_container_identifiers") or {}).get("uniprot_ids") or []

    mine = [e for e in proteins if accession and accession in uniprots(e)] or proteins
    entity = max(mine, key=lambda e: len((e.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can") or ""))
    poly = entity.get("entity_poly") or {}
    sequence = re.sub(r"\s+", "", poly.get("pdbx_seq_one_letter_code_can") or "")

    uniprot_range = None
    for align in entity.get("rcsb_polymer_entity_align") or []:
        if align.get("reference_database_name") != "UniProt":
            continue
        if accession and align.get("reference_database_accession") != accession:
            continue
        regions = [r for r in align.get("aligned_regions") or []
                   if isinstance(r.get("ref_beg_seq_id"), int) and isinstance(r.get("length"), int)]
        if regions:
            first, last = regions[0], regions[-1]
            uniprot_range = [first["ref_beg_seq_id"], last["ref_beg_seq_id"] + last["length"] - 1]
        break

    info = entry.get("rcsb_entry_info") or {}
    resolutions = info.get("resolution_combined") or []
    ligands = []
    for e in entry.get("nonpolymer_entities") or []:
        comp = ((e.get("nonpolymer_comp") or {}).get("chem_comp")) or {}
        if comp.get("id"):
            name = comp.get("name") or "?"
            ligands.append(f"{comp['id']} ({name if len(name) <= 60 else name[:57] + '...'})")
    return {
        "pdb_id": entry.get("rcsb_id"),
        "title": (entry.get("struct") or {}).get("title"),
        "method": info.get("experimental_method"),
        "resolution": resolutions[0] if resolutions else None,
        "entity": (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description"),
        "chains": (entity.get("rcsb_polymer_entity_container_identifiers") or {}).get("auth_asym_ids"),
        "length": poly.get("rcsb_sample_sequence_length") or len(sequence),
        "uniprot_range": uniprot_range,
        "ligands": ligands,
        "sequence": sequence,
    }


def fetch_constructs(pdb_ids: list[str], accession: Optional[str] = None) -> dict[str, Any]:
    """The crystallised construct of each PDB entry, from the RCSB data API.

    One GraphQL request for all entries, bounded by ``_RCSB_TIMEOUT``. The
    sequence is the deposited one (SEQRES), tags and mutations included.
    """
    ids = [i.strip().upper() for i in pdb_ids if i and i.strip()][:_MAX_CONSTRUCTS]
    if not ids:
        return {"constructs": []}
    try:
        resp = requests.post(
            _RCSB_GRAPHQL,
            json={"query": _CONSTRUCTS_QUERY, "variables": {"ids": ids}},
            timeout=_RCSB_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json() or {}
    except (requests.RequestException, ValueError) as exc:
        logger.warning("RCSB construct lookup failed: %s", exc)
        return {"constructs": [], "constructs_error": f"RCSB data API unreachable ({exc}); constructs not retrieved."}
    entries = (payload.get("data") or {}).get("entries") or []
    constructs = [c for c in (_construct(e, accession) for e in entries if e) if c]
    out: dict[str, Any] = {"constructs": constructs}
    if payload.get("errors") and not constructs:
        out["constructs_error"] = f"RCSB data API error: {payload['errors'][0].get('message')}"
    return out


def _with_constructs(info: dict[str, Any], pdb_ids: Optional[list[str]]) -> dict[str, Any]:
    ids = pdb_ids or [s["pdb_id"] for s in info.get("pdb_structures", [])[:_MAX_CONSTRUCTS]]
    info.update(fetch_constructs(ids, info.get("accession")))
    constructs = info.get("constructs") or []
    example = (
        f"e.g. {constructs[0]['pdb_id']}: {constructs[0]['length']} aa"
        + (f", UniProt {constructs[0]['uniprot_range'][0]}-{constructs[0]['uniprot_range'][1]}"
           if constructs[0].get("uniprot_range") else "")
        if constructs else "or a domain range from `pdb_structures[].uniprot_range`"
    )
    info["cofold_advice"] = _COFOLD_ADVICE.format(full=info.get("sequence_length"), example=example)
    return info


def fetch_target_info(query: str, pdb_ids: Optional[list[str]] = None) -> dict[str, Any]:
    """Look up a protein target on UniProt by accession or free-text query.

    Args:
        query: A UniProt accession (e.g. ``"O60885"``) or free text such as a
            gene name plus organism (e.g. ``"BRD4 human"``).
        pdb_ids: Entries whose construct to report; by default the top
            ranked ones (at most ``_MAX_CONSTRUCTS``).

    Returns:
        A dict with accession, names, organism, a short ``function`` summary,
        and:

        * ``sequence``: the canonical FULL-LENGTH sequence
          (``sequence_kind="full_length"``). Cofold it only when full length
          is asked for: deposited structures hold one domain or construct.
        * ``pdb_ids`` / ``pdb_structures``: PDB entries ranked best first
          (domain-sized, X-ray, resolution), with the UniProt residue range
          each covers.
        * ``constructs``: for the top entries, the crystallised polymer
          entity from RCSB -- its ``sequence``, ``length``, ``uniprot_range``
          and ligands. This is the sequence to cofold by default.
        * ``cofold_advice``: the above in one sentence for the agent.

        No structure is prepared. On failure returns ``{"error": ...}``; an
        unreachable RCSB leaves ``constructs`` empty with ``constructs_error``.
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
                return _with_constructs(_parse_entry(resp.json()), pdb_ids)
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
        return _with_constructs(info, pdb_ids)

    except requests.RequestException as exc:
        logger.warning("UniProt request failed: %s", exc)
        return {
            "error": (
                f"Could not reach UniProt ({exc}). The target sequence was NOT "
                "retrieved; do not fabricate one."
            )
        }
