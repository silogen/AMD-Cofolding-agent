"""Annotation microservice — UniProt + InterPro + RCSB nucleic-acid annotations.

Thin caching proxy in front of external annotation APIs.  All results are
persisted under ``/shared/annotations/`` so repeated calls for the same
accession are served from disk without hitting the upstream APIs again.

Key feature: ``POST /uniprot/features`` returns both ``features[]`` AND
``sequence`` in a single upstream request, eliminating the double API hit
that occurs when callers fetch them separately.

Endpoints
---------
GET  /health               liveness + cache statistics
POST /uniprot/features     UniProt features + sequence (cached)
POST /interpro/domains     InterPro domain entries (cached, 204 → [])
POST /nucleic/info         RCSB polymer entity DNA/RNA + RNAcentral
GET  /cache/status         list all cached accessions per category

Run:
    uvicorn app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("annotation_service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Annotation Service",
    description=(
        "Caching proxy for UniProt, InterPro and RCSB nucleic-acid annotations. "
        "Results are stored at ANNOTATION_CACHE_DIR (/shared/annotations by default)."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CACHE_ROOT = Path(os.environ.get("ANNOTATION_CACHE_DIR", "/shared/annotations"))
_UNIPROT_BASE  = "https://rest.uniprot.org/uniprotkb"
_INTERPRO_BASE = "https://www.ebi.ac.uk/interpro/api"
_RCSB_BASE     = "https://data.rcsb.org/rest/v1/core"
_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(category: str, key: str) -> Path:
    """Return the cache file path for a (category, key) pair.

    Cache is stored at ``{CACHE_ROOT}/{category}/{KEY}.json``.
    The directory is created on first use.
    """
    subdir = _CACHE_ROOT / category
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{key.upper()}.json"


def _load_cache(category: str, key: str) -> dict[str, Any] | None:
    p = _cache_path(category, key)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Cache read failed %s: %s", p, exc)
        return None


def _save_cache(category: str, key: str, data: dict[str, Any]) -> None:
    p = _cache_path(category, key)
    try:
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("Cache write failed %s: %s", p, exc)


def _list_cached(category: str) -> list[str]:
    subdir = _CACHE_ROOT / category
    if not subdir.is_dir():
        return []
    return sorted(p.stem for p in subdir.glob("*.json"))


# ---------------------------------------------------------------------------
# UniProt helpers
# ---------------------------------------------------------------------------

def _parse_features(raw: list[dict]) -> list[dict]:
    out = []
    for feat in raw or []:
        loc = feat.get("location") or {}
        entry: dict[str, Any] = {"type": feat.get("type", "")}
        start = (loc.get("start") or {}).get("value")
        end   = (loc.get("end")   or {}).get("value")
        if start is not None:
            entry["start"] = start
        if end is not None:
            entry["end"] = end
        desc = feat.get("description", "")
        if desc:
            entry["description"] = desc
        out.append(entry)
    return out


def _parse_cross_references(refs: list[dict]) -> dict[str, list]:
    interpro, pfam, panther = [], [], []
    for ref in refs or []:
        db = (ref.get("database") or "").lower()
        rid = ref.get("id", "")
        if not rid:
            continue
        if db == "interpro":
            name = ""
            for prop in ref.get("properties") or []:
                if prop.get("key") == "EntryName":
                    name = prop.get("value", "")
                    break
            interpro.append({"id": rid, "name": name} if name else {"id": rid})
        elif db == "pfam":
            pfam.append({"id": rid})
        elif db == "panther":
            panther.append({"id": rid})
    return {"interpro": interpro, "pfam": pfam, "panther": panther}


def _fetch_uniprot_live(accession: str) -> dict[str, Any]:
    """Single upstream hit: returns features[], sequence, and metadata together."""
    url = f"{_UNIPROT_BASE}/{accession}.json"
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        return {"error": f"UniProt unreachable for {accession!r}: {exc}"}

    if resp.status_code == 404:
        return {"error": f"UniProt accession {accession!r} not found (HTTP 404)."}
    if resp.status_code != 200:
        return {"error": f"UniProt returned HTTP {resp.status_code} for {accession!r}."}

    try:
        data = resp.json()
    except ValueError as exc:
        return {"error": f"UniProt returned non-JSON: {exc}"}

    # Protein name
    pdesc = data.get("proteinDescription") or {}
    rec = pdesc.get("recommendedName") or {}
    protein_name = (rec.get("fullName") or {}).get("value", "")
    if not protein_name:
        submitted = pdesc.get("submissionNames") or []
        if submitted:
            protein_name = (submitted[0].get("fullName") or {}).get("value", "")

    seq_obj = data.get("sequence") or {}
    return {
        "accession":      accession.upper(),
        "protein_name":   protein_name,
        "sequence_length": seq_obj.get("length"),
        "sequence":        seq_obj.get("value", ""),       # ← returned alongside features
        "features":        _parse_features(data.get("features") or []),
        "cross_references": _parse_cross_references(data.get("uniProtKBCrossReferences") or []),
    }


# ---------------------------------------------------------------------------
# InterPro helpers
# ---------------------------------------------------------------------------

def _fetch_interpro_live(accession: str) -> dict[str, Any]:
    url = (
        f"{_INTERPRO_BASE}/entry/interpro/protein/uniprot"
        f"/{accession}/?format=json&page_size=100"
    )
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        return {"error": f"InterPro unreachable for {accession!r}: {exc}"}

    # 204 = valid accession but no InterPro entries — not an error
    if resp.status_code == 204:
        return {"uniprot_accession": accession.upper(), "interpro_entries": []}
    if resp.status_code != 200:
        return {"error": f"InterPro returned HTTP {resp.status_code} for {accession!r}."}

    try:
        payload = resp.json()
    except ValueError as exc:
        return {"error": f"InterPro returned non-JSON: {exc}"}

    entries: list[dict] = []
    for item in payload.get("results") or []:
        meta = item.get("metadata") or {}
        ipr_acc  = meta.get("accession", "")
        ipr_name = meta.get("name", "")
        ipr_type = meta.get("type", "")

        # Residue ranges for this specific protein
        residue_ranges: list[dict] = []
        for prot in item.get("proteins") or []:
            if (prot.get("accession") or "").upper() != accession.upper():
                continue
            for loc in prot.get("entry_protein_locations") or []:
                for frag in loc.get("fragments") or []:
                    s, e = frag.get("start"), frag.get("end")
                    if s is not None and e is not None:
                        residue_ranges.append({"start": s, "end": e})

        # Source databases (Pfam, PANTHER, …)
        source_dbs: list[dict] = []
        for db_name, db_list in (item.get("entry_integrated") or {}).items():
            for db_entry in db_list or []:
                db_acc = db_entry.get("accession", "")
                if db_acc:
                    source_dbs.append({"db": db_name, "accession": db_acc})
        for xref in item.get("cross_references") or []:
            if xref.get("db") and xref.get("accession"):
                source_dbs.append({"db": xref["db"], "accession": xref["accession"]})

        entries.append({
            "accession":        ipr_acc,
            "name":             ipr_name,
            "type":             ipr_type,
            "residue_ranges":   residue_ranges,
            "source_databases": source_dbs,
        })

    return {"uniprot_accession": accession.upper(), "interpro_entries": entries}


# ---------------------------------------------------------------------------
# RCSB nucleic-acid helper
# ---------------------------------------------------------------------------

def _fetch_nucleic_live(pdb_id: str, chain_id: str) -> dict[str, Any]:
    """Walk RCSB polymer entities to find the chain and return sequence + type."""
    pdb_id   = pdb_id.strip().upper()
    chain_id = chain_id.strip()

    try:
        entry_resp = requests.get(f"{_RCSB_BASE}/entry/{pdb_id}", timeout=_TIMEOUT)
    except requests.RequestException as exc:
        return {"error": f"RCSB unreachable for {pdb_id!r}: {exc}"}

    if entry_resp.status_code == 404:
        return {"error": f"PDB entry {pdb_id!r} not found (HTTP 404)."}
    if entry_resp.status_code != 200:
        return {"error": f"RCSB returned HTTP {entry_resp.status_code} for {pdb_id!r}."}

    try:
        entry_data = entry_resp.json()
    except ValueError as exc:
        return {"error": f"RCSB returned non-JSON for {pdb_id!r}: {exc}"}

    entity_ids = (
        (entry_data.get("rcsb_entry_container_identifiers") or {})
        .get("polymer_entity_ids") or []
    )

    for eid in entity_ids:
        try:
            e_resp = requests.get(f"{_RCSB_BASE}/polymer_entity/{pdb_id}/{eid}", timeout=_TIMEOUT)
            if e_resp.status_code != 200:
                continue
            e_data = e_resp.json()
        except Exception:
            continue

        chain_ids_in_entity = (
            (e_data.get("rcsb_polymer_entity_container_identifiers") or {})
            .get("auth_asym_ids") or []
        )
        if chain_id not in chain_ids_in_entity:
            continue

        entity_type = (e_data.get("entity") or {}).get("type", "").lower()
        poly_type   = (e_data.get("rcsb_polymer_entity") or {}).get("type", "").lower()
        combined    = f"{entity_type} {poly_type}"

        if "rna" in combined:
            nuc_type = "rna"
        elif "dna" in combined:
            nuc_type = "dna"
        else:
            nuc_type = "unknown"

        sequence = (
            (e_data.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can")
            or (e_data.get("entity_poly") or {}).get("pdbx_seq_one_letter_code")
            or ""
        ).replace("\n", "").replace(" ", "").upper()

        if nuc_type == "unknown" and sequence:
            if "U" in sequence:
                nuc_type = "rna"
            elif set(sequence).issubset(set("ATCGN-")):
                nuc_type = "dna"

        rnacentral_id = None
        if nuc_type == "rna":
            for xref in e_data.get("rcsb_external_references") or []:
                if (xref.get("type") or "").upper() == "RNACENTRAL":
                    rnacentral_id = xref.get("reference_id")
                    break

        return {
            "pdb_id":        pdb_id,
            "chain_id":      chain_id,
            "type":          nuc_type,
            "sequence":      sequence,
            "length":        len(sequence),
            "rnacentral_id": rnacentral_id,
        }

    return {
        "error": (
            f"Chain {chain_id!r} not found in polymer entities of PDB entry {pdb_id!r}. "
            "Verify the chain ID and that the chain is a nucleic acid."
        )
    }


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class UniprotFeaturesRequest(BaseModel):
    accession: str = Field(description="UniProt accession, e.g. 'P24941'.")
    feature_types: list[str] | None = Field(
        default=None,
        description=(
            "Optional filter; e.g. ['Active site', 'Binding site', 'Domain']. "
            "None returns every feature type."
        ),
    )


class InterproDomainRequest(BaseModel):
    uniprot_accession: str = Field(description="UniProt accession, e.g. 'P24941'.")
    entry_types: list[str] | None = Field(
        default=None,
        description="Optional filter: ['Domain', 'Family', 'Site']. None = all types.",
    )


class NucleicInfoRequest(BaseModel):
    pdb_id:   str = Field(description="Four-letter PDB identifier, e.g. '1BNA'.")
    chain_id: str = Field(description="Author asymmetric chain ID, e.g. 'A'.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check with cache statistics."""
    cached_uniprot  = _list_cached("uniprot")
    cached_interpro = _list_cached("interpro")
    return {
        "status": "ok",
        "service": "annotation",
        "version": "1.0.0",
        "cache_dir": str(_CACHE_ROOT),
        "cached_uniprot_count":  len(cached_uniprot),
        "cached_interpro_count": len(cached_interpro),
    }


@app.post("/uniprot/features")
def uniprot_features(req: UniprotFeaturesRequest) -> dict[str, Any]:
    """Fetch UniProt features and sequence for an accession.

    Returns features[], sequence, protein_name, sequence_length, and
    cross_references in a **single upstream API hit**.  Results are cached
    at ``/shared/annotations/uniprot/{ACC}.json``; subsequent calls are
    served entirely from disk.

    When ``feature_types`` is provided the returned ``features[]`` list is
    filtered, but the full unfiltered payload is always persisted to cache
    so that different callers with different filters share the same file.
    """
    acc = req.accession.strip().upper()
    if not acc:
        raise HTTPException(status_code=400, detail="accession must not be empty.")

    # Cache lookup (unfiltered data)
    cached = _load_cache("uniprot", acc)
    if cached is None:
        t0 = time.monotonic()
        cached = _fetch_uniprot_live(acc)
        elapsed = time.monotonic() - t0
        if "error" not in cached:
            _save_cache("uniprot", acc, cached)
            logger.info("UniProt %s fetched in %.2fs and cached.", acc, elapsed)
        else:
            logger.warning("UniProt fetch error for %s: %s", acc, cached["error"])
            raise HTTPException(status_code=502, detail=cached["error"])
    else:
        logger.debug("UniProt %s served from cache.", acc)

    # Apply optional feature_types filter (never mutates the cached payload)
    if req.feature_types is not None:
        requested = {ft.lower() for ft in req.feature_types}
        result = dict(cached)
        result["features"] = [
            f for f in cached.get("features", [])
            if f.get("type", "").lower() in requested
        ]
        return result

    return cached


@app.post("/interpro/domains")
def interpro_domains(req: InterproDomainRequest) -> dict[str, Any]:
    """Fetch InterPro domain entries for a UniProt accession.

    HTTP 204 from InterPro (valid accession, no entries) is normalised to an
    empty list — never treated as an error.  Results are cached at
    ``/shared/annotations/interpro/{ACC}.json``.
    """
    acc = req.uniprot_accession.strip().upper()
    if not acc:
        raise HTTPException(status_code=400, detail="uniprot_accession must not be empty.")

    cached = _load_cache("interpro", acc)
    if cached is None:
        t0 = time.monotonic()
        cached = _fetch_interpro_live(acc)
        elapsed = time.monotonic() - t0
        if "error" not in cached:
            _save_cache("interpro", acc, cached)
            logger.info("InterPro %s fetched in %.2fs and cached.", acc, elapsed)
        else:
            logger.warning("InterPro fetch error for %s: %s", acc, cached["error"])
            raise HTTPException(status_code=502, detail=cached["error"])
    else:
        logger.debug("InterPro %s served from cache.", acc)

    # Apply optional entry_types filter
    if req.entry_types is not None:
        requested = {t.lower() for t in req.entry_types}
        result = dict(cached)
        result["interpro_entries"] = [
            e for e in cached.get("interpro_entries", [])
            if e.get("type", "").lower() in requested
        ]
        return result

    return cached


@app.post("/nucleic/info")
def nucleic_info(req: NucleicInfoRequest) -> dict[str, Any]:
    """Fetch DNA/RNA chain metadata from RCSB.

    Walks the polymer entity list of ``pdb_id`` to find the chain whose
    author asymmetric ID matches ``chain_id``, determines the nucleic-acid
    type (dna / rna / unknown), returns the canonical sequence, and provides
    an RNAcentral cross-reference when available.

    Results are **not cached** (RCSB data changes with PDB updates).
    """
    pdb_id   = req.pdb_id.strip().upper()
    chain_id = req.chain_id.strip()
    if not pdb_id or not chain_id:
        raise HTTPException(status_code=400, detail="pdb_id and chain_id must not be empty.")

    result = _fetch_nucleic_live(pdb_id, chain_id)
    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])
    return result


@app.get("/cache/status")
def cache_status() -> dict[str, Any]:
    """List all accessions currently in the on-disk cache."""
    return {
        "cache_dir":        str(_CACHE_ROOT),
        "uniprot_cached":   _list_cached("uniprot"),
        "interpro_cached":  _list_cached("interpro"),
        "n_uniprot":        len(_list_cached("uniprot")),
        "n_interpro":       len(_list_cached("interpro")),
    }
