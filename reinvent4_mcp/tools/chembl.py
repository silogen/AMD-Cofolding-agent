"""ChEMBL REST API fetcher — 11 typed MCP operations.

Provides provenance-preserving access to ChEMBL data: molecules, targets,
activities, assays, mechanisms, drug indications, documents, similarity
searches, and substructure searches.

This is a data-retrieval tool. It never:
- Fabricates chemical data.
- Silently converts IC50/Ki/Kd/EC50 to pIC50 (records original values).
- Accepts or constructs arbitrary remote URLs.
- Imports structure preparation, profiler, REINVENT, or GNINA code.

Registration: imported by registry.py and activated via register_tools(mcp).

ChEMBL REST API v3 base: https://www.ebi.ac.uk/chembl/api/data
Documentation: https://chembl.gitbook.io/chembl-interface-documentation/web-services
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote, urlencode

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BASE_URL: str = os.environ.get(
    "CHEMBL_API_URL", "https://www.ebi.ac.uk/chembl/api/data"
).rstrip("/")
_TIMEOUT_CONNECT = 5.0
_TIMEOUT_READ    = 30.0
_TIMEOUT = (_TIMEOUT_CONNECT, _TIMEOUT_READ)
_MAX_RETRIES = 3
_RETRY_STATUSES = {429, 502, 503, 504}
_MAX_LIMIT = 1000   # ChEMBL hard cap
_DEFAULT_LIMIT = 20
_MAX_PAGES = 50     # prevent runaway pagination in single calls
_USER_AGENT = "Cofolding-Agent/1.0 ChEMBL-Fetcher/1.0"

# Common LLM aliases for standard_type values.
# ChEMBL stores them in mixed case; __iexact handles it but this map
# ensures the right canonical form is sent regardless of what the LLM writes.
_TYPE_ALIASES: dict[str, str] = {
    "ic50": "IC50", "ic 50": "IC50",
    "ki": "Ki",
    "kd": "Kd",
    "ec50": "EC50", "ec 50": "EC50",
    "gi50": "GI50",
    "logd": "logD", "log d": "logD",
    "logp": "LogP", "log p": "LogP",
    "papp": "Papp", "papp a-b": "Papp",
    "solubility": "Solubility",
    "clearance": "Clearance",
    "t1/2": "t1/2", "t½": "t1/2", "half life": "t1/2",
}


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class ChEMBLError(Exception):
    """Base ChEMBL fetcher error."""
    code: str = "chembl_error"

class ChEMBLNotFoundError(ChEMBLError):
    code = "not_found"

class ChEMBLRateLimitError(ChEMBLError):
    code = "rate_limit"

class ChEMBLUnavailableError(ChEMBLError):
    code = "service_unavailable"

class ChEMBLMalformedResponse(ChEMBLError):
    code = "malformed_response"

class ChEMBLValidationError(ChEMBLError):
    code = "validation_error"


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

def _http_get(path: str, params: dict | None = None) -> dict[str, Any]:
    """GET a ChEMBL endpoint with retry on transient errors.

    Parameters
    ----------
    path:
        Relative path after the base URL (e.g. ``/molecule/CHEMBL25``).
    params:
        Optional query parameters dict.

    Returns
    -------
    Parsed JSON dict.

    Raises
    ------
    ChEMBLNotFoundError, ChEMBLRateLimitError, ChEMBLUnavailableError,
    ChEMBLMalformedResponse, ChEMBLError
    """
    base_params = {"format": "json"}
    if params:
        base_params.update(params)

    url = f"{_BASE_URL}{path}"
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}

    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=base_params, headers=headers, timeout=_TIMEOUT)
        except requests.Timeout as exc:
            last_exc = exc
            logger.warning("ChEMBL timeout on attempt %d: %s %s", attempt + 1, url, exc)
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** attempt)
            continue
        except requests.ConnectionError as exc:
            raise ChEMBLUnavailableError(
                f"Connection error reaching ChEMBL ({_BASE_URL}): {exc}"
            ) from exc
        except requests.RequestException as exc:
            raise ChEMBLError(f"Request failed: {exc}") from exc

        # Handle transient server errors
        if resp.status_code in _RETRY_STATUSES:
            if resp.status_code == 429 and attempt < _MAX_RETRIES:
                retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                logger.warning(
                    "ChEMBL rate limit (429); waiting %.1fs before retry %d.",
                    retry_after, attempt + 2,
                )
                time.sleep(retry_after)
                last_exc = ChEMBLRateLimitError("Rate limited by ChEMBL API.")
                continue
            elif resp.status_code in (502, 503, 504) and attempt < _MAX_RETRIES:
                wait = 2 ** attempt
                logger.warning(
                    "ChEMBL transient error %d; waiting %ds (attempt %d).",
                    resp.status_code, wait, attempt + 1,
                )
                time.sleep(wait)
                last_exc = ChEMBLUnavailableError(f"ChEMBL service error {resp.status_code}.")
                continue

        # HTTP 500 can be either a legitimate API error (JSON body with a message)
        # or an EBI infrastructure outage that returns an HTML error page.
        # Retry the HTML variant — it's transient; raise immediately for JSON errors.
        if resp.status_code == 500:
            ct = resp.headers.get("Content-Type", "")
            body_start = resp.text[:200].lstrip()
            is_html_500 = "html" in ct.lower() or body_start.lower().startswith(("<!doctype", "<html"))
            if is_html_500 and attempt < _MAX_RETRIES:
                wait = 2 ** attempt
                logger.warning(
                    "ChEMBL infrastructure 500 (HTML response); waiting %ds before retry %d.",
                    wait, attempt + 1,
                )
                time.sleep(wait)
                last_exc = ChEMBLUnavailableError("ChEMBL service error 500 (infrastructure outage).")
                continue
            # JSON 500 → raise immediately with the API's own error message.
            raise ChEMBLError(
                f"ChEMBL returned HTTP 500 for {url}: {resp.text[:300]}"
            )

        if resp.status_code == 404:
            raise ChEMBLNotFoundError(f"Not found: {url}")
        if resp.status_code == 400:
            raise ChEMBLValidationError(
                f"Bad request (HTTP 400) for {url}: {resp.text[:300]}"
            )
        if not resp.ok:
            raise ChEMBLError(
                f"ChEMBL returned HTTP {resp.status_code} for {url}: {resp.text[:300]}"
            )

        ct = resp.headers.get("Content-Type", "")
        if "json" not in ct:
            raise ChEMBLMalformedResponse(
                f"Expected JSON from ChEMBL but got Content-Type: {ct!r}"
            )

        try:
            return resp.json()
        except ValueError as exc:
            raise ChEMBLMalformedResponse(
                f"ChEMBL returned non-JSON response: {resp.text[:300]}"
            ) from exc

    # All retries exhausted
    raise (last_exc or ChEMBLError("All retries exhausted"))


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provenance(endpoint: str, params: dict | None = None) -> dict[str, Any]:
    return {
        "_chembl_api_url": _BASE_URL,
        "_endpoint": endpoint,
        "_query_params": params or {},
        "_retrieved_at": _now_iso(),
    }


# ---------------------------------------------------------------------------
# Identifier validation
# ---------------------------------------------------------------------------

def _validate_chembl_id(value: str, prefix: str) -> None:
    """Raise ChEMBLValidationError if *value* is not a valid ChEMBL ID."""
    if not value or not value.upper().startswith(prefix.upper()):
        raise ChEMBLValidationError(
            f"Expected a ChEMBL ID starting with {prefix!r}, got {value!r}."
        )


# ---------------------------------------------------------------------------
# Pagination helper
# ---------------------------------------------------------------------------

def _fetch_smiles_bulk(chembl_ids: list[str]) -> dict[str, str | None]:
    """Fetch canonical SMILES for a list of ChEMBL molecule IDs in batches of 1000.

    Returns a dict {chembl_id: canonical_smiles_or_None}.
    Missing / invalid molecules map to None.
    """
    BATCH = 1000
    result: dict[str, str | None] = {cid: None for cid in chembl_ids}
    unique = list(dict.fromkeys(chembl_ids))  # deduplicate, preserve order

    for i in range(0, len(unique), BATCH):
        batch = unique[i:i + BATCH]
        id_list = ",".join(batch)
        try:
            data = _http_get("/molecule", {
                "molecule_chembl_id__in": id_list,
                "only": "molecule_chembl_id,molecule_structures",
                "limit": BATCH,
            })
        except ChEMBLError as exc:
            logger.warning("Bulk SMILES fetch failed for batch %d: %s", i, exc)
            continue
        for mol in data.get("molecules") or []:
            cid = mol.get("molecule_chembl_id")
            structs = mol.get("molecule_structures") or {}
            smiles = structs.get("canonical_smiles")
            if cid and smiles:
                result[cid] = smiles
    return result


def _paginate(
    path: str,
    params: dict,
    limit: int,
    offset: int,
    collection_key: str,
) -> dict[str, Any]:
    """Fetch one page from a ChEMBL collection endpoint.

    Parameters
    ----------
    path, params: forwarded to _http_get.
    limit, offset: pagination.
    collection_key: the key in the response that holds the records list
                    (e.g. ``"molecules"``, ``"targets"``, ``"activities"``).

    Returns a unified dict with ``records``, ``page``, and ``meta``.
    """
    params = dict(params)
    params["limit"]  = min(int(limit), _MAX_LIMIT)
    params["offset"] = max(int(offset), 0)

    data = _http_get(path, params)
    records = data.get(collection_key) or []
    page_meta = data.get("page_meta") or {}
    total = page_meta.get("total_count") or data.get("count") or len(records)
    next_url = page_meta.get("next")

    return {
        "records":   records,
        "page": {
            "offset":    params["offset"],
            "limit":     params["limit"],
            "count":     len(records),
            "total":     total,
            "has_more":  bool(next_url),
            "next_offset": params["offset"] + len(records) if next_url else None,
        },
    }


# ---------------------------------------------------------------------------
# Tool results factory
# ---------------------------------------------------------------------------

def _ok(operation: str, data: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Wrap a successful result with provenance."""
    return {"operation": operation, "ok": True, **data, **extra}


def _err(operation: str, code: str, message: str) -> dict[str, Any]:
    """Wrap a failure without fabricating any data."""
    return {"operation": operation, "ok": False, "error": {"code": code, "message": message}}


# ---------------------------------------------------------------------------
# register_tools — all 11 MCP operations
# ---------------------------------------------------------------------------

def register_tools(mcp: Any) -> None:
    """Register ChEMBL MCP tools with the server."""

    # ── 1. search_chembl_molecules ─────────────────────────────────────────

    @mcp.tool()
    def search_chembl_molecules(
        query: str,
        molecule_type: Optional[str] = None,
        max_phase: Optional[int] = None,
        therapeutic_flag: Optional[bool] = None,
        limit: int = _DEFAULT_LIMIT,
        offset: int = 0,
        only: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Search ChEMBL molecules by name, synonym, or text query.

        Args:
            query: Free-text search (name, synonym, InChIKey fragment, …).
            molecule_type: Filter by molecule type, e.g. ``"Small molecule"``,
                ``"Protein"``, ``"Antibody"``.
            max_phase: Filter by maximum clinical phase (0–4; 4 = approved).
            therapeutic_flag: If True, only return molecules with therapeutic
                applications.
            limit: Records per page (1–1000; default 20).
            offset: Number of records to skip (for pagination).
            only: Optional subset of fields to return, e.g.
                ``["molecule_chembl_id", "pref_name", "max_phase"]``.

        Returns:
            ``{ok, operation, records[], page{offset,limit,count,total,has_more},
              provenance}``
            Each record preserves the raw ChEMBL JSON plus
            ``molecule_chembl_id``, ``pref_name``, ``max_phase``,
            ``molecule_type``, ``canonical_smiles``, ``inchi_key``,
            ``molecular_formula``, ``molecular_weight``, ``synonyms``.
        """
        op = "search_chembl_molecules"
        if not query or not query.strip():
            return _err(op, "validation_error", "query must not be empty.")

        params: dict[str, Any] = {"q": query.strip()}
        if molecule_type:
            params["molecule_type__iexact"] = molecule_type
        if max_phase is not None:
            params["max_phase"] = int(max_phase)
        if therapeutic_flag is not None:
            params["therapeutic_flag"] = str(therapeutic_flag).lower()
        if only:
            params["only"] = ",".join(only)

        try:
            result = _paginate("/molecule/search", params, limit, offset, "molecules")
        except ChEMBLNotFoundError:
            return _ok(op, {"records": [], "page": {"count": 0, "total": 0}},
                       provenance=_provenance("/molecule/search", params))
        except ChEMBLError as exc:
            return _err(op, getattr(exc, "code", "chembl_error"), str(exc))

        return _ok(op, result, provenance=_provenance("/molecule/search", params))

    # ── 2. get_chembl_molecule ─────────────────────────────────────────────

    @mcp.tool()
    def get_chembl_molecule(
        molecule_chembl_id: str,
        include_raw: bool = False,
    ) -> dict[str, Any]:
        """Retrieve a single ChEMBL molecule by its ChEMBL ID.

        Args:
            molecule_chembl_id: ChEMBL molecule identifier, e.g. ``"CHEMBL25"``.
            include_raw: If True, include the full raw ChEMBL JSON under
                ``raw_record`` in the response.

        Returns:
            ``{ok, operation, molecule{…}, provenance}``
            Preserves: ``molecule_chembl_id``, ``pref_name``, ``max_phase``,
            ``molecule_type``, ``canonical_smiles``, ``standard_inchi``,
            ``inchi_key``, ``molecular_formula``, ``molecular_weight``,
            ``alogp``, ``hbd_count``, ``hba_count``, ``ro5_violations``,
            ``synonyms[]``, ``cross_references[]``.
        """
        op = "get_chembl_molecule"
        cid = (molecule_chembl_id or "").strip().upper()
        if not cid.startswith("CHEMBL"):
            return _err(op, "validation_error",
                        f"Expected a ChEMBL ID (CHEMBLXXX), got {molecule_chembl_id!r}.")
        try:
            data = _http_get(f"/molecule/{quote(cid)}")
        except ChEMBLNotFoundError:
            return _err(op, "not_found",
                        f"No ChEMBL molecule found for ID {cid!r}.")
        except ChEMBLError as exc:
            return _err(op, getattr(exc, "code", "chembl_error"), str(exc))

        mol = _normalize_molecule(data)
        result: dict[str, Any] = {"molecule": mol}
        if include_raw:
            result["raw_record"] = data
        result["provenance"] = _provenance(f"/molecule/{cid}")
        return _ok(op, result)

    # ── 3. search_chembl_targets ───────────────────────────────────────────

    @mcp.tool()
    def search_chembl_targets(
        query: str,
        target_type: Optional[str] = None,
        organism: Optional[str] = None,
        limit: int = _DEFAULT_LIMIT,
        offset: int = 0,
        only: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Search ChEMBL targets by name, gene, or organism.

        Args:
            query: Free-text search (target name, gene symbol, UniProt ID, …).
            target_type: Filter by type, e.g. ``"SINGLE PROTEIN"``,
                ``"PROTEIN COMPLEX"``, ``"CELL-LINE"``, ``"ORGANISM"``.
            organism: Filter by organism, e.g. ``"Homo sapiens"``,
                ``"Mus musculus"``.
            limit: Records per page (1–1000; default 20).
            offset: Number of records to skip.
            only: Optional subset of fields to return.

        Returns:
            ``{ok, operation, records[], page{…}, provenance}``
            Each record includes: ``target_chembl_id``, ``pref_name``,
            ``target_type``, ``organism``, ``target_components[]``
            (with ``accession``, ``component_type``, ``component_id``).
        """
        op = "search_chembl_targets"
        if not query or not query.strip():
            return _err(op, "validation_error", "query must not be empty.")

        params: dict[str, Any] = {"q": query.strip()}
        if target_type:
            # ChEMBL /target/search returns 500 when target_type__iexact is
            # combined with a text query (server-side Django bug).
            # Use icontains which goes through a different code path.
            params["target_type__icontains"] = target_type
        if organism:
            params["organism__icontains"] = organism
        if only:
            params["only"] = ",".join(only)

        try:
            result = _paginate("/target/search", params, limit, offset, "targets")
        except ChEMBLNotFoundError:
            return _ok(op, {"records": [], "page": {"count": 0, "total": 0}},
                       provenance=_provenance("/target/search", params))
        except ChEMBLError as exc:
            return _err(op, getattr(exc, "code", "chembl_error"), str(exc))

        return _ok(op, result, provenance=_provenance("/target/search", params))

    # ── 4. get_chembl_target ───────────────────────────────────────────────

    @mcp.tool()
    def get_chembl_target(
        target_chembl_id: str,
        include_raw: bool = False,
    ) -> dict[str, Any]:
        """Retrieve a single ChEMBL target by its ChEMBL ID.

        Args:
            target_chembl_id: ChEMBL target ID, e.g. ``"CHEMBL301"``.
            include_raw: If True, include full raw ChEMBL JSON under
                ``raw_record``.

        Returns:
            ``{ok, operation, target{…}, provenance}``
            Includes: ``target_chembl_id``, ``pref_name``, ``target_type``,
            ``organism``, ``target_components[]`` (UniProt accessions, gene
            names, component type).
        """
        op = "get_chembl_target"
        tid = (target_chembl_id or "").strip().upper()
        if not tid.startswith("CHEMBL"):
            return _err(op, "validation_error",
                        f"Expected a ChEMBL target ID (CHEMBLXXX), got {target_chembl_id!r}.")
        try:
            data = _http_get(f"/target/{quote(tid)}")
        except ChEMBLNotFoundError:
            return _err(op, "not_found",
                        f"No ChEMBL target found for ID {tid!r}.")
        except ChEMBLError as exc:
            return _err(op, getattr(exc, "code", "chembl_error"), str(exc))

        target = _normalize_target(data)
        result: dict[str, Any] = {"target": target}
        if include_raw:
            result["raw_record"] = data
        result["provenance"] = _provenance(f"/target/{tid}")
        return _ok(op, result)

    # ── 5. search_chembl_activities ────────────────────────────────────────

    @mcp.tool()
    def search_chembl_activities(
        molecule_chembl_id: Optional[str] = None,
        target_chembl_id: Optional[str] = None,
        assay_chembl_id: Optional[str] = None,
        standard_type: Optional[str] = None,
        assay_type: Optional[str] = None,
        pchembl_value_gte: Optional[float] = None,
        pchembl_value_lte: Optional[float] = None,
        standard_value_lte: Optional[float] = None,
        standard_value_gte: Optional[float] = None,
        target_organism: Optional[str] = None,
        limit: int = _DEFAULT_LIMIT,
        offset: int = 0,
        only: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Retrieve bioactivity data from ChEMBL with explicit filters.

        Activity values are returned with BOTH the original (unconverted)
        value/units/operator AND the ChEMBL-standardized values. The tool
        never silently converts IC50 → pIC50.

        Args:
            molecule_chembl_id: Filter by compound ChEMBL ID, e.g.
                ``"CHEMBL25"`` (aspirin).
            target_chembl_id: Filter by target ChEMBL ID, e.g.
                ``"CHEMBL301"`` (EGFR).
            assay_chembl_id: Filter by assay ChEMBL ID.
            standard_type: Filter by activity type, e.g. ``"IC50"``,
                ``"Ki"``, ``"Kd"``, ``"EC50"``, ``"GI50"``.
            assay_type: Filter by assay type:
                ``"B"`` (Binding), ``"F"`` (Functional), ``"A"`` (ADME),
                ``"T"`` (Toxicity), ``"P"`` (Physicochemical),
                ``"U"`` (Unclassified).
            pchembl_value_gte: Minimum pChEMBL value (≥ threshold).
            pchembl_value_lte: Maximum pChEMBL value (≤ threshold).
            standard_value_lte: Maximum standard_value.
            standard_value_gte: Minimum standard_value.
            target_organism: Filter by target organism, e.g.
                ``"Homo sapiens"``. Exact match required — the ChEMBL
                /activity endpoint does not support partial matching on
                this field.
            limit: Records per page (1–1000; default 20).
            offset: Number of records to skip.
            only: Optional subset of fields to return.

        Returns:
            ``{ok, operation, records[], page{…}, provenance}``
            Each record preserves: ``activity_id``, ``molecule_chembl_id``,
            ``target_chembl_id``, ``assay_chembl_id``, ``assay_type``,
            ``standard_type``, ``standard_relation``, ``standard_value``,
            ``standard_units``, ``type`` (original), ``relation`` (original),
            ``value`` (original), ``units`` (original), ``pchembl_value``,
            ``confidence_score``, ``target_organism``, ``document_chembl_id``.
        """
        op = "search_chembl_activities"
        params: dict[str, Any] = {}
        if molecule_chembl_id:
            params["molecule_chembl_id"] = molecule_chembl_id.strip().upper()
        if target_chembl_id:
            params["target_chembl_id"] = target_chembl_id.strip().upper()
        if assay_chembl_id:
            params["assay_chembl_id"] = assay_chembl_id.strip().upper()
        if standard_type:
            st = standard_type.strip()
            st = _TYPE_ALIASES.get(st.lower(), st)
            params["standard_type__iexact"] = st
        if assay_type:
            params["assay_type__iexact"] = assay_type.strip()
        if pchembl_value_gte is not None:
            params["pchembl_value__gte"] = float(pchembl_value_gte)
        if pchembl_value_lte is not None:
            params["pchembl_value__lte"] = float(pchembl_value_lte)
        if standard_value_lte is not None:
            params["standard_value__lte"] = float(standard_value_lte)
        if standard_value_gte is not None:
            params["standard_value__gte"] = float(standard_value_gte)
        if target_organism:
            params["target_organism"] = target_organism.strip()  # exact match only — ChEMBL /activity rejects icontains
        if only:
            params["only"] = ",".join(only)

        if not params:
            return _err(op, "validation_error",
                        "At least one filter (molecule_chembl_id, target_chembl_id, "
                        "standard_type, etc.) must be provided to avoid downloading "
                        "the entire ChEMBL activity table.")

        try:
            result = _paginate("/activity", params, limit, offset, "activities")
        except ChEMBLError as exc:
            return _err(op, getattr(exc, "code", "chembl_error"), str(exc))

        return _ok(op, result, provenance=_provenance("/activity", params))

    # ── 5b. fetch_all_activities ───────────────────────────────────────────

    @mcp.tool()
    def fetch_all_activities(
        molecule_chembl_id: Optional[str] = None,
        target_chembl_id: Optional[str] = None,
        standard_type: Optional[str] = None,
        assay_type: Optional[str] = None,
        pchembl_value_gte: Optional[float] = None,
        target_organism: Optional[str] = None,
        page_size: int = 1000,
        max_records: int = 50000,
        save_path: Optional[str] = None,
        with_smiles: bool = False,
    ) -> dict[str, Any]:
        """Fetch ALL ChEMBL activities for a target or molecule, paginating
        internally. Returns a statistics summary instead of raw records to
        avoid context-window overflow.

        The full record set is saved to a JSONL file (one record per line)
        which can be read by downstream tools or scripts.

        Args:
            molecule_chembl_id: Filter by compound, e.g. ``"CHEMBL221959"``.
            target_chembl_id: Filter by target, e.g. ``"CHEMBL2835985"``.
            standard_type: Filter by activity type, e.g. ``"IC50"``, ``"Ki"``.
            assay_type: ``"B"`` Binding, ``"F"`` Functional, ``"A"`` ADME, …
            pchembl_value_gte: Minimum pChEMBL value (quality filter,
                e.g. 5.0 excludes weak binders).
            target_organism: Exact match, e.g. ``"Homo sapiens"``.
            page_size: Records per API call (max 1000; default 1000).
            max_records: Hard cap on total records fetched (default 50 000).
            save_path: Optional path to write the JSONL output.  Defaults to
                ``/tmp/chembl_activities_<target|mol>_<type>.jsonl``.
            with_smiles: If True, enriches every record with
                ``canonical_smiles`` by bulk-fetching molecule structures
                after activity collection (~12 extra API calls per 10 000
                unique molecules). The JSONL rows will contain a
                ``canonical_smiles`` field ready for ADMET prediction or
                model training. Default False.

        Returns:
            ``{ok, operation, summary{…}, saved_to, provenance}`` where
            ``summary`` contains:
            ``total_fetched``, ``total_available``, ``standard_types``
            (counts per type), ``units`` (counts), ``pchembl_stats``
            (count/min/max/mean of records with a pChEMBL value),
            ``molecules_count`` (unique molecules), ``assay_count``
            (unique assays), ``phase4_molecule_count`` (approved drugs only).
        """
        import json, math, os, statistics as _stats

        op = "fetch_all_activities"

        # --- Build filter params (same logic as search_chembl_activities) ---
        params: dict[str, Any] = {
            "only": ",".join([
                "activity_id", "molecule_chembl_id", "target_chembl_id",
                "assay_chembl_id", "standard_type", "standard_relation",
                "standard_value", "standard_units", "pchembl_value",
                "assay_type", "confidence_score", "target_organism",
                "document_chembl_id",
            ]),
        }
        if molecule_chembl_id:
            params["molecule_chembl_id"] = molecule_chembl_id.strip().upper()
        if target_chembl_id:
            params["target_chembl_id"] = target_chembl_id.strip().upper()
        if standard_type:
            st = standard_type.strip()
            st = _TYPE_ALIASES.get(st.lower(), st)
            params["standard_type__iexact"] = st
        if assay_type:
            params["assay_type__iexact"] = assay_type.strip()
        if pchembl_value_gte is not None:
            params["pchembl_value__gte"] = float(pchembl_value_gte)
        if target_organism:
            params["target_organism"] = target_organism.strip()

        if not any(k in params for k in (
            "molecule_chembl_id", "target_chembl_id",
            "standard_type__iexact", "assay_type__iexact",
        )):
            return _err(op, "validation_error",
                        "Provide at least molecule_chembl_id or target_chembl_id.")

        # --- Determine output path ---
        slug = (
            (molecule_chembl_id or "").replace("CHEMBL", "mol") or
            (target_chembl_id or "").replace("CHEMBL", "tgt")
        )
        type_slug = (standard_type or "all").lower()
        default_path = f"/tmp/chembl_activities_{slug}_{type_slug}.jsonl"
        out_path = (save_path or default_path).strip()

        # --- Pagination loop ---
        page_sz = min(int(page_size), _MAX_LIMIT)
        cap = max(1, int(max_records))
        offset = 0
        total_available = None
        all_records: list[dict] = []

        try:
            while True:
                page = _paginate("/activity", params, page_sz, offset, "activities")
                records = page["records"]
                if total_available is None:
                    total_available = page["page"]["total"]
                all_records.extend(records)
                offset += len(records)
                if not records or not page["page"]["has_more"] or offset >= cap:
                    break
        except ChEMBLError as exc:
            return _err(op, getattr(exc, "code", "chembl_error"), str(exc))

        # --- Optionally enrich with SMILES ---
        smiles_map: dict[str, str | None] = {}
        if with_smiles and all_records:
            mol_ids = [r["molecule_chembl_id"] for r in all_records if r.get("molecule_chembl_id")]
            logger.info("Fetching SMILES for %d unique molecules…", len(set(mol_ids)))
            smiles_map = _fetch_smiles_bulk(mol_ids)
            for rec in all_records:
                rec["canonical_smiles"] = smiles_map.get(rec.get("molecule_chembl_id"))

        # --- Write JSONL ---
        try:
            os.makedirs(os.path.dirname(out_path) or "/tmp", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                for rec in all_records:
                    fh.write(json.dumps(rec) + "\n")
        except OSError as exc:
            logger.warning("Could not write activity JSONL to %s: %s", out_path, exc)
            out_path = None  # type: ignore[assignment]

        # --- Build statistics summary ---
        type_counts: dict[str, int] = {}
        unit_counts: dict[str, int] = {}
        pchembl_vals: list[float] = []
        molecules: set[str] = set()
        assays: set[str] = set()

        for rec in all_records:
            st = rec.get("standard_type") or "unknown"
            type_counts[st] = type_counts.get(st, 0) + 1
            su = rec.get("standard_units") or "unknown"
            unit_counts[su] = unit_counts.get(su, 0) + 1
            pv = rec.get("pchembl_value")
            if pv is not None:
                try:
                    pchembl_vals.append(float(pv))
                except (TypeError, ValueError):
                    pass
            mol = rec.get("molecule_chembl_id")
            if mol:
                molecules.add(mol)
            assay = rec.get("assay_chembl_id")
            if assay:
                assays.add(assay)

        pchembl_stats: dict[str, Any] = {"count": len(pchembl_vals)}
        if pchembl_vals:
            pchembl_stats.update({
                "min":  round(min(pchembl_vals), 2),
                "max":  round(max(pchembl_vals), 2),
                "mean": round(_stats.mean(pchembl_vals), 2),
                "median": round(_stats.median(pchembl_vals), 2),
            })

        summary = {
            "total_fetched":    len(all_records),
            "total_available":  total_available,
            "capped":           len(all_records) >= cap,
            "standard_types":   dict(sorted(type_counts.items(),
                                            key=lambda x: -x[1])),
            "units":            dict(sorted(unit_counts.items(),
                                            key=lambda x: -x[1])),
            "pchembl_stats":    pchembl_stats,
            "molecules_count":  len(molecules),
            "assay_count":      len(assays),
            "smiles_enriched":  with_smiles,
            "smiles_resolved":  (
                sum(1 for v in smiles_map.values() if v)
                if with_smiles else None
            ),
        }

        return _ok(op, {"summary": summary, "saved_to": out_path},
                   provenance=_provenance("/activity", params))

    # ── 6. search_chembl_assays ────────────────────────────────────────────

    @mcp.tool()
    def search_chembl_assays(
        query: Optional[str] = None,
        assay_type: Optional[str] = None,
        target_chembl_id: Optional[str] = None,
        assay_organism: Optional[str] = None,
        document_chembl_id: Optional[str] = None,
        limit: int = _DEFAULT_LIMIT,
        offset: int = 0,
        only: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Search ChEMBL assays by text query or filter by target/type.

        Args:
            query: Free-text search across assay description.  Optional when
                other filters are provided.
            assay_type: Filter by assay type: ``"B"`` (Binding), ``"F"``,
                ``"A"``, ``"T"``, ``"P"``, ``"U"``.
            target_chembl_id: Filter by target, e.g. ``"CHEMBL301"``.
            assay_organism: Filter by assay organism.
            document_chembl_id: Filter by source document ChEMBL ID.
            limit: Records per page (1–1000; default 20).
            offset: Number of records to skip.
            only: Optional field subset.

        Returns:
            ``{ok, operation, records[], page{…}, provenance}``
        """
        op = "search_chembl_assays"
        if not query and not assay_type and not target_chembl_id and not document_chembl_id:
            return _err(op, "validation_error",
                        "Provide at least one of: query, assay_type, "
                        "target_chembl_id, document_chembl_id.")

        params: dict[str, Any] = {}
        if assay_type:
            params["assay_type__iexact"] = assay_type.strip()
        if target_chembl_id:
            params["target_chembl_id"] = target_chembl_id.strip().upper()
        if assay_organism:
            params["assay_organism__icontains"] = assay_organism.strip()
        if document_chembl_id:
            params["document_chembl_id"] = document_chembl_id.strip().upper()
        if only:
            params["only"] = ",".join(only)

        try:
            if query:
                params["q"] = query.strip()
                result = _paginate("/assay/search", params, limit, offset, "assays")
            else:
                result = _paginate("/assay", params, limit, offset, "assays")
        except ChEMBLNotFoundError:
            return _ok(op, {"records": [], "page": {"count": 0, "total": 0}},
                       provenance=_provenance("/assay", params))
        except ChEMBLError as exc:
            return _err(op, getattr(exc, "code", "chembl_error"), str(exc))

        endpoint = "/assay/search" if query else "/assay"
        return _ok(op, result, provenance=_provenance(endpoint, params))

    # ── 7. get_chembl_mechanisms ───────────────────────────────────────────

    @mcp.tool()
    def get_chembl_mechanisms(
        molecule_chembl_id: Optional[str] = None,
        target_chembl_id: Optional[str] = None,
        limit: int = _DEFAULT_LIMIT,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Retrieve drug mechanism-of-action data from ChEMBL.

        Args:
            molecule_chembl_id: Filter by compound ChEMBL ID.
            target_chembl_id: Filter by target ChEMBL ID.
            limit: Records per page.
            offset: Number of records to skip.

        Returns:
            ``{ok, operation, records[], page{…}, provenance}``
            Each record includes: ``mec_id``, ``molecule_chembl_id``,
            ``target_chembl_id``, ``mechanism_of_action``, ``action_type``,
            ``mechanism_comment``, ``selectivity_comment``,
            ``binding_site_comment``.
        """
        op = "get_chembl_mechanisms"
        if not molecule_chembl_id and not target_chembl_id:
            return _err(op, "validation_error",
                        "Provide molecule_chembl_id and/or target_chembl_id.")

        params: dict[str, Any] = {}
        if molecule_chembl_id:
            params["molecule_chembl_id"] = molecule_chembl_id.strip().upper()
        if target_chembl_id:
            params["target_chembl_id"] = target_chembl_id.strip().upper()

        try:
            result = _paginate("/mechanism", params, limit, offset, "mechanisms")
        except ChEMBLError as exc:
            return _err(op, getattr(exc, "code", "chembl_error"), str(exc))

        return _ok(op, result, provenance=_provenance("/mechanism", params))

    # ── 8. get_chembl_indications ──────────────────────────────────────────

    @mcp.tool()
    def get_chembl_indications(
        molecule_chembl_id: Optional[str] = None,
        efo_term: Optional[str] = None,
        max_phase_for_ind: Optional[int] = None,
        limit: int = _DEFAULT_LIMIT,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Retrieve clinical indication data for drugs from ChEMBL.

        Args:
            molecule_chembl_id: Filter by compound ChEMBL ID.
            efo_term: Filter by EFO disease term (partial match), e.g.
                ``"cancer"``, ``"diabetes"``.
            max_phase_for_ind: Minimum clinical phase for the indication
                (1–4; 4 = approved).
            limit: Records per page.
            offset: Number of records to skip.

        Returns:
            ``{ok, operation, records[], page{…}, provenance}``
            Each record: ``drugind_id``, ``molecule_chembl_id``,
            ``max_phase_for_ind``, ``mesh_id``, ``mesh_heading``,
            ``efo_id``, ``efo_term``.
        """
        op = "get_chembl_indications"
        if not molecule_chembl_id and not efo_term:
            return _err(op, "validation_error",
                        "Provide molecule_chembl_id and/or efo_term.")

        params: dict[str, Any] = {}
        if molecule_chembl_id:
            params["molecule_chembl_id"] = molecule_chembl_id.strip().upper()
        if efo_term:
            params["efo_term__icontains"] = efo_term.strip()
        if max_phase_for_ind is not None:
            params["max_phase_for_ind__gte"] = int(max_phase_for_ind)

        try:
            result = _paginate("/drug_indication", params, limit, offset, "drug_indications")
        except ChEMBLError as exc:
            return _err(op, getattr(exc, "code", "chembl_error"), str(exc))

        return _ok(op, result, provenance=_provenance("/drug_indication", params))

    # ── 9. get_chembl_documents ────────────────────────────────────────────

    @mcp.tool()
    def get_chembl_documents(
        document_chembl_id: Optional[str] = None,
        journal: Optional[str] = None,
        year: Optional[int] = None,
        authors: Optional[str] = None,
        limit: int = _DEFAULT_LIMIT,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Retrieve ChEMBL document/publication records.

        Args:
            document_chembl_id: Retrieve a specific document by ChEMBL ID,
                e.g. ``"CHEMBL1121364"``.
            journal: Filter by journal name (partial match), e.g.
                ``"J Med Chem"``.
            year: Filter by publication year, e.g. ``2020``.
            authors: Filter by author name (partial match).
            limit: Records per page.
            offset: Number of records to skip.

        Returns:
            ``{ok, operation, records[] or document{}, page{…}, provenance}``
            Each record: ``document_chembl_id``, ``title``, ``authors``,
            ``journal``, ``year``, ``volume``, ``issue``, ``first_page``,
            ``last_page``, ``pubmed_id``, ``doi``.
        """
        op = "get_chembl_documents"

        # Single-record fetch
        if document_chembl_id:
            did = document_chembl_id.strip().upper()
            try:
                data = _http_get(f"/document/{quote(did)}")
            except ChEMBLNotFoundError:
                return _err(op, "not_found",
                            f"No ChEMBL document found for {did!r}.")
            except ChEMBLError as exc:
                return _err(op, getattr(exc, "code", "chembl_error"), str(exc))
            return _ok(op, {"document": data,
                            "provenance": _provenance(f"/document/{did}")})

        params: dict[str, Any] = {}
        if journal:
            params["journal__icontains"] = journal.strip()
        if year is not None:
            params["year"] = int(year)
        if authors:
            params["authors__icontains"] = authors.strip()

        if not params:
            return _err(op, "validation_error",
                        "Provide document_chembl_id, journal, year, or authors.")

        try:
            result = _paginate("/document", params, limit, offset, "documents")
        except ChEMBLError as exc:
            return _err(op, getattr(exc, "code", "chembl_error"), str(exc))

        return _ok(op, result, provenance=_provenance("/document", params))

    # ── 10. search_chembl_similarity ───────────────────────────────────────

    @mcp.tool()
    def search_chembl_similarity(
        smiles: str,
        threshold: float = 0.7,
        limit: int = _DEFAULT_LIMIT,
        offset: int = 0,
        only: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Similarity search — find ChEMBL molecules similar to a query SMILES.

        Uses Tanimoto similarity on Morgan (circular) fingerprints.
        The query representation and threshold are always explicit in the
        response — the tool never silently changes either.

        Args:
            smiles: Query molecule SMILES. Must be a valid, parseable SMILES.
            threshold: Tanimoto similarity threshold (0.0–1.0; default 0.7).
                Lower thresholds return more, less similar results.
            limit: Records per page (1–100; default 20; capped at 100 for
                similarity to limit response size).
            offset: Number of records to skip.
            only: Optional field subset.

        Returns:
            ``{ok, operation, query_smiles, threshold, records[], page{…},
              provenance}``
        """
        op = "search_chembl_similarity"
        smiles = (smiles or "").strip()
        if not smiles:
            return _err(op, "validation_error", "smiles must not be empty.")
        if not (0.0 <= float(threshold) <= 1.0):
            return _err(op, "validation_error",
                        f"threshold must be between 0.0 and 1.0; got {threshold}.")

        thr_pct = int(float(threshold) * 100)
        path = f"/similarity/{quote(smiles, safe='')}/{thr_pct}"
        params: dict[str, Any] = {}
        if only:
            params["only"] = ",".join(only)
        limit = min(int(limit), 100)   # cap for similarity

        try:
            result = _paginate(path, params, limit, offset, "molecules")
        except ChEMBLNotFoundError:
            return _ok(op, {"records": [], "page": {"count": 0, "total": 0},
                            "query_smiles": smiles, "threshold": float(threshold)},
                       provenance=_provenance(path, params))
        except ChEMBLError as exc:
            return _err(op, getattr(exc, "code", "chembl_error"), str(exc))

        return _ok(op, {**result, "query_smiles": smiles, "threshold": float(threshold)},
                   provenance=_provenance(path, params))

    # ── 11. search_chembl_substructure ────────────────────────────────────

    @mcp.tool()
    def search_chembl_substructure(
        smiles: str,
        limit: int = _DEFAULT_LIMIT,
        offset: int = 0,
        only: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Substructure search — find ChEMBL molecules containing a query fragment.

        Args:
            smiles: Query substructure SMILES (must be a valid substructure).
            limit: Records per page (1–100; default 20; capped at 100).
            offset: Number of records to skip.
            only: Optional field subset.

        Returns:
            ``{ok, operation, query_smiles, records[], page{…}, provenance}``
        """
        op = "search_chembl_substructure"
        smiles = (smiles or "").strip()
        if not smiles:
            return _err(op, "validation_error", "smiles must not be empty.")

        path = f"/substructure/{quote(smiles, safe='')}"
        params: dict[str, Any] = {}
        if only:
            params["only"] = ",".join(only)
        limit = min(int(limit), 100)

        try:
            result = _paginate(path, params, limit, offset, "molecules")
        except ChEMBLNotFoundError:
            return _ok(op, {"records": [], "page": {"count": 0, "total": 0},
                            "query_smiles": smiles},
                       provenance=_provenance(path, params))
        except ChEMBLError as exc:
            return _err(op, getattr(exc, "code", "chembl_error"), str(exc))

        return _ok(op, {**result, "query_smiles": smiles},
                   provenance=_provenance(path, params))


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_molecule(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract key fields from a raw ChEMBL molecule record."""
    props = raw.get("molecule_properties") or {}
    structs = raw.get("molecule_structures") or {}
    syns = [s.get("molecule_synonym") for s in raw.get("molecule_synonyms") or []
            if s.get("molecule_synonym")]
    xrefs = raw.get("cross_references") or []
    return {
        "molecule_chembl_id": raw.get("molecule_chembl_id"),
        "pref_name":          raw.get("pref_name"),
        "max_phase":          raw.get("max_phase"),
        "molecule_type":      raw.get("molecule_type"),
        "first_approval":     raw.get("first_approval"),
        "oral":               raw.get("oral"),
        "parenteral":         raw.get("parenteral"),
        "topical":            raw.get("topical"),
        "black_box_warning":  raw.get("black_box_warning"),
        "canonical_smiles":   structs.get("canonical_smiles"),
        "standard_inchi":     structs.get("standard_inchi"),
        "inchi_key":          structs.get("standard_inchi_key"),
        "molecular_formula":  props.get("full_molformula"),
        "molecular_weight":   props.get("full_mwt"),
        "alogp":              props.get("alogp"),
        "hbd_count":          props.get("hbd"),
        "hba_count":          props.get("hba"),
        "psa":                props.get("psa"),
        "ro5_violations":     props.get("num_ro5_violations"),
        "rtb":                props.get("rtb"),
        "synonyms":           syns,
        "cross_references":   xrefs,
    }


def _normalize_target(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract key fields from a raw ChEMBL target record."""
    components = []
    for comp in raw.get("target_components") or []:
        components.append({
            "component_id":   comp.get("component_id"),
            "component_type": comp.get("component_type"),
            "accession":      comp.get("accession"),
            "pref_name":      comp.get("pref_name"),
            "description":    comp.get("description"),
        })
    return {
        "target_chembl_id": raw.get("target_chembl_id"),
        "pref_name":        raw.get("pref_name"),
        "target_type":      raw.get("target_type"),
        "organism":         raw.get("organism"),
        "tax_id":           raw.get("tax_id"),
        "species_group_flag": raw.get("species_group_flag"),
        "target_components": components,
    }
