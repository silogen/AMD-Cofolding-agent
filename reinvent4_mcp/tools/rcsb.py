"""rcsb.py — RCSB/PDB search, metadata-enrichment, and file-download module.

Standalone implementation (lives outside source/ for isolated testing).
When merged, this file moves to reinvent4_mcp/tools/rcsb.py.

Architecture — six layers, no circular imports:
  Layer 0  constants + error hierarchy
  Layer 1  RCSBHTTPClient  (requests.Session + retry/backoff)
  Layer 2  QueryBuilder    (rcsbapi.search → serialized JSON dict)
  Layer 3  MetadataClient  (rcsbapi.data.DataQuery)
  Layer 4  normalize_entry (pure transforms, no network)
  Layer 5  FileClient      (streaming download + SHA-256)
  Layer 6  register_tools  (MCP tool registration)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import concurrent.futures
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal

import requests

# ─────────────────────────────────────────────────────────────────────────────
# Thread-runner — rcsbapi.data.DataQuery.exec() calls asyncio.run() internally.
# When invoked from inside the MCP server's already-running event loop this
# raises "asyncio.run() cannot be called from a running event loop".  Running
# exec() in a fresh worker thread (which has no event loop) sidesteps this.
# ─────────────────────────────────────────────────────────────────────────────

_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="rcsb_data")


def _exec_in_thread(fn, *args, **kwargs):
    """Run *fn* in a thread that has no asyncio event loop, return its result."""
    future = _THREAD_POOL.submit(fn, *args, **kwargs)
    return future.result()


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 0 — Constants & errors
# ─────────────────────────────────────────────────────────────────────────────

RCSB_SEARCH_URL = os.getenv(
    "RCSB_SEARCH_URL", "https://search.rcsb.org/rcsbsearch/v2/query"
)
RCSB_FILES_BASE = os.getenv(
    "RCSB_FILES_BASE", "https://files.rcsb.org/download"
)
RCSB_DATA_GQL = os.getenv(
    "RCSB_DATA_GQL", "https://data.rcsb.org/graphql"
)

SCHEMA_VERSION = "1.3"
PDB_ATOM_LIMIT = 99_999      # PDB format hard limit — silently truncated above this
USER_AGENT = "Cofolding-Agent/1.0 (rcsb-fetcher)"
CONNECT_TIMEOUT = 10          # seconds
READ_TIMEOUT = 60             # seconds
STREAM_TIMEOUT = 300          # seconds, file downloads
MAX_RETRIES = 4
BACKOFF_BASE = 2.0            # exponential backoff multiplier
TRANSIENT_CODES = {429, 502, 503, 504}
MAX_RESPONSE_BYTES = 50 * 1024 * 1024  # 50 MB
DEFAULT_PAGE_SIZE = 25


class RCSBError(Exception):
    """Base for all RCSB tool errors."""


class RCSBQueryError(RCSBError):
    """Invalid query structure, unknown attribute, or bad request (4xx)."""


class RCSBNoResultsError(RCSBError):
    """Valid query returned zero hits."""


class RCSBRateLimitError(RCSBError):
    """HTTP 429 rate-limit; retry_after_seconds attribute when available."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RCSBUnavailableError(RCSBError):
    """5xx or connection failure; service temporarily unavailable."""


class RCSBMalformedResponse(RCSBError):
    """Unexpected JSON shape or non-JSON body."""


class RCSBLocalError(RCSBError):
    """Local write failure, disk full, or checksum mismatch."""


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — HTTP client
# ─────────────────────────────────────────────────────────────────────────────

class RCSBHTTPClient:
    """Reusable requests.Session with retry/backoff and error classification."""

    def __init__(
        self,
        search_url: str = RCSB_SEARCH_URL,
        files_base: str = RCSB_FILES_BASE,
        timeout: tuple[int, int] = (CONNECT_TIMEOUT, READ_TIMEOUT),
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.search_url = search_url
        self.files_base = files_base
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

    # ── public ──────────────────────────────────────────────────────────────

    def post_json(self, url: str, payload: dict) -> dict:
        """POST JSON payload; retry transient codes; return parsed dict."""

        def _call() -> requests.Response:
            return self._session.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )

        resp = self._retry_loop(_call)
        return self._parse_json_response(resp)

    def download_stream(self, url: str, dest: Path) -> tuple[str, int]:
        """Stream file to *dest* with atomic write and streaming SHA-256.

        Returns (hex_sha256, size_bytes).
        Partial file is deleted on any error.
        """
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        dest.parent.mkdir(parents=True, exist_ok=True)

        def _call() -> requests.Response:
            return self._session.get(
                url,
                stream=True,
                timeout=(CONNECT_TIMEOUT, STREAM_TIMEOUT),
            )

        resp = self._retry_loop(_call)
        sha256 = hashlib.sha256()
        size = 0

        try:
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        fh.write(chunk)
                        sha256.update(chunk)
                        size += len(chunk)
                        if size > MAX_RESPONSE_BYTES:
                            raise RCSBLocalError(
                                f"Response exceeded {MAX_RESPONSE_BYTES} bytes for {url}"
                            )
            tmp.rename(dest)
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise

        return sha256.hexdigest(), size

    # ── private ─────────────────────────────────────────────────────────────

    def _retry_loop(self, fn: Callable[[], requests.Response]) -> requests.Response:
        """Attempt fn() up to max_retries times; backoff on transient codes."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = fn()
            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt)
                    continue
                raise RCSBUnavailableError(
                    f"Connection failed after {self.max_retries} retries: {exc}"
                ) from exc

            if resp.status_code == 200:
                return resp

            # 204 No Content = zero results (RCSB returns this for empty searches)
            if resp.status_code == 204:
                return resp

            if resp.status_code not in TRANSIENT_CODES:
                raise self._classify_error(resp)

            # Transient — decide whether to retry
            if attempt >= self.max_retries:
                raise self._classify_error(resp)

            if resp.status_code == 429:
                retry_after = self._parse_retry_after(resp)
                time.sleep(retry_after)
            else:
                self._sleep_backoff(attempt)

        # should not reach here
        raise RCSBUnavailableError("Retry loop exhausted unexpectedly")

    def _classify_error(self, resp: requests.Response) -> RCSBError:
        code = resp.status_code
        body = resp.text[:500]
        if code == 400:
            return RCSBQueryError(f"Bad request (400): {body}")
        if code == 401:
            return RCSBQueryError(f"Unauthorized (401): {body}")
        if code == 403:
            return RCSBQueryError(f"Forbidden (403): {body}")
        if code == 404:
            return RCSBQueryError(f"Not found (404): {body}")
        if code == 429:
            retry_after = self._parse_retry_after(resp)
            return RCSBRateLimitError(
                f"Rate limited (429), retry after {retry_after}s",
                retry_after=retry_after,
            )
        if code in {502, 503, 504}:
            return RCSBUnavailableError(f"Service unavailable ({code}): {body}")
        return RCSBUnavailableError(f"Unexpected HTTP {code}: {body}")

    @staticmethod
    def _parse_retry_after(resp: requests.Response) -> float:
        header = resp.headers.get("Retry-After", "")
        try:
            return float(header)
        except (ValueError, TypeError):
            return 1.0

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        delay = BACKOFF_BASE ** attempt + random.uniform(0, 0.5)
        time.sleep(delay)

    def _parse_json_response(self, resp: requests.Response) -> dict:
        # 204 No Content = zero results
        if resp.status_code == 204:
            return {"total_count": 0, "result_list": [], "result_set": []}
        ct = resp.headers.get("Content-Type", "")
        if "json" not in ct and not resp.text.strip().startswith("{"):
            raise RCSBMalformedResponse(
                f"Expected JSON response, got Content-Type: {ct!r}"
            )
        try:
            return resp.json()
        except Exception as exc:
            raise RCSBMalformedResponse(f"Failed to parse JSON: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — Query builder
# ─────────────────────────────────────────────────────────────────────────────

class QueryBuilder:
    """Translates agent-supplied query dicts through rcsbapi.search for
    validation, then serializes them to plain RCSB Search API JSON dicts."""

    @classmethod
    def from_dict(cls, query_spec: dict) -> dict:
        """Entry point.

        query_spec is the agent's RCSB query node tree
        (``type`` = 'terminal' or 'group').
        Returns a validated, serialized RCSB query JSON dict.
        Raises RCSBQueryError on invalid structure.
        """
        if not isinstance(query_spec, dict):
            raise RCSBQueryError("query_spec must be a dict")
        try:
            node_type = query_spec.get("type", "terminal")
            if node_type == "group":
                obj = cls._build_group(query_spec)
            else:
                obj = cls._build_terminal(query_spec)
            return cls._serialize(obj)
        except RCSBQueryError:
            raise
        except Exception as exc:
            raise RCSBQueryError(f"Query construction failed: {exc}") from exc

    @classmethod
    def _build_terminal(cls, node: dict) -> Any:
        """Build an rcsbapi.search terminal query object from a node dict."""
        from rcsbapi.search import (
            TextQuery,
            AttributeQuery,
            SeqSimilarityQuery,
            SeqMotifQuery,
            StructSimilarityQuery,
            StructMotifQuery,
            StructMotifResidue,
            ChemSimilarityQuery,
        )

        service = node.get("service", "text")
        params = node.get("parameters", node)  # allow flat spec too

        if service in ("text", "full_text"):
            value = params.get("value", "")
            # If the params look like an attribute query (have 'attribute' +
            # 'operator' keys) but the wrong service name was used, auto-reroute
            # rather than raising — this is the most common LLM mistake.
            if "attribute" in params and "operator" in params:
                service = "attribute"
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                # Numeric value with no attribute name — raise a clear error.
                raise RCSBQueryError(
                    f"full_text / text service requires a string value, "
                    f"but got a number ({value!r}). "
                    "To filter on a numeric field use service='attribute' with "
                    "'attribute' (e.g. 'rcsb_entry_info.resolution_combined'), "
                    "'operator' (e.g. 'less_or_equal'), and 'value'."
                )
            else:
                return TextQuery(value=str(value))

        # Common LLM aliases for the attribute service.
        if service in ("text_attribute", "structure_attribute", "attr",
                       "attribute_text", "rcsb_attribute", "entry_attribute"):
            service = "attribute"

        if service == "text_chem":
            value = params.get("value", "")
            # TextQuery with service="text_chem" for chemical text search
            return AttributeQuery(
                attribute=params.get("attribute", "chem_comp.name"),
                operator=params.get("operator", "contains_words"),
                value=value,
                service="text_chem",
            )

        if service == "attribute":
            attribute = params.get("attribute")
            operator = params.get("operator", "exact_match")
            value = params.get("value")
            svc = params.get("service")  # optional disambiguation
            if not attribute:
                raise RCSBQueryError("attribute service requires 'attribute'")
            # Fix common wrong attribute paths the LLM generates.
            # Note: ligand/non-polymer comp_id is NOT a searchable attribute —
            # use service='chemical' with query_type='comp_id' and value=<CCD> instead.
            _COMP_ID_PATHS = {
                "rcsb_nonpolymer_entity.comp_id",
                "rcsb_polymer_entity_annotation.comp_id",
                "nonpolymer_entity.comp_id",
                "rcsb_nonpolymer_entity_container_identifiers.comp_id",
                "chem_comp_id",
                "comp_id",
            }
            if attribute in _COMP_ID_PATHS:
                raise RCSBQueryError(
                    f"Attribute {attribute!r} is not a valid RCSB search attribute. "
                    "To search by ligand CCD code (e.g. 'RXB' for ruxolitinib), use "
                    "service='chemical' with parameters={'value': '<CCD_CODE>', "
                    "'type': 'descriptor', 'descriptor_type': 'SMILES', 'match_type': 'similar'} "
                    "OR service='chemical' with {'value': '<CCD_CODE>', 'type': 'comp_id'} "
                    "OR use full-text search: service='full_text' with value='<ligand name>'. "
                    "To count entries WITH any ligand use attribute='rcsb_entry_info.nonpolymer_entity_count' "
                    "with operator='greater' and value=0."
                )
            _ATTR_PATH_ALIASES: dict[str, str] = {
                # Deposition / release dates
                "rcsb_entry_info.deposit_date":
                    "rcsb_accession_info.deposit_date",
                "rcsb_entry_info.deposition_date":
                    "rcsb_accession_info.deposit_date",
                "rcsb_entry_info.release_date":
                    "rcsb_accession_info.initial_release_date",
                "rcsb_entry_info.initial_release_date":
                    "rcsb_accession_info.initial_release_date",
                "deposit_date": "rcsb_accession_info.deposit_date",
                "deposition_date": "rcsb_accession_info.deposit_date",
                "release_date": "rcsb_accession_info.initial_release_date",
            }
            attribute = _ATTR_PATH_ALIASES.get(attribute, attribute)
            kwargs: dict[str, Any] = {
                "attribute": attribute,
                "operator": operator,
            }
            if value is not None:
                kwargs["value"] = value
            if svc:
                kwargs["service"] = svc
            return AttributeQuery(**kwargs)

        if service == "sequence":
            sequence = params.get("sequence") or params.get("value")
            if not sequence:
                raise RCSBQueryError("sequence service requires 'sequence'")
            return SeqSimilarityQuery(
                sequence,
                evalue_cutoff=params.get("evalue_cutoff", 0.1),
                identity_cutoff=params.get("identity_cutoff", 0.0),
                sequence_type=params.get("sequence_type", "protein"),
            )

        if service == "seqmotif":
            value = params.get("value") or params.get("pattern")
            if not value:
                raise RCSBQueryError("seqmotif service requires 'value'")
            return SeqMotifQuery(
                value,
                pattern_type=params.get("pattern_type", "simple"),
                sequence_type=params.get("sequence_type", "protein"),
            )

        if service == "structure":
            entry_id = params.get("entry_id")
            file_url = params.get("file_url")
            file_path = params.get("file_path")
            file_format = params.get("file_format")
            if entry_id:
                return StructSimilarityQuery(
                    structure_search_type="entry_id",
                    entry_id=entry_id,
                    assembly_id=params.get("assembly_id", "1"),
                    chain_id=params.get("chain_id"),
                    similarity_type=params.get("similarity_type", "local"),
                    target_search_space=params.get("target_search_space", "assembly"),
                    number_of_candidates=params.get("number_of_candidates", 10000),
                    ptmscore_cutoff=params.get("ptmscore_cutoff", 0.8),
                )
            if file_url:
                return StructSimilarityQuery(
                    structure_search_type="file_url",
                    file_url=file_url,
                    file_format=file_format,
                    similarity_type=params.get("similarity_type", "local"),
                    target_search_space=params.get("target_search_space", "assembly"),
                )
            if file_path:
                return StructSimilarityQuery(
                    structure_search_type="file_upload",
                    file_path=file_path,
                    file_format=file_format,
                    similarity_type=params.get("similarity_type", "local"),
                    target_search_space=params.get("target_search_space", "assembly"),
                )
            raise RCSBQueryError(
                "structure service requires entry_id, file_url, or file_path"
            )

        if service == "strucmotif":
            residues_raw = params.get("residue_ids", [])
            residues = [
                StructMotifResidue(
                    struct_oper_id=r.get("struct_oper_id", "1"),
                    chain_id=r["chain_id"],
                    label_seq_id=r["label_seq_id"],
                    exchanges=r.get("exchanges", []),
                )
                for r in residues_raw
            ]
            if len(residues) < 2:
                raise RCSBQueryError("strucmotif requires at least 2 residues")
            entry_id = params.get("entry_id")
            file_url = params.get("file_url")
            file_path = params.get("file_path")
            file_format = params.get("file_format")
            kwargs2: dict[str, Any] = {"residue_ids": residues}
            if entry_id:
                kwargs2["entry_id"] = entry_id
            elif file_url:
                kwargs2["structure_search_type"] = "file_url"
                kwargs2["file_url"] = file_url
                kwargs2["file_format"] = file_format
            elif file_path:
                kwargs2["structure_search_type"] = "file_upload"
                kwargs2["file_path"] = file_path
                kwargs2["file_format"] = file_format
            for opt in (
                "backbone_distance_tolerance",
                "side_chain_distance_tolerance",
                "angle_tolerance",
                "rmsd_cutoff",
                "atom_pairing_scheme",
                "motif_pruning_strategy",
                "allowed_structures",
                "excluded_structures",
                "limit",
            ):
                if opt in params:
                    kwargs2[opt] = params[opt]
            return StructMotifQuery(**kwargs2)

        if service == "chemical":
            value = params.get("value")
            if not value:
                raise RCSBQueryError("chemical service requires 'value'")
            query_type = params.get("query_type", "formula")
            kwargs3: dict[str, Any] = {"value": value, "query_type": query_type}
            if query_type == "descriptor":
                kwargs3["descriptor_type"] = params.get("descriptor_type", "SMILES")
                mt = params.get("match_type")
                if mt:
                    kwargs3["match_type"] = mt
            else:
                kwargs3["match_subset"] = params.get("match_subset", False)
            from rcsbapi.search import ChemSimilarityQuery
            return ChemSimilarityQuery(**kwargs3)

        raise RCSBQueryError(f"Unknown search service: {service!r}")

    @classmethod
    def _build_group(cls, node: dict) -> Any:
        """Recursively build a boolean group node."""
        from rcsbapi.search import NestedAttributeQuery

        operator = node.get("logical_operator", "and").lower()
        nodes_raw = node.get("nodes", [])
        if not nodes_raw:
            raise RCSBQueryError("Group node has no children")

        children: list[Any] = []
        for child in nodes_raw:
            child_type = child.get("type", "terminal")
            if child_type == "group":
                children.append(cls._build_group(child))
            else:
                children.append(cls._build_terminal(child))

        # Apply boolean operator
        is_nested = node.get("nested", False)
        if is_nested and len(children) == 2:
            return NestedAttributeQuery(*children)

        combined = children[0]
        for other in children[1:]:
            if operator == "and":
                combined = combined & other
            elif operator == "or":
                combined = combined | other
            elif operator == "not":
                combined = combined & (~other)
            else:
                raise RCSBQueryError(f"Unknown logical_operator: {operator!r}")

        if node.get("negate", False):
            combined = ~combined

        return combined

    @classmethod
    def _serialize(cls, query_obj: Any) -> dict:
        """Extract the JSON dict from an rcsbapi.search query object."""
        # rcsbapi.search objects expose to_dict() in the new package
        if hasattr(query_obj, "to_dict"):
            return query_obj.to_dict()
        # Fallback: try the internal query attribute used by the library
        for attr in ("query", "_query", "__dict__"):
            val = getattr(query_obj, attr, None)
            if isinstance(val, dict):
                return val
        # Last resort: try calling and extracting the first page's query
        raise RCSBQueryError(
            "Cannot serialize query object — rcsbapi version may be incompatible"
        )


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — Metadata client
# ─────────────────────────────────────────────────────────────────────────────

class MetadataClient:
    """Fetches entry/entity/component metadata using rcsbapi.data.DataQuery."""

    ENTRY_RETURN_DATA: ClassVar[list[str]] = [
        "struct.title",
        "struct_keywords.pdbx_keywords",
        "exptl.method",
        "refine.ls_d_res_high",
        "refine.ls_d_res_low",
        "refine.ls_R_factor_R_free",
        "refine.ls_R_factor_R_work",
        "cell.length_a",
        "cell.length_b",
        "cell.length_c",
        "cell.angle_alpha",
        "cell.angle_beta",
        "cell.angle_gamma",
        "symmetry.space_group_name_H_M",
        "rcsb_entry_info",
        "rcsb_accession_info",
        "rcsb_entry_container_identifiers",
        "rcsb_entry_container_identifiers.emdb_ids",
        "audit_author.name",
        "rcsb_primary_citation.title",
        "rcsb_primary_citation.pdbx_database_id_PubMed",
        "rcsb_primary_citation.pdbx_database_id_DOI",
        "rcsb_primary_citation.journal_abbrev",
        "rcsb_primary_citation.year",
        "pdbx_vrpt_summary",
        "pdbx_vrpt_summary_geometry",
        "pdbx_vrpt_summary_diffraction",
        "reflns.percent_possible_obs",
        "reflns.pdbx_redundancy",
        "reflns.pdbx_netI_over_sigmaI",
        "reflns.pdbx_Rmerge_I_obs",
        "reflns.d_resolution_low",
        "pdbx_nmr_ensemble.conformers_submitted_total_number",
        "pdbx_nmr_representative.conformer_id",
        "assemblies.rcsb_struct_symmetry.symbol",
        "assemblies.rcsb_struct_symmetry.kind",
        "assemblies.rcsb_struct_symmetry.stoichiometry",
        "assemblies.rcsb_assembly_info",
        "polymer_entities.rcsb_id",
        "polymer_entities.entity_poly",
        "polymer_entities.rcsb_entity_source_organism",
        "polymer_entities.rcsb_polymer_entity",
        "polymer_entities.rcsb_polymer_entity_container_identifiers"
        ".reference_sequence_identifiers",
        "polymer_entities.entity_src_gen.pdbx_host_org_scientific_name",
        "polymer_entities.entity_src_gen.pdbx_gene_src_gene",
        "nonpolymer_entities.rcsb_id",
        "nonpolymer_entities.pdbx_entity_nonpoly",
        "nonpolymer_entities.nonpolymer_comp.chem_comp",
        "nonpolymer_entities.nonpolymer_comp.chem_comp.pdbx_formal_charge",
        "nonpolymer_entities.nonpolymer_entity_instances"
        ".rcsb_nonpolymer_entity_instance_container_identifiers",
        "nonpolymer_entities.nonpolymer_entity_instances"
        ".rcsb_nonpolymer_instance_validation_score",
        "branched_entities.rcsb_id",
        "branched_entities.pdbx_entity_branch",
        "branched_entities.pdbx_entity_branch_descriptor",
        "rcsb_ma_qa_metric_global.ma_qa_metric_global.type",
        "rcsb_ma_qa_metric_global.ma_qa_metric_global.value",
    ]

    CSM_EXTRA_FIELDS: ClassVar[list[str]] = [
        "rcsb_ma_qa_metric_global",
    ]

    def get_entries(
        self,
        entry_ids: list[str],
        extra_fields: list[str] | None = None,
    ) -> dict[str, dict | None]:
        """Returns {entry_id: raw_data_api_dict}. Missing IDs map to None."""
        from rcsbapi.data import DataQuery

        fields = list(self.ENTRY_RETURN_DATA)
        if extra_fields:
            fields = fields + extra_fields

        query = DataQuery(
            input_type="entries",
            input_ids=entry_ids,
            return_data_list=fields,
        )
        try:
            result = _exec_in_thread(query.exec)
        except Exception as exc:
            raise RCSBMalformedResponse(
                f"Data API query failed for {entry_ids}: {exc}"
            ) from exc

        entries_list = result.get("data", {}).get("entries", [])
        by_id: dict[str, dict | None] = {eid: None for eid in entry_ids}
        for entry in entries_list:
            eid = entry.get("rcsb_id") or entry.get("entry_id")
            if eid and eid in by_id:
                by_id[eid] = entry
        return by_id

    # Common wrong field names the LLM produces → correct RCSB GraphQL paths.
    _FIELD_ALIASES: ClassVar[dict[str, str]] = {
        "rcsb_entry_info.deposit_date":       "rcsb_accession_info.deposit_date",
        "rcsb_entry_info.deposition_date":    "rcsb_accession_info.deposit_date",
        "rcsb_entry_info.release_date":       "rcsb_accession_info.initial_release_date",
        "rcsb_entry_info.initial_release_date":"rcsb_accession_info.initial_release_date",
        "rcsb_entry_info.revision_date":      "rcsb_accession_info.revision_date",
        "deposit_date":                       "rcsb_accession_info.deposit_date",
        "deposition_date":                    "rcsb_accession_info.deposit_date",
        "release_date":                       "rcsb_accession_info.initial_release_date",
        "resolution":                         "rcsb_entry_info.resolution_combined",
        "experimental_method":                "rcsb_entry_info.experimental_method",
        "title":                              "struct.title",
        "organism":         "polymer_entities.rcsb_entity_source_organism.scientific_name",
    }

    @classmethod
    def _normalize_fields(cls, fields: list[str]) -> list[str]:
        """Remap common LLM field aliases to valid RCSB GraphQL paths."""
        return [cls._FIELD_ALIASES.get(f, f) for f in fields]

    def get_summary_fields(
        self,
        identifiers: list[str],
        return_data_list: list[str],
        input_type: str = "entries",
    ) -> dict[str, dict]:
        """Batch-fetch specific fields for inline search result summaries."""
        from rcsbapi.data import DataQuery

        if not identifiers:
            return {}
        return_data_list = self._normalize_fields(return_data_list)
        try:
            # DataQuery validates fields against the schema at __init__ time,
            # so wrap both construction and exec in the same try/except.
            query = DataQuery(
                input_type=input_type,
                input_ids=identifiers,
                return_data_list=return_data_list,
            )
            result = _exec_in_thread(query.exec)
        except Exception:
            return {}

        items = result.get("data", {}).get(input_type, [])
        out: dict[str, dict] = {}
        for item in items:
            iid = item.get("rcsb_id") or item.get("entry_id")
            if iid:
                out[iid] = item
        return out

    def get_chem_comps(self, comp_ids: list[str]) -> dict[str, dict | None]:
        """Fetch chemical component definitions (CCD) including SMILES, InChI, and formal charge."""
        from rcsbapi.data import DataQuery

        query = DataQuery(
            input_type="chem_comps",
            input_ids=comp_ids,
            return_data_list=[
                "chem_comp.type",
                "chem_comp.formula_weight",
                "chem_comp.name",
                "chem_comp.formula",
                "chem_comp.pdbx_formal_charge",
                "pdbx_chem_comp_descriptor",
                "rcsb_chem_comp_info.initial_release_date",
            ],
        )
        try:
            result = _exec_in_thread(query.exec)
        except Exception as exc:
            raise RCSBMalformedResponse(f"chem_comps query failed: {exc}") from exc

        items = result.get("data", {}).get("chem_comps", [])
        by_id: dict[str, dict | None] = {cid: None for cid in comp_ids}
        for item in items:
            cid = item.get("rcsb_id") or item.get("comp_id")
            if cid and cid in by_id:
                by_id[cid] = item
        return by_id

    def get_pubmed(self, pubmed_ids: list[int]) -> dict[int, dict | None]:
        """Fetch PubMed abstracts for citation enrichment."""
        from rcsbapi.data import DataQuery

        query = DataQuery(
            input_type="pubmed",
            input_ids=pubmed_ids,
            return_data_list=["rcsb_pubmed_abstract_text"],
        )
        try:
            result = _exec_in_thread(query.exec)
        except Exception as exc:
            raise RCSBMalformedResponse(f"pubmed query failed: {exc}") from exc

        items = result.get("data", {}).get("pubmed", [])
        by_id: dict[int, dict | None] = {pid: None for pid in pubmed_ids}
        for item in items:
            pid = item.get("rcsb_id") or item.get("pubmed_id")
            if pid is not None:
                by_id[int(pid)] = item
        return by_id


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4 — Normalizer (pure functions)
# ─────────────────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_get(d: dict | None, *keys: str, default: Any = None) -> Any:
    """Navigate nested dicts safely; return default if any key is missing."""
    curr: Any = d
    for k in keys:
        if not isinstance(curr, dict):
            return default
        curr = curr.get(k, default)
        if curr is None:
            return default
    return curr


def normalize_entry(raw: dict | None, entry_id: str, ccd_map: dict | None = None) -> dict:
    """Transform a raw DataQuery response dict into the normalized JSON schema.

    All missing fields become None; never raises on missing data.
    raw is the dict for one entry from the DataQuery response.
    ccd_map is an optional pre-fetched CCD dict {comp_id: ccd_data} for ligand enrichment.
    """
    if raw is None:
        return {
            "entry_id": entry_id,
            "source": "rcsb",
            "schema_version": SCHEMA_VERSION,
            "retrieved_at": _utcnow(),
            "error": "entry_not_found",
        }

    identity = _norm_identity(raw)
    emdb_ids = identity.get("emdb_ids") or []

    return {
        "entry_id": entry_id,
        "source": "rcsb",
        "schema_version": SCHEMA_VERSION,
        "retrieved_at": _utcnow(),
        "identity": identity,
        "history": _norm_history(raw),
        "citation": _norm_citation(raw),
        "experiment": _norm_experiment_or_model(raw),
        "validation": _norm_validation(raw),
        "composition": _norm_composition(raw),
        "assemblies": _norm_assemblies(raw),
        "polymer_entities": _norm_polymer_entities(raw),
        "nonpolymer_entities": _norm_nonpolymer_entities(raw, ccd_map),
        "branched_entities": _norm_branched_entities(raw),
        "download_urls": _build_download_urls(entry_id, emdb_ids=emdb_ids),
        "raw_rcsb_response_artifact": None,
    }


def _norm_identity(raw: dict) -> dict:
    info = raw.get("rcsb_entry_info", {}) or {}
    kw = raw.get("struct_keywords", {}) or {}
    struct = raw.get("struct", {}) or {}
    cid = raw.get("rcsb_entry_container_identifiers", {}) or {}
    emdb_ids = cid.get("emdb_ids") or []
    return {
        "title": _safe_get(struct, "title"),
        "pdb_doi": f"https://doi.org/10.2210/pdb{raw.get('rcsb_id', '').lower()}/pdb",
        "classification": _safe_get(kw, "pdbx_keywords"),
        "keywords": [k.strip() for k in (_safe_get(kw, "pdbx_keywords") or "").split(",") if k.strip()],
        "entry_type": _safe_get(info, "structure_determination_methodology", default="experimental"),
        "emdb_ids": emdb_ids,
    }


def _norm_history(raw: dict) -> dict:
    acc = raw.get("rcsb_accession_info", {}) or {}
    return {
        "deposited": _safe_get(acc, "deposit_date"),
        "released": _safe_get(acc, "initial_release_date"),
        "last_modified": _safe_get(acc, "revision_date"),
        "version": _safe_get(acc, "major_revision"),
        "replaces": [],
        "replaced_by": None,
    }


def _norm_citation(raw: dict) -> dict:
    cit = raw.get("rcsb_primary_citation", {}) or {}
    authors_raw = raw.get("audit_author", []) or []
    return {
        "title": _safe_get(cit, "title"),
        "authors": [a.get("name") for a in authors_raw if isinstance(a, dict)],
        "journal": _safe_get(cit, "journal_abbrev"),
        "year": _safe_get(cit, "year"),
        "doi": _safe_get(cit, "pdbx_database_id_DOI"),
        "pubmed_id": _safe_get(cit, "pdbx_database_id_PubMed"),
    }


def _is_csm(raw: dict) -> bool:
    info = raw.get("rcsb_entry_info", {}) or {}
    methodology = info.get("structure_determination_methodology", "")
    return str(methodology).lower() == "computational"


def _norm_experiment_or_model(raw: dict) -> dict:
    if _is_csm(raw):
        return _norm_model_block(raw)
    return _norm_experiment_block(raw)


def _norm_experiment_block(raw: dict) -> dict:
    exptl_list = raw.get("exptl", []) or []
    method = exptl_list[0].get("method") if exptl_list else None
    refine_list = raw.get("refine", []) or []
    refine = refine_list[0] if refine_list else {}
    cell = raw.get("cell", {}) or {}
    sym = raw.get("symmetry", {}) or {}
    # Support both old lowercase keys (fixtures) and correct API keys
    r_free = _safe_get(refine, "ls_R_factor_R_free") or _safe_get(refine, "ls_rfactor_rfree")
    r_work = _safe_get(refine, "ls_R_factor_R_work") or _safe_get(refine, "ls_rfactor_r_work")
    space_group = (
        _safe_get(sym, "space_group_name_H_M")
        or _safe_get(sym, "space_group_name_hm")
    )

    # NMR-specific metrics
    is_nmr = method and "NMR" in method
    nmr_ens = raw.get("pdbx_nmr_ensemble") or {}
    if isinstance(nmr_ens, list):
        nmr_ens = nmr_ens[0] if nmr_ens else {}
    nmr_rep = raw.get("pdbx_nmr_representative") or {}
    if isinstance(nmr_rep, list):
        nmr_rep = nmr_rep[0] if nmr_rep else {}

    # X-ray data collection stats from reflns table
    is_xray = method == "X-RAY DIFFRACTION"
    reflns_list = raw.get("reflns", []) or []
    reflns = reflns_list[0] if reflns_list else {}

    return {
        "type": "experimental",
        "method": method,
        "resolution_A": _safe_get(refine, "ls_d_res_high"),
        "r_work": r_work,
        "r_free": r_free,
        "space_group": space_group,
        "unit_cell": {
            "a": _safe_get(cell, "length_a"),
            "b": _safe_get(cell, "length_b"),
            "c": _safe_get(cell, "length_c"),
            "alpha": _safe_get(cell, "angle_alpha"),
            "beta": _safe_get(cell, "angle_beta"),
            "gamma": _safe_get(cell, "angle_gamma"),
        },
        "data_collection": {
            "resolution_low_A":  _safe_get(reflns, "d_resolution_low"),
            "completeness_pct":  _safe_get(reflns, "percent_possible_obs"),
            "redundancy":        _safe_get(reflns, "pdbx_redundancy"),
            "i_over_sigma":      _safe_get(reflns, "pdbx_netI_over_sigmaI"),
            "r_merge":           _safe_get(reflns, "pdbx_Rmerge_I_obs"),
        } if is_xray else None,
        "nmr_ensemble_size":        _safe_get(nmr_ens, "conformers_submitted_total_number") if is_nmr else None,
        "nmr_representative_model": _safe_get(nmr_rep, "conformer_id") if is_nmr else None,
    }


def _norm_model_block(raw: dict) -> dict:
    # Live DataQuery path: rcsb_ma_qa_metric_global is a list of objects,
    # each containing a nested ma_qa_metric_global list.
    # e.g. [{"ma_qa_metric_global": [{"type": "pLDDT", "value": 97.18}]}]
    metrics_raw: list = []
    qa_wrapper = raw.get("rcsb_ma_qa_metric_global")
    if isinstance(qa_wrapper, list):
        for item in qa_wrapper:
            if isinstance(item, dict):
                nested = item.get("ma_qa_metric_global", []) or []
                metrics_raw.extend(nested)
    elif isinstance(qa_wrapper, dict):
        metrics_raw = qa_wrapper.get("ma_qa_metric_global", []) or []

    # Fallback: top-level ma_qa_metric_global (offline fixtures)
    if not metrics_raw:
        metrics_raw = raw.get("ma_qa_metric_global", []) or []

    metrics = [
        {"type": m.get("type"), "value": m.get("value")}
        for m in metrics_raw
        if isinstance(m, dict)
    ]
    return {
        "type": "computational",
        "method": "computational",
        "resolution_A": None,
        "r_work": None,
        "r_free": None,
        "space_group": None,
        "unit_cell": None,
        "confidence_metrics": metrics,
    }


def _norm_validation(raw: dict) -> dict:
    geom = (raw.get("pdbx_vrpt_summary_geometry") or [{}])[0]
    diff = (raw.get("pdbx_vrpt_summary_diffraction") or [{}])[0]
    return {
        "clashscore":              _safe_get(geom, "clashscore"),
        "ramachandran_outlier_pct": _safe_get(geom, "percent_ramachandran_outliers"),
        "rotamer_outlier_pct":     _safe_get(geom, "percent_rotamer_outliers"),
        "bond_length_rmsz":        _safe_get(geom, "bonds_RMSZ"),
        "bond_angle_rmsz":         _safe_get(geom, "angles_RMSZ"),
        "percent_rsrz_outliers":   _safe_get(diff, "percent_RSRZ_outliers"),
        "data_completeness_pct":   _safe_get(diff, "data_completeness"),
        "percentiles":             None,
    }


def _norm_composition(raw: dict) -> dict:
    info = raw.get("rcsb_entry_info", {}) or {}
    deposited = _safe_get(info, "deposited_polymer_monomer_count")
    modeled = _safe_get(info, "deposited_modeled_polymer_monomer_count")
    unmodeled = (deposited - modeled) if (deposited is not None and modeled is not None) else None
    return {
        "total_weight_kda": _safe_get(info, "molecular_weight"),
        "atom_count": _safe_get(info, "deposited_atom_count"),
        "modeled_residue_count": modeled,
        "deposited_residue_count": deposited,
        "unmodeled_residue_count": unmodeled,
        "unique_protein_chains": _safe_get(info, "polymer_entity_count_protein"),
        "unique_dna_chains": _safe_get(info, "polymer_entity_count_DNA"),
        "unique_rna_chains": _safe_get(info, "polymer_entity_count_RNA"),
        "nonpolymer_entity_count": _safe_get(info, "nonpolymer_entity_count"),
        "solvent_entity_count": _safe_get(info, "solvent_entity_count_ion"),
    }


def _norm_assemblies(raw: dict) -> list[dict]:
    result = []
    # New DataQuery path: assemblies is a top-level list of assembly objects
    assemblies_list = raw.get("assemblies", []) or []
    for asm in assemblies_list:
        if not isinstance(asm, dict):
            continue
        sym_list = asm.get("rcsb_struct_symmetry", []) or []
        sym_info = sym_list[0] if sym_list else {}
        asm_info_raw = asm.get("rcsb_assembly_info") or {}
        asm_info = asm_info_raw if isinstance(asm_info_raw, dict) else (asm_info_raw[0] if isinstance(asm_info_raw, list) and asm_info_raw else {})
        result.append({
            "assembly_id": _safe_get(asm_info, "assembly_id") or asm.get("rcsb_id"),
            "stoichiometry": _safe_get(sym_info, "stoichiometry"),
            "symmetry": _safe_get(sym_info, "symbol"),
        })

    # Fallback: old flat shape from offline fixtures (rcsb_assembly_info at entry level)
    if not result:
        items = raw.get("rcsb_assembly_info", []) or []
        if not isinstance(items, list):
            items = [items]
        sym_list = raw.get("rcsb_struct_symmetry", []) or []
        sym_info = sym_list[0] if sym_list else {}
        for a in items:
            if not isinstance(a, dict):
                continue
            result.append({
                "assembly_id": _safe_get(a, "assembly_id"),
                "stoichiometry": _safe_get(sym_info, "stoichiometry"),
                "symmetry": _safe_get(sym_info, "symbol"),
            })
    return result


def _norm_polymer_entities(raw: dict) -> list[dict]:
    entities = raw.get("polymer_entities", []) or []
    result = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        poly = e.get("entity_poly", {}) or {}
        org_list = e.get("rcsb_entity_source_organism", []) or []
        org = org_list[0] if org_list else {}
        pe = e.get("rcsb_polymer_entity", {}) or {}
        pci = e.get("rcsb_polymer_entity_container_identifiers", {}) or {}
        ref_ids = pci.get("reference_sequence_identifiers", []) or []
        # G1: full UniProt mapping with sequence ranges
        uniprot_mappings = [
            {
                "uniprot_id":       r.get("database_accession"),
                "isoform":          r.get("database_isoform"),
                "db_seq_begin":     r.get("database_seq_begin"),
                "db_seq_end":       r.get("database_seq_end"),
                "entity_seq_begin": r.get("entity_seq_begin"),
                "entity_seq_end":   r.get("entity_seq_end"),
            }
            for r in ref_ids
            if isinstance(r, dict) and r.get("database_name") in ("UniProt", "uniprot")
        ]
        uniprot_ids = [m["uniprot_id"] for m in uniprot_mappings if m["uniprot_id"]]
        # G2: expression system from entity_src_gen
        src_gen_list = e.get("entity_src_gen", []) or []
        src_gen = src_gen_list[0] if src_gen_list else {}
        gene_raw = _safe_get(src_gen, "pdbx_gene_src_gene") or ""
        result.append({
            "entity_id": e.get("rcsb_id"),
            "name": _safe_get(pe, "pdbx_description"),
            "type": _safe_get(poly, "type"),
            "polymer_type": _safe_get(poly, "rcsb_entity_polymer_type"),
            "sequence": _safe_get(poly, "pdbx_seq_one_letter_code_can"),
            "length": _safe_get(poly, "rcsb_sample_sequence_length"),
            "organism": {
                "scientific_name": _safe_get(org, "scientific_name"),
                "taxonomy_id": _safe_get(org, "ncbi_taxonomy_id"),
            },
            "engineered": _safe_get(pe, "pdbx_fragment"),
            "mutations": _safe_get(pe, "pdbx_mutation"),
            "uniprot_ids": uniprot_ids,
            "uniprot_mappings": uniprot_mappings,
            "expression_system": _safe_get(src_gen, "pdbx_host_org_scientific_name"),
            "gene_names": [g.strip() for g in gene_raw.split(",") if g.strip()],
        })
    return result


def _pick_descriptor(descriptors: list, dtype: str) -> str | None:
    """Extract best SMILES/InChI from CCD descriptor list by type and program priority."""
    priority = {"OpenEye OEToolkits": 0, "CACTVS": 1, "ACDLabs": 2}
    matches = [d for d in descriptors if isinstance(d, dict) and d.get("type") == dtype]
    if not matches:
        return None
    return min(matches, key=lambda d: priority.get(d.get("program", ""), 99)).get("descriptor")


def _norm_nonpolymer_entities(raw: dict, ccd_map: dict | None = None) -> list[dict]:
    entities = raw.get("nonpolymer_entities", []) or []
    result = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        npoly = e.get("pdbx_entity_nonpoly", {}) or {}
        # New API path: nonpolymer_comp.chem_comp; old fixture path: chem_comp
        nonpoly_comp = e.get("nonpolymer_comp", {}) or {}
        chem = nonpoly_comp.get("chem_comp", {}) or e.get("chem_comp", {}) or {}
        cid_raw = e.get("rcsb_chem_comp_container_identifiers", {}) or {}
        comp_id = _safe_get(cid_raw, "comp_id") or _safe_get(npoly, "comp_id")
        # B2/B3: pull CCD-sourced charge and descriptors
        ccd = (ccd_map or {}).get(comp_id) or {}
        ccd_chem = ccd.get("chem_comp", {}) or {}
        descriptors = ccd.get("pdbx_chem_comp_descriptor", []) or []
        # G6: per-instance data (chain, sequence ID, occupancy)
        instances_raw = e.get("nonpolymer_entity_instances", []) or []
        instances = []
        for inst in instances_raw:
            if not isinstance(inst, dict):
                continue
            cid_inst = inst.get("rcsb_nonpolymer_entity_instance_container_identifiers", {}) or {}
            score = inst.get("rcsb_nonpolymer_instance_validation_score", {}) or {}
            instances.append({
                "chain_id":    _safe_get(cid_inst, "auth_asym_id"),
                "auth_seq_id": _safe_get(cid_inst, "auth_seq_id"),
                "occupancy":   _safe_get(score, "average_occupancy"),
            })
        result.append({
            "entity_id":       e.get("rcsb_id"),
            "comp_id":         comp_id,
            "name":            _safe_get(npoly, "name") or _safe_get(chem, "name"),
            "formula":         _safe_get(chem, "formula"),
            "formula_weight_da": _safe_get(chem, "formula_weight"),
            "type":            _safe_get(chem, "type"),
            "classification":  ccd_chem.get("type") or _safe_get(chem, "type"),
            "formal_charge":   ccd_chem.get("pdbx_formal_charge"),
            "smiles":          _pick_descriptor(descriptors, "SMILES_CANONICAL"),
            "smiles_isomeric": _pick_descriptor(descriptors, "SMILES"),
            "inchi":           _pick_descriptor(descriptors, "InChI"),
            "inchikey":        _pick_descriptor(descriptors, "InChIKey"),
            "instances":       instances,
        })
    return result


def _norm_branched_entities(raw: dict) -> list[dict]:
    entities = raw.get("branched_entities", []) or []
    result = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        branch = e.get("pdbx_entity_branch", {}) or {}
        descriptors = e.get("pdbx_entity_branch_descriptor", []) or []
        result.append({
            "entity_id": e.get("rcsb_id"),
            "type": _safe_get(branch, "type"),
            "descriptors": [
                {"type": d.get("type"), "descriptor": d.get("descriptor")}
                for d in descriptors
                if isinstance(d, dict)
            ],
        })
    return result


def _build_download_urls(entry_id: str, emdb_ids: list | None = None) -> dict:
    base = RCSB_FILES_BASE.rstrip("/")
    eid = entry_id.upper()
    urls: dict = {
        "pdb": f"{base}/{eid}.pdb",
        "cif": f"{base}/{eid}.cif",
        "assembly_cif": f"{base}/{eid}-assembly1.cif",
        "validation_report": (
            f"https://files.rcsb.org/pub/pdb/validation_reports/"
            f"{eid[1:3].lower()}/{eid.lower()}/{eid.lower()}_validation.pdf"
        ),
        "validation_xml": (
            f"https://files.rcsb.org/pub/pdb/validation_reports/"
            f"{eid[1:3].lower()}/{eid.lower()}/{eid.lower()}_validation.xml.gz"
        ),
    }
    if emdb_ids:
        emid = emdb_ids[0]
        eid_lower = emid.lower()
        urls["emdb_map"] = (
            f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/"
            f"{emid.upper()}/map/{eid_lower}.map.gz"
        )
    return urls


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 5 — File client
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FileArtifact:
    artifact_id: str
    kind: str
    entry_id: str
    format: str
    assembly_id: str | None
    uri: str
    sha256: str
    size_bytes: int
    source_url: str
    retrieved_at: str
    rcsb_version: str | None
    schema_version: str
    metadata_artifact_id: str | None
    error: str | None = None


class FileClient:
    """Downloads coordinate files from RCSB files server."""

    def __init__(self, http: RCSBHTTPClient, output_dir: Path) -> None:
        self.http = http
        self.output_dir = Path(output_dir)

    def fetch(
        self,
        entry_ids: list[str],
        formats: list[str] | None = None,
        assembly_ids: list[str] | None = None,
        include_validation_report: bool = False,
        metadata_cache: dict | None = None,
    ) -> list[FileArtifact]:
        """Download files for all entry_ids × formats × assembly_ids.

        metadata_cache: optional {entry_id: normalized_entry_dict} for atom-count guard.
        Per-file errors do not abort other downloads.
        """
        if formats is None:
            formats = ["cif"]

        artifacts: list[FileArtifact] = []
        for entry_id in entry_ids:
            atom_count: int | None = None
            if metadata_cache:
                meta = metadata_cache.get(entry_id.upper()) or metadata_cache.get(entry_id)
                if meta:
                    atom_count = (meta.get("composition") or {}).get("atom_count")
            for fmt in formats:
                # Asymmetric unit
                artifacts.append(
                    self._fetch_one(entry_id, fmt, assembly_id=None, atom_count=atom_count)
                )
                # Assemblies
                for asm in (assembly_ids or []):
                    artifacts.append(
                        self._fetch_one(entry_id, fmt, assembly_id=asm, atom_count=atom_count)
                    )
        return artifacts

    def _fetch_one(
        self, entry_id: str, fmt: str, assembly_id: str | None, atom_count: int | None = None
    ) -> FileArtifact:
        url = self._url_for(entry_id, fmt, assembly_id)

        # B4: guard against PDB format limit (silently empty/corrupt above 99,999 atoms)
        if fmt == "pdb" and atom_count is not None and atom_count > PDB_ATOM_LIMIT:
            return FileArtifact(
                artifact_id="",
                kind="raw_structure",
                entry_id=entry_id.upper(),
                format=fmt,
                assembly_id=assembly_id,
                uri="",
                sha256="",
                size_bytes=0,
                source_url=url,
                retrieved_at=_utcnow(),
                rcsb_version=None,
                schema_version=SCHEMA_VERSION,
                metadata_artifact_id=None,
                error=(
                    f"pdb_format_unsupported: entry has {atom_count:,} atoms; "
                    f"PDB format hard limit is {PDB_ATOM_LIMIT:,}; use cif instead"
                ),
            )
        fname = Path(url).name
        dest = self.output_dir / entry_id.upper() / fname

        # Cache hit: file exists and we already have it
        if dest.exists():
            sha256 = self._sha256_file(dest)
            return FileArtifact(
                artifact_id=self._make_artifact_id(entry_id, fmt, assembly_id, sha256),
                kind="raw_structure",
                entry_id=entry_id.upper(),
                format=fmt,
                assembly_id=assembly_id,
                uri=str(dest.resolve()),
                sha256=sha256,
                size_bytes=dest.stat().st_size,
                source_url=url,
                retrieved_at=_utcnow(),
                rcsb_version=None,
                schema_version=SCHEMA_VERSION,
                metadata_artifact_id=None,
            )

        try:
            sha256, size = self.http.download_stream(url, dest)
            return FileArtifact(
                artifact_id=self._make_artifact_id(entry_id, fmt, assembly_id, sha256),
                kind="raw_structure",
                entry_id=entry_id.upper(),
                format=fmt,
                assembly_id=assembly_id,
                uri=str(dest.resolve()),
                sha256=sha256,
                size_bytes=size,
                source_url=url,
                retrieved_at=_utcnow(),
                rcsb_version=None,
                schema_version=SCHEMA_VERSION,
                metadata_artifact_id=None,
            )
        except Exception as exc:
            return FileArtifact(
                artifact_id="",
                kind="raw_structure",
                entry_id=entry_id.upper(),
                format=fmt,
                assembly_id=assembly_id,
                uri="",
                sha256="",
                size_bytes=0,
                source_url=url,
                retrieved_at=_utcnow(),
                rcsb_version=None,
                schema_version=SCHEMA_VERSION,
                metadata_artifact_id=None,
                error=str(exc),
            )

    def _url_for(self, entry_id: str, fmt: str, assembly_id: str | None) -> str:
        base = self.http.files_base.rstrip("/")
        eid = entry_id.upper()
        if assembly_id is not None:
            # Only CIF is available for assemblies
            return f"{base}/{eid}-assembly{assembly_id}.cif"
        return f"{base}/{eid}.{fmt}"

    @staticmethod
    def _make_artifact_id(
        entry_id: str, fmt: str, assembly_id: str | None, sha256: str
    ) -> str:
        raw = f"{entry_id.upper()}:{fmt}:{assembly_id or 'asym'}:{sha256[:16]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared by MCP tools
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_percentiles_rest(entry_id: str, session: Any) -> dict | None:
    """Fetch wwPDB percentile ranks via RCSB REST v1 core endpoint.

    Returns a dict with clashscore_abs, ramachandran_abs, rotamer_abs, or None
    on failure. Not in the critical path — called only when include_percentiles=True.
    """
    try:
        url = f"https://data.rcsb.org/rest/v1/core/entry/{entry_id.upper()}"
        resp = session.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        pct_list = data.get("pdbx_percentile_conditions", []) or []
        raw_pct = pct_list[0] if pct_list else {}
        return {
            "clashscore_abs":    raw_pct.get("clashscore_percentile_rank_absolute"),
            "ramachandran_abs":  raw_pct.get("percent_ramachandran_outliers_percentile_rank_absolute"),
            "rotamer_abs":       raw_pct.get("percent_rotamer_outliers_percentile_rank_absolute"),
        } if raw_pct else None
    except Exception:
        return None


def _canonical_query_id(query_dict: dict) -> str:
    canonical = json.dumps(query_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _parse_result_list(raw: dict) -> list[dict]:
    """Parse result list from RCSB Search API response.

    The API returns either 'result_list' (newer) or 'result_set' (older/some versions).
    """
    result_list = raw.get("result_list") or raw.get("result_set") or []
    out = []
    for r in result_list:
        out.append({
            "identifier": r.get("identifier", ""),
            "score": r.get("score"),
            "services": r.get("services"),
            "summary": {},
        })
    return out


def _build_search_response(
    raw: dict,
    query_dict: dict,
    results: list[dict],
    page_start: int,
    rows: int,
) -> dict:
    total_hits = raw.get("total_count", 0)
    total_pages = math.ceil(total_hits / rows) if rows > 0 else 0
    has_more = (page_start + rows) < total_hits

    # Parse facets from RCSB response
    facets_raw = raw.get("facets", []) or []
    facets: dict[str, list] = {}
    for f in facets_raw:
        name = f.get("name", "")
        buckets = f.get("buckets", []) or f.get("ranges", [])
        facets[name] = buckets

    return {
        "rcsb_query_id": _canonical_query_id(query_dict),
        "query_forwarded": query_dict,
        "total_hits": total_hits,
        "page": {
            "start": page_start,
            "rows": rows,
            "has_more": has_more,
            "total_pages": total_pages,
        },
        "facets": facets,
        "results": results,
        "editor_link": None,  # populated if library exposes session.get_editor_link()
        "errors": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 6 — MCP tool registration
# ─────────────────────────────────────────────────────────────────────────────

def register_tools(mcp: Any) -> None:
    """Register all four RCSB MCP tools on *mcp* (a FastMCP instance)."""

    @mcp.tool()
    def rcsb_search(
        query: dict,
        return_type: str = "entry",
        request_options: dict | None = None,
        summary_fields: list[str] | None = None,
    ) -> dict:
        """Search the RCSB PDB.

        Returns one page of results + facets + total count.
        `query` is the native RCSB Search API query node tree.
        `request_options` supports: paginate, results_content_type,
        results_verbosity, return_explain_metadata, scoring_strategy,
        sort, facets, group_by, group_by_return_type.
        Advance pages by incrementing request_options.paginate.start.
        """
        client = RCSBHTTPClient()
        query_dict = QueryBuilder.from_dict(query)
        opts = dict(request_options or {})
        paginate = opts.pop("paginate", {})
        page_start = paginate.get("start", 0)
        rows = paginate.get("rows", DEFAULT_PAGE_SIZE)
        # Normalise sort directions: RCSB requires "asc"/"desc", not full words.
        _DIR = {"ascending": "asc", "descending": "desc",
                "ASC": "asc", "DESC": "desc", "ASCENDING": "asc", "DESCENDING": "desc"}
        for s in opts.get("sort", []):
            if isinstance(s, dict) and "direction" in s:
                s["direction"] = _DIR.get(s["direction"], s["direction"])
        payload = {
            "query": query_dict,
            "return_type": return_type,
            "request_options": {
                "paginate": {"start": page_start, "rows": rows},
                **opts,
            },
        }
        raw = client.post_json(RCSB_SEARCH_URL, payload)
        results = _parse_result_list(raw)
        if summary_fields and results:
            ids = [r["identifier"] for r in results]
            summaries = MetadataClient().get_summary_fields(ids, summary_fields)
            for r in results:
                r["summary"] = summaries.get(r["identifier"], {})
        return _build_search_response(raw, query_dict, results, page_start, rows)

    @mcp.tool()
    def rcsb_count(
        query: dict,
        return_type: str = "entry",
        results_content_type: list[str] | None = None,
    ) -> dict:
        """Return only total_hits for a query without fetching results.

        Useful for evaluating constraint tightness before a full search.
        """
        client = RCSBHTTPClient()
        query_dict = QueryBuilder.from_dict(query)
        opts: dict[str, Any] = {"return_counts": True}
        if results_content_type:
            opts["results_content_type"] = results_content_type
        payload = {
            "query": query_dict,
            "return_type": return_type,
            "request_options": opts,
        }
        raw = client.post_json(RCSB_SEARCH_URL, payload)
        total = raw.get("total_count", 0)
        return {
            "rcsb_query_id": _canonical_query_id(query_dict),
            "total_hits": total,
            "results_content_type": results_content_type or ["experimental"],
        }

    @mcp.tool()
    def rcsb_get_entries(
        entry_ids: list[str],
        include_raw_response: bool = False,
        include_percentiles: bool = False,
    ) -> dict:
        """Return full normalized metadata for selected entry IDs.

        Each entry is normalized into a hierarchical JSON record covering
        identity, history, citation, experiment/model, validation,
        composition, assemblies, polymer entities, nonpolymer entities,
        and branched entities.

        include_percentiles: when True, fetch wwPDB percentile ranks from the
        RCSB REST v1 core endpoint and populate validation.percentiles per entry.
        Adds one extra HTTP round-trip per entry; avoid in large batch calls.
        """
        mc = MetadataClient()
        raw_map = mc.get_entries(entry_ids)

        # B2/B3: collect comp_ids for CCD enrichment across all entries
        comp_ids: list[str] = []
        for raw in raw_map.values():
            if raw is None:
                continue
            for npoly_ent in (raw.get("nonpolymer_entities") or []):
                npoly = (npoly_ent.get("pdbx_entity_nonpoly") or {}) if isinstance(npoly_ent, dict) else {}
                cid = npoly.get("comp_id") or (
                    (npoly_ent.get("rcsb_chem_comp_container_identifiers") or {}).get("comp_id")
                    if isinstance(npoly_ent, dict) else None
                )
                if cid and cid not in comp_ids:
                    comp_ids.append(cid)
        ccd_map = mc.get_chem_comps(comp_ids) if comp_ids else {}

        # G8: optional percentile REST session
        pct_session = None
        if include_percentiles:
            pct_session = mc  # we'll use a plain requests session

        normalized: dict[str, dict] = {}
        for eid, raw in raw_map.items():
            rec = normalize_entry(raw, eid, ccd_map=ccd_map)
            # G8: merge percentile ranks if requested
            if include_percentiles and raw is not None:
                from requests import Session as _Session
                _sess = getattr(mc, "_session", None) or _Session()
                pct = _fetch_percentiles_rest(eid, _sess)
                if pct and rec.get("validation") is not None:
                    rec["validation"]["percentiles"] = pct
            if include_raw_response:
                rec["raw_rcsb_response_artifact"] = raw
            normalized[eid] = rec
        return {"entries": normalized, "schema_version": SCHEMA_VERSION}

    @mcp.tool()
    def rcsb_fetch_files(
        entry_ids: list[str],
        formats: list[str] | None = None,
        assembly_ids: list[str] | None = None,
        include_validation_report: bool = False,
        output_dir: str | None = None,
    ) -> dict:
        """Download coordinate files; return artifact records with SHA-256.

        formats: ["cif"] | ["pdb"] | ["cif", "pdb"]
        assembly_ids: list of assembly numbers, e.g. ["1", "2"]
        output_dir: destination directory (defaults to /shared/rcsb_structures)
        """
        out_path = Path(output_dir or "/shared/rcsb_structures")
        client = RCSBHTTPClient()
        # Pre-fetch metadata to get atom counts for B4 PDB size guard
        # Gracefully handled: if metadata is unavailable, size guard is skipped.
        metadata_cache: dict[str, dict] = {}
        try:
            mc = MetadataClient()
            raw_map = mc.get_entries(entry_ids)
            for eid, raw in raw_map.items():
                if raw is not None:
                    metadata_cache[eid.upper()] = normalize_entry(raw, eid)
        except Exception:
            pass  # size guard will be skipped silently
        fc = FileClient(http=client, output_dir=out_path)
        artifacts = fc.fetch(
            entry_ids=entry_ids,
            formats=formats or ["cif"],
            assembly_ids=assembly_ids,
            include_validation_report=include_validation_report,
            metadata_cache=metadata_cache,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "output_dir": str(out_path),
            "artifacts": [asdict(a) for a in artifacts],
            "errors": [asdict(a) for a in artifacts if a.error],
        }
