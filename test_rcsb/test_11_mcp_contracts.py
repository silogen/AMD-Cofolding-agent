"""test_11_mcp_contracts.py — Tests for MCP tool registration and contracts.

Verifies that:
  - register_tools() registers exactly the expected four tool functions
  - Each tool is callable with the correct signature/docstring
  - schema_version appears on artifact and entry results
  - The registry's load_extensions() isolates failures per module
  - Partial download failures are surfaced per-artifact, not per-call
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rcsb import register_tools, RCSBUnavailableError, SCHEMA_VERSION
from registry import load_extensions
from test_rcsb.utils import make_search_response


# ── shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture()
def mcp_and_tools():
    """Return (mcp_mock, registered_tools_dict) after register_tools()."""
    mcp = MagicMock()
    registered: dict = {}

    def tool():
        def deco(fn):
            registered[fn.__name__] = fn
            return fn
        return deco

    mcp.tool = tool
    register_tools(mcp)
    return mcp, registered


# ── tool registration ─────────────────────────────────────────────────────────

def test_rcsb_search_registered(mcp_and_tools):
    _, registered = mcp_and_tools
    assert "rcsb_search" in registered


def test_rcsb_count_registered(mcp_and_tools):
    _, registered = mcp_and_tools
    assert "rcsb_count" in registered


def test_rcsb_get_entries_registered(mcp_and_tools):
    _, registered = mcp_and_tools
    assert "rcsb_get_entries" in registered


def test_rcsb_fetch_files_registered(mcp_and_tools):
    _, registered = mcp_and_tools
    assert "rcsb_fetch_files" in registered


def test_rcsb_search_callable(mcp_and_tools):
    _, registered = mcp_and_tools
    assert callable(registered["rcsb_search"])


# ── docstrings ────────────────────────────────────────────────────────────────

def test_rcsb_search_docstring(mcp_and_tools):
    """rcsb_search docstring must mention RCSB."""
    _, registered = mcp_and_tools
    doc = registered["rcsb_search"].__doc__ or ""
    assert "RCSB" in doc


def test_rcsb_count_docstring(mcp_and_tools):
    """rcsb_count docstring must mention total_hits."""
    _, registered = mcp_and_tools
    doc = registered["rcsb_count"].__doc__ or ""
    assert "total_hits" in doc


# ── isolated extension failure ────────────────────────────────────────────────

def test_load_extensions_isolated_failure(monkeypatch):
    """A broken module in EXTENSION_MODULES is logged but does not prevent
    the working module from registering its tools."""
    import registry as _registry_mod

    monkeypatch.setattr(
        _registry_mod,
        "EXTENSION_MODULES",
        [
            "rcsb",                              # valid standalone module
            "nonexistent_broken_module_xyz_abc", # will raise ImportError
        ],
    )

    mcp = MagicMock()
    registered: dict = {}

    def tool():
        def deco(fn):
            registered[fn.__name__] = fn
            return fn
        return deco

    mcp.tool = tool
    # Must not raise; broken module is isolated
    load_extensions(mcp)
    # Tools from the valid module must still be registered
    assert "rcsb_search" in registered


# ── schema_version on results ─────────────────────────────────────────────────

def test_schema_version_in_fetch_result(mock_http, tmp_output_dir):
    """rcsb_fetch_files result always carries schema_version == SCHEMA_VERSION."""
    mcp = MagicMock()
    registered: dict = {}

    def tool():
        def deco(fn):
            registered[fn.__name__] = fn
            return fn
        return deco

    mcp.tool = tool
    register_tools(mcp)

    result = registered["rcsb_fetch_files"](
        entry_ids=["3PXY"],
        formats=["cif"],
        output_dir=str(tmp_output_dir),
    )
    assert result["schema_version"] == SCHEMA_VERSION


def test_schema_version_in_entries_result(mock_data_api):
    """rcsb_get_entries result always carries schema_version == SCHEMA_VERSION."""
    mcp = MagicMock()
    registered: dict = {}

    def tool():
        def deco(fn):
            registered[fn.__name__] = fn
            return fn
        return deco

    mcp.tool = tool
    register_tools(mcp)

    result = registered["rcsb_get_entries"](entry_ids=["3PXY"])
    assert result["schema_version"] == SCHEMA_VERSION


# ── search result stable fields ───────────────────────────────────────────────

def test_search_result_stable_fields(mock_http):
    """rcsb_search always returns: rcsb_query_id, total_hits, page, facets, results."""
    mock_http.set_response(
        "default",
        make_search_response(total_count=3, result_ids=["3PXY", "4HHB", "1HSA"]),
    )

    mcp = MagicMock()
    registered: dict = {}

    def tool():
        def deco(fn):
            registered[fn.__name__] = fn
            return fn
        return deco

    mcp.tool = tool
    register_tools(mcp)

    result = registered["rcsb_search"](
        query={
            "type": "terminal",
            "service": "text",
            "parameters": {"value": "CDK2"},
        },
    )
    for field in ("rcsb_query_id", "total_hits", "page", "facets", "results"):
        assert field in result, f"Missing field '{field}' in rcsb_search result"


# ── null / error entries ──────────────────────────────────────────────────────

def test_get_entries_null_for_missing(mock_data_api):
    """Unknown entry ID returns error='entry_not_found', not a hard failure."""
    mcp = MagicMock()
    registered: dict = {}

    def tool():
        def deco(fn):
            registered[fn.__name__] = fn
            return fn
        return deco

    mcp.tool = tool
    register_tools(mcp)

    result = registered["rcsb_get_entries"](entry_ids=["XXXX"])
    assert result["entries"]["XXXX"]["error"] == "entry_not_found"


# ── partial download failure ──────────────────────────────────────────────────

def test_fetch_files_partial_failure(mock_http, tmp_output_dir, mcp_and_tools):
    """When one entry's download fails, other entries still produce valid artifacts."""
    _, registered = mcp_and_tools
    # mock_http already patches RCSBHTTPClient → mock_http (the mock_http_client)

    original_ds = mock_http.download_stream

    def patched_ds(url: str, dest: Path) -> tuple[str, int]:
        if "4HHB" in url:
            raise RCSBUnavailableError("mocked 4HHB failure")
        return original_ds(url, dest)

    mock_http.download_stream = patched_ds

    result = registered["rcsb_fetch_files"](
        entry_ids=["3PXY", "4HHB"],
        formats=["cif"],
        output_dir=str(tmp_output_dir),
    )
    artifacts = result["artifacts"]
    assert len(artifacts) == 2

    by_entry = {a["entry_id"]: a for a in artifacts}
    assert by_entry["3PXY"]["error"] is None, "3PXY should succeed"
    assert by_entry["4HHB"]["error"] is not None, "4HHB should carry an error string"


# ── include_percentiles parameter ─────────────────────────────────────────────

def test_rcsb_get_entries_accepts_include_percentiles_param(mcp_and_tools):
    """rcsb_get_entries accepts include_percentiles=False without error."""
    _, registered = mcp_and_tools
    fn = registered["rcsb_get_entries"]
    import inspect
    sig = inspect.signature(fn)
    assert "include_percentiles" in sig.parameters, (
        "rcsb_get_entries must accept include_percentiles parameter"
    )
    # Default must be False
    default = sig.parameters["include_percentiles"].default
    assert default is False, "include_percentiles must default to False"
