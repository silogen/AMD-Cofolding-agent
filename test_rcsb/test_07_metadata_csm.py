"""test_07_metadata_csm.py — Tests for computed structure model (CSM) handling.

CSMs (AlphaFold, ESMFold, etc.) are identified by:
  rcsb_entry_info.structure_determination_methodology == "computational"

The normalizer returns a "computational" experiment block (with confidence_metrics
and resolution_A==None) instead of a standard experimental block.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from rcsb import normalize_entry, _is_csm, register_tools
from test_rcsb.utils import load_fixture, make_search_response


# ── CSM detection ────────────────────────────────────────────────────────────

def test_csm_detected_by_methodology(fixture_af):
    """An AlphaFold entry (computational methodology) is detected as CSM."""
    assert _is_csm(fixture_af) is True


def test_experimental_not_csm(fixture_3pxy):
    """fixture_3pxy is the raw entry dict (from conftest); not a CSM."""
    assert _is_csm(fixture_3pxy) is False


# ── CSM normalization: experiment block ──────────────────────────────────────

def test_csm_model_block(fixture_af):
    """normalize_entry returns experiment.type == 'computational' for CSMs."""
    entry = normalize_entry(fixture_af, "AF_AFP68871F1")
    assert entry["experiment"]["type"] == "computational"


def test_csm_no_resolution(fixture_af):
    """CSMs have no crystallographic resolution."""
    entry = normalize_entry(fixture_af, "AF_AFP68871F1")
    assert entry["experiment"]["resolution_A"] is None


def test_csm_no_space_group(fixture_af):
    """CSMs have no space group (no crystal)."""
    entry = normalize_entry(fixture_af, "AF_AFP68871F1")
    assert entry["experiment"]["space_group"] is None


def test_csm_plddt_present(fixture_af):
    """CSM experiment block has confidence_metrics list with at least one entry."""
    entry = normalize_entry(fixture_af, "AF_AFP68871F1")
    metrics = entry["experiment"]["confidence_metrics"]
    assert isinstance(metrics, list)
    assert len(metrics) >= 1


def test_csm_plddt_value(fixture_af):
    """First confidence metric is pLDDT with a positive value."""
    entry = normalize_entry(fixture_af, "AF_AFP68871F1")
    first = entry["experiment"]["confidence_metrics"][0]
    assert first["type"] == "pLDDT"
    assert first["value"] > 0


# ── CSM normalization: identity block ────────────────────────────────────────

def test_csm_entry_type_computational(fixture_af):
    """identity.entry_type is 'computational' for CSMs."""
    entry = normalize_entry(fixture_af, "AF_AFP68871F1")
    assert entry["identity"]["entry_type"] == "computational"


# ── CSM download URL ─────────────────────────────────────────────────────────

def test_csm_download_url(fixture_af):
    """download_urls.cif contains the full entry ID (including AF_ prefix)."""
    entry = normalize_entry(fixture_af, "AF_AFP68871F1")
    assert "AF_AFP68871F1" in entry["download_urls"]["cif"]


# ── CSM content type in search payload ───────────────────────────────────────

@pytest.fixture()
def tools(mock_http, monkeypatch):
    """Register all RCSB MCP tools and return (registered_dict, mock_http_client).

    QueryBuilder.from_dict is patched to bypass rcsbapi when not installed.
    """
    import rcsb as _rcsb
    monkeypatch.setattr(
        _rcsb.QueryBuilder, "from_dict",
        staticmethod(lambda spec: {"type": "terminal", "service": "text",
                                   "parameters": {"value": "test"}}),
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
    return registered, mock_http


def test_csm_content_type_in_search(tools):
    """rcsb_search forwards results_content_type=['computational'] to RCSB POST."""
    registered, mock_http = tools
    # Provide a response so post_json doesn't raise
    mock_http.set_response("default", make_search_response(total_count=5))

    registered["rcsb_search"](
        query={
            "type": "terminal",
            "service": "text",
            "parameters": {"value": "CDK2"},
        },
        request_options={"results_content_type": ["computational"]},
    )

    payload = mock_http.last_post_payload()
    assert payload["request_options"]["results_content_type"] == ["computational"]
