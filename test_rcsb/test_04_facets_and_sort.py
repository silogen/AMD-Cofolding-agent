"""test_04_facets_and_sort.py — Tests for facet parsing and sort/group_by forwarding.

Uses both the direct helper functions (_build_search_response, _parse_result_list)
and the full rcsb_search MCP tool via mock_http.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rcsb import _build_search_response, _parse_result_list, register_tools
from test_rcsb.utils import load_fixture


# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

_SIMPLE_QUERY = {
    "type": "terminal",
    "service": "text",
    "parameters": {"value": "CDK2"},
}


@pytest.fixture()
def tools(mock_http, monkeypatch):
    """Register all MCP tools; stub QueryBuilder.from_dict to identity."""
    monkeypatch.setattr("rcsb.QueryBuilder.from_dict", lambda spec: spec)

    mcp = MagicMock()
    registered: dict = {}

    def _tool():
        def decorator(fn):
            registered[fn.__name__] = fn
            return fn
        return decorator

    mcp.tool = _tool
    register_tools(mcp)
    return registered, mock_http


@pytest.fixture()
def cdk2_p1_response():
    """Load the CDK2 page-1 search fixture once for all facet tests."""
    return load_fixture("search", "cdk2_text_p1.json")


def _build_response_from_fixture(raw: dict, page_start: int = 0, rows: int = 25) -> dict:
    results = _parse_result_list(raw)
    return _build_search_response(raw, _SIMPLE_QUERY, results, page_start, rows)


def _call_search(tools_fixture, query=None, return_type="entry", **kwargs):
    registered, _ = tools_fixture
    return registered["rcsb_search"](
        query=query or _SIMPLE_QUERY,
        return_type=return_type,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Facet parsing — helper functions
# ---------------------------------------------------------------------------

def test_facet_terms_organism(cdk2_p1_response):
    """Organism facet must be present with at least one bucket."""
    resp = _build_response_from_fixture(cdk2_p1_response)
    assert "Organism" in resp["facets"], "Expected 'Organism' facet in parsed response"
    buckets = resp["facets"]["Organism"]
    assert len(buckets) >= 1
    # Each bucket must carry a key and a count
    for b in buckets:
        assert "key" in b, f"Bucket missing 'key': {b}"
        assert "count" in b, f"Bucket missing 'count': {b}"


def test_facet_terms_method(cdk2_p1_response):
    """Method facet must be present with at least one bucket."""
    resp = _build_response_from_fixture(cdk2_p1_response)
    assert "Method" in resp["facets"]
    buckets = resp["facets"]["Method"]
    assert len(buckets) >= 1
    for b in buckets:
        assert "key" in b
        assert "count" in b


def test_facet_range_resolution(cdk2_p1_response):
    """Resolution facet (range type) must be present with key+count per bucket."""
    resp = _build_response_from_fixture(cdk2_p1_response)
    assert "Resolution" in resp["facets"]
    buckets = resp["facets"]["Resolution"]
    assert len(buckets) >= 1
    for b in buckets:
        # The fixture uses "key" (e.g. "0-1.5") and "count" for range buckets
        assert "key" in b, f"Range bucket missing 'key': {b}"
        assert "count" in b, f"Range bucket missing 'count': {b}"


def test_facets_absent_ok(tools):
    """A response without a 'facets' field must return facets=={} without error."""
    _, mock_http = tools
    mock_http.set_response(
        "default",
        {
            "total_count": 2,
            "result_list": [
                {"identifier": "3PXY", "score": 1.0},
                {"identifier": "4HHB", "score": 0.9},
            ],
            # deliberately no "facets" key
        },
    )
    result = _call_search(tools)
    assert result["facets"] == {}


# ---------------------------------------------------------------------------
# Sort forwarding — via full rcsb_search tool
# ---------------------------------------------------------------------------

def test_sort_by_score_desc(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    sort_spec = [{"sort_by": "score", "direction": "desc"}]
    _call_search(tools, request_options={"sort": sort_spec})
    payload = mock_http.last_post_payload()
    assert payload["request_options"]["sort"] == sort_spec


def test_sort_by_release_date(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    sort_spec = [
        {
            "sort_by": "rcsb_accession_info.initial_release_date",
            "direction": "desc",
        }
    ]
    _call_search(tools, request_options={"sort": sort_spec})
    payload = mock_http.last_post_payload()
    assert payload["request_options"]["sort"] == sort_spec


def test_sort_multiple_keys(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    sort_spec = [
        {"sort_by": "score", "direction": "desc"},
        {"sort_by": "rcsb_accession_info.initial_release_date", "direction": "asc"},
    ]
    _call_search(tools, request_options={"sort": sort_spec})
    payload = mock_http.last_post_payload()
    assert len(payload["request_options"]["sort"]) == 2
    assert payload["request_options"]["sort"] == sort_spec


# ---------------------------------------------------------------------------
# group_by_return_type forwarding
# ---------------------------------------------------------------------------

def test_group_by_return_type_all(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    _call_search(tools, request_options={"group_by_return_type": "all"})
    payload = mock_http.last_post_payload()
    assert payload["request_options"].get("group_by_return_type") == "all"


def test_group_by_return_type_reps(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    _call_search(tools, request_options={"group_by_return_type": "representatives"})
    payload = mock_http.last_post_payload()
    assert payload["request_options"].get("group_by_return_type") == "representatives"
