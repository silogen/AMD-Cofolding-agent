"""test_02_search_execution.py — Tests for the rcsb_search MCP tool.

Uses the mock_http fixture to capture POST payloads without touching the
network.  QueryBuilder.from_dict is stubbed to the identity function so
these tests do not require rcsbapi to be installed.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rcsb import register_tools, _build_search_response, _parse_result_list
from test_rcsb.utils import load_fixture, make_search_response


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SIMPLE_QUERY = {
    "type": "terminal",
    "service": "text",
    "parameters": {"value": "CDK2"},
}


@pytest.fixture()
def tools(mock_http, monkeypatch):
    """Register all MCP tools with a fake MCP object.

    * QueryBuilder.from_dict is patched to the identity function so that
      tests do not require rcsbapi.
    * mock_http is already patching RCSBHTTPClient.
    """
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_search(tools_fixture, query=None, return_type="entry", **kwargs):
    registered, _ = tools_fixture
    return registered["rcsb_search"](
        query=query or _SIMPLE_QUERY,
        return_type=return_type,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Pagination & result count
# ---------------------------------------------------------------------------

def test_page_one_25_results(tools):
    registered, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    result = _call_search(tools)
    assert len(result["results"]) == 25
    assert result["page"]["start"] == 0


def test_page_two_start_offset(tools):
    registered, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p2.json"))
    _call_search(tools, request_options={"paginate": {"start": 25, "rows": 25}})
    payload = mock_http.last_post_payload()
    assert payload["request_options"]["paginate"]["start"] == 25


def test_has_more_true_on_page_one(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    result = _call_search(tools)
    # total=638, rows=25, start=0 → (0+25) < 638
    assert result["page"]["has_more"] is True
    assert result["page"]["total_pages"] == math.ceil(638 / 25)


def test_has_more_false_on_last_page(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    result = _call_search(
        tools,
        request_options={"paginate": {"start": 625, "rows": 25}},
    )
    # 625 + 25 = 650 > 638 → no more pages
    assert result["page"]["has_more"] is False


def test_total_pages_calculated(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    result = _call_search(tools)
    assert result["page"]["total_pages"] == math.ceil(638 / 25)  # == 26


# ---------------------------------------------------------------------------
# Payload forwarding
# ---------------------------------------------------------------------------

def test_query_forwarded_unchanged(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    result = _call_search(tools, query=_SIMPLE_QUERY)
    # Because QueryBuilder.from_dict is stubbed to identity, query_forwarded
    # must equal the original dict passed in.
    assert result["query_forwarded"] == _SIMPLE_QUERY


def test_return_type_entry(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    _call_search(tools, return_type="entry")
    payload = mock_http.last_post_payload()
    assert payload["return_type"] == "entry"


def test_return_type_nonpolymer(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    _call_search(tools, return_type="non_polymer_entity")
    payload = mock_http.last_post_payload()
    assert payload["return_type"] == "non_polymer_entity"


def test_results_verbosity_minimal(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    _call_search(
        tools,
        request_options={"results_verbosity": "minimal"},
    )
    payload = mock_http.last_post_payload()
    assert payload["request_options"].get("results_verbosity") == "minimal"


def test_content_type_experimental(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    _call_search(
        tools,
        request_options={"results_content_type": ["experimental"]},
    )
    payload = mock_http.last_post_payload()
    assert payload["request_options"]["results_content_type"] == ["experimental"]


def test_content_type_computational(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    _call_search(
        tools,
        request_options={"results_content_type": ["computational"]},
    )
    payload = mock_http.last_post_payload()
    assert payload["request_options"]["results_content_type"] == ["computational"]


def test_content_type_both(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    _call_search(
        tools,
        request_options={"results_content_type": ["experimental", "computational"]},
    )
    payload = mock_http.last_post_payload()
    assert payload["request_options"]["results_content_type"] == [
        "experimental",
        "computational",
    ]


def test_scoring_strategy_forwarded(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    _call_search(
        tools,
        request_options={"scoring_strategy": "sequence"},
    )
    payload = mock_http.last_post_payload()
    assert payload["request_options"].get("scoring_strategy") == "sequence"


def test_return_explain_metadata(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    _call_search(
        tools,
        request_options={"return_explain_metadata": True},
    )
    payload = mock_http.last_post_payload()
    assert payload["request_options"].get("return_explain_metadata") is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_no_results_empty_list(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "no_results.json"))
    result = _call_search(tools)
    assert result["results"] == []
    assert result["total_hits"] == 0


def test_rcsb_query_id_stable(tools):
    """Same query called twice must produce the same rcsb_query_id."""
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    r1 = _call_search(tools, query=_SIMPLE_QUERY)
    r2 = _call_search(tools, query=_SIMPLE_QUERY)
    assert r1["rcsb_query_id"] == r2["rcsb_query_id"]


def test_rcsb_query_id_changes(tools):
    """Different queries must produce different rcsb_query_ids."""
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    q1 = {"type": "terminal", "service": "text", "parameters": {"value": "CDK2"}}
    q2 = {"type": "terminal", "service": "text", "parameters": {"value": "kinase"}}
    r1 = _call_search(tools, query=q1)
    r2 = _call_search(tools, query=q2)
    assert r1["rcsb_query_id"] != r2["rcsb_query_id"]


def test_missing_facets_field_ok(tools):
    """A response without a 'facets' key must not raise and must yield facets=={}."""
    _, mock_http = tools
    response_without_facets = {
        "total_count": 3,
        "result_list": [
            {"identifier": "3PXY", "score": 1.0},
            {"identifier": "4HHB", "score": 0.9},
            {"identifier": "1FQV", "score": 0.8},
        ],
        # no "facets" key
    }
    mock_http.set_response("default", response_without_facets)
    result = _call_search(tools)
    assert result["facets"] == {}


def test_group_by_forwarded(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    group_by_opts = {
        "aggregation_method": "sequence_identity",
        "similarity_cutoff": 0.9,
        "ranking_criteria_type": [{"sort_by": "coverage", "direction": "desc"}],
    }
    _call_search(
        tools,
        request_options={"group_by": group_by_opts},
    )
    payload = mock_http.last_post_payload()
    assert payload["request_options"]["group_by"] == group_by_opts
