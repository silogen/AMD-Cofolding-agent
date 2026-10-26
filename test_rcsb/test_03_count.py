"""test_03_count.py — Tests for the rcsb_count MCP tool.

Uses the same mock_http / tools pattern as test_02.  QueryBuilder.from_dict
is stubbed to the identity function.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rcsb import register_tools
from test_rcsb.utils import load_fixture


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_count(tools_fixture, query=None, **kwargs):
    registered, _ = tools_fixture
    return registered["rcsb_count"](
        query=query or _SIMPLE_QUERY,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_count_returns_integer(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    result = _call_count(tools)
    assert result["total_hits"] == 638


def test_count_zero(tools):
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "no_results.json"))
    result = _call_count(tools)
    assert result["total_hits"] == 0


def test_count_no_paginate_in_payload(tools):
    """rcsb_count must NOT include a 'paginate' key in request_options."""
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    _call_count(tools)
    payload = mock_http.last_post_payload()
    assert "paginate" not in payload.get("request_options", {})


def test_count_return_counts_in_payload(tools):
    """rcsb_count must send return_counts=True in request_options."""
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    _call_count(tools)
    payload = mock_http.last_post_payload()
    assert payload["request_options"].get("return_counts") is True


def test_count_content_type_forwarded(tools):
    """results_content_type must be forwarded into request_options."""
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    _call_count(tools, results_content_type=["computational"])
    payload = mock_http.last_post_payload()
    assert payload["request_options"].get("results_content_type") == [
        "computational"
    ]


def test_count_query_id_returned(tools):
    """rcsb_count result must include a non-empty rcsb_query_id string."""
    _, mock_http = tools
    mock_http.set_response("default", load_fixture("search", "cdk2_text_p1.json"))
    result = _call_count(tools)
    qid = result.get("rcsb_query_id")
    assert isinstance(qid, str) and len(qid) > 0
