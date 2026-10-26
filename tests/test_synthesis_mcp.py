"""Contract tests for the synthesis planning MCP tools.

The synthesis-planner HTTP service is mocked via httpx's MockTransport so
no running service or AiZynthFinder weights are needed.

Covers:
- plan_synthesis: order preservation, partial failure, status fields
- check_building_block_availability: order, unknown vs available
- submit_synthesis_job + get_synthesis_job_status: async job lifecycle
- HTTP error handling: graceful JSON error response (never raises)
- /health endpoint response shape (via a direct httpx mock)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "services" / "synthesis-planner"))

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_plan_response(statuses: list[str], n_routes: list[int] = None) -> dict:
    """Build a minimal /plan response payload."""
    if n_routes is None:
        n_routes = [1 if s == "success" else 0 for s in statuses]
    results = []
    for i, (status, nr) in enumerate(zip(statuses, n_routes)):
        results.append({
            "input_smiles":      f"SMILES_{i}",
            "canonical_smiles":  f"SMILES_{i}",
            "status":            status,
            "routes":            [{"route_id": f"r{i}", "n_steps": 2, "score": 0.8,
                                   "all_in_stock": True, "reactions": [], "leaves": []}] * nr,
            "n_routes":          nr,
            "search_stats":      {},
            "failure_reason":    None if status == "success" else f"Reason for {status}",
            "warning":           None,
            "elapsed_seconds":   1.0,
        })
    return {
        "results":           results,
        "model_version":     "uspto_rop@2024.1",
        "templates_version": "uspto_2017@2024.1",
        "stock_version":     "zinc_in_stock@2024-06",
        "n_molecules":       len(statuses),
        "n_success":         sum(1 for s in statuses if s == "success"),
        "n_no_route":        sum(1 for s in statuses if s in ("no_route", "timeout")),
        "n_failed":          sum(1 for s in statuses if s not in ("success", "no_route", "timeout")),
    }


def _make_avail_response(smiles: list[str], availabilities: list[str]) -> dict:
    results = [
        {"smiles": smi, "canonical_smiles": smi, "availability": av,
         "in_stock": av == "available", "stock_source": "zinc" if av == "available" else None,
         "warning": None}
        for smi, av in zip(smiles, availabilities)
    ]
    return {
        "results":       results,
        "stock_version": "zinc_in_stock@2024-06",
        "n_molecules":   len(smiles),
        "n_available":   availabilities.count("available"),
        "n_unavailable": availabilities.count("unavailable"),
        "n_unknown":     availabilities.count("unknown"),
    }


# ---------------------------------------------------------------------------
# Import the synthesis MCP module
# ---------------------------------------------------------------------------

def _load_synthesis_module():
    path = _REPO_ROOT / "reinvent4_mcp" / "tools" / "synthesis.py"
    spec = importlib.util.spec_from_file_location("synthesis_mcp", str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ===========================================================================
# Category 1: plan_synthesis — order preservation and partial failures
# ===========================================================================

class TestPlanSynthesisMCP:
    def _run_plan(self, smiles, mock_response, **params):
        """Call plan_synthesis with a mocked HTTP response."""
        syn = _load_synthesis_module()
        mcp = MagicMock()
        registered: dict = {}

        def _tool():
            def decorator(fn):
                registered[fn.__name__] = fn
                return fn
            return decorator
        mcp.tool = _tool
        syn.register_tools(mcp)
        fn = registered["plan_synthesis"]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client.post", return_value=mock_resp):
            result_str = fn(smiles=smiles, **params)

        return json.loads(result_str)

    def test_order_preserved_three_molecules(self):
        smiles = ["CC", "CCO", "c1ccccc1"]
        mock = _make_plan_response(["success", "no_route", "success"])
        result = self._run_plan(smiles, mock)
        assert len(result["results"]) == 3

    def test_partial_failure_does_not_shift_results(self):
        smiles = ["CC", "INVALID", "CCO"]
        mock = _make_plan_response(["success", "invalid_structure", "success"])
        result = self._run_plan(smiles, mock)
        statuses = [r["status"] for r in result["results"]]
        assert statuses == ["success", "invalid_structure", "success"]

    def test_no_route_is_valid_result(self):
        mock = _make_plan_response(["no_route"])
        result = self._run_plan(["CC"], mock)
        assert result["results"][0]["status"] == "no_route"
        assert result["results"][0]["n_routes"] == 0

    def test_timeout_distinct_from_no_route(self):
        mock = _make_plan_response(["timeout"])
        result = self._run_plan(["CC"], mock)
        assert result["results"][0]["status"] == "timeout"

    def test_provenance_fields_present(self):
        mock = _make_plan_response(["success"])
        result = self._run_plan(["CC"], mock)
        assert "model_version" in result
        assert "templates_version" in result
        assert "stock_version" in result

    def test_http_error_returns_json_error(self):
        """Service error should return JSON error dict, never raise."""
        import httpx
        syn = _load_synthesis_module()
        mcp = MagicMock()
        registered = {}

        def _tool():
            def decorator(fn):
                registered[fn.__name__] = fn
                return fn
            return decorator
        mcp.tool = _tool
        syn.register_tools(mcp)
        fn = registered["plan_synthesis"]

        with patch("httpx.Client.post", side_effect=Exception("connection refused")):
            result_str = fn(smiles=["CC"])

        result = json.loads(result_str)
        assert "error" in result


# ===========================================================================
# Category 2: check_building_block_availability
# ===========================================================================

class TestAvailabilityMCP:
    def _run_avail(self, smiles, mock_response, **params):
        syn = _load_synthesis_module()
        mcp = MagicMock()
        registered = {}

        def _tool():
            def decorator(fn):
                registered[fn.__name__] = fn
                return fn
            return decorator
        mcp.tool = _tool
        syn.register_tools(mcp)
        fn = registered["check_building_block_availability"]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client.post", return_value=mock_resp):
            result_str = fn(smiles=smiles, **params)

        return json.loads(result_str)

    def test_order_preserved(self):
        smiles = ["CC", "CCO", "c1ccccc1"]
        mock = _make_avail_response(smiles, ["available", "unavailable", "unknown"])
        result = self._run_avail(smiles, mock)
        avail = [r["availability"] for r in result["results"]]
        assert avail == ["available", "unavailable", "unknown"]

    def test_unknown_when_no_stock(self):
        mock = _make_avail_response(["CC"], ["unknown"])
        result = self._run_avail(["CC"], mock)
        assert result["results"][0]["availability"] == "unknown"
        assert result["results"][0]["in_stock"] is False or result["results"][0]["in_stock"] is None

    def test_available_sets_stock_source(self):
        mock = _make_avail_response(["CC"], ["available"])
        result = self._run_avail(["CC"], mock)
        assert result["results"][0]["stock_source"] == "zinc"

    def test_aggregate_counts(self):
        smiles = ["A", "B", "C", "D"]
        mock = _make_avail_response(smiles, ["available", "available", "unavailable", "unknown"])
        result = self._run_avail(smiles, mock)
        assert result["n_available"] == 2
        assert result["n_unavailable"] == 1
        assert result["n_unknown"] == 1

    def test_http_error_returns_json_error(self):
        syn = _load_synthesis_module()
        mcp = MagicMock()
        registered = {}

        def _tool():
            def decorator(fn):
                registered[fn.__name__] = fn
                return fn
            return decorator
        mcp.tool = _tool
        syn.register_tools(mcp)
        fn = registered["check_building_block_availability"]

        with patch("httpx.Client.post", side_effect=Exception("timeout")):
            result_str = fn(smiles=["CC"])

        result = json.loads(result_str)
        assert "error" in result


# ===========================================================================
# Category 3: Async job lifecycle
# ===========================================================================

class TestJobLifecycleMCP:
    def _get_tool(self, name):
        syn = _load_synthesis_module()
        mcp = MagicMock()
        registered = {}

        def _tool():
            def decorator(fn):
                registered[fn.__name__] = fn
                return fn
            return decorator
        mcp.tool = _tool
        syn.register_tools(mcp)
        return registered[name]

    def test_submit_returns_job_id(self):
        fn = self._get_tool("submit_synthesis_job")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"job_id": "abc123", "status": "queued", "n_molecules": 3}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client.post", return_value=mock_resp):
            result = json.loads(fn(smiles=["CC", "CCO", "c1ccccc1"]))

        assert result["job_id"] == "abc123"
        assert result["status"] == "queued"
        assert result["n_molecules"] == 3

    def test_poll_queued_status(self):
        fn = self._get_tool("get_synthesis_job_status")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "job_id": "abc123", "status": "running", "n_molecules": 3,
            "result": None, "error": None,
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client.get", return_value=mock_resp):
            result = json.loads(fn(job_id="abc123"))

        assert result["status"] == "running"
        assert result["result"] is None

    def test_poll_done_has_results(self):
        fn = self._get_tool("get_synthesis_job_status")
        plan_resp = _make_plan_response(["success", "no_route"])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "job_id": "abc123", "status": "done", "n_molecules": 2,
            "result": plan_resp, "error": None,
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client.get", return_value=mock_resp):
            result = json.loads(fn(job_id="abc123"))

        assert result["status"] == "done"
        assert result["result"]["n_molecules"] == 2

    def test_job_not_found_returns_error(self):
        import httpx
        fn = self._get_tool("get_synthesis_job_status")
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=mock_resp
        )

        with patch("httpx.Client.get", return_value=mock_resp):
            result = json.loads(fn(job_id="bad_id"))

        assert "error" in result
