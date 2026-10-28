# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the gateway's summary of what Boltz is cofolding.

A small local HTTP server stands in for the Boltz service; the MCP server
and the LLM are not involved.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from reinvent4_agent.web import cofold_status, server  # noqa: E402

BUSY_HEALTH = {
    "status": "ok",
    "mode": "pool",
    "workers": [
        {"id": "gpu0-w0", "device": 0, "state": "busy", "request_id": "a1", "completed": 3},
        {"id": "gpu0-w1", "device": 0, "state": "busy", "request_id": "a1", "completed": 2},
        {"id": "gpu1-w0", "device": 1, "state": "busy", "request_id": "b2", "completed": 0},
        {"id": "gpu1-w1", "device": 1, "state": "ready", "completed": 0},
        {"id": "gpu2-w0", "device": 2, "state": "loading", "completed": 0},
        {"id": "gpu2-w1", "device": 2, "state": "failed", "error": "HIP error", "completed": 0},
    ],
    "queue_length": 1,
    "requests": [
        {"id": "a1", "molecules": 4, "done": 1, "queued": 1, "running": 2, "state": "running",
         "started_at": 1793100000.0, "elapsed_s": 41.26},
        {"id": "b2", "molecules": 1, "done": 0, "queued": 0, "running": 1, "state": "running",
         "started_at": 1793100030.0, "elapsed_s": 11.0},
    ],
}


class FakeBoltz:
    """Serves ``/health`` with whatever ``body`` is, after ``delay`` seconds."""

    def __init__(self) -> None:
        self.body: object = {"status": "ok"}
        self.delay = 0.0
        self.calls = 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                fake.calls += 1
                time.sleep(fake.delay)
                payload = json.dumps(fake.body).encode()
                self.send_response(200 if self.path == "/health" else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/score"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def boltz(monkeypatch, tmp_path):
    from reinvent4_mcp import config as mcp_config

    fake = FakeBoltz()
    config = tmp_path / "mcp.yaml"
    config.write_text(f"scoring:\n  boltz:\n    service_url: \"{fake.url}\"\n")
    monkeypatch.setenv("REINVENT4_MCP_CONFIG", str(config))
    mcp_config.reset_config()
    yield fake
    mcp_config.reset_config()
    fake.close()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("REINVENT4_WEB_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.delenv(server.chat_models.CONTROL_DIR_ENV, raising=False)
    monkeypatch.delenv("REINVENT4_WEB_TOKEN", raising=False)

    async def start(self) -> None:
        self.ready = True

    monkeypatch.setattr(server.AgentGateway, "start", start)
    with TestClient(server.create_app()) as test_client:
        yield test_client


def test_the_health_url_comes_from_the_mcp_config(boltz):
    assert cofold_status.boltz_health_url() == boltz.url[: -len("/score")] + "/health"


def test_busy_service_is_summarised(boltz, client):
    boltz.body = BUSY_HEALTH
    body = client.get("/api/services/cofold").json()
    assert body == {
        "available": True,
        "status": "ok",
        "mode": "pool",
        "moleculeSeconds": None,
        "workers": {"total": 6, "busy": 3, "loading": 1, "failed": 1},
        "gpus": {"total": 3, "busy": 2},
        "queueLength": 1,
        "requests": [
            {"id": "a1", "molecules": 4, "done": 1, "queued": 1, "running": 2, "state": "running", "elapsedSeconds": 41.3,
             "startedAt": 1793100000.0, "runningSeconds": None},
            {"id": "b2", "molecules": 1, "done": 0, "queued": 0, "running": 1, "state": "running", "elapsedSeconds": 11.0,
             "startedAt": 1793100030.0, "runningSeconds": None},
        ],
        "active": True,
    }


def test_molecule_times_are_passed_on(boltz, client):
    first = {**BUSY_HEALTH["requests"][0], "running_s": [12.04, 30.0, "x", -1]}
    boltz.body = {**BUSY_HEALTH, "molecule_s": 85.25, "requests": [first]}
    body = client.get("/api/services/cofold").json()
    assert body["moleculeSeconds"] == 85.2
    assert body["requests"][0]["runningSeconds"] == [12.0, 30.0]


def test_idle_pool_is_not_active(boltz, client):
    boltz.body = {**BUSY_HEALTH, "workers": [{"id": "gpu0-w0", "device": 0, "state": "ready"}], "queue_length": 0, "requests": []}
    body = client.get("/api/services/cofold").json()
    assert body["available"] is True
    assert body["active"] is False
    assert body["workers"] == {"total": 1, "busy": 0, "loading": 0, "failed": 0}


def test_a_service_without_request_tracking_still_reports_busy_workers(boltz, client):
    boltz.body = {key: value for key, value in BUSY_HEALTH.items() if key != "requests"}
    body = client.get("/api/services/cofold").json()
    assert body["requests"] == []
    assert body["active"] is True


def test_a_subprocess_service_reports_its_requests(boltz, client):
    boltz.body = {"status": "ok", "requests": [{"id": "c3", "molecules": 2, "done": 0, "queued": 0, "running": 2, "state": "running", "elapsed_s": 5}]}
    body = client.get("/api/services/cofold").json()
    assert body["mode"] == "subprocess"
    assert body["requests"][0]["molecules"] == 2
    assert body["active"] is True


def test_malformed_fields_are_dropped(boltz, client):
    boltz.body = {"status": "ok", "workers": ["x", {"state": "busy", "device": 0}], "queue_length": "many",
                  "requests": ["x", {"id": 7, "molecules": -1, "done": True, "state": "weird"}]}
    body = client.get("/api/services/cofold").json()
    assert body["workers"]["total"] == 1
    assert body["queueLength"] == 0
    assert body["requests"] == [
        {"id": "7", "molecules": 0, "done": 0, "queued": 0, "running": 0, "state": "running", "elapsedSeconds": None,
         "startedAt": None, "runningSeconds": None},
    ]


def test_a_slow_service_is_unavailable_within_the_timeout(boltz, client, monkeypatch):
    monkeypatch.setattr(cofold_status, "TIMEOUT_S", 0.2)
    boltz.delay = 1.0
    started = time.monotonic()
    assert client.get("/api/services/cofold").json() == {"available": False}
    assert time.monotonic() - started < 1.0


def test_an_unreachable_service_is_unavailable(boltz, client):
    boltz.close()
    assert client.get("/api/services/cofold").json() == {"available": False}


def test_a_non_object_answer_is_unavailable(boltz, client):
    boltz.body = ["not", "an", "object"]
    assert client.get("/api/services/cofold").json() == {"available": False}


def test_without_boltz_enabled_nothing_is_fetched(client, monkeypatch, tmp_path):
    from reinvent4_mcp import config as mcp_config

    config = tmp_path / "mcp.yaml"
    config.write_text("scoring:\n  boltz:\n    enabled: false\n")
    monkeypatch.setenv("REINVENT4_MCP_CONFIG", str(config))
    mcp_config.reset_config()
    try:
        assert client.get("/api/services/cofold").json() == {"available": False}
    finally:
        mcp_config.reset_config()


def test_the_access_token_applies(boltz, monkeypatch, tmp_path):
    monkeypatch.setenv("REINVENT4_WEB_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.delenv(server.chat_models.CONTROL_DIR_ENV, raising=False)
    monkeypatch.setenv("REINVENT4_WEB_TOKEN", "s3cret")

    async def start(self) -> None:
        self.ready = True

    monkeypatch.setattr(server.AgentGateway, "start", start)
    with TestClient(server.create_app()) as test_client:
        assert test_client.get("/api/services/cofold").status_code == 401
        boltz.body = BUSY_HEALTH
        answer = test_client.get("/api/services/cofold", headers={"X-DD-Token": "s3cret"})
        assert answer.status_code == 200
        assert answer.json()["active"] is True
    assert boltz.calls == 1
