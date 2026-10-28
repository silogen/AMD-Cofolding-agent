# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for which REINVENT runs each conversation launched (GET /api/sessions ``runs``).

Everything is built in temporary directories::

    python -m pytest tests/test_session_runs.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from reinvent4_agent.web import session_data
from reinvent4_agent.web.session_data import Roots, RunIndex

SID = "s-20260101-000000-aaaaaa"
OTHER = "s-20260102-000000-bbbbbb"


@pytest.fixture
def roots(tmp_path) -> Roots:
    for name in ("shared", "jobs", "sessions"):
        (tmp_path / name).mkdir()
    return Roots(shared=tmp_path / "shared", jobs=tmp_path / "jobs")


@pytest.fixture
def store(tmp_path) -> Path:
    return tmp_path / "sessions"


def make_job(roots: Roots, job_id: str, state: str, config: str = "[parameters]\n") -> Path:
    directory = roots.jobs / f"job_{job_id}"
    directory.mkdir(exist_ok=True)
    (directory / "config.toml").write_text(config)
    (directory / "job_metadata.json").write_text(json.dumps({"job_id": job_id, "state": state}))
    return directory


def launch(seq: int, job_id: str, tool: str = "run_optimization") -> list[dict]:
    return [
        {"type": "tool_call", "id": f"c{seq}", "name": tool, "arguments": {"ttp_id": "brd4"}, "seq": seq},
        {"type": "tool_result", "id": f"c{seq}", "name": tool, "result": json.dumps({"job_id": job_id, "state": "running"}), "isError": False, "seq": seq + 1},
    ]


def look(seq: int, job_id: str) -> list[dict]:
    return [
        {"type": "tool_call", "id": f"c{seq}", "name": "get_optimization_status", "arguments": {"job_id": job_id}, "seq": seq},
        {"type": "tool_result", "id": f"c{seq}", "name": "get_optimization_status", "result": "{}", "isError": False, "seq": seq + 1},
    ]


def write_events(store: Path, session_id: str, events: list[dict]) -> Path:
    path = store / session_id / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    return path


def test_counts_the_runs_a_conversation_launched(roots, store):
    make_job(roots, "aaaa1111", "running")
    make_job(roots, "bbbb2222", "completed")
    make_job(roots, "cccc3333", "queued")
    write_events(store, SID, launch(1, "aaaa1111") + launch(3, "bbbb2222") + launch(5, "cccc3333", "run_transfer_learning"))

    runs = RunIndex(store).runs([SID], roots)

    assert runs[SID]["running"] == 2
    assert runs[SID]["total"] == 3
    assert [(j["id"], j["state"], j["launched"]) for j in runs[SID]["jobs"]] == [
        ("aaaa1111", "running", True),
        ("bbbb2222", "completed", True),
        ("cccc3333", "queued", True),
    ]


NONE = {"running": 0, "total": 0, "jobs": []}


def test_a_conversation_without_runs_says_so(roots, store):
    write_events(store, SID, [{"type": "user_message", "content": "hello", "seq": 1}])
    assert RunIndex(store).runs([SID, OTHER], roots) == {SID: NONE, OTHER: NONE}


def test_referenced_jobs_are_listed_but_not_counted(roots, store):
    make_job(roots, "aaaa1111", "running")
    write_events(store, OTHER, launch(1, "aaaa1111"))
    write_events(store, SID, look(1, "aaaa1111"))

    runs = RunIndex(store).runs([SID, OTHER], roots)

    assert runs[SID] == {"running": 0, "total": 0, "jobs": [{"id": "aaaa1111", "state": "running", "launched": False}]}
    assert runs[OTHER]["running"] == 1


def test_jobs_that_are_gone_are_dropped(roots, store):
    write_events(store, SID, launch(1, "dead0000") + look(3, "beef0000"))
    assert RunIndex(store).runs([SID], roots) == {SID: NONE}


def test_a_run_continued_outside_the_chat_belongs_to_the_conversation_of_its_parent(roots, store):
    make_job(roots, "aaaa1111", "stopped")
    make_job(roots, "bbbb2222", "running", f'agent_file = "{roots.jobs}/job_aaaa1111/stage.chkpt"\n')
    make_job(roots, "cccc3333", "queued", f'agent_file = "/data/reinvent4_mcp/jobs/job_bbbb2222/stage.chkpt"\n')
    write_events(store, SID, launch(1, "aaaa1111"))

    runs = RunIndex(store).runs([SID], roots)

    assert [j["id"] for j in runs[SID]["jobs"]] == ["aaaa1111", "bbbb2222", "cccc3333"]
    assert runs[SID]["running"] == 2


def test_a_run_launched_elsewhere_is_not_inherited(roots, store):
    make_job(roots, "aaaa1111", "completed")
    make_job(roots, "bbbb2222", "running", f'agent_file = "/jobs/job_aaaa1111/stage.chkpt"\n')
    write_events(store, SID, launch(1, "aaaa1111"))
    write_events(store, OTHER, launch(1, "bbbb2222"))

    runs = RunIndex(store).runs([SID, OTHER], roots)

    assert [j["id"] for j in runs[SID]["jobs"]] == ["aaaa1111"]
    assert [j["id"] for j in runs[OTHER]["jobs"]] == ["bbbb2222"]


def test_events_are_read_again_only_when_they_change(roots, store, monkeypatch):
    make_job(roots, "aaaa1111", "running")
    path = write_events(store, SID, launch(1, "aaaa1111"))
    index = RunIndex(store)
    reads: list[Path] = []
    real = session_data._read_events
    monkeypatch.setattr(session_data, "_read_events", lambda p: reads.append(p) or real(p))

    index.runs([SID], roots)
    index.runs([SID], roots)
    assert len(reads) == 1

    make_job(roots, "bbbb2222", "running")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(e) + "\n" for e in launch(3, "bbbb2222")))
    assert index.runs([SID], roots)[SID]["total"] == 2
    assert len(reads) == 2


def test_a_state_change_is_seen_without_rereading_the_events(roots, store):
    directory = make_job(roots, "aaaa1111", "running")
    write_events(store, SID, launch(1, "aaaa1111"))
    index = RunIndex(store)
    assert index.runs([SID], roots)[SID]["running"] == 1

    meta = directory / "job_metadata.json"
    meta.write_text(json.dumps({"job_id": "aaaa1111", "state": "completed", "finished_at": "later"}))
    stat = meta.stat()
    os.utime(meta, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert index.runs([SID], roots)[SID] == {"running": 0, "total": 1, "jobs": [{"id": "aaaa1111", "state": "completed", "launched": True}]}


def test_the_sessions_listing_carries_the_runs(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from reinvent4_agent.web import server

    sessions_dir = tmp_path / "sessions"
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setenv("REINVENT4_WEB_SESSION_DIR", str(sessions_dir))
    monkeypatch.setenv("REINVENT4_JOBS_DIR", str(jobs_dir))
    monkeypatch.delenv(server.chat_models.CONTROL_DIR_ENV, raising=False)

    async def start(self) -> None:
        self.ready = True

    monkeypatch.setattr(server.AgentGateway, "start", start)
    monkeypatch.setattr(server.AgentGateway, "new_agent", lambda self, handler: None)

    make_job(Roots(shared=tmp_path, jobs=jobs_dir), "aaaa1111", "running")
    for sid, events in ((SID, launch(1, "aaaa1111")), (OTHER, [])):
        write_events(sessions_dir, sid, [{"type": "user_message", "content": "go", "seq": 0}, *events])
        (sessions_dir / sid / "meta.json").write_text(json.dumps({"id": sid, "title": "t", "created": "x", "updated": sid, "turns": 1}))

    with TestClient(server.create_app()) as client:
        listed = {s["id"]: s for s in client.get("/api/sessions").json()}

    assert listed[SID]["runs"] == {"running": 1, "total": 1, "jobs": [{"id": "aaaa1111", "state": "running", "launched": True}]}
    assert listed[OTHER]["runs"] == NONE
