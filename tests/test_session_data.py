# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for a conversation's data view and for deleting it with its data.

Everything is built in temporary directories; nothing outside them is read
or removed.  Run from the repository root::

    python -m pytest tests/test_session_data.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from reinvent4_agent.web import session_data
from reinvent4_agent.web.session_data import Roots, collect_references, inventory, purge, stored_references, unsafe_target

SID = "s-20260101-000000-aaaaaa"
OTHER = "s-20260102-000000-bbbbbb"


@pytest.fixture
def roots(tmp_path) -> Roots:
    shared = tmp_path / "shared"
    jobs = tmp_path / "jobs"
    shared.mkdir()
    jobs.mkdir()
    return Roots(shared=shared, jobs=jobs)


def write(path: Path, data: bytes | str = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode()
    path.write_bytes(data)
    return path


def make_job(roots: Roots, job_id: str, state: str | None, config: str = "", size: int = 10) -> Path:
    directory = roots.jobs / f"job_{job_id}"
    directory.mkdir()
    write(directory / "results_1.csv", b"S" * size)
    write(directory / "config.toml", config or "[parameters]\n")
    if state is not None:
        write(directory / "job_metadata.json", json.dumps({"job_id": job_id, "state": state, "ttp_id": "brd4"}))
    return directory


def call(seq: int, name: str, arguments: dict, result, panels=None, ts: str = "2026-01-01T10:00:00+00:00"):
    call_id = f"call-{seq}"
    return [
        {"type": "tool_call", "id": call_id, "name": name, "arguments": arguments, "seq": seq, "ts": ts},
        {
            "type": "tool_result",
            "id": call_id,
            "name": name,
            "result": result if isinstance(result, str) else json.dumps(result),
            "isError": False,
            "panels": panels or [],
            "seq": seq + 1,
            "ts": ts,
        },
    ]


def conversation(job_id: str = "abc12345", run: str = "out_1111") -> list[dict]:
    """A conversation that uploads a pocket, cofolds, profiles, makes a TTP and launches a job."""
    pose = f"/shared/predictions/{run}/boltz_results_batch_1111/predictions/mol_0/mol_0_model_0.cif"
    return [
        {"type": "user_message", "content": f"Dock into this\nStructure: /shared/uploads/{SID}/pocket.pdb", "seq": 1, "ts": "2026-01-01T09:00:00+00:00"},
        *call(2, "score_molecules", {"smiles": ["CCO"], "components": ["boltz", "posebusters"]},
              {"scores": [0.8], "smiles": ["CCO"], "metadata": {"structure_path": [pose]}},
              panels=[{"kind": "structure", "path": pose[len("/shared/"):], "url": "/api/artifacts/x",
                       "metrics": {"posebustersPassed": 18, "posebustersTotal": 18}}]),
        *call(4, "profile_interactions", {"structure_paths": [pose[len("/shared/"):]]}, {"molecules": [{"interactions": []}]},
              panels=[{"kind": "interactions", "structurePath": pose[len("/shared/"):]}]),
        *call(6, "create_ttp", {"name": "BRD4 lead"}, {"status": "created", "id": "brd4", "name": "BRD4 lead", "version": 1}),
        *call(8, "run_optimization", {"ttp_id": "brd4", "max_steps": 50}, {"job_id": job_id, "state": "running", "working_dir": f"/data/reinvent4_mcp/jobs/job_{job_id}"}),
        *call(10, "get_optimization_status", {"job_id": job_id}, {"job_id": job_id, "state": "completed"}),
        *call(12, "list_shared_assets", {}, {"structures": ["predictions/out_9999/x.pdb", "uploads/s-20991231-000000-cccccc/z.pdb"]}),
        *call(14, "list_jobs", {}, {"jobs": [{"job_id": "fff00000"}]}),
    ]


def populate(roots: Roots, job_state: str | None = "completed", job_id: str = "abc12345", run: str = "out_1111") -> None:
    write(roots.shared / "uploads" / SID / "pocket.pdb", b"P" * 100)
    write(roots.shared / "uploads" / SID / "unused.sdf", b"U" * 7)
    write(roots.shared / "predictions" / run / "boltz_results_batch_1111" / "predictions" / "mol_0" / "mol_0_model_0.cif", b"C" * 300)
    write(roots.shared / "predictions" / run / "boltz_results_batch_1111" / "predictions" / "mol_0" / "confidence.json", b"{}")
    write(roots.shared / "batches" / "batch_1111" / "input.yaml", b"y")
    write(roots.shared / "msa_cache" / "abc.a3m", b"m")
    make_job(roots, job_id, job_state, size=50)


def items_by_id(view: dict) -> dict[str, dict]:
    return {item["id"]: item for group in view["groups"] for item in group["items"]}


# -- association -----------------------------------------------------------


def test_references_come_from_messages_calls_and_results(roots):
    refs = collect_references(conversation(), roots.shared)
    assert f"uploads/{SID}/pocket.pdb" in refs.paths
    assert refs.units() >= {f"uploads/{SID}/pocket.pdb", "predictions/out_1111"}
    assert set(refs.launched) == {"abc12345"}
    assert refs.ttps["brd4"]["role"] == "created"
    assert [a["name"] for a in refs.analyses] == ["PoseBusters checks", "Interaction profile"]


def test_inventory_tools_do_not_count_as_use(roots):
    refs = collect_references(conversation(), roots.shared)
    assert "predictions/out_9999" not in refs.units()
    assert "fff00000" not in refs.jobs


def test_a_job_named_in_a_resume_path_is_referenced(roots):
    events = call(1, "run_optimization", {"ttp_id": "t", "resume_from": "/data/reinvent4_mcp/jobs/job_0badc0de/stage.chkpt"}, {"error": "busy"})
    refs = collect_references(events, roots.shared)
    assert "0badc0de" in refs.jobs
    assert refs.launched == {}


@pytest.mark.parametrize(
    "token, expected",
    [
        ("/shared/predictions/out_1/a.cif", "predictions/out_1/a.cif"),
        ("/home/me/dd/data/shared/uploads/s-1/a.pdb", "uploads/s-1/a.pdb"),
        ("predictions/out_1/a.cif", "predictions/out_1/a.cif"),
        ("/shared/../etc/passwd.pdb", None),
        ("predictions/../../etc", None),
        ("/etc/hosts", None),
        ("C:/elsewhere/x.pdb", None),
    ],
)
def test_paths_are_rerooted_on_the_shared_volume(roots, token, expected):
    assert session_data.shared_relative(token, roots.shared) == expected


def test_the_view_groups_items_with_sizes_dates_and_links(roots):
    populate(roots)
    view = inventory(SID, conversation(), roots, [], conversation_bytes=1234)
    items = items_by_id(view)

    upload = items[f"upload:uploads/{SID}/pocket.pdb"]
    assert upload["bytes"] == 100 and upload["deletable"] and upload["url"] == f"/api/artifacts/uploads/{SID}/pocket.pdb"
    assert items[f"upload:uploads/{SID}/unused.sdf"]["deletable"], "the uploads folder is listed even when unmentioned"

    run = items["prediction:predictions/out_1111"]
    assert run["bytes"] == 302 and run["deletable"]
    assert run["links"][0]["url"].endswith("mol_0_model_0.cif")

    job = items["job:abc12345"]
    assert job["bytes"] == 50 + len("[parameters]\n") + len(json.dumps({"job_id": "abc12345", "state": "completed", "ttp_id": "brd4"}))
    assert job["deletable"] and job["state"] == "completed"
    assert {link["label"] for link in job["links"]} == {"results_1.csv", "config.toml"}

    assert items["ttp:brd4"]["deletable"] is False
    assert all(item["modified"] for item in items.values())
    assert view["totals"]["conversationBytes"] == 1234
    assert view["totals"]["deletableBytes"] == 100 + 7 + 302 + job["bytes"]
    assert [g["kind"] for g in view["groups"]] == ["uploads", "predictions", "structures", "analyses", "jobs", "ttps"]


# -- deletion and its rules --------------------------------------------------


def test_purge_removes_the_selection_and_reports_space_freed(roots):
    populate(roots)
    view = inventory(SID, conversation(), roots, [])
    chosen = [item["id"] for group in view["groups"] for item in group["items"] if item["deletable"]]

    summary = purge(SID, conversation(), roots, [], chosen)

    assert sorted(entry["id"] for entry in summary["removed"]) == sorted(chosen)
    assert summary["freedBytes"] == view["totals"]["deletableBytes"]
    assert not (roots.shared / "uploads" / SID).exists()
    assert not (roots.shared / "predictions" / "out_1111").exists()
    assert not (roots.jobs / "job_abc12345").exists()
    assert (roots.shared / "batches" / "batch_1111" / "input.yaml").exists()
    assert (roots.shared / "msa_cache" / "abc.a3m").exists()
    assert {entry["id"]: entry["reason"] for entry in summary["kept"]} == {"ttp:brd4": "TTPs are shared and never deleted here"}


def test_unselected_items_are_kept_and_say_so(roots):
    populate(roots)
    summary = purge(SID, conversation(), roots, [], [f"upload:uploads/{SID}/unused.sdf"])
    kept = {entry["id"]: entry["reason"] for entry in summary["kept"]}
    assert kept["prediction:predictions/out_1111"] == "Not selected"
    assert (roots.shared / "uploads" / SID / "pocket.pdb").exists()
    assert not (roots.shared / "uploads" / SID / "unused.sdf").exists()


@pytest.mark.parametrize("state", ["running", "queued", None, "weird"])
def test_a_job_that_is_not_finished_is_never_deleted(roots, state):
    populate(roots, job_state=state)
    summary = purge(SID, conversation(), roots, [], ["job:abc12345"])
    assert (roots.jobs / "job_abc12345" / "results_1.csv").exists()
    reason = {entry["id"]: entry["reason"] for entry in summary["kept"]}["job:abc12345"]
    assert ("stop the run first" in reason) if state in ("running", "queued") else ("State unknown" in reason)
    assert summary["freedBytes"] == 0


def test_a_job_only_checked_here_was_not_launched_here_and_is_kept(roots):
    populate(roots)
    make_job(roots, "0badc0de", "completed")
    events = conversation() + call(20, "get_optimization_status", {"job_id": "0badc0de"}, {"state": "completed"})
    summary = purge(SID, events, roots, [], ["job:0badc0de"])
    assert (roots.jobs / "job_0badc0de").exists()
    assert {e["id"]: e["reason"] for e in summary["kept"]}["job:0badc0de"] == "Launched outside this conversation"


def test_anything_another_conversation_references_is_kept(roots, tmp_path):
    populate(roots)
    store = tmp_path / "sessions"
    other = store / OTHER
    other.mkdir(parents=True)
    other_events = [
        {"type": "user_message", "content": f"Compare with /shared/uploads/{SID}/pocket.pdb", "seq": 1},
        *call(2, "profile_interactions", {"structure_paths": ["predictions/out_1111/x/y.cif"]}, {"molecules": []}),
        *call(4, "get_top_candidates", {"job_id": "abc12345"}, {"candidates": []}),
    ]
    (other / "events.jsonl").write_text("\n".join(json.dumps(e) for e in other_events) + "\n", encoding="utf-8")
    others = stored_references(store, roots.shared, SID)

    view = items_by_id(inventory(SID, conversation(), roots, others))
    for item_id in (f"upload:uploads/{SID}/pocket.pdb", "prediction:predictions/out_1111", "job:abc12345"):
        assert view[item_id]["reason"] == "Also used by 1 other conversation"

    purge(SID, conversation(), roots, others, list(view))
    assert (roots.shared / "uploads" / SID / "pocket.pdb").exists()
    assert (roots.shared / "predictions" / "out_1111").exists()
    assert (roots.jobs / "job_abc12345").exists()
    assert not (roots.shared / "uploads" / SID / "unused.sdf").exists()


def test_the_conversation_itself_is_not_counted_as_another(roots, tmp_path):
    populate(roots)
    store = tmp_path / "sessions"
    (store / SID).mkdir(parents=True)
    (store / SID / "events.jsonl").write_text("\n".join(json.dumps(e) for e in conversation()), encoding="utf-8")
    assert stored_references(store, roots.shared, SID) == []


def test_listing_assets_elsewhere_does_not_pin_this_conversations_files(roots):
    populate(roots)
    from reinvent4_agent.web.session_data import References

    lister = collect_references(call(1, "list_shared_assets", {}, {"structures": [f"uploads/{SID}/pocket.pdb", "predictions/out_1111/a.cif"]}), roots.shared)
    assert lister.units() == References().units()
    assert items_by_id(inventory(SID, conversation(), roots, [lister]))["prediction:predictions/out_1111"]["deletable"]


def test_a_job_another_job_resumes_from_is_kept(roots):
    populate(roots)
    make_job(roots, "def67890", "running", config='agent_file = "/data/reinvent4_mcp/jobs/job_abc12345/stage.chkpt"\n')
    summary = purge(SID, conversation(), roots, [], ["job:abc12345"])
    assert (roots.jobs / "job_abc12345").exists()
    assert "def67890" in {e["id"]: e["reason"] for e in summary["kept"]}["job:abc12345"]


def test_ttps_and_shared_caches_are_only_listed(roots):
    populate(roots)
    events = conversation() + call(30, "score_molecules", {"msa": "/shared/msa_cache/abc.a3m", "protein": "/shared/msa_cache/t.pdb"}, {})
    write(roots.shared / "msa_cache" / "t.pdb", b"t")
    items = items_by_id(inventory(SID, events, roots, []))
    assert items["structure:msa_cache/t.pdb"]["reason"] == "Shared cache, never deleted here"
    summary = purge(SID, events, roots, [], ["ttp:brd4", "structure:msa_cache/t.pdb"])
    assert summary["removed"] == []
    assert (roots.shared / "msa_cache" / "t.pdb").exists()


@pytest.mark.parametrize(
    "forged",
    [
        "upload:uploads/../../outside.txt",
        "prediction:batches/batch_1111",
        "prediction:msa_cache",
        f"upload:uploads/{OTHER}/theirs.pdb",
        "job:../jobs",
    ],
)
def test_ids_the_browser_makes_up_remove_nothing(roots, tmp_path, forged):
    populate(roots)
    outside = write(tmp_path / "outside.txt", b"keep")
    theirs = write(roots.shared / "uploads" / OTHER / "theirs.pdb", b"keep")
    summary = purge(SID, conversation(), roots, [], [forged])
    assert {e["id"]: e["reason"] for e in summary["kept"]}[forged] == "Not associated with this conversation"
    assert outside.exists() and theirs.exists()
    assert (roots.shared / "batches" / "batch_1111").exists()


def test_an_upload_from_another_folder_is_listed_but_kept(roots):
    populate(roots)
    write(roots.shared / "uploads" / "new" / "early.pdb", b"e")
    events = conversation() + [{"type": "user_message", "content": "Structure: /shared/uploads/new/early.pdb", "seq": 40}]
    item = items_by_id(inventory(SID, events, roots, []))["upload:uploads/new/early.pdb"]
    assert item["reason"] == "Uploaded outside this conversation's folder"


@pytest.mark.parametrize(
    "relative",
    ["msa_cache/abc.a3m", "batches/batch_1111", "predictions", "uploads", f"uploads/{OTHER}/x.pdb", "predictions/out_1111/boltz_results_batch_1111"],
)
def test_the_safety_check_refuses_everything_but_its_three_targets(roots, relative):
    populate(roots)
    write(roots.shared / "uploads" / OTHER / "x.pdb")
    reason = unsafe_target(roots.shared / relative, roots.shared, session_data._allowed_shared(SID))
    assert reason is not None


def test_the_safety_check_refuses_paths_outside_the_root(roots, tmp_path):
    write(tmp_path / "elsewhere" / "x")
    assert unsafe_target(tmp_path / "elsewhere", roots.shared, session_data._allowed_shared(SID)) is not None
    assert unsafe_target(roots.shared / ".." / "elsewhere", roots.shared, session_data._allowed_shared(SID)) is not None
    assert unsafe_target(roots.jobs / "job_abc12345", roots.jobs, session_data._allowed_job) == "Already gone"


def symlink_or_skip(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError):
        pytest.skip("This system does not let the tests create symbolic links")


def test_a_prediction_run_that_is_a_link_out_of_the_volume_is_kept(roots, tmp_path):
    outside = tmp_path / "elsewhere" / "run"
    write(outside / "mol_0_model_0.cif", b"keep")
    (roots.shared / "predictions").mkdir()
    symlink_or_skip(roots.shared / "predictions" / "out_2222", outside)
    events = [{"type": "user_message", "content": "Look at /shared/predictions/out_2222/mol_0_model_0.cif", "seq": 1}]

    summary = purge(SID, events, roots, [], ["prediction:predictions/out_2222"])

    assert summary["removed"] == []
    assert "symbolic link" in summary["kept"][0]["reason"]
    assert (outside / "mol_0_model_0.cif").read_bytes() == b"keep"


def test_a_link_inside_a_deleted_job_is_not_followed(roots, tmp_path):
    populate(roots)
    outside = write(tmp_path / "precious" / "model.prior", b"keep")
    symlink_or_skip(roots.jobs / "job_abc12345" / "prior", outside.parent)

    summary = purge(SID, conversation(), roots, [], ["job:abc12345"])

    assert [e["id"] for e in summary["removed"]] == ["job:abc12345"]
    assert outside.read_bytes() == b"keep"


def test_an_uploads_folder_that_is_a_link_is_never_entered(roots, tmp_path):
    outside = tmp_path / "elsewhere"
    write(outside / "a.pdb", b"keep")
    (roots.shared / "uploads").mkdir()
    symlink_or_skip(roots.shared / "uploads" / SID, outside)
    events = [{"type": "user_message", "content": f"Structure: /shared/uploads/{SID}/a.pdb", "seq": 1}]

    summary = purge(SID, events, roots, [], [f"upload:uploads/{SID}/a.pdb"])

    assert summary["removed"] == []
    assert (outside / "a.pdb").exists()
