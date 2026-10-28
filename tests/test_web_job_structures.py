# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The cofolded complex of each molecule of a REINVENT run, from its results CSV."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from reinvent4_agent.web import job_structures, server  # noqa: E402

HEADER = (
    "Agent,Prior,Target,Score,SMILES,SMILES_state,Boltz affinity,Boltz affinity (raw),"
    "structure_path (Boltz affinity),ligand_iptm (Boltz affinity),confidence_score (Boltz affinity),"
    "affinity_pred_value (Boltz affinity),skip_reason (Boltz affinity),"
    "n_checks_passed (Pose quality),n_checks_total (Pose quality),step"
)
PDB = "/shared/predictions/out_{b}/boltz_results_batch_{b}/predictions/mol_{i}/mol_{i}_model_0.pdb"


def row(score, smiles, path, iptm="0.81", affinity="-1.2", step=1):
    return f"1,2,3,{score},{smiles},1,0.5,{affinity},{path},{iptm},0.9,{affinity},,20,22,{step}"


def write_job(root: Path, *lines: str, name="results_1.csv", job="abc123") -> Path:
    directory = root / f"job_{job}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text("\n".join([HEADER, *lines]) + "\n")
    return directory


@pytest.fixture(autouse=True)
def fresh_cache():
    job_structures._cache.clear()


def test_each_cofolded_molecule_comes_with_its_complex_and_boltz_values(tmp_path):
    job = write_job(
        tmp_path,
        row(0.4, "CCO", PDB.format(b="aa11", i=0)),
        row(0.7, "c1ccccc1", PDB.format(b="aa11", i=1), iptm="0.65", affinity="-2.5", step=2),
    )
    entries = job_structures.job_structures(job, Path("/shared"))
    assert [e["smiles"] for e in entries] == ["c1ccccc1", "CCO"]
    assert entries[0] == {
        "smiles": "c1ccccc1",
        "step": 2,
        "score": 0.7,
        "path": "predictions/out_aa11/boltz_results_batch_aa11/predictions/mol_1/mol_1_model_0.pdb",
        "metrics": {"ligandIptm": 0.65, "confidence": 0.9, "affinity": -2.5, "posebustersPassed": 20, "posebustersTotal": 22},
    }


def test_a_molecule_sampled_twice_keeps_its_best_scoring_complex(tmp_path):
    job = write_job(
        tmp_path,
        row(0.3, "CCO", PDB.format(b="aa11", i=0)),
        row(0.9, "CCO", PDB.format(b="bb22", i=3), step=5),
        row(0.5, "CCO", PDB.format(b="cc33", i=1), step=7),
    )
    [entry] = job_structures.job_structures(job, Path("/shared"))
    assert entry["step"] == 5 and "out_bb22" in entry["path"]


def test_rows_without_a_structure_or_outside_the_volume_are_left_out(tmp_path):
    job = write_job(
        tmp_path,
        row(0.0, "CCO", ""),
        row(0.4, "CCC", "/etc/passwd.pdb"),
        row(0.4, "CCN", "/shared/predictions/../../secret.pdb"),
        row(0.6, "CCCl", PDB.format(b="aa11", i=0)),
        "1,2,3,0.9,CCBr,1",
    )
    assert [e["smiles"] for e in job_structures.job_structures(job, Path("/shared"))] == ["CCCl"]


def test_every_stage_is_read_and_a_run_without_boltz_has_none(tmp_path):
    job = write_job(tmp_path, row(0.2, "CCO", PDB.format(b="aa11", i=0)))
    write_job(tmp_path, row(0.8, "CCC", PDB.format(b="bb22", i=0)), name="results_2.csv")
    assert [e["smiles"] for e in job_structures.job_structures(job, Path("/shared"))] == ["CCC", "CCO"]

    plain = tmp_path / "job_def456"
    plain.mkdir()
    (plain / "results_1.csv").write_text("Score,SMILES,QED,step\n0.5,CCO,0.5,1\n")
    assert job_structures.job_structures(plain, Path("/shared")) == []
    assert job_structures.job_structures(tmp_path / "job_missing", Path("/shared")) is None


def test_a_csv_that_grows_is_read_again(tmp_path):
    job = write_job(tmp_path, row(0.2, "CCO", PDB.format(b="aa11", i=0)))
    assert len(job_structures.job_structures(job, Path("/shared"))) == 1
    with (job / "results_1.csv").open("a") as handle:
        handle.write(row(0.5, "CCC", PDB.format(b="bb22", i=0), step=2) + "\n")
    assert len(job_structures.job_structures(job, Path("/shared"))) == 2


# -- endpoint --------------------------------------------------------------


@pytest.fixture
def gateway(monkeypatch, tmp_path):
    async def start(self) -> None:
        self.ready = True

    monkeypatch.setenv("REINVENT4_WEB_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("REINVENT4_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.delenv(server.chat_models.CONTROL_DIR_ENV, raising=False)
    monkeypatch.setattr(server.AgentGateway, "start", start)
    monkeypatch.setattr(server.shared_files, "SHARED_ROOT", Path("/shared"))
    return tmp_path / "jobs"


def test_the_gateway_lists_a_runs_structures(gateway):
    write_job(gateway, row(0.6, "CCO", PDB.format(b="aa11", i=0)))
    with TestClient(server.create_app()) as client:
        body = client.get("/api/jobs/abc123/structures").json()
        assert body[0]["path"].startswith("predictions/out_aa11/")
        assert client.get("/api/jobs/fff999/structures").status_code == 404
        assert client.get("/api/jobs/ABC/structures").status_code == 404
        assert client.get("/api/jobs/..%2F..%2Fetc/structures").status_code == 404
