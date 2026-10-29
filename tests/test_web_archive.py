# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Many shared files as one ZIP, under the rules of /api/artifacts."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pytest

from reinvent4_agent.web import archive


@pytest.fixture(autouse=True)
def no_cached_listing(monkeypatch):
    monkeypatch.setattr(archive, "_listing", None)


@pytest.fixture
def shared(monkeypatch, tmp_path):
    root = tmp_path / "shared"
    (root / "predictions" / "out_aa11").mkdir(parents=True)
    (root / "predictions" / "out_aa11" / "mol_0_model_0.pdb").write_bytes(b"ATOM\n" * 100)
    (root / "ligands.sdf").write_bytes(b"$$$$\n")
    (root / ".ligands.sdf.part").write_text("partial")
    (tmp_path / "secret.txt").write_text("no")
    monkeypatch.setattr(archive.shared_files, "SHARED_ROOT", root)
    return root


def unzip(chunks) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(b"".join(chunks)))


def test_two_files_come_back_as_one_zip_with_their_folders(shared, tmp_path):
    files = archive.select_files(["predictions/out_aa11/mol_0_model_0.pdb", "ligands.sdf", "ligands.sdf"], tmp_path)
    zipped = unzip(archive.zip_chunks(files))
    assert sorted(zipped.namelist()) == ["ligands.sdf", "predictions/out_aa11/mol_0_model_0.pdb"]
    assert zipped.read("predictions/out_aa11/mol_0_model_0.pdb") == b"ATOM\n" * 100
    assert zipped.testzip() is None


def test_a_job_file_is_read_from_the_jobs_directory(shared, tmp_path):
    jobs = tmp_path / "jobs"
    (jobs / "job_abc123").mkdir(parents=True)
    (jobs / "job_abc123" / "results_1.csv").write_bytes(b"SMILES\nCCO\n")
    files = archive.select_files(["job_abc123/results_1.csv"], jobs)
    assert unzip(archive.zip_chunks(files)).read("jobs/job_abc123/results_1.csv") == b"SMILES\nCCO\n"
    with pytest.raises(archive.ArchiveError):
        archive.select_files(["job_abc123/secret.txt"], jobs)


@pytest.mark.parametrize("path", ["../secret.txt", "/etc/passwd", "predictions/../../secret.txt", "missing.pdb"])
def test_a_path_outside_the_volume_refuses_the_whole_archive(shared, tmp_path, path):
    with pytest.raises(archive.ArchiveError) as refused:
        archive.select_files(["ligands.sdf", path], tmp_path)
    assert refused.value.status == 404


def test_a_symlink_out_of_the_volume_is_refused(shared, tmp_path):
    try:
        os.symlink(tmp_path / "secret.txt", shared / "escape.txt")
    except OSError:
        pytest.skip("symlinks are not available here")
    with pytest.raises(archive.ArchiveError):
        archive.select_files(["escape.txt"], tmp_path)


def test_too_many_files_or_bytes_are_refused(shared, tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "MAX_FILES", 1)
    with pytest.raises(archive.ArchiveError) as refused:
        archive.select_files(["ligands.sdf", "predictions/out_aa11/mol_0_model_0.pdb"], tmp_path)
    assert refused.value.status == 413
    monkeypatch.setattr(archive, "MAX_FILES", 10)
    monkeypatch.setattr(archive, "MAX_BYTES", 100)
    with pytest.raises(archive.ArchiveError) as refused:
        archive.select_files(["ligands.sdf", "predictions/out_aa11/mol_0_model_0.pdb"], tmp_path)
    assert refused.value.status == 413


def test_a_malformed_request_is_refused(tmp_path):
    for paths in (None, [], "ligands.sdf", [1]):
        with pytest.raises(archive.ArchiveError) as refused:
            archive.select_files(paths, tmp_path)
        assert refused.value.status == 400


def test_the_download_name_is_safe():
    assert archive.archive_name("all predictions") == "all_predictions.zip"
    assert archive.archive_name("../x.zip") == "x.zip"
    assert archive.archive_name(None) == "assets.zip"


def test_the_listing_leaves_out_hidden_files(shared):
    listed = archive.list_files(shared)
    assert [f["path"] for f in listed["files"]] == ["ligands.sdf", "predictions/out_aa11/mol_0_model_0.pdb"]
    assert listed["files"][0]["size"] == 5
    assert archive.list_files(shared, limit=1)["truncated"] is True


def test_the_listing_is_reused_for_a_few_seconds(shared, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(archive.time, "monotonic", lambda: now[0])
    first = archive.cached_listing(shared)
    (shared / "new.sdf").write_bytes(b"$$$$\n")
    assert archive.cached_listing(shared) is first
    now[0] += archive.LISTING_TTL_S + 1
    assert "new.sdf" in [f["path"] for f in archive.cached_listing(shared)["files"]]


# -- endpoint --------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path, shared):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from reinvent4_agent.web import server

    async def start(self) -> None:
        self.ready = True

    monkeypatch.setenv("REINVENT4_WEB_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("REINVENT4_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.delenv(server.chat_models.CONTROL_DIR_ENV, raising=False)
    monkeypatch.setattr(server.AgentGateway, "start", start)
    with TestClient(server.create_app()) as test_client:
        yield test_client


def test_the_gateway_streams_the_zip(client):
    response = client.post("/api/archive", json={"paths": ["ligands.sdf", "predictions/out_aa11/mol_0_model_0.pdb"], "name": "all files"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert 'filename="all_files.zip"' in response.headers["content-disposition"]
    assert len(zipfile.ZipFile(io.BytesIO(response.content)).namelist()) == 2


def test_the_gateway_refuses_bad_archives(client, monkeypatch):
    assert client.post("/api/archive", json={"paths": ["../secret.txt"]}).status_code == 404
    assert client.post("/api/archive", json={"paths": []}).status_code == 400
    monkeypatch.setattr(archive, "MAX_FILES", 1)
    refused = client.post("/api/archive", json={"paths": ["ligands.sdf", "predictions/out_aa11/mol_0_model_0.pdb"]})
    assert refused.status_code == 413
    assert "at most 1" in refused.json()["detail"]


def test_the_gateway_lists_the_volume(client):
    assert [f["path"] for f in client.get("/api/files").json()["files"]] == ["ligands.sdf", "predictions/out_aa11/mol_0_model_0.pdb"]
