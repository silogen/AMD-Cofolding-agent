"""test_08_file_download.py — Tests for FileClient file downloads.

All tests are fully offline: the _MockHTTPClient in conftest writes synthetic
CIF/PDB content and returns a real SHA-256 without hitting any network.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pytest

from rcsb import (
    FileArtifact,
    FileClient,
    RCSBHTTPClient,
    RCSBUnavailableError,
    SCHEMA_VERSION,
)
from test_rcsb.utils import assert_artifact_valid, sha256_of_file


@pytest.fixture()
def file_client(mock_http_client, tmp_output_dir):
    """FileClient wired to the offline mock HTTP client."""
    return FileClient(http=mock_http_client, output_dir=tmp_output_dir)


# ── format tests ─────────────────────────────────────────────────────────────

def test_download_cif_single(file_client, tmp_output_dir):
    artifacts = file_client.fetch(["3PXY"], formats=["cif"])
    assert len(artifacts) == 1
    assert artifacts[0].format == "cif"
    assert artifacts[0].error is None


def test_download_pdb_single(file_client):
    artifacts = file_client.fetch(["3PXY"], formats=["pdb"])
    assert len(artifacts) == 1
    assert artifacts[0].format == "pdb"


def test_download_both_formats(file_client):
    artifacts = file_client.fetch(["3PXY"], formats=["cif", "pdb"])
    assert len(artifacts) == 2
    formats = {a.format for a in artifacts}
    assert "cif" in formats
    assert "pdb" in formats


# ── assembly tests ────────────────────────────────────────────────────────────

def test_download_assembly(file_client):
    """With assembly_ids=['1'] we get asymmetric unit + assembly = 2 artifacts."""
    artifacts = file_client.fetch(["3PXY"], formats=["cif"], assembly_ids=["1"])
    assert len(artifacts) == 2
    assembly_artifacts = [a for a in artifacts if a.assembly_id == "1"]
    assert len(assembly_artifacts) == 1


# ── artifact field tests ──────────────────────────────────────────────────────

def test_artifact_fields_present(file_client):
    """All required artifact schema fields are present and valid."""
    artifacts = file_client.fetch(["3PXY"], formats=["cif"])
    assert_artifact_valid(asdict(artifacts[0]))


def test_atomic_write(file_client, tmp_output_dir):
    """Files are written atomically: no .tmp files remain after a successful fetch."""
    file_client.fetch(["3PXY"], formats=["cif"])
    tmp_files = list(tmp_output_dir.rglob("*.tmp"))
    assert tmp_files == [], f"Unexpected .tmp files: {tmp_files}"
    cif_files = list(tmp_output_dir.rglob("*.cif"))
    assert len(cif_files) >= 1


def test_sha256_stored(file_client, tmp_output_dir):
    """Artifact sha256 matches the actual SHA-256 of the written file."""
    artifacts = file_client.fetch(["3PXY"], formats=["cif"])
    artifact = artifacts[0]
    actual = sha256_of_file(Path(artifact.uri))
    assert artifact.sha256 == actual


def test_size_bytes_recorded(file_client):
    """size_bytes is positive after a successful download."""
    artifacts = file_client.fetch(["3PXY"], formats=["cif"])
    assert artifacts[0].size_bytes > 0


def test_source_url_rcsb(file_client):
    """source_url starts with the canonical RCSB files base URL."""
    artifacts = file_client.fetch(["3PXY"], formats=["cif"])
    assert artifacts[0].source_url.startswith("https://files.rcsb.org")


def test_retrieved_at_iso8601(file_client):
    """retrieved_at is a parseable ISO-8601 datetime string."""
    artifacts = file_client.fetch(["3PXY"], formats=["cif"])
    retrieved = artifacts[0].retrieved_at
    dt = datetime.fromisoformat(retrieved)
    assert dt is not None


def test_schema_version_present(file_client):
    """schema_version on every artifact matches the module constant."""
    artifacts = file_client.fetch(["3PXY"], formats=["cif"])
    assert artifacts[0].schema_version == SCHEMA_VERSION


# ── multi-entry tests ─────────────────────────────────────────────────────────

def test_multiple_entries(file_client):
    """Fetching two entries (default CIF format) yields exactly two artifacts."""
    artifacts = file_client.fetch(["3PXY", "4HHB"])
    assert len(artifacts) == 2


def test_one_failure_preserves_others(mock_http_client, tmp_output_dir):
    """A download failure for one entry does not abort others."""
    original = mock_http_client.download_stream

    def patched(url: str, dest: Path) -> tuple[str, int]:
        if "4HHB" in url:
            raise RCSBUnavailableError("mocked 4HHB failure")
        return original(url, dest)

    mock_http_client.download_stream = patched
    fc = FileClient(http=mock_http_client, output_dir=tmp_output_dir)
    artifacts = fc.fetch(["3PXY", "4HHB"], formats=["cif"])

    by_entry = {a.entry_id: a for a in artifacts}
    assert by_entry["3PXY"].error is None
    assert by_entry["4HHB"].error is not None


# ── artifact identity / cache tests ──────────────────────────────────────────

def test_artifact_id_stable(file_client):
    """Two fetches of the same file produce the same artifact_id (cache hit)."""
    a1 = file_client.fetch(["3PXY"], formats=["cif"])
    a2 = file_client.fetch(["3PXY"], formats=["cif"])
    assert a1[0].artifact_id == a2[0].artifact_id


def test_output_dir_created(mock_http_client, tmp_path):
    """FileClient creates the output directory if it doesn't exist."""
    new_dir = tmp_path / "nonexistent" / "subdir"
    assert not new_dir.exists()
    fc = FileClient(http=mock_http_client, output_dir=new_dir)
    fc.fetch(["3PXY"], formats=["cif"])
    assert new_dir.exists()


def test_no_overwrite_existing(mock_http_client, tmp_output_dir):
    """Second fetch with file already present is a cache hit: no re-download."""
    fc = FileClient(http=mock_http_client, output_dir=tmp_output_dir)

    a1 = fc.fetch(["3PXY"], formats=["cif"])
    sha256_first = a1[0].sha256
    calls_after_first = len(mock_http_client.calls)

    a2 = fc.fetch(["3PXY"], formats=["cif"])
    sha256_second = a2[0].sha256

    # SHA-256 is unchanged
    assert sha256_first == sha256_second
    # download_stream was NOT called a second time
    assert len(mock_http_client.calls) == calls_after_first


# ── B4: PDB format size guard ─────────────────────────────────────────────────

def test_pdb_size_guard_blocks_large_entry(mock_http_client, tmp_output_dir):
    """PDB format is blocked with an error when atom_count > 99,999."""
    from rcsb import FileClient, PDB_ATOM_LIMIT
    fc = FileClient(http=mock_http_client, output_dir=tmp_output_dir)
    # Simulate a 125,000 atom entry (like 7NSS ribosome)
    big_metadata_cache = {
        "7NSS": {"composition": {"atom_count": 125_000}}
    }
    artifacts = fc.fetch(["7NSS"], formats=["pdb"], metadata_cache=big_metadata_cache)
    assert len(artifacts) == 1
    art = artifacts[0]
    assert art.error is not None, "Large PDB download must produce an error"
    assert "pdb_format_unsupported" in art.error
    assert "use cif instead" in art.error


def test_pdb_size_guard_allows_small_entry(mock_http_client, tmp_output_dir):
    """PDB format is allowed when atom_count <= 99,999."""
    from rcsb import FileClient, PDB_ATOM_LIMIT
    fc = FileClient(http=mock_http_client, output_dir=tmp_output_dir)
    small_metadata_cache = {
        "3PXY": {"composition": {"atom_count": 2412}}
    }
    artifacts = fc.fetch(["3PXY"], formats=["pdb"], metadata_cache=small_metadata_cache)
    assert len(artifacts) == 1
    assert artifacts[0].error is None, "Small PDB download must succeed"
