"""test_09_cache.py — Tests for file-cache hit behavior.

FileClient checks whether the destination file already exists before calling
download_stream.  These tests verify that:
  - a cache hit suppresses the second download call
  - SHA-256 and artifact_id are preserved across fetches
  - deleting the file forces a fresh download
  - distinct (format, entry) pairs produce distinct artifact_ids
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rcsb import FileClient
from test_rcsb.utils import sha256_of_file


# ── cache hit: no second download ─────────────────────────────────────────────

def test_file_cache_hit_no_second_download(mock_http_client, tmp_output_dir):
    """After the file is downloaded once, a second fetch makes no new calls."""
    fc = FileClient(http=mock_http_client, output_dir=tmp_output_dir)
    fc.fetch(["3PXY"], formats=["cif"])
    calls_before = len(mock_http_client.calls)

    fc.fetch(["3PXY"], formats=["cif"])

    assert len(mock_http_client.calls) == calls_before, (
        "download_stream should not be called on a cache hit"
    )


def test_file_cache_hit_sha256_preserved(mock_http_client, tmp_output_dir):
    """Two fetches of the same file return the same sha256."""
    fc = FileClient(http=mock_http_client, output_dir=tmp_output_dir)
    a1 = fc.fetch(["3PXY"], formats=["cif"])
    a2 = fc.fetch(["3PXY"], formats=["cif"])
    assert a1[0].sha256 == a2[0].sha256


# ── cache miss after deletion ─────────────────────────────────────────────────

def test_file_cache_miss_file_deleted(mock_http_client, tmp_output_dir):
    """Deleting the cached file forces a fresh download on next fetch."""
    fc = FileClient(http=mock_http_client, output_dir=tmp_output_dir)
    artifacts = fc.fetch(["3PXY"], formats=["cif"])

    # Delete the cached file to simulate a cache miss
    Path(artifacts[0].uri).unlink()

    calls_before = len(mock_http_client.calls)
    fc.fetch(["3PXY"], formats=["cif"])

    assert len(mock_http_client.calls) > calls_before, (
        "download_stream must be called again when the file is missing"
    )


# ── artifact_id stability ─────────────────────────────────────────────────────

def test_artifact_id_stable_across_calls(mock_http_client, tmp_output_dir):
    """artifact_id is identical across two fetches of the same file."""
    fc = FileClient(http=mock_http_client, output_dir=tmp_output_dir)
    a1 = fc.fetch(["3PXY"], formats=["cif"])
    a2 = fc.fetch(["3PXY"], formats=["cif"])
    assert a1[0].artifact_id == a2[0].artifact_id


# ── distinct artifact_ids for different keys ──────────────────────────────────

def test_different_format_different_artifact(mock_http_client, tmp_output_dir):
    """CIF and PDB of the same entry have different artifact_ids."""
    fc = FileClient(http=mock_http_client, output_dir=tmp_output_dir)
    cif_arts = fc.fetch(["3PXY"], formats=["cif"])
    pdb_arts = fc.fetch(["3PXY"], formats=["pdb"])
    assert cif_arts[0].artifact_id != pdb_arts[0].artifact_id


def test_different_entry_different_artifact(mock_http_client, tmp_output_dir):
    """Two different PDB entries have different artifact_ids for the same format."""
    fc = FileClient(http=mock_http_client, output_dir=tmp_output_dir)
    a1 = fc.fetch(["3PXY"], formats=["cif"])
    a2 = fc.fetch(["4HHB"], formats=["cif"])
    assert a1[0].artifact_id != a2[0].artifact_id
