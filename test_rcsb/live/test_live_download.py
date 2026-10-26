"""test_live_download.py — Live integration tests for coordinate file downloads.

All tests are skipped by default.  Run with:

    RCSB_LIVE_TESTS=1 pytest test_rcsb/live/test_live_download.py -v

These tests download real files from https://files.rcsb.org and verify
that FileClient correctly streams, checksums, and stores them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rcsb import (
    RCSB_SEARCH_URL,
    FileClient,
    QueryBuilder,
    RCSBHTTPClient,
)


# ── CIF download ──────────────────────────────────────────────────────────────

def test_live_download_3pxy_cif(tmp_path):
    """Download 3PXY.cif: file exists on disk, size > 50 KB, sha256 non-empty."""
    client = RCSBHTTPClient()
    fc = FileClient(http=client, output_dir=tmp_path)
    artifacts = fc.fetch(["3PXY"], formats=["cif"])

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.error is None, f"Download failed: {artifact.error}"

    dest = Path(artifact.uri)
    assert dest.exists(), f"Expected file at {dest}"
    assert artifact.size_bytes > 50_000, (
        f"Expected size > 50 KB, got {artifact.size_bytes} bytes"
    )
    assert artifact.sha256, "sha256 must be non-empty"
    assert len(artifact.sha256) == 64, "sha256 must be a 64-char hex string"


# ── biological assembly download ──────────────────────────────────────────────

def test_live_download_3pxy_assembly(tmp_path):
    """Download 3PXY assembly 1: assembly artifact is present without error."""
    client = RCSBHTTPClient()
    fc = FileClient(http=client, output_dir=tmp_path)
    artifacts = fc.fetch(["3PXY"], formats=["cif"], assembly_ids=["1"])

    assembly_artifacts = [a for a in artifacts if a.assembly_id == "1"]
    assert len(assembly_artifacts) == 1, (
        f"Expected exactly one assembly artifact, got {len(assembly_artifacts)}"
    )
    asm = assembly_artifacts[0]
    assert asm.error is None, f"Assembly download failed: {asm.error}"
    assert Path(asm.uri).exists(), f"Assembly file not found at {asm.uri}"


# ── count / search consistency ────────────────────────────────────────────────

def test_live_count_then_search_consistent():
    """count('CDK2') and search('CDK2') total_count agree within 5 % tolerance."""
    client = RCSBHTTPClient()
    query_dict = QueryBuilder.from_dict({
        "type": "terminal",
        "service": "text",
        "parameters": {"value": "CDK2"},
    })
    opts_common = {
        "results_content_type": ["experimental"],
    }

    count_result = client.post_json(RCSB_SEARCH_URL, {
        "query": query_dict,
        "return_type": "entry",
        "request_options": {**opts_common, "return_counts": True},
    })
    count = count_result.get("total_count", 0)

    search_result = client.post_json(RCSB_SEARCH_URL, {
        "query": query_dict,
        "return_type": "entry",
        "request_options": {**opts_common, "paginate": {"start": 0, "rows": 1}},
    })
    total_hits = search_result.get("total_count", 0)

    # Allow 5 % tolerance to account for index updates between the two calls
    tolerance = 0.05
    larger = max(count, total_hits)
    assert larger == 0 or abs(count - total_hits) <= larger * tolerance, (
        f"count={count} and total_count={total_hits} differ by more than 5 %"
    )
