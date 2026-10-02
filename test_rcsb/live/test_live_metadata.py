"""test_live_metadata.py — Live integration tests for the RCSB Data API.

All tests are skipped by default.  Run with:

    RCSB_LIVE_TESTS=1 pytest test_rcsb/live/test_live_metadata.py -v

These tests call the real RCSB Data API (rcsbapi.data.DataQuery) and verify
that normalize_entry() correctly maps the live response into the expected schema.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rcsb import MetadataClient, normalize_entry


@pytest.fixture(scope="module")
def mc():
    return MetadataClient()


# ── 3PXY: CDK2 + inhibitor, X-ray at 1.8 Å ──────────────────────────────────

def test_live_metadata_3pxy(mc):
    """Live 3PXY entry: resolution == 1.8 Å and method == 'X-RAY DIFFRACTION'."""
    raw_map = mc.get_entries(["3PXY"])
    assert raw_map.get("3PXY") is not None, "3PXY not returned by Data API"

    entry = normalize_entry(raw_map["3PXY"], "3PXY")
    exp = entry["experiment"]
    assert exp["resolution_A"] == pytest.approx(1.8, abs=0.05), (
        f"Expected resolution ≈ 1.8 Å, got {exp['resolution_A']}"
    )
    assert exp["method"] == "X-RAY DIFFRACTION", (
        f"Expected X-RAY DIFFRACTION, got {exp['method']!r}"
    )


# ── 4HHB: deoxyhemoglobin, 4-chain tetramer ──────────────────────────────────

def test_live_metadata_4hhb_chains(mc):
    """Live 4HHB entry: has multiple polymer entities (alpha + beta chains)."""
    raw_map = mc.get_entries(["4HHB"])
    assert raw_map.get("4HHB") is not None, "4HHB not returned by Data API"

    entry = normalize_entry(raw_map["4HHB"], "4HHB")
    # 4HHB has 2 unique entity types (alpha and beta hemoglobin chains)
    assert len(entry["polymer_entities"]) >= 2, (
        f"Expected ≥2 polymer entities for 4HHB, got {len(entry['polymer_entities'])}"
    )


# ── AF_AFP68871F1: AlphaFold hemoglobin beta ──────────────────────────────────

def test_live_metadata_alphafold(mc):
    """Live AlphaFold entry: entry_type=='computational' and pLDDT is present."""
    raw_map = mc.get_entries(["AF_AFP68871F1"])
    assert raw_map.get("AF_AFP68871F1") is not None, (
        "AF_AFP68871F1 not returned by Data API"
    )

    entry = normalize_entry(raw_map["AF_AFP68871F1"], "AF_AFP68871F1")
    assert entry["identity"]["entry_type"] == "computational", (
        f"Expected entry_type='computational', got {entry['identity']['entry_type']!r}"
    )

    metrics = entry["experiment"].get("confidence_metrics", [])
    assert len(metrics) >= 1, "Expected at least one confidence metric (pLDDT)"
    metric_types = [m.get("type") for m in metrics]
    assert "pLDDT" in metric_types, (
        f"Expected 'pLDDT' in confidence_metrics, got {metric_types}"
    )
