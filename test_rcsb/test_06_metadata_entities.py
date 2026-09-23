"""test_06_metadata_entities.py — Tests for MetadataClient methods.

Uses the mock_data_api fixture to avoid real rcsbapi.data calls.
The mock patches MetadataClient.get_entries / get_chem_comps /
get_pubmed / get_summary_fields directly so no network is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rcsb import MetadataClient
from test_rcsb.utils import load_fixture


# ---------------------------------------------------------------------------
# Helper – build a minimal entries fixture inline
# ---------------------------------------------------------------------------

def _entries_fixture(*entry_dicts) -> dict:
    """Wrap one or more raw entry dicts in the data-API envelope."""
    return {"data": {"entries": list(entry_dicts)}}


# ---------------------------------------------------------------------------
# get_entries
# ---------------------------------------------------------------------------

def test_get_entries_returns_dict(mock_data_api):
    """A registered 3PXY entry must be returned under key '3PXY'."""
    fixture = load_fixture("data", "3pxy_entry.json")
    mock_data_api.register("entries", "3PXY", fixture)

    mc = MetadataClient()
    result = mc.get_entries(["3PXY"])

    assert "3PXY" in result, "Expected '3PXY' key in result dict"
    assert result["3PXY"] is not None, "Entry for '3PXY' must not be None"


def test_get_entries_missing_id(mock_data_api):
    """An unregistered entry ID must map to None, not raise."""
    # Nothing registered for XXXX
    mc = MetadataClient()
    result = mc.get_entries(["XXXX"])

    assert "XXXX" in result, "Result must contain the requested ID even when missing"
    assert result["XXXX"] is None, "Missing entry must map to None"


def test_partial_batch_preserves_ok(mock_data_api):
    """In a batch with one valid and one missing ID, each is handled correctly."""
    fixture = load_fixture("data", "3pxy_entry.json")
    # The patched get_entries uses sorted(entry_ids) as the key, so register
    # with the combined, sorted key that would be looked up for ["3PXY", "XXXX"].
    mock_data_api.register("entries", "3PXY,XXXX", fixture)

    mc = MetadataClient()
    result = mc.get_entries(["3PXY", "XXXX"])

    assert result["3PXY"] is not None, "Known entry must not be None"
    assert result["XXXX"] is None, "Unknown entry must be None"


# ---------------------------------------------------------------------------
# get_chem_comps
# ---------------------------------------------------------------------------

def test_get_chem_comps_nag(mock_data_api):
    """NAG chemical component must be returned and be non-None."""
    fixture = load_fixture("data", "nag_chemcomp.json")
    mock_data_api.register("chem_comps", "NAG", fixture)

    mc = MetadataClient()
    result = mc.get_chem_comps(["NAG"])

    assert "NAG" in result, "Expected 'NAG' key in chem_comps result"
    assert result["NAG"] is not None, "NAG chem_comp must not be None"


def test_get_chem_comps_nag_formal_charge(mock_data_api):
    """NAG formal charge must be 0."""
    fixture = load_fixture("data", "nag_chemcomp.json")
    mock_data_api.register("chem_comps", "NAG", fixture)

    mc = MetadataClient()
    result = mc.get_chem_comps(["NAG"])
    nag = result["NAG"]
    assert nag is not None
    assert nag["chem_comp"]["pdbx_formal_charge"] == 0, "NAG formal charge must be 0"


def test_get_chem_comps_nag_has_descriptors(mock_data_api):
    """NAG must have pdbx_chem_comp_descriptor with InChI."""
    fixture = load_fixture("data", "nag_chemcomp.json")
    mock_data_api.register("chem_comps", "NAG", fixture)

    mc = MetadataClient()
    result = mc.get_chem_comps(["NAG"])
    nag = result["NAG"]
    assert nag is not None
    descriptors = nag.get("pdbx_chem_comp_descriptor", [])
    assert len(descriptors) >= 1, "NAG must have at least one descriptor"
    types = {d["type"] for d in descriptors}
    assert "InChI" in types, "NAG must have InChI descriptor"


def test_get_chem_comps_atp(mock_data_api):
    """ATP chemical component must be returned and be non-None."""
    fixture = load_fixture("data", "atp_chemcomp.json")
    mock_data_api.register("chem_comps", "ATP", fixture)

    mc = MetadataClient()
    result = mc.get_chem_comps(["ATP"])

    assert "ATP" in result, "Expected 'ATP' key in chem_comps result"
    assert result["ATP"] is not None, "ATP chem_comp must not be None"


def test_get_chem_comps_atp_formal_charge(mock_data_api):
    """ATP formal charge must be -4."""
    fixture = load_fixture("data", "atp_chemcomp.json")
    mock_data_api.register("chem_comps", "ATP", fixture)

    mc = MetadataClient()
    result = mc.get_chem_comps(["ATP"])
    atp = result["ATP"]
    assert atp is not None
    assert atp["chem_comp"]["pdbx_formal_charge"] == -4, "ATP formal charge must be -4"


# ---------------------------------------------------------------------------
# get_pubmed
# ---------------------------------------------------------------------------

def test_get_pubmed_abstract(mock_data_api):
    """PubMed record for a known PMID must be returned and keyed by integer."""
    pmid = 21291286
    fixture = {
        "data": {
            "pubmed": [
                {
                    "rcsb_id": str(pmid),
                    "rcsb_pubmed_abstract_text": (
                        "Crystal structure of CDK2 in complex with a potent "
                        "pyrazolopyrimidine inhibitor JWS648."
                    ),
                }
            ]
        }
    }
    mock_data_api.register("pubmed", str(pmid), fixture)

    mc = MetadataClient()
    result = mc.get_pubmed([pmid])

    assert pmid in result, f"Expected integer key {pmid} in pubmed result"
    assert result[pmid] is not None, f"PubMed record for {pmid} must not be None"


# ---------------------------------------------------------------------------
# get_summary_fields
# ---------------------------------------------------------------------------

def test_get_summary_fields(mock_data_api):
    """get_summary_fields must return a dict keyed by entry ID."""
    fixture = load_fixture("data", "3pxy_entry.json")
    mock_data_api.register("entries", "3PXY", fixture)

    mc = MetadataClient()
    result = mc.get_summary_fields(["3PXY"], ["struct.title"])

    assert "3PXY" in result, "Expected '3PXY' key in summary_fields result"
    # The returned item is whatever the fixture contains for that entry
    assert result["3PXY"] is not None
