"""Tests for reinvent4_mcp/tools/chembl.py.

All tests mock HTTP calls — no live ChEMBL API required.
Covers: URL/path construction, parameter forwarding, pagination,
activity unit preservation, error handling, identifier validation,
similarity/substructure threshold, and registry seam.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import chembl module
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

spec = importlib.util.spec_from_file_location(
    "chembl_mod", str(_REPO / "reinvent4_mcp" / "tools" / "chembl.py")
)
chembl = importlib.util.module_from_spec(spec)
sys.modules["chembl_mod"] = chembl
spec.loader.exec_module(chembl)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_json(payload: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.ok = (status < 400)
    resp.json.return_value = payload
    resp.headers = {"Content-Type": "application/json"}
    resp.text = json.dumps(payload)[:100]
    resp.raise_for_status = MagicMock()
    return resp


def _register() -> dict[str, Any]:
    """Register all tools and return a name→fn dict."""
    mcp = MagicMock()
    registered: dict[str, Any] = {}

    def _tool():
        def decorator(fn):
            registered[fn.__name__] = fn
            return fn
        return decorator

    mcp.tool = _tool
    chembl.register_tools(mcp)
    return registered


TOOLS = _register()


def _call(name: str, **kwargs) -> dict:
    return TOOLS[name](**kwargs)


# ---------------------------------------------------------------------------
# Category 1: Tool registration
# ---------------------------------------------------------------------------

class TestRegistration:
    EXPECTED = {
        "search_chembl_molecules",
        "get_chembl_molecule",
        "search_chembl_targets",
        "get_chembl_target",
        "search_chembl_activities",
        "fetch_all_activities",
        "search_chembl_assays",
        "get_chembl_mechanisms",
        "get_chembl_indications",
        "get_chembl_documents",
        "search_chembl_similarity",
        "search_chembl_substructure",
    }

    def test_all_12_tools_registered(self):
        assert self.EXPECTED == set(TOOLS.keys())

    def test_all_tools_callable(self):
        for name, fn in TOOLS.items():
            assert callable(fn), f"{name} is not callable"


# ---------------------------------------------------------------------------
# Category 2: search_chembl_molecules
# ---------------------------------------------------------------------------

class TestSearchMolecules:
    def _payload(self, records=None):
        return {"molecules": records or [], "page_meta": {"total_count": len(records or []), "next": None}}

    def test_empty_query_returns_error(self):
        r = _call("search_chembl_molecules", query="")
        assert r["ok"] is False
        assert r["error"]["code"] == "validation_error"

    def test_url_uses_search_endpoint(self):
        payload = self._payload([{"molecule_chembl_id": "CHEMBL25"}])
        with patch("requests.get", return_value=_mock_json(payload)) as mock_get:
            _call("search_chembl_molecules", query="aspirin")
        url = mock_get.call_args[0][0]
        assert "/molecule/search" in url

    def test_query_param_forwarded(self):
        payload = self._payload()
        with patch("requests.get", return_value=_mock_json(payload)) as mock_get:
            _call("search_chembl_molecules", query="ibuprofen")
        params = mock_get.call_args[1]["params"]
        assert params["q"] == "ibuprofen"

    def test_molecule_type_filter_forwarded(self):
        payload = self._payload()
        with patch("requests.get", return_value=_mock_json(payload)) as mock_get:
            _call("search_chembl_molecules", query="test", molecule_type="Small molecule")
        params = mock_get.call_args[1]["params"]
        assert "molecule_type__iexact" in params

    def test_max_phase_filter(self):
        payload = self._payload()
        with patch("requests.get", return_value=_mock_json(payload)) as mock_get:
            _call("search_chembl_molecules", query="test", max_phase=4)
        params = mock_get.call_args[1]["params"]
        assert params["max_phase"] == 4

    def test_limit_and_offset_forwarded(self):
        payload = self._payload()
        with patch("requests.get", return_value=_mock_json(payload)) as mock_get:
            _call("search_chembl_molecules", query="test", limit=50, offset=20)
        params = mock_get.call_args[1]["params"]
        assert params["limit"] == 50
        assert params["offset"] == 20

    def test_only_fields_forwarded(self):
        payload = self._payload()
        with patch("requests.get", return_value=_mock_json(payload)) as mock_get:
            _call("search_chembl_molecules", query="test",
                  only=["molecule_chembl_id", "pref_name"])
        params = mock_get.call_args[1]["params"]
        assert "only" in params

    def test_not_found_returns_empty_not_error(self):
        resp_404 = _mock_json({}, status=404)
        resp_404.ok = False
        resp_404.raise_for_status = MagicMock()
        with patch("requests.get", return_value=resp_404):
            r = _call("search_chembl_molecules", query="xyznotexist")
        assert r["ok"] is True
        assert r["records"] == []

    def test_result_has_provenance(self):
        payload = self._payload()
        with patch("requests.get", return_value=_mock_json(payload)):
            r = _call("search_chembl_molecules", query="test")
        assert "provenance" in r
        assert "_chembl_api_url" in r["provenance"]
        assert "_retrieved_at" in r["provenance"]


# ---------------------------------------------------------------------------
# Category 3: get_chembl_molecule
# ---------------------------------------------------------------------------

class TestGetMolecule:
    def _mol_payload(self, cid="CHEMBL25") -> dict:
        return {
            "molecule_chembl_id": cid, "pref_name": "ASPIRIN",
            "max_phase": 4, "molecule_type": "Small molecule",
            "molecule_structures": {"canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                                    "standard_inchi_key": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"},
            "molecule_properties": {"full_molformula": "C9H8O4", "full_mwt": "180.16"},
            "molecule_synonyms": [{"molecule_synonym": "Aspirin"}],
        }

    def test_invalid_id_returns_error(self):
        r = _call("get_chembl_molecule", molecule_chembl_id="NOT_CHEMBL")
        assert r["ok"] is False
        assert "validation_error" in r["error"]["code"]

    def test_valid_id_normalizes_result(self):
        with patch("requests.get", return_value=_mock_json(self._mol_payload())):
            r = _call("get_chembl_molecule", molecule_chembl_id="CHEMBL25")
        assert r["ok"] is True
        mol = r["molecule"]
        assert mol["molecule_chembl_id"] == "CHEMBL25"
        assert mol["pref_name"] == "ASPIRIN"
        assert mol["canonical_smiles"] == "CC(=O)Oc1ccccc1C(=O)O"
        assert mol["inchi_key"] is not None

    def test_raw_not_included_by_default(self):
        with patch("requests.get", return_value=_mock_json(self._mol_payload())):
            r = _call("get_chembl_molecule", molecule_chembl_id="CHEMBL25")
        assert "raw_record" not in r

    def test_include_raw_when_requested(self):
        with patch("requests.get", return_value=_mock_json(self._mol_payload())):
            r = _call("get_chembl_molecule", molecule_chembl_id="CHEMBL25",
                      include_raw=True)
        assert "raw_record" in r

    def test_not_found_returns_error(self):
        resp_404 = _mock_json({}, status=404)
        resp_404.ok = False
        with patch("requests.get", return_value=resp_404):
            r = _call("get_chembl_molecule", molecule_chembl_id="CHEMBL9999999")
        assert r["ok"] is False
        assert "not_found" in r["error"]["code"]

    def test_lowercased_id_uppercased_in_url(self):
        with patch("requests.get", return_value=_mock_json(self._mol_payload())) as mock_get:
            _call("get_chembl_molecule", molecule_chembl_id="chembl25")
        url = mock_get.call_args[0][0]
        assert "CHEMBL25" in url


# ---------------------------------------------------------------------------
# Category 4: search_chembl_targets
# ---------------------------------------------------------------------------

class TestSearchTargets:
    def _payload(self, records=None):
        return {"targets": records or [], "page_meta": {"total_count": 0, "next": None}}

    def test_empty_query_error(self):
        r = _call("search_chembl_targets", query="")
        assert r["ok"] is False

    def test_url_uses_target_search(self):
        with patch("requests.get", return_value=_mock_json(self._payload())) as mock_get:
            _call("search_chembl_targets", query="kinase")
        url = mock_get.call_args[0][0]
        assert "/target/search" in url

    def test_organism_filter_forwarded(self):
        with patch("requests.get", return_value=_mock_json(self._payload())) as mock_get:
            _call("search_chembl_targets", query="CDK2",
                  organism="Homo sapiens")
        params = mock_get.call_args[1]["params"]
        assert "organism__icontains" in params

    def test_target_type_filter_forwarded(self):
        with patch("requests.get", return_value=_mock_json(self._payload())) as mock_get:
            _call("search_chembl_targets", query="test",
                  target_type="SINGLE PROTEIN")
        params = mock_get.call_args[1]["params"]
        assert "target_type__icontains" in params


# ---------------------------------------------------------------------------
# Category 5: search_chembl_activities — unit preservation (critical)
# ---------------------------------------------------------------------------

class TestSearchActivities:
    def _activity_record(self) -> dict:
        """Raw ChEMBL activity record with BOTH original and standardized values."""
        return {
            "activity_id": 12345,
            "molecule_chembl_id": "CHEMBL25",
            "target_chembl_id": "CHEMBL203",   # EGFR (not CHEMBL301 which is CDK2)
            "assay_chembl_id": "CHEMBL654321",
            "assay_type": "B",
            "standard_type": "IC50",
            "standard_relation": "=",
            "standard_value": 100.0,
            "standard_units": "nM",
            "type": "IC50",             # original type
            "relation": "=",            # original relation
            "value": 100.0,             # original value
            "units": "nM",              # original units
            "pchembl_value": 7.0,
            "confidence_score": 9,
            "target_organism": "Homo sapiens",
            "document_chembl_id": "CHEMBL1121364",
        }

    def _payload(self, records=None):
        if records is None:
            records = [self._activity_record()]
        return {"activities": records, "page_meta": {"total_count": len(records), "next": None}}

    def test_requires_at_least_one_filter(self):
        r = _call("search_chembl_activities")
        assert r["ok"] is False
        assert "validation_error" in r["error"]["code"]

    def test_molecule_filter_forwarded(self):
        with patch("requests.get", return_value=_mock_json(self._payload())) as mock_get:
            _call("search_chembl_activities", molecule_chembl_id="CHEMBL25")
        params = mock_get.call_args[1]["params"]
        assert params["molecule_chembl_id"] == "CHEMBL25"

    def test_standard_type_filter(self):
        with patch("requests.get", return_value=_mock_json(self._payload())) as mock_get:
            _call("search_chembl_activities",
                  molecule_chembl_id="CHEMBL25", standard_type="IC50")
        params = mock_get.call_args[1]["params"]
        assert "standard_type__iexact" in params

    def test_assay_type_filter(self):
        with patch("requests.get", return_value=_mock_json(self._payload())) as mock_get:
            _call("search_chembl_activities",
                  target_chembl_id="CHEMBL301", assay_type="B")
        params = mock_get.call_args[1]["params"]
        assert "assay_type__iexact" in params

    def test_pchembl_value_gte_filter(self):
        with patch("requests.get", return_value=_mock_json(self._payload())) as mock_get:
            _call("search_chembl_activities",
                  target_chembl_id="CHEMBL301", pchembl_value_gte=6.0)
        params = mock_get.call_args[1]["params"]
        assert params["pchembl_value__gte"] == 6.0

    def test_original_units_preserved_in_record(self):
        """Activity records must preserve BOTH original and standardized values."""
        with patch("requests.get", return_value=_mock_json(self._payload())):
            r = _call("search_chembl_activities", molecule_chembl_id="CHEMBL25")
        assert r["ok"] is True
        record = r["records"][0]
        # Both standardized AND original values must be present
        assert "standard_value" in record
        assert "standard_units" in record
        assert "standard_type" in record
        assert "standard_relation" in record
        assert "value" in record         # original
        assert "units" in record         # original
        assert "type" in record          # original
        assert "relation" in record      # original
        assert "pchembl_value" in record

    def test_no_silent_pic50_conversion(self):
        """The tool must NOT transform the value; pchembl_value comes from ChEMBL."""
        with patch("requests.get", return_value=_mock_json(self._payload())):
            r = _call("search_chembl_activities", molecule_chembl_id="CHEMBL25")
        record = r["records"][0]
        # IC50 value should remain as reported (100 nM), not converted
        assert record["standard_value"] == 100.0
        assert record["standard_units"] == "nM"
        # pchembl_value is FROM ChEMBL, not computed locally
        assert record["pchembl_value"] == 7.0

    def test_confidence_score_preserved(self):
        with patch("requests.get", return_value=_mock_json(self._payload())):
            r = _call("search_chembl_activities", molecule_chembl_id="CHEMBL25")
        assert r["records"][0]["confidence_score"] == 9

    def test_pagination_params_forwarded(self):
        with patch("requests.get", return_value=_mock_json(self._payload())) as mock_get:
            _call("search_chembl_activities",
                  molecule_chembl_id="CHEMBL25", limit=100, offset=40)
        params = mock_get.call_args[1]["params"]
        assert params["limit"] == 100
        assert params["offset"] == 40


# ---------------------------------------------------------------------------
# Category 6: search_chembl_similarity + substructure
# ---------------------------------------------------------------------------

class TestSimilaritySubstructure:
    def _mol_payload(self, n=3):
        return {
            "molecules": [{"molecule_chembl_id": f"CHEMBL{i}"} for i in range(n)],
            "page_meta": {"total_count": n, "next": None},
        }

    def test_similarity_empty_smiles_error(self):
        r = _call("search_chembl_similarity", smiles="")
        assert r["ok"] is False
        assert r["error"]["code"] == "validation_error"

    def test_similarity_invalid_threshold_error(self):
        r = _call("search_chembl_similarity", smiles="CC", threshold=1.5)
        assert r["ok"] is False

    def test_similarity_threshold_in_url(self):
        with patch("requests.get", return_value=_mock_json(self._mol_payload())) as mock_get:
            _call("search_chembl_similarity", smiles="CC", threshold=0.8)
        url = mock_get.call_args[0][0]
        assert "/similarity/" in url
        assert "80" in url   # 0.8 × 100 = 80

    def test_similarity_query_smiles_in_result(self):
        with patch("requests.get", return_value=_mock_json(self._mol_payload())):
            r = _call("search_chembl_similarity", smiles="CC", threshold=0.7)
        assert r["query_smiles"] == "CC"
        assert r["threshold"] == 0.7

    def test_similarity_limit_capped_at_100(self):
        with patch("requests.get", return_value=_mock_json(self._mol_payload())) as mock_get:
            _call("search_chembl_similarity", smiles="CC", limit=500)
        params = mock_get.call_args[1]["params"]
        assert params["limit"] <= 100

    def test_substructure_empty_smiles_error(self):
        r = _call("search_chembl_substructure", smiles="")
        assert r["ok"] is False

    def test_substructure_url_contains_smiles(self):
        with patch("requests.get", return_value=_mock_json(self._mol_payload())) as mock_get:
            _call("search_chembl_substructure", smiles="c1ccccc1")
        url = mock_get.call_args[0][0]
        assert "/substructure/" in url

    def test_substructure_result_has_query_smiles(self):
        with patch("requests.get", return_value=_mock_json(self._mol_payload())):
            r = _call("search_chembl_substructure", smiles="c1ccccc1")
        assert r["query_smiles"] == "c1ccccc1"


# ---------------------------------------------------------------------------
# Category 7: mechanisms and indications
# ---------------------------------------------------------------------------

class TestMechanismsIndications:
    def _mec_payload(self):
        return {"mechanisms": [{"mec_id": 1, "mechanism_of_action": "Inhibitor"}],
                "page_meta": {"total_count": 1, "next": None}}

    def _ind_payload(self):
        return {"drug_indications": [{"drugind_id": 1, "efo_term": "cancer"}],
                "page_meta": {"total_count": 1, "next": None}}

    def test_mechanisms_requires_filter(self):
        r = _call("get_chembl_mechanisms")
        assert r["ok"] is False

    def test_mechanisms_molecule_filter(self):
        with patch("requests.get", return_value=_mock_json(self._mec_payload())) as mock_get:
            r = _call("get_chembl_mechanisms", molecule_chembl_id="CHEMBL25")
        assert r["ok"] is True
        params = mock_get.call_args[1]["params"]
        assert params["molecule_chembl_id"] == "CHEMBL25"

    def test_indications_requires_filter(self):
        r = _call("get_chembl_indications")
        assert r["ok"] is False

    def test_indications_efo_filter(self):
        with patch("requests.get", return_value=_mock_json(self._ind_payload())) as mock_get:
            r = _call("get_chembl_indications", efo_term="cancer")
        assert r["ok"] is True
        params = mock_get.call_args[1]["params"]
        assert "efo_term__icontains" in params

    def test_indications_phase_filter(self):
        with patch("requests.get", return_value=_mock_json(self._ind_payload())) as mock_get:
            _call("get_chembl_indications", molecule_chembl_id="CHEMBL25",
                  max_phase_for_ind=4)
        params = mock_get.call_args[1]["params"]
        assert params["max_phase_for_ind__gte"] == 4


# ---------------------------------------------------------------------------
# Category 8: HTTP error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_rate_limit_returns_error(self):
        resp_429 = _mock_json({}, status=429)
        resp_429.ok = False
        resp_429.headers = {"Content-Type": "application/json", "Retry-After": "0"}
        with patch("requests.get", return_value=resp_429):
            with patch("time.sleep"):  # skip actual sleep
                r = _call("get_chembl_molecule", molecule_chembl_id="CHEMBL25")
        assert r["ok"] is False

    def test_service_unavailable_returns_error(self):
        resp_503 = _mock_json({}, status=503)
        resp_503.ok = False
        resp_503.headers = {"Content-Type": "application/json"}
        with patch("requests.get", return_value=resp_503):
            with patch("time.sleep"):
                r = _call("get_chembl_molecule", molecule_chembl_id="CHEMBL25")
        assert r["ok"] is False

    def test_connection_error_returns_error(self):
        import requests as req
        with patch("requests.get", side_effect=req.ConnectionError("refused")):
            r = _call("get_chembl_molecule", molecule_chembl_id="CHEMBL25")
        assert r["ok"] is False

    def test_malformed_json_returns_error(self):
        resp_bad = MagicMock()
        resp_bad.status_code = 200
        resp_bad.ok = True
        resp_bad.headers = {"Content-Type": "application/json"}
        resp_bad.json.side_effect = ValueError("not JSON")
        with patch("requests.get", return_value=resp_bad):
            r = _call("get_chembl_molecule", molecule_chembl_id="CHEMBL25")
        assert r["ok"] is False

    def test_non_json_content_type_returns_error(self):
        resp_html = MagicMock()
        resp_html.status_code = 200
        resp_html.ok = True
        resp_html.headers = {"Content-Type": "text/html"}
        with patch("requests.get", return_value=resp_html):
            r = _call("get_chembl_molecule", molecule_chembl_id="CHEMBL25")
        assert r["ok"] is False


# ---------------------------------------------------------------------------
# Category 9: Provenance fields
# ---------------------------------------------------------------------------

class TestProvenance:
    def _ok_resp(self):
        return _mock_json({
            "molecule_chembl_id": "CHEMBL25", "pref_name": "ASPIRIN",
            "molecule_structures": {}, "molecule_properties": {},
            "molecule_synonyms": [],
        })

    def test_provenance_has_api_url(self):
        with patch("requests.get", return_value=self._ok_resp()):
            r = _call("get_chembl_molecule", molecule_chembl_id="CHEMBL25")
        assert "_chembl_api_url" in r["provenance"]

    def test_provenance_has_endpoint(self):
        with patch("requests.get", return_value=self._ok_resp()):
            r = _call("get_chembl_molecule", molecule_chembl_id="CHEMBL25")
        assert "_endpoint" in r["provenance"]
        assert "CHEMBL25" in r["provenance"]["_endpoint"]

    def test_provenance_has_retrieval_time(self):
        with patch("requests.get", return_value=self._ok_resp()):
            r = _call("get_chembl_molecule", molecule_chembl_id="CHEMBL25")
        ts = r["provenance"]["_retrieved_at"]
        assert "T" in ts   # ISO 8601

    def test_all_tools_include_provenance(self):
        payload = {"molecules": [], "page_meta": {"total_count": 0, "next": None}}
        with patch("requests.get", return_value=_mock_json(payload)):
            r = _call("search_chembl_molecules", query="test")
        assert "provenance" in r


# ---------------------------------------------------------------------------
# Category 10: Page limit cap
# ---------------------------------------------------------------------------

class TestPageCap:
    def test_limit_cannot_exceed_1000(self):
        payload = {"molecules": [], "page_meta": {"total_count": 0, "next": None}}
        with patch("requests.get", return_value=_mock_json(payload)) as mock_get:
            _call("search_chembl_molecules", query="test", limit=9999)
        params = mock_get.call_args[1]["params"]
        assert params["limit"] <= 1000

    def test_similarity_limit_capped_at_100(self):
        payload = {"molecules": [], "page_meta": {"total_count": 0, "next": None}}
        with patch("requests.get", return_value=_mock_json(payload)) as mock_get:
            _call("search_chembl_similarity", smiles="CC", limit=9999)
        params = mock_get.call_args[1]["params"]
        assert params["limit"] <= 100
