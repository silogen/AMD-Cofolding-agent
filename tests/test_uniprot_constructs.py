# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""fetch_target_info: the full-length sequence is labelled as such and the
crystallised constructs of the top PDB entries come from RCSB (HTTP stubbed)."""

from __future__ import annotations

from typing import Any

import pytest
import requests

from reinvent4_mcp.tools import uniprot

FULL = "M" + "A" * 1209
KINASE = "GEAPNQALLRILKETEFKKIKVLGSGAFGTVYKG"


def _xref(pdb_id: str, method: str, resolution: str | None, chains: str) -> dict[str, Any]:
    props = [{"key": "Method", "value": method}, {"key": "Chains", "value": chains}]
    if resolution:
        props.append({"key": "Resolution", "value": resolution})
    return {"database": "PDB", "id": pdb_id, "properties": props}


ENTRY = {
    "primaryAccession": "P00533",
    "uniProtkbId": "EGFR_HUMAN",
    "entryType": "UniProtKB reviewed (Swiss-Prot)",
    "proteinDescription": {"recommendedName": {"fullName": {"value": "Epidermal growth factor receptor"}}},
    "genes": [{"geneName": {"value": "EGFR"}}],
    "organism": {"scientificName": "Homo sapiens"},
    "sequence": {"value": FULL, "length": len(FULL)},
    "uniProtKBCrossReferences": [
        _xref("1IVO", "X-ray", "3.30 A", "A/B=25-646"),
        _xref("2J5E", "X-ray", "3.10 A", "A=696-1022"),
        _xref("1XKK", "X-ray", "2.40 A", "A=695-1022"),
        _xref("5X2A", "NMR", None, "A=669-721"),
        _xref("2KS1", "X-ray", "1.00 A", "B=1-12"),
        {"database": "ChEMBL", "id": "CHEMBL203"},
    ],
}


def _rcsb_entry(pdb_id: str, start: int) -> dict[str, Any]:
    return {
        "rcsb_id": pdb_id,
        "struct": {"title": f"EGFR kinase domain {pdb_id}"},
        "rcsb_entry_info": {"resolution_combined": [2.4], "experimental_method": "X-ray"},
        "polymer_entities": [
            {
                "rcsb_polymer_entity": {"pdbx_description": "EPIDERMAL GROWTH FACTOR RECEPTOR"},
                "entity_poly": {"pdbx_seq_one_letter_code_can": KINASE[:20] + "\n" + KINASE[20:], "rcsb_sample_sequence_length": len(KINASE), "type": "polypeptide(L)"},
                "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["A"], "uniprot_ids": ["P00533"]},
                "rcsb_polymer_entity_align": [
                    {"reference_database_name": "UniProt", "reference_database_accession": "P00533",
                     "aligned_regions": [{"entity_beg_seq_id": 1, "ref_beg_seq_id": start, "length": len(KINASE)}]},
                ],
            },
            {
                "rcsb_polymer_entity": {"pdbx_description": "Fab heavy chain"},
                "entity_poly": {"pdbx_seq_one_letter_code_can": "Q" * 200, "type": "polypeptide(L)"},
                "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["H"], "uniprot_ids": None},
            },
        ],
        "nonpolymer_entities": [{"nonpolymer_comp": {"chem_comp": {"id": "CY0", "name": "inhibitor"}}}],
    }


class _Response:
    def __init__(self, payload: Any, status: int = 200):
        self._payload = payload
        self.status_code = status

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def http(monkeypatch):
    calls: dict[str, list] = {"get": [], "post": []}
    state: dict[str, Any] = {"post": lambda url, **kw: _Response({"data": {"entries": [_rcsb_entry(i, 696) for i in kw["json"]["variables"]["ids"]]}})}

    def get(url, **kw):
        calls["get"].append((url, kw))
        return _Response(ENTRY)

    def post(url, **kw):
        calls["post"].append((url, kw))
        return state["post"](url, **kw)

    monkeypatch.setattr(uniprot.requests, "get", get)
    monkeypatch.setattr(uniprot.requests, "post", post)
    return calls, state


def test_full_length_sequence_is_labelled_and_constructs_reported(http):
    calls, _ = http
    info = uniprot.fetch_target_info("P00533")

    assert info["sequence"] == FULL
    assert info["sequence_kind"] == "full_length"
    assert "FULL-LENGTH" in info["cofold_advice"] and "1210 aa" in info["cofold_advice"]

    # Domain-sized X-ray entries first by resolution; peptide and NMR entries last.
    assert info["pdb_ids"][:3] == ["1XKK", "2J5E", "1IVO"]
    assert info["pdb_ids"][-1] == "2KS1"
    assert info["pdb_structures"][0] == {"pdb_id": "1XKK", "method": "X-ray", "resolution": 2.4, "uniprot_range": [695, 1022]}
    assert info["pdb_structures"][2]["uniprot_range"] == [25, 646]

    [(url, kw)] = calls["post"]
    assert url == uniprot._RCSB_GRAPHQL
    assert kw["json"]["variables"]["ids"] == ["1XKK", "2J5E", "1IVO"]
    assert kw["timeout"] == uniprot._RCSB_TIMEOUT

    first = info["constructs"][0]
    assert first["pdb_id"] == "1XKK"
    assert first["sequence"] == KINASE
    assert first["length"] == len(KINASE)
    assert first["uniprot_range"] == [696, 696 + len(KINASE) - 1]
    assert first["chains"] == ["A"]
    assert first["entity"] == "EPIDERMAL GROWTH FACTOR RECEPTOR"
    assert first["ligands"] == ["CY0 (inhibitor)"]
    assert "1XKK: 34 aa, UniProt 696-729" in info["cofold_advice"]


def test_named_entries_are_described(http):
    calls, _ = http
    info = uniprot.fetch_target_info("P00533", pdb_ids=["2j5e"])
    assert calls["post"][0][1]["json"]["variables"]["ids"] == ["2J5E"]
    assert [c["pdb_id"] for c in info["constructs"]] == ["2J5E"]


def test_construct_lookup_is_bounded():
    many = [f"{i}ABC" for i in range(10)]
    seen = {}

    def post(url, **kw):
        seen["ids"] = kw["json"]["variables"]["ids"]
        return _Response({"data": {"entries": []}})

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(uniprot.requests, "post", post)
        uniprot.fetch_constructs(many, "P00533")
    assert len(seen["ids"]) == uniprot._MAX_CONSTRUCTS


def test_unreachable_rcsb_keeps_the_uniprot_answer(http):
    _, state = http

    def down(url, **kw):
        raise requests.ConnectTimeout("timed out")

    state["post"] = down
    info = uniprot.fetch_target_info("P00533")
    assert info["accession"] == "P00533"
    assert info["constructs"] == []
    assert "RCSB" in info["constructs_error"]
    assert "pdb_structures[].uniprot_range" in info["cofold_advice"]


def test_graphql_error_is_reported(http):
    _, state = http
    state["post"] = lambda url, **kw: _Response({"errors": [{"message": "bad field"}]})
    info = uniprot.fetch_target_info("P00533")
    assert info["constructs"] == []
    assert info["constructs_error"] == "RCSB data API error: bad field"


def test_uniprot_failure_still_errors(monkeypatch):
    def down(url, **kw):
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(uniprot.requests, "get", down)
    assert "NOT retrieved" in uniprot.fetch_target_info("P00533")["error"]
