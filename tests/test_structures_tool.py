# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for prepare_structure and read_molecules.

The shared volume is a tmp_path, so nothing here touches /shared, and the PDB
is never called: the one test that fetches stubs ``requests.get``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reinvent4_mcp.tools import structures

#: Twelve residues a chain, since a run of fewer than ten is not read as a
#: protein, and a water to be dropped from the template.
CHAIN_A = ["MET", "ALA", "GLY", "SER", "LYS", "LEU", "VAL", "ILE", "GLU", "ASP", "PHE", "TRP"]
CHAIN_B = ["GLY", "GLY", "ALA", "PRO", "THR", "TYR", "ASN", "GLN", "ARG", "HIS", "CYS", "MET"]
SEQUENCE_A = "MAGSKLVIEDFW"


def _atom(serial: int, name: str, residue: str, chain: str, number: int, record: str = "ATOM  ") -> str:
    """One PDB atom record, in the columns the reader counts on."""
    return (
        f"{record}{serial:5d} {name:<4s} {residue:>3s} {chain}{number:4d}    "
        f"{10.0 + number:8.3f}{10.0:8.3f}{10.0:8.3f}  1.00 20.00          {name[0]:>2s}"
    )


def _pdb() -> str:
    lines = []
    serial = 1
    for chain, residues in (("A", CHAIN_A), ("B", CHAIN_B)):
        for number, residue in enumerate(residues, start=1):
            for name in ("N", "CA", "C", "O"):
                lines.append(_atom(serial, name, residue, chain, number))
                serial += 1
    lines.append(_atom(serial, "O", "HOH", "A", 101, record="HETATM"))
    return "\n".join([*lines, "END", ""])


PDB = _pdb()

#: What the PDB serves for an entry: an mmCIF, whose construct is enough to
#: read the chain off without any coordinates.
CIF = "\n".join(
    [
        "data_4ZC9",
        "#",
        "_entity_poly.entity_id 1",
        "_entity_poly.type 'polypeptide(L)'",
        "_entity_poly.pdbx_strand_id A",
        f"_entity_poly.pdbx_seq_one_letter_code_can {SEQUENCE_A}",
        "#",
        "",
    ]
)


def _shared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the tools at a shared volume under *tmp_path*."""
    monkeypatch.setenv("IP_SHARED_DIR", str(tmp_path))
    return tmp_path


def _write(tmp_path: Path, name: str, text: str) -> str:
    (tmp_path / name).write_text(text)
    return name


class TestPrepareStructure:
    def test_either_an_entry_or_a_path_is_required(self, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)
        assert "error" in structures.prepare_structure()
        assert "error" in structures.prepare_structure(pdb_id="4ZC9", path="x.cif")

    def test_a_file_on_the_volume_reports_its_chains(self, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)
        result = structures.prepare_structure(path=_write(tmp_path, "two_chains.pdb", PDB))

        assert result["structure_path"] == "two_chains.pdb"
        assert result["format"] == "pdb"
        assert [c["chain_id"] for c in result["chains"]] == ["A", "B"]
        # No chain was asked for, so no sequence is spelled out.
        assert "sequence" not in result

    def test_a_named_chain_comes_back_with_its_sequence(self, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)
        result = structures.prepare_structure(path=_write(tmp_path, "two_chains.pdb", PDB), chain="A")

        assert result["structure_chain"] == "A"
        assert result["sequence"] == SEQUENCE_A
        assert result["sequence_length"] == len(SEQUENCE_A)

    def test_an_unknown_chain_is_refused_with_the_chains_there_are(self, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)
        result = structures.prepare_structure(path=_write(tmp_path, "two_chains.pdb", PDB), chain="Z")

        assert "no protein chain 'Z'" in result["error"]
        assert "A (12 aa)" in result["error"]

    def test_a_template_is_written_and_described(self, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)
        result = structures.prepare_structure(path=_write(tmp_path, "4ZC9.pdb", PDB), chain="A", template=True)

        template = result["template"]
        assert template["path"] == "uploads/agent/4ZC9_A.cif"
        assert template["chain_id"] == "A"
        assert template["atoms"] == 4 * len(CHAIN_A)
        assert template["residues"] == len(CHAIN_A)
        written = tmp_path / template["path"]
        assert written.is_file()
        # Boltz reads it back as one chain: nothing of chain B or the water.
        assert "PRO" not in written.read_text()
        assert "HOH" not in written.read_text()

    def test_a_path_outside_the_volume_is_refused(self, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)
        result = structures.prepare_structure(path="../escaped.cif")

        assert "outside the shared volume" in result["error"]

    def test_a_missing_file_says_how_to_find_the_real_paths(self, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)
        assert "list_shared_assets" in structures.prepare_structure(path="nope.cif")["error"]

    def test_a_file_that_is_not_a_structure_is_refused(self, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)
        result = structures.prepare_structure(path=_write(tmp_path, "notes.txt", "hello"))

        assert "error" in result

    @pytest.mark.parametrize("entry", ["ZZZZ", "4ZC", "4zc9x", ""])
    def test_something_that_is_not_a_pdb_id_is_refused(self, entry, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)
        assert "error" in structures.prepare_structure(pdb_id=entry)

    def test_an_entry_is_fetched_once_and_reused(self, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)
        calls: list[str] = []

        class Response:
            status_code = 200
            ok = True
            content = CIF.encode()

        def fake_get(url: str, **kwargs: object) -> Response:
            calls.append(url)
            return Response()

        monkeypatch.setattr(structures.requests, "get", fake_get)
        first = structures.prepare_structure(pdb_id="4zc9")

        assert first["pdb_id"] == "4ZC9"
        assert first["structure_path"] == "uploads/agent/4ZC9.cif"
        assert calls == ["https://files.rcsb.org/download/4ZC9.cif"]

        structures.prepare_structure(pdb_id="4ZC9")
        assert len(calls) == 1

    def test_an_entry_the_pdb_does_not_have_is_reported(self, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)

        class Missing:
            status_code = 404
            ok = False
            content = b""

        monkeypatch.setattr(structures.requests, "get", lambda url, **kwargs: Missing())
        assert "no entry 1ABC" in structures.prepare_structure(pdb_id="1ABC")["error"]


class TestReadMolecules:
    def test_a_smi_file_keeps_the_smiles_and_drops_the_names(self, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)
        name = _write(tmp_path, "seeds.smi", "# a comment\nCCO ethanol\nc1ccccc1\tbenzene\n\n")
        result = structures.read_molecules(name)

        assert result["smiles"] == ["CCO", "c1ccccc1"]
        assert result["count"] == 2

    def test_a_csv_is_read_from_its_smiles_column(self, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)
        name = _write(tmp_path, "series.csv", "id,SMILES,score\n1,CCO,0.4\n2,CCN,0.7\n")
        result = structures.read_molecules(name)

        assert result["smiles"] == ["CCO", "CCN"]

    def test_a_csv_without_a_header_still_finds_the_column(self, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)
        name = _write(tmp_path, "bare.csv", "CCO,0.4\nCCN,0.7\n")

        assert structures.read_molecules(name)["smiles"] == ["CCO", "CCN"]

    def test_the_list_is_capped_and_says_so(self, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)
        name = _write(tmp_path, "many.smi", "\n".join(["CCO"] * 5))
        result = structures.read_molecules(name, limit=2)

        assert result["count"] == 5
        assert len(result["smiles"]) == 2
        assert "5 molecules" in result["note"]

    def test_a_file_with_no_molecules_is_an_error(self, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)
        name = _write(tmp_path, "empty.smi", "# nothing here\n")

        assert "No SMILES" in structures.read_molecules(name)["error"]

    def test_a_structure_file_is_not_a_molecule_file(self, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)
        name = _write(tmp_path, "protein.pdb", PDB)

        assert "error" in structures.read_molecules(name)


class TestTheToolsAreRegistered:
    def test_the_server_exposes_both_as_json(self, tmp_path, monkeypatch):
        _shared(tmp_path, monkeypatch)
        from reinvent4_mcp import server

        payload = json.loads(server.prepare_structure(path=_write(tmp_path, "two_chains.pdb", PDB)))
        assert payload["structure_path"] == "two_chains.pdb"

        name = _write(tmp_path, "seeds.smi", "CCO\n")
        assert json.loads(server.read_molecules(path=name))["smiles"] == ["CCO"]
