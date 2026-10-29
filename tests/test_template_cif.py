# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the template mmCIF writer."""

from __future__ import annotations

import pytest

from reinvent4_mcp.ttp.structure import cif_chains, read_cif
from reinvent4_mcp.ttp.template import CleanedChain, cif_value, clean_chain, template_name


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEQ = ["MSE", "GLY", "SER", "LYS", "ALA", "LEU", "VAL", "ILE", "GLU", "ASP", "PHE", "TRP"]
SEQ_ONE_LETTER = "MGSKALVIEDFW"

ATOM_SITE_FIELDS = (
    "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id", "label_asym_id",
    "label_entity_id", "label_seq_id", "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
    "B_iso_or_equiv", "auth_seq_id", "auth_comp_id", "auth_asym_id", "auth_atom_id", "pdbx_PDB_model_num",
)


def _atom(
    serial: int,
    element: str,
    name: str,
    alt: str,
    comp: str,
    asym: str,
    entity: str,
    seq: str,
    auth_seq: str,
    auth_asym: str,
    model: int = 1,
    group: str = "ATOM",
) -> str:
    fields = (
        group, serial, element, name, alt, comp, asym, entity, seq, "?",
        "1.0", "2.0", "3.0", "1.00", "10.0", auth_seq, comp, auth_asym, name, model,
    )
    return " ".join(str(f) for f in fields)


def _cif(rows: list[str], poly_seq: list[str] = SEQ) -> str:
    """A two-chain mmCIF: a protein (entity 1, chains A and B), a ligand and water."""
    return "\n".join(
        [
            "data_1ABC",
            "#",
            "loop_",
            "_entity.id",
            "_entity.type",
            "1 polymer",
            "2 non-polymer",
            "3 water",
            "#",
            "_entity_poly.entity_id 1",
            "_entity_poly.type 'polypeptide(L)'",
            "_entity_poly.pdbx_strand_id A,B",
            f"_entity_poly.pdbx_seq_one_letter_code_can {SEQ_ONE_LETTER}",
            "#",
            "loop_",
            "_entity_poly_seq.entity_id",
            "_entity_poly_seq.num",
            "_entity_poly_seq.mon_id",
            "_entity_poly_seq.hetero",
            *(f"1 {i} {mon} n" for i, mon in enumerate(poly_seq, 1)),
            "#",
            "loop_",
            *(f"_atom_site.{f}" for f in ATOM_SITE_FIELDS),
            *rows,
            "#",
            "",
        ]
    )


CIF_ROWS = [
    _atom(1, "N", "N", ".", "MSE", "C", "1", "1", "671", "A", group="HETATM"),
    _atom(2, "SE", "SE", ".", "MSE", "C", "1", "1", "671", "A", group="HETATM"),
    _atom(3, "N", "N", "A", "GLY", "C", "1", "2", "672", "A"),
    _atom(4, "N", "N", "B", "GLY", "C", "1", "2", "672", "A"),
    _atom(5, "C", "CA", "B", "GLY", "C", "1", "2", "672", "A"),
    _atom(6, "C", "CA", "A", "GLY", "C", "1", "2", "672", "A"),
    _atom(7, "N", "N", ".", "SER", "C", "1", "3", "673", "A"),
    _atom(8, "H", "H", ".", "SER", "C", "1", "3", "673", "A"),
    _atom(9, "C", "C1", ".", "ATP", "D", "2", ".", "901", "A", group="HETATM"),
    _atom(10, "O", "O", ".", "HOH", "E", "3", ".", "1001", "A", group="HETATM"),
    _atom(11, "N", "N", ".", "LYS", "F", "1", "4", "674", "B"),
    _atom(12, "N", "N", ".", "LYS", "C", "1", "4", "674", "A", model=2),
]

CIF = _cif(CIF_ROWS)


def _pdb_atom(
    serial: int,
    name: str,
    comp: str,
    chain: str,
    resseq: int,
    alt: str = " ",
    icode: str = " ",
    element: str = "C",
    record: str = "ATOM  ",
) -> str:
    label = f" {name:<3}" if len(name) < 4 else name
    return (
        f"{record}{serial:>5} {label}{alt}{comp:<3} {chain}{resseq:>4}{icode}   "
        f"{1.0:>8.3f}{2.0:>8.3f}{3.0:>8.3f}{1.0:>6.2f}{10.0:>6.2f}          {element:>2}"
    )


def _atom_site(text: str) -> list[dict[str, str | None]]:
    return read_cif(text, ("atom_site",)).get("atom_site", [])


# ---------------------------------------------------------------------------
# mmCIF input
# ---------------------------------------------------------------------------


class TestCleanChainFromCif:
    def test_keeps_one_chain_of_the_first_model(self):
        out = clean_chain("x/1abc.cif", CIF, "A", "1ABC_A")
        assert out == CleanedChain(out.text, atoms=5, residues=3)
        atoms = _atom_site(out.text)
        assert [f"{a['label_comp_id']}:{a['label_atom_id']}:{a['label_seq_id']}" for a in atoms] == [
            "MSE:N:1",
            "MSE:SE:1",
            "GLY:N:2",
            "GLY:CA:2",
            "SER:N:3",
        ]
        assert [a["auth_seq_id"] for a in atoms] == ["671", "671", "672", "672", "673"]
        assert atoms[0]["group_pdb"] == "HETATM"
        assert atoms[0]["cartn_x"] == "1.0"

    def test_gives_the_chain_the_author_and_label_id_the_caller_picked(self):
        atoms = _atom_site(clean_chain("x/1abc.cif", CIF, "A", "1ABC_A").text)
        assert all(a["label_asym_id"] == "A" and a["auth_asym_id"] == "A" for a in atoms)
        assert all(a["label_entity_id"] == "1" and a["label_alt_id"] is None for a in atoms)

    def test_refuses_a_chain_without_protein_atoms(self):
        with pytest.raises(ValueError, match=r"Chain Z has no protein atoms\."):
            clean_chain("x/1abc.cif", CIF, "Z", "1ABC_Z")

    def test_reads_back_as_the_whole_construct(self):
        text = clean_chain("x/1abc.cif", CIF, "A", "1ABC_A").text
        assert text.startswith("data_1ABC_A\n")
        assert [(c.id, c.sequence, c.source) for c in cif_chains(text)] == [("A", SEQ_ONE_LETTER, "construct")]
        assert read_cif(text, ("struct_asym",))["struct_asym"] == [{"id": "A", "entity_id": "1"}]
        assert len(read_cif(text, ("entity_poly_seq",))["entity_poly_seq"]) == len(SEQ)


class TestNumbering:
    ROWS = [
        _atom(1, "C", "CA", ".", "LYS", "C", "1", "4", "604", "A"),
        _atom(2, "C", "CA", ".", "ALA", "C", "1", "5", "605", "A"),
    ]

    def test_entity_poly_seq_numbering_is_kept(self):
        out = clean_chain("x.cif", _cif(self.ROWS), "A", "x")
        assert [a["label_seq_id"] for a in _atom_site(out.text)] == ["4", "5"]
        monomers = read_cif(out.text, ("entity_poly_seq",))["entity_poly_seq"]
        assert [(m["num"], m["mon_id"]) for m in monomers] == [(str(i), mon) for i, mon in enumerate(SEQ, 1)]

    def test_observed_residues_renumbered_when_the_sequence_is_short(self):
        out = clean_chain("x.cif", _cif(self.ROWS, poly_seq=SEQ[:2]), "A", "x")
        assert [a["label_seq_id"] for a in _atom_site(out.text)] == ["1", "2"]
        monomers = read_cif(out.text, ("entity_poly_seq",))["entity_poly_seq"]
        assert [(m["num"], m["mon_id"]) for m in monomers] == [("1", "LYS"), ("2", "ALA")]


# ---------------------------------------------------------------------------
# PDB input
# ---------------------------------------------------------------------------


class TestCleanChainFromPdb:
    """Chain B holds the 12 residues of SEQ, numbered with a gap so that the
    output's 1..n numbering is not the file's own."""

    @staticmethod
    def _text() -> str:
        residues = [
            _pdb_atom(i + 1, "CA", comp, "B", i + 10 if i < 6 else i + 20, record="HETATM" if comp == "MSE" else "ATOM  ")
            for i, comp in enumerate(SEQ)
        ]
        return "\n".join(
            [
                _pdb_atom(100, "N", "GLY", "A", 1, element="N"),
                *residues,
                _pdb_atom(30, "CB", "TRP", "B", 31, alt="A"),
                _pdb_atom(31, "CB", "TRP", "B", 31, alt="B"),
                _pdb_atom(32, "HA", "TRP", "B", 31, element="H"),
                _pdb_atom(40, "C1", "ATP", "B", 401, record="HETATM"),
                _pdb_atom(41, "O", "HOH", "B", 501, element="O", record="HETATM"),
                "ENDMDL",
                _pdb_atom(50, "CA", "GLY", "B", 99),
            ]
        )

    def test_numbers_the_observed_residues_from_one(self):
        out = clean_chain("shared/5uf0_protein.pdb", self._text(), "B", "5uf0_protein_B")
        assert (out.atoms, out.residues) == (13, len(SEQ))
        atoms = _atom_site(out.text)
        assert list(dict.fromkeys(a["label_seq_id"] for a in atoms)) == [str(i) for i in range(1, len(SEQ) + 1)]
        assert all(a["label_asym_id"] == "B" and a["auth_asym_id"] == "B" for a in atoms)

    def test_drops_the_second_alt_loc_hydrogens_ligands_and_water(self):
        atoms = _atom_site(clean_chain("shared/5uf0_protein.pdb", self._text(), "B", "5uf0_protein_B").text)
        assert {a["label_comp_id"] for a in atoms} == set(SEQ)
        assert [a["label_atom_id"] for a in atoms].count("CB") == 1
        assert atoms[-1]["label_atom_id"] == "CB"
        assert (atoms[-1]["auth_seq_id"], atoms[-1]["label_seq_id"]) == ("31", "12")

    def test_reads_back_as_the_observed_sequence(self):
        text = clean_chain("shared/5uf0_protein.pdb", self._text(), "B", "5uf0_protein_B").text
        assert [(c.id, c.sequence, c.source) for c in cif_chains(text)] == [("B", SEQ_ONE_LETTER, "construct")]

    def test_refuses_a_chain_that_is_not_there(self):
        with pytest.raises(ValueError, match=r"Chain C has no protein atoms\."):
            clean_chain("shared/5uf0_protein.pdb", self._text(), "C", "5uf0_protein_C")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestCifValue:
    def test_plain_token_is_written_as_is(self):
        assert cif_value("CA") == "CA"
        assert cif_value("1.0") == "1.0"

    def test_nothing_is_a_dot(self):
        assert cif_value(None) == "."
        assert cif_value("") == "."

    def test_space_and_apostrophe_are_quoted(self):
        assert cif_value("a b") == "'a b'"
        assert cif_value("O5'") == '"O5\'"'

    def test_reserved_words_are_quoted(self):
        assert cif_value("loop_") == "'loop_'"
        assert cif_value("data_x") == "'data_x'"
        assert cif_value("_x") == "'_x'"
        assert cif_value(".") == "'.'"


class TestTemplateName:
    def test_named_after_the_source_and_chain(self):
        assert template_name("4ZC9.cif", "A") == "4ZC9_A.cif"
        assert template_name("a/b/1m17_protein.pdb", "A") == "1m17_protein_A.cif"
        assert template_name("1M17", "A") == "1M17_A.cif"

    def test_odd_characters_become_underscores(self):
        assert template_name("structures\\egfr model.pdb", "B") == "egfr_model_B.cif"
        assert template_name("x/1abc (copy).cif", "A/1") == "1abc__copy__A_1.cif"
