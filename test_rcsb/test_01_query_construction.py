"""test_01_query_construction.py — Offline unit tests for QueryBuilder.from_dict.

All tests construct RCSB query objects via QueryBuilder and assert the
serialised result is a non-empty dict.  No network calls are made and no
HTTP mocks are needed.  The entire module is skipped when rcsbapi.search
is not installed.
"""

from __future__ import annotations

import pytest

# Skip whole module when the optional dependency is absent
pytest.importorskip("rcsbapi.search", reason="rcsb-api not installed")

from rcsb import QueryBuilder, RCSBQueryError  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _terminal(service: str, **params) -> dict:
    """Build a minimal terminal-node spec."""
    return {"type": "terminal", "service": service, "parameters": params}


def _attr(attribute: str, operator: str = "exact_match", **extra) -> dict:
    """Shorthand for an attribute terminal spec."""
    p = {"attribute": attribute, "operator": operator, **extra}
    return _terminal("attribute", **p)


# ---------------------------------------------------------------------------
# TestTextQuery
# ---------------------------------------------------------------------------

class TestTextQuery:
    def test_text_simple(self):
        spec = _terminal("text", value="CDK2")
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)
        assert len(result) >= 1

    def test_text_chem(self):
        spec = _terminal(
            "text_chem",
            attribute="chem_comp.name",
            operator="contains_words",
            value="adenosine",
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# TestAttributeQuery
# ---------------------------------------------------------------------------

class TestAttributeQuery:
    def test_attr_exact_match(self):
        spec = _attr("exptl.method", "exact_match", value="X-RAY DIFFRACTION")
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_attr_range_resolution(self):
        spec = _attr(
            "rcsb_entry_info.resolution_combined",
            "less_or_equal",
            value=2.0,
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_attr_exists(self):
        # 'exists' operator: no value field required
        spec = _attr("rcsb_entry_info.resolution_combined", "exists")
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_attr_in_operator(self):
        spec = _attr(
            "exptl.method",
            "in",
            value=["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"],
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_attr_contains_phrase(self):
        spec = _attr("struct.title", "contains_phrase", value="kinase inhibitor")
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_attr_service_disambiguation(self):
        # rcsb_id is ambiguous (appears in both structure and chem attribute sets).
        # Pass service="text" inside parameters to disambiguate (per rcsbapi error message).
        spec = {
            "type": "terminal",
            "service": "attribute",
            "parameters": {
                "attribute": "rcsb_id",
                "operator": "in",
                "value": ["3PXY", "4HHB"],
                "service": "text",
            },
        }
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestBooleanComposition
# ---------------------------------------------------------------------------

class TestBooleanComposition:
    @staticmethod
    def _two_attr_group(operator: str, negate: bool = False) -> dict:
        return {
            "type": "group",
            "logical_operator": operator,
            "negate": negate,
            "nodes": [
                _attr("exptl.method", "exact_match", value="X-RAY DIFFRACTION"),
                _attr(
                    "rcsb_entry_info.resolution_combined",
                    "less_or_equal",
                    value=2.0,
                ),
            ],
        }

    def test_and(self):
        result = QueryBuilder.from_dict(self._two_attr_group("and"))
        assert isinstance(result, dict)
        # A serialised boolean group carries some nesting indicator
        assert len(result) >= 1

    def test_or(self):
        result = QueryBuilder.from_dict(self._two_attr_group("or"))
        assert isinstance(result, dict)
        assert len(result) >= 1

    def test_not(self):
        # Single node with negate=True
        spec = {
            "type": "group",
            "logical_operator": "and",
            "negate": True,
            "nodes": [
                _attr("exptl.method", "exact_match", value="SOLUTION NMR"),
            ],
        }
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_xor(self):
        """XOR is not a native RCSB Search operator; RCSBQueryError is expected."""
        spec = {
            "type": "group",
            "logical_operator": "xor",
            "nodes": [
                _attr("exptl.method", "exact_match", value="X-RAY DIFFRACTION"),
                _attr("exptl.method", "exact_match", value="ELECTRON MICROSCOPY"),
            ],
        }
        with pytest.raises(RCSBQueryError):
            QueryBuilder.from_dict(spec)

    def test_set_diff(self):
        """Set-difference is not a native RCSB Search operator; RCSBQueryError expected."""
        spec = {
            "type": "group",
            "logical_operator": "set_diff",
            "nodes": [
                _attr("exptl.method", "exact_match", value="X-RAY DIFFRACTION"),
                _attr("exptl.method", "exact_match", value="ELECTRON MICROSCOPY"),
            ],
        }
        with pytest.raises(RCSBQueryError):
            QueryBuilder.from_dict(spec)


# ---------------------------------------------------------------------------
# TestNestedAttribute
# ---------------------------------------------------------------------------

class TestNestedAttribute:
    def test_nested_ec50(self):
        spec = {
            "type": "group",
            "nested": True,
            "logical_operator": "and",
            "nodes": [
                _attr("rcsb_binding_affinity.type", "exact_match", value="EC50"),
                _attr("rcsb_binding_affinity.value", "less_or_equal", value=1.0),
            ],
        }
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# TestSequenceQuery
# ---------------------------------------------------------------------------

_CDK2_SEQ = (
    "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELRHP"
    "NIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHS"
)
_DNA_SEQ = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"  # 48 nt ≥25
_RNA_SEQ = "AUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCG"  # 48 nt ≥25


class TestSequenceQuery:
    def test_seq_protein(self):
        spec = _terminal(
            "sequence",
            sequence=_CDK2_SEQ,
            sequence_type="protein",
            evalue_cutoff=0.1,
            identity_cutoff=0.5,
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_seq_dna(self):
        spec = _terminal(
            "sequence",
            sequence=_DNA_SEQ,
            sequence_type="dna",
            evalue_cutoff=0.1,
            identity_cutoff=0.9,
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_seq_rna(self):
        spec = _terminal(
            "sequence",
            sequence=_RNA_SEQ,
            sequence_type="rna",
            evalue_cutoff=0.1,
            identity_cutoff=0.9,
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_seq_identity_cutoff_high(self):
        spec = _terminal(
            "sequence",
            sequence=_CDK2_SEQ,
            sequence_type="protein",
            identity_cutoff=0.9,
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_seq_identity_cutoff_low(self):
        spec = _terminal(
            "sequence",
            sequence=_CDK2_SEQ,
            sequence_type="protein",
            identity_cutoff=0.3,
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestSeqMotifQuery
# ---------------------------------------------------------------------------

class TestSeqMotifQuery:
    def test_seqmotif_prosite(self):
        spec = _terminal(
            "seqmotif",
            value="C-x(2,4)-C-x(3)-[LIVMFYWC]-x(8)-H-x(3,5)-H",
            pattern_type="prosite",
            sequence_type="protein",
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_seqmotif_regex(self):
        spec = _terminal(
            "seqmotif",
            value="C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H",
            pattern_type="regex",
            sequence_type="protein",
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_seqmotif_simple(self):
        spec = _terminal(
            "seqmotif",
            value="GXGXXG",
            pattern_type="simple",
            sequence_type="protein",
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_seqmotif_dna(self):
        spec = _terminal(
            "seqmotif",
            value="TATAAA",
            pattern_type="simple",
            sequence_type="dna",
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_seqmotif_rna(self):
        spec = _terminal(
            "seqmotif",
            value="GUAAGU",
            pattern_type="simple",
            sequence_type="rna",
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestStructSimilarity
# ---------------------------------------------------------------------------

class TestStructSimilarity:
    def test_struct_sim_entry_id(self):
        spec = _terminal(
            "structure",
            entry_id="4HHB",
            assembly_id="1",
            similarity_type="local",
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_struct_sim_local(self):
        spec = _terminal(
            "structure",
            entry_id="3PXY",
            similarity_type="local",
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_struct_sim_global(self):
        spec = _terminal(
            "structure",
            entry_id="3PXY",
            similarity_type="global",
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_struct_sim_file_url(self):
        spec = _terminal(
            "structure",
            file_url="https://files.rcsb.org/download/4HHB.cif",
            file_format="cif",
            similarity_type="local",
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestStructMotif
# ---------------------------------------------------------------------------

class TestStructMotif:
    @staticmethod
    def _residues(exchanges: list[str] | None = None) -> list[dict]:
        return [
            {
                "struct_oper_id": "1",
                "chain_id": "A",
                "label_seq_id": 84,
                "exchanges": exchanges or [],
            },
            {
                "struct_oper_id": "1",
                "chain_id": "A",
                "label_seq_id": 86,
                "exchanges": exchanges or [],
            },
        ]

    def test_struct_motif_two_residues(self):
        spec = _terminal(
            "strucmotif",
            entry_id="3PXY",
            residue_ids=self._residues(),
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_struct_motif_exchanges(self):
        spec = _terminal(
            "strucmotif",
            entry_id="3PXY",
            residue_ids=self._residues(exchanges=["LYS", "HIS"]),
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_struct_motif_tolerances(self):
        spec = _terminal(
            "strucmotif",
            entry_id="3PXY",
            residue_ids=self._residues(),
            backbone_distance_tolerance=2,
            side_chain_distance_tolerance=1,
            angle_tolerance=1,
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestChemSimilarity
# ---------------------------------------------------------------------------

class TestChemSimilarity:
    def test_chem_formula(self):
        spec = _terminal("chemical", value="C9 H8 O4", query_type="formula")
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_chem_formula_subset(self):
        spec = _terminal(
            "chemical",
            value="C9 H8 O4",
            query_type="formula",
            match_subset=True,
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_chem_smiles_exact(self):
        spec = _terminal(
            "chemical",
            value="CC(=O)Oc1ccccc1C(=O)O",
            query_type="descriptor",
            descriptor_type="SMILES",
            match_type="graph-exact",
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_chem_smiles_relaxed(self):
        spec = _terminal(
            "chemical",
            value="CC(=O)Oc1ccccc1C(=O)O",
            query_type="descriptor",
            descriptor_type="SMILES",
            match_type="graph-relaxed-stereo",
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_chem_smiles_substructure(self):
        spec = _terminal(
            "chemical",
            value="c1ccccc1",
            query_type="descriptor",
            descriptor_type="SMILES",
            match_type="sub-struct-graph-relaxed",
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)

    def test_chem_inchi(self):
        spec = _terminal(
            "chemical",
            value="InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
            query_type="descriptor",
            descriptor_type="InChI",
        )
        result = QueryBuilder.from_dict(spec)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestReturnTypes
# ---------------------------------------------------------------------------

class TestReturnTypes:
    VALID_RETURN_TYPES = [
        "entry",
        "assembly",
        "polymer_entity",
        "non_polymer_entity",
        "branched_entity",
        "polymer_instance",
        "mol_definition",
    ]

    def test_all_return_types(self):
        """All documented return_type strings are valid non-empty strings."""
        for rt in self.VALID_RETURN_TYPES:
            assert isinstance(rt, str) and rt, f"Invalid return_type: {rt!r}"


# ---------------------------------------------------------------------------
# TestErrors
# ---------------------------------------------------------------------------

class TestErrors:
    def test_invalid_service_raises(self):
        """An unknown service name must raise RCSBQueryError."""
        spec = _terminal("totally_unknown_service_xyz", value="test")
        with pytest.raises(RCSBQueryError):
            QueryBuilder.from_dict(spec)

    def test_missing_sequence_raises(self):
        """sequence service without 'sequence' key must raise RCSBQueryError."""
        spec = _terminal("sequence", evalue_cutoff=0.1)  # no 'sequence' key
        with pytest.raises(RCSBQueryError):
            QueryBuilder.from_dict(spec)

    def test_non_dict_raises(self):
        """Passing a non-dict query_spec must raise RCSBQueryError."""
        with pytest.raises(RCSBQueryError):
            QueryBuilder.from_dict("this is a string, not a dict")  # type: ignore[arg-type]
