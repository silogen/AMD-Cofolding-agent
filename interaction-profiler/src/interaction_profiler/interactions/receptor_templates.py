"""Residue templates for receptor pharmacophore features.

Perceiving protein/nucleic chemistry by reading a PDB into RDKit is
fragile (bond orders / aromaticity / explicit-H connectivity are all
unreliable straight from coordinates). Standard residues are far more
robustly annotated by atom name, which is what tools like PLIP / ProLIF
encode internally. These tables map ``(residue, atom_name)`` to feature
roles; hydrogen positions come from the actual prepared structure.
"""

from __future__ import annotations

# Protein backbone
BACKBONE_DONOR = {"N"}          # amide N-H (except proline)
BACKBONE_ACCEPTOR = {"O", "OXT"}

# Sidechain hydrogen-bond donors (heavy atom bearing H)
SC_DONOR: dict[str, set[str]] = {
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}, "CYS": {"SG"},
    "ASN": {"ND2"}, "GLN": {"NE2"}, "TRP": {"NE1"},
    "HIS": {"ND1", "NE2"}, "HID": {"ND1"}, "HIE": {"NE2"}, "HIP": {"ND1", "NE2"},
    "LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"},
    "SER1": set(),
}

# Sidechain acceptors
SC_ACCEPTOR: dict[str, set[str]] = {
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "ASN": {"OD1"}, "GLN": {"OE1"},
    "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"},
    "HIS": {"ND1", "NE2"}, "HID": {"NE2"}, "HIE": {"ND1"},
    "MET": {"SD"}, "CYS": {"SG"},
}

SC_POS: dict[str, set[str]] = {
    "ARG": {"NE", "NH1", "NH2", "CZ"},
    "LYS": {"NZ"},
    "HIP": {"ND1", "NE2", "CE1"},
}
SC_NEG: dict[str, set[str]] = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
}

# Aromatic ring atom-name tuples (one entry per ring; TRP/HIS fused handled).
AROM_RINGS: dict[str, list[tuple[str, ...]]] = {
    "PHE": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
    "TYR": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
    "HIS": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "HID": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "HIE": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "HIP": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "TRP": [
        ("CG", "CD1", "NE1", "CE2", "CD2"),
        ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    ],
}

# Hydrophobic sidechain carbons
SC_HYDROPHOBE: dict[str, set[str]] = {
    "ALA": {"CB"},
    "VAL": {"CB", "CG1", "CG2"},
    "LEU": {"CB", "CG", "CD1", "CD2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"},
    "PRO": {"CB", "CG", "CD"},
    "MET": {"CB", "CG", "CE"},
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2"},
    "TRP": {"CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "LYS": {"CB", "CG", "CD"},
    "ARG": {"CB", "CG"},
}

# -------- Nucleic acids (RNA / DNA) --------
_NUC_DONOR = {
    "A": {"N6"}, "G": {"N1", "N2"}, "C": {"N4"}, "U": {"N3"}, "T": {"N3"}, "I": {"N1"},
}
_NUC_ACCEPTOR = {
    "A": {"N1", "N3", "N7"},
    "G": {"N3", "N7", "O6"},
    "C": {"O2", "N3"},
    "U": {"O2", "O4"},
    "T": {"O2", "O4"},
    "I": {"N3", "N7", "O6"},
}
_NUC_RINGS = {
    "A": [("N1", "C2", "N3", "C4", "C5", "C6"), ("C4", "C5", "N7", "C8", "N9")],
    "G": [("N1", "C2", "N3", "C4", "C5", "C6"), ("C4", "C5", "N7", "C8", "N9")],
    "I": [("N1", "C2", "N3", "C4", "C5", "C6"), ("C4", "C5", "N7", "C8", "N9")],
    "C": [("N1", "C2", "N3", "C4", "C5", "C6")],
    "U": [("N1", "C2", "N3", "C4", "C5", "C6")],
    "T": [("N1", "C2", "N3", "C4", "C5", "C6")],
}
_NUC_NEG_ATOMS = {"OP1", "OP2", "OP3", "O1P", "O2P", "O3P"}


def _nuc_key(resname: str) -> str | None:
    rn = resname.strip().upper()
    for base in ("A", "G", "C", "U", "T", "I"):
        if rn in (base, "R" + base, "D" + base, base + "3", base + "5"):
            return base
    return None


def nucleic_donor(resname: str) -> set[str]:
    k = _nuc_key(resname)
    return set(_NUC_DONOR.get(k, set())) if k else set()


def nucleic_acceptor(resname: str) -> set[str]:
    k = _nuc_key(resname)
    return set(_NUC_ACCEPTOR.get(k, set())) if k else set()


def nucleic_rings(resname: str) -> list[tuple[str, ...]]:
    k = _nuc_key(resname)
    return list(_NUC_RINGS.get(k, [])) if k else []


def nucleic_neg_atoms() -> set[str]:
    return set(_NUC_NEG_ATOMS)


__all__ = [
    "BACKBONE_DONOR",
    "BACKBONE_ACCEPTOR",
    "SC_DONOR",
    "SC_ACCEPTOR",
    "SC_POS",
    "SC_NEG",
    "AROM_RINGS",
    "SC_HYDROPHOBE",
    "nucleic_donor",
    "nucleic_acceptor",
    "nucleic_rings",
    "nucleic_neg_atoms",
]
