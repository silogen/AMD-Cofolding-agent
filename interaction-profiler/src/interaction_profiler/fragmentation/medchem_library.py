# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Curated SMARTS library to enrich AccFG coverage.

AccFG provides a strong hierarchical functional-group taxonomy but can
miss some medchem-relevant motifs (covalent warheads, common bioisosteres,
linker/handle groups). These SMARTS supplement it. Each entry is applied
with ``GetSubstructMatches`` and merged into the same ``Fragment`` schema,
preserving atom indices for the interaction join.

Patterns are intentionally specific; overlaps with AccFG are fine (the
join layer tolerates an atom belonging to multiple fragments).
"""

from __future__ import annotations

# name -> SMARTS
MEDCHEM_SMARTS: dict[str, str] = {
    # ---- H-bonding / polar groups often central to binding ----
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "carboxylate": "[CX3](=O)[O-]",
    "primary_amide": "[NX3H2][CX3]=[OX1]",
    "secondary_amide": "[NX3H1][CX3]=[OX1]",
    "tertiary_amide": "[NX3H0]([#6])[CX3]=[OX1]",
    "urea": "[NX3][CX3](=[OX1])[NX3]",
    "carbamate": "[NX3][CX3](=[OX1])[OX2]",
    "guanidine": "[NX3][CX3](=[NX2,NX3+])[NX3]",
    "amidine": "[NX3][CX3]=[NX2,NX3+]",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])[NX3]",
    "sulfone": "[SX4](=[OX1])(=[OX1])([#6])[#6]",
    "sulfonate": "[SX4](=[OX1])(=[OX1])[OX1,OX2]",
    "phosphate": "[PX4](=[OX1])([OX2,OX1])([OX2,OX1])[OX2,OX1]",
    "phosphonate": "[PX4](=[OX1])([#6])([OX2,OX1])[OX2,OX1]",
    "nitro": "[NX3+](=[OX1])[O-]",
    "nitrile": "[NX1]#[CX2]",
    "hydroxyl": "[OX2H][CX4]",
    "phenol": "[OX2H][c]",
    "1,2_diol": "[OX2H][CX4][CX4][OX2H]",
    "ether": "[OD2]([#6])[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "primary_amine": "[NX3;H2][CX4]",
    "secondary_amine": "[NX3;H1]([CX4])[CX4]",
    "tertiary_amine": "[NX3;H0]([CX4])([CX4])[CX4]",
    # ---- halogens ----
    "aryl_halide": "[c][F,Cl,Br,I]",
    "alkyl_halide": "[CX4][F,Cl,Br,I]",
    "trifluoromethyl": "[CX4]([F])([F])[F]",
    # ---- covalent warheads ----
    "acrylamide_warhead": "[CX3]=[CX3][CX3](=[OX1])[NX3]",
    "vinyl_sulfone_warhead": "[CX3]=[CX3][SX4](=O)(=O)",
    "chloroacetamide_warhead": "[Cl][CX4][CX3](=[OX1])[NX3]",
    "nitrile_warhead": "[CX4,c][CX2]#[NX1]",
    "boronic_acid": "[BX3]([OX2H])[OX2H]",
    "epoxide": "[OX2r3]1[CX4r3][CX4r3]1",
    "aldehyde_warhead": "[CX3H1](=O)[#6,#7]",
    # ---- common heterocycles (5-membered) ----
    "imidazole": "c1cnc[nH]1",
    "pyrazole": "c1cc[nH]n1",
    "triazole_1_2_3": "c1cnn[nH]1",
    "triazole_1_2_4": "c1n[nH]cn1",
    "tetrazole": "c1nnn[nH]1",
    "oxazole": "c1ocnc1",
    "isoxazole": "c1cc(on1)",
    "thiazole": "c1cscn1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "pyrrole": "c1cc[nH]c1",
    # ---- common heterocycles (6-membered) ----
    "pyridine": "c1ccncc1",
    "pyrimidine": "c1cncnc1",
    "pyrazine": "c1cnccn1",
    "pyridazine": "c1ccnnc1",
    "triazine": "c1ncncn1",
    "morpholine": "C1COCCN1",
    "piperazine": "C1CNCCN1",
    "piperidine": "C1CCNCC1",
    "benzene": "c1ccccc1",
    # ---- fused / privileged scaffolds ----
    "purine": "c1ncc2ncnc2n1",
    "indole": "c1ccc2[nH]ccc2c1",
    "benzimidazole": "c1ccc2[nH]cnc2c1",
    "quinoline": "c1ccc2ncccc2c1",
    "naphthalene": "c1ccc2ccccc2c1",
}

_COMPILED: list[tuple[str, str, object]] | None = None


def compiled_patterns():
    """Return ``list[(name, smarts, rdkit_mol)]`` of compiled SMARTS (cached)."""
    global _COMPILED
    if _COMPILED is not None:
        return _COMPILED

    from rdkit import Chem

    out = []
    for name, smarts in MEDCHEM_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None:
            out.append((name, smarts, patt))
    _COMPILED = out
    return out


__all__ = ["MEDCHEM_SMARTS", "compiled_patterns"]
