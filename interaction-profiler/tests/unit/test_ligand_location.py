# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Finding the ligand: fused into the chain, under another chain id, and
its bond orders from the hydrogens the receptor preparation adds."""

from __future__ import annotations

import pytest

from interaction_profiler.core import _is_chain_contact
from interaction_profiler.io.structure import LigandSelection
from interaction_profiler.schemas import Interaction


def _line(serial, name, resname, chain, resnum, xyz, element, record="ATOM"):
    x, y, z = xyz
    return (
        f"{record:<6}{serial:>5} {name if len(name) == 4 else ' ' + name:<4} {resname:>3} {chain:1}{resnum:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2}"
    )


def _pdb(atoms) -> str:
    return "\n".join(_line(i + 1, *a) for i, a in enumerate(atoms)) + "\nEND\n"


def _chain_with_adduct():
    """GLY 1, a fused residue XYZ 2 (24 heavy atoms), ALA 3, a small
    modified SEP 4, ALA 5 and a separate ligand LIG 100."""
    atoms = [
        ("N", "GLY", "A", 1, (0.0, 0.0, 0.0), "N"),
        ("CA", "GLY", "A", 1, (1.45, 0.0, 0.0), "C"),
        ("C", "GLY", "A", 1, (2.5, 1.0, 0.0), "C"),
        ("O", "GLY", "A", 1, (2.5, 2.2, 0.0), "O"),
        ("N", "XYZ", "A", 2, (3.83, 1.0, 0.0), "N", "HETATM"),
        ("CA", "XYZ", "A", 2, (5.2, 1.0, 0.0), "C", "HETATM"),
        ("C", "XYZ", "A", 2, (6.2, 2.0, 0.0), "C", "HETATM"),
        ("O", "XYZ", "A", 2, (6.2, 3.2, 0.0), "O", "HETATM"),
    ]
    atoms += [(f"C{k}", "XYZ", "A", 2, (5.2, 1.0, 1.5 * k), "C", "HETATM") for k in range(1, 21)]
    atoms += [
        ("N", "ALA", "A", 3, (7.53, 2.0, 0.0), "N"),
        ("CA", "ALA", "A", 3, (8.9, 2.0, 0.0), "C"),
        ("C", "ALA", "A", 3, (9.9, 3.0, 0.0), "C"),
        ("O", "ALA", "A", 3, (9.9, 4.2, 0.0), "O"),
        ("CB", "ALA", "A", 3, (8.9, 0.5, 0.0), "C"),
        ("N", "SEP", "A", 4, (11.23, 3.0, 0.0), "N", "HETATM"),
        ("CA", "SEP", "A", 4, (12.6, 3.0, 0.0), "C", "HETATM"),
        ("C", "SEP", "A", 4, (13.6, 4.0, 0.0), "C", "HETATM"),
        ("O", "SEP", "A", 4, (13.6, 5.2, 0.0), "O", "HETATM"),
        ("CB", "SEP", "A", 4, (12.6, 1.5, 0.0), "C", "HETATM"),
        ("OG", "SEP", "A", 4, (12.6, 0.1, 0.0), "O", "HETATM"),
        ("P", "SEP", "A", 4, (12.6, -1.4, 0.0), "P", "HETATM"),
        ("O1P", "SEP", "A", 4, (13.9, -2.1, 0.0), "O", "HETATM"),
        ("O2P", "SEP", "A", 4, (11.3, -2.1, 0.0), "O", "HETATM"),
        ("O3P", "SEP", "A", 4, (12.6, -1.4, 1.5), "O", "HETATM"),
        ("N", "ALA", "A", 5, (14.93, 4.0, 0.0), "N"),
        ("CA", "ALA", "A", 5, (16.3, 4.0, 0.0), "C"),
        ("C", "ALA", "A", 5, (17.3, 5.0, 0.0), "C"),
        ("O", "ALA", "A", 5, (17.3, 6.2, 0.0), "O"),
    ]
    atoms += [(f"C{k}", "LIG", "A", 100, (0.0, 4.0, 1.5 * k), "C", "HETATM") for k in range(8)]
    return atoms


@pytest.fixture
def mda():
    return pytest.importorskip("MDAnalysis")


def test_fused_residue_is_a_ligand_and_small_modified_residue_is_not(mda, tmp_path):
    from interaction_profiler.io.structure import load_structure

    path = tmp_path / "adduct.pdb"
    path.write_text(_pdb(_chain_with_adduct()))
    found = {(s.resname, s.resnum): s.in_polymer for s in load_structure(path).detect_ligands()}
    assert found == {("XYZ", 2): True, ("LIG", 100): False}


def test_modified_residue_named_explicitly_is_flagged(mda, tmp_path):
    from interaction_profiler.io.structure import load_structure

    path = tmp_path / "adduct.pdb"
    path.write_text(_pdb(_chain_with_adduct()))
    [sel] = load_structure(path).detect_ligands(resname="SEP")
    assert sel.in_polymer and sel.resnum == 4


def test_ligand_found_under_another_chain_id(mda, tmp_path):
    from interaction_profiler.core import _find_ligand

    atoms = [(f"C{k}", "MI1", "B", 1, (0.0, 0.0, 1.5 * k), "C", "HETATM") for k in range(6)]
    atoms += [(f"C{k}", "MI1", "C", 1, (20.0, 0.0, 1.5 * k), "C", "HETATM") for k in range(6)]
    path = tmp_path / "prepared.pdb"
    path.write_text(_pdb(atoms))
    u = mda.Universe(str(path))
    sel = LigandSelection(chain="A", resname="MI1", resnum=1, n_heavy=6)
    assert _find_ligand(u, sel, near=[20.0, 0.0, 3.75]).n_atoms == 6
    assert set(_find_ligand(u, sel, near=[20.0, 0.0, 3.75]).chainIDs) == {"C"}
    assert set(_find_ligand(u, sel, near=[0.0, 0.0, 3.75]).chainIDs) == {"B"}


def test_chain_contacts_of_a_fused_residue_are_dropped():
    sel = LigandSelection(chain="A", resname="CY0", resnum=797, n_heavy=28, in_polymer=True)

    def contact(ligand_atom, resnum, receptor_atom):
        return Interaction(interaction_type="vdw_contact", ligand_atom_idx=0, ligand_atom_name=ligand_atom,
                           receptor_chain="A", receptor_resname="X", receptor_resnum=resnum,
                           receptor_atom_name=receptor_atom)

    assert _is_chain_contact(contact("N", 745, "NZ"), sel)
    assert _is_chain_contact(contact("CB", 796, "C"), sel)
    assert _is_chain_contact(contact("CB", 798, "N"), sel)
    assert not _is_chain_contact(contact("CB", 798, "CD1"), sel)
    assert not _is_chain_contact(contact("CAF", 790, "O"), sel)


def test_bond_orders_perceived_from_the_hydrogens(mda, tmp_path):
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    from interaction_profiler.prep.ligand import prepare_ligand_from_structure

    mol = Chem.AddHs(Chem.MolFromSmiles("N#Cc1ccncc1"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    for atom in mol.GetAtoms():
        info = Chem.AtomPDBResidueInfo(f"{atom.GetSymbol()}{atom.GetIdx()}".ljust(4)[:4])
        info.SetResidueName("LIG")
        info.SetResidueNumber(1)
        info.SetChainId("A")
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)
    path = tmp_path / "lig.pdb"
    path.write_text(Chem.MolToPDBBlock(mol, flavor=2))
    ag = mda.Universe(str(path)).atoms

    with_h = prepare_ligand_from_structure(ag)
    assert with_h.bond_assignment.startswith("perceived")
    assert Chem.CanonSmiles(with_h.smiles_prepared) == Chem.CanonSmiles("N#Cc1ccncc1")

    heavy_only = prepare_ligand_from_structure(ag.select_atoms("not element H"))
    assert Chem.CanonSmiles(heavy_only.smiles_prepared) != Chem.CanonSmiles("N#Cc1ccncc1")
