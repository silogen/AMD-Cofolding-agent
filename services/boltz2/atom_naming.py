"""Boltz ligand atom-name helpers.

Boltz assigns atom names by: MolFromSmiles → AddHs → CanonicalRankAtoms
and then names each atom as element.upper() + str(canonical_rank + 1).

The **order matters**: AddHs must happen before CanonicalRankAtoms or the
ranks (and thus names) come out differently.  This module replicates the exact
naming so callers can derive contact-constraint atom tokens without running
Boltz itself.
"""

from __future__ import annotations


def boltz_atom_names(smiles: str) -> dict[str, int]:
    """Return {boltz_atom_name: atom_index} for all heavy atoms in smiles.

    The index is the RDKit atom index (0-based) in the AddHs molecule.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles!r}")
    mol = AllChem.AddHs(mol)
    order = AllChem.CanonicalRankAtoms(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    names: dict[str, int] = {}
    for atom, rank in zip(mol.GetAtoms(), order):
        if atom.GetAtomicNum() == 1:  # skip hydrogen
            continue
        name = atom.GetSymbol().upper() + str(rank + 1)
        names[name] = atom.GetIdx()
    return names


def heavy_atom_count(smiles: str) -> int:
    """Return the number of non-hydrogen atoms in smiles."""
    from rdkit import Chem  # noqa: PLC0415
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles!r}")
    return mol.GetNumHeavyAtoms()


def mcs_atom_map(
    parent_smiles: str,
    child_smiles: str,
) -> dict[int, int]:
    """Map parent heavy-atom indices to child heavy-atom indices via MCS (E6).

    Returns {parent_idx: child_idx}.  Indices are RDKit atom indices in the
    respective molecules (without explicit H).
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import rdFMCS  # noqa: PLC0415

    parent = Chem.MolFromSmiles(parent_smiles)
    child = Chem.MolFromSmiles(child_smiles)
    if parent is None or child is None:
        raise ValueError("Cannot parse one or both SMILES for MCS.")

    result = rdFMCS.FindMCS(
        [parent, child],
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=False,
        timeout=30,
    )
    if result.numAtoms == 0:
        return {}

    query = Chem.MolFromSmarts(result.smartsString)
    parent_match = parent.GetSubstructMatch(query)
    child_match = child.GetSubstructMatch(query)

    return {p: c for p, c in zip(parent_match, child_match)}
