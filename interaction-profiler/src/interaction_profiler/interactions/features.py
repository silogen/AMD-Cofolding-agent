# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pharmacophore feature perception for ligand and receptor.

Ligand features come from the prepared RDKit Mol (explicit Hs, correct
bond orders). Receptor features come from an RDKit Mol built from the
cropped pocket PDB, which carries per-atom ``AtomPDBResidueInfo`` so each
site knows its residue name / number / chain / atom name / backbone-vs-
sidechain classification.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from interaction_profiler.io.structure import is_backbone, residue_kind

logger = logging.getLogger(__name__)

_FACTORY = None


def _factory():
    global _FACTORY
    if _FACTORY is None:
        from rdkit import RDConfig
        from rdkit.Chem import ChemicalFeatures

        fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        _FACTORY = ChemicalFeatures.BuildFeatureFactory(fdef)
    return _FACTORY


HALOGENS = {"CL", "BR", "I", "F"}


@dataclass
class Donor:
    heavy_idx: int
    heavy_xyz: np.ndarray
    h_xyz: list[np.ndarray] = field(default_factory=list)


@dataclass
class Aromatic:
    centroid: np.ndarray
    normal: np.ndarray
    atom_idxs: list[int]


@dataclass
class Charged:
    center: np.ndarray
    atom_idxs: list[int]


@dataclass
class Halogen:
    x_idx: int
    x_xyz: np.ndarray
    c_xyz: np.ndarray


@dataclass
class MetalSite:
    xyz: np.ndarray
    meta: dict


@dataclass
class WaterSite:
    xyz: np.ndarray
    meta: dict


@dataclass
class LigandFeatures:
    coords: np.ndarray
    elements: list[str]
    donors: list[Donor] = field(default_factory=list)
    acceptors: list[int] = field(default_factory=list)
    cations: list[Charged] = field(default_factory=list)
    anions: list[Charged] = field(default_factory=list)
    hydrophobes: list[int] = field(default_factory=list)
    aromatics: list[Aromatic] = field(default_factory=list)
    halogens: list[Halogen] = field(default_factory=list)
    chalcogens: list[Halogen] = field(default_factory=list)  # S/Se with C neighbour (x=chalcogen)
    amides: list[Aromatic] = field(default_factory=list)      # planar amide groups (centroid/normal)


@dataclass
class ReceptorFeatures:
    coords: np.ndarray
    meta: list[dict]
    donors: list[Donor] = field(default_factory=list)
    acceptors: list[int] = field(default_factory=list)
    cations: list[Charged] = field(default_factory=list)
    anions: list[Charged] = field(default_factory=list)
    hydrophobes: list[int] = field(default_factory=list)
    aromatics: list[Aromatic] = field(default_factory=list)
    sulfurs: list[int] = field(default_factory=list)  # Met SD / Cys SG (for S-arene)
    metals: list[MetalSite] = field(default_factory=list)
    waters: list[WaterSite] = field(default_factory=list)


def _conf_coords(mol: Any) -> np.ndarray:
    conf = mol.GetConformer()
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float64)


def _bonded_h(mol: Any, idx: int, coords: np.ndarray) -> list[np.ndarray]:
    atom = mol.GetAtomWithIdx(idx)
    return [coords[n.GetIdx()] for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]


def _aromatic_rings(mol: Any, coords: np.ndarray) -> list[Aromatic]:
    rings: list[Aromatic] = []
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        try:
            if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                continue
        except Exception:
            continue
        pts = coords[list(ring)]
        centroid = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centroid)
        normal = vt[2]
        rings.append(Aromatic(centroid=centroid, normal=normal, atom_idxs=list(ring)))
    return rings


def perceive_ligand(mol: Any) -> LigandFeatures:

    coords = _conf_coords(mol)
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    feats = LigandFeatures(coords=coords, elements=elements)

    try:
        chem_feats = _factory().GetFeaturesForMol(mol)
    except Exception as exc:
        logger.debug("ligand feature factory failed: %s", exc)
        chem_feats = []

    for f in chem_feats:
        fam = f.GetFamily()
        ids = list(f.GetAtomIds())
        if fam == "Donor":
            for aid in ids:
                feats.donors.append(Donor(aid, coords[aid], _bonded_h(mol, aid, coords)))
        elif fam == "Acceptor":
            feats.acceptors.extend(ids)
        elif fam == "PosIonizable":
            feats.cations.append(Charged(center=coords[ids].mean(axis=0), atom_idxs=ids))
        elif fam == "NegIonizable":
            feats.anions.append(Charged(center=coords[ids].mean(axis=0), atom_idxs=ids))
        elif fam in ("Hydrophobe", "LumpedHydrophobe"):
            feats.hydrophobes.extend(ids)

    for atom in mol.GetAtoms():
        sym = atom.GetSymbol().upper()
        if sym in HALOGENS:
            carbons = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 6]
            if carbons:
                feats.halogens.append(
                    Halogen(atom.GetIdx(), coords[atom.GetIdx()], coords[carbons[0].GetIdx()])
                )
        elif sym in ("S", "SE"):
            # chalcogen with a carbon neighbour -> sigma-hole donor (S/Se-arene, chalcogen bond)
            carbons = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 6]
            if carbons:
                feats.chalcogens.append(
                    Halogen(atom.GetIdx(), coords[atom.GetIdx()], coords[carbons[0].GetIdx()])
                )

    feats.aromatics = _aromatic_rings(mol, coords)
    feats.amides = _amide_planes(mol, coords)
    return feats


def _amide_planes(mol: Any, coords: np.ndarray) -> list[Aromatic]:
    """Planar amide groups (N-C=O) as centroid + plane normal, for amide-pi."""
    from rdkit import Chem

    patt = Chem.MolFromSmarts("[NX3][CX3]=[OX1]")
    out: list[Aromatic] = []
    if patt is None:
        return out
    for match in mol.GetSubstructMatches(patt):
        pts = coords[list(match)]
        centroid = pts.mean(axis=0)
        try:
            _, _, vt = np.linalg.svd(pts - centroid)
            normal = vt[2]
        except Exception:
            continue
        out.append(Aromatic(centroid=centroid, normal=normal, atom_idxs=list(match)))
    return out


def _guess_element(name: str, given: str) -> str:
    if given and given.strip():
        return given.strip().capitalize()
    nm = (name or "").strip()
    if not nm:
        return ""
    # PDB atom names: element usually first 1-2 alpha chars, digits stripped
    alpha = "".join(c for c in nm if c.isalpha())
    if len(alpha) >= 2 and alpha[:2].upper() in ("CL", "BR", "NA", "MG", "ZN", "FE", "MN", "CA", "CU", "NI", "CO"):
        return alpha[:2].capitalize()
    return alpha[:1].upper() if alpha else ""


def _mda_meta(atom: Any, chain: str) -> dict:
    name = (getattr(atom, "name", "") or "").strip()
    resname = (getattr(atom, "resname", "") or "").strip()
    element = _guess_element(name, getattr(atom, "element", "") or getattr(atom, "type", ""))
    kind = residue_kind(resname)
    if kind in ("protein", "rna", "dna", "nucleic"):
        moiety = "backbone" if is_backbone(kind, name) else "sidechain"
    elif kind == "ion":
        moiety = "ion"
    elif kind == "water":
        moiety = "water"
    else:
        moiety = None
    return {
        "name": name,
        "resname": resname,
        "resnum": int(getattr(atom, "resid", 0)),
        "chain": chain,
        "reskind": kind,
        "moiety": moiety,
        "element": element,
    }


def _chain_of_atom(atom: Any) -> str:
    for attr in ("chainID", "segid"):
        v = getattr(atom, attr, None)
        if v:
            return str(v).strip()
    return ""


def perceive_receptor(ag: Any) -> ReceptorFeatures:
    """Perceive receptor features from an MDAnalysis pocket AtomGroup.

    Uses residue templates (robust for standard protein/nucleic residues)
    plus the actual explicit hydrogens present in the prepared structure.
    Metals/waters are supplied separately by the caller.
    """
    from interaction_profiler.interactions import receptor_templates as T

    coords = np.ascontiguousarray(ag.positions, dtype=np.float64)
    atoms = list(ag.atoms)
    ix_to_local = {a.ix: i for i, a in enumerate(atoms)}
    meta = [_mda_meta(a, _chain_of_atom(a)) for a in atoms]
    feats = ReceptorFeatures(coords=coords, meta=meta)

    h_local = [i for i, m in enumerate(meta) if m["element"].upper() == "H"]
    h_xyz = coords[h_local] if h_local else np.empty((0, 3))

    def bonded_h(idx: int) -> list[np.ndarray]:
        if h_xyz.shape[0] == 0:
            return []
        d = np.linalg.norm(h_xyz - coords[idx], axis=1)
        return [h_xyz[j] for j in np.where(d <= 1.3)[0]]

    for res in ag.residues:
        rn = (res.resname or "").strip().upper()
        kind = residue_kind(rn)
        if kind not in ("protein", "rna", "dna", "nucleic"):
            continue
        name_to_local: dict[str, int] = {}
        for a in res.atoms:
            li = ix_to_local.get(a.ix)
            if li is not None:
                name_to_local[(a.name or "").strip().upper()] = li

        if kind == "protein":
            donor_names = set(T.SC_DONOR.get(rn, set()))
            if rn != "PRO":
                donor_names |= T.BACKBONE_DONOR
            acc_names = set(T.SC_ACCEPTOR.get(rn, set())) | T.BACKBONE_ACCEPTOR
            pos_names = T.SC_POS.get(rn, set())
            neg_names = T.SC_NEG.get(rn, set())
            rings = T.AROM_RINGS.get(rn, [])
            hyd_names = T.SC_HYDROPHOBE.get(rn, set())
        else:
            donor_names = T.nucleic_donor(rn)
            acc_names = T.nucleic_acceptor(rn)
            pos_names = set()
            neg_names = T.nucleic_neg_atoms()
            rings = T.nucleic_rings(rn)
            hyd_names = set()

        for nm in donor_names:
            li = name_to_local.get(nm)
            if li is not None:
                feats.donors.append(Donor(li, coords[li], bonded_h(li)))
        for nm in acc_names:
            li = name_to_local.get(nm)
            if li is not None:
                feats.acceptors.append(li)
        pos_sel = [name_to_local[n] for n in pos_names if n in name_to_local]
        if pos_sel:
            feats.cations.append(Charged(center=coords[pos_sel].mean(axis=0), atom_idxs=pos_sel))
        neg_sel = [name_to_local[n] for n in neg_names if n in name_to_local]
        if neg_sel:
            feats.anions.append(Charged(center=coords[neg_sel].mean(axis=0), atom_idxs=neg_sel))
        for nm in hyd_names:
            li = name_to_local.get(nm)
            if li is not None:
                feats.hydrophobes.append(li)
        # Met/Cys sulfur for sulfur-arene interactions
        if kind == "protein":
            for nm in ("SD", "SG"):
                li = name_to_local.get(nm)
                if li is not None:
                    feats.sulfurs.append(li)
        for ring in rings:
            ring_idx = [name_to_local[n] for n in ring if n in name_to_local]
            if len(ring_idx) >= 4:
                pts = coords[ring_idx]
                centroid = pts.mean(axis=0)
                _, _, vt = np.linalg.svd(pts - centroid)
                feats.aromatics.append(Aromatic(centroid=centroid, normal=vt[2], atom_idxs=ring_idx))

    return feats


__all__ = [
    "LigandFeatures",
    "ReceptorFeatures",
    "Donor",
    "Aromatic",
    "Charged",
    "Halogen",
    "MetalSite",
    "WaterSite",
    "perceive_ligand",
    "perceive_receptor",
]
