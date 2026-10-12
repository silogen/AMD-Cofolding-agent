# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stereochemistry perception and validation for a prepared ligand."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from interaction_profiler.schemas import StereoInfo

logger = logging.getLogger(__name__)


def _canonical(smiles: str) -> str | None:
    try:
        from rdkit import Chem

        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        return Chem.MolToSmiles(Chem.RemoveHs(m))
    except Exception:
        return None


def analyze_stereo(mol: Any, input_smiles: str | None = None) -> StereoInfo:
    """Perceive stereocentres / double-bond geometry and compare to input SMILES."""
    from rdkit import Chem

    info = StereoInfo()
    with contextlib.suppress(Exception):
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)

    try:
        centers = Chem.FindMolChiralCenters(
            mol, useLegacyImplementation=False, includeUnassigned=True
        )
        info.n_stereocenters = len(centers)
        undefined = [idx for idx, code in centers if code in ("?", None)]
        info.n_undefined = len(undefined)
        info.n_defined = info.n_stereocenters - info.n_undefined
        info.undefined_atom_indices = undefined
    except Exception as exc:
        logger.debug("chiral center perception failed: %s", exc)

    try:
        n_db = 0
        n_db_unspec = 0
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE and not bond.GetIsAromatic():
                a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
                if a1.GetDegree() > 1 and a2.GetDegree() > 1 and bond.IsInRing() is False:
                    n_db += 1
                    if bond.GetStereo() == Chem.BondStereo.STEREONONE:
                        n_db_unspec += 1
        info.n_double_bond_stereo = n_db
        info.n_double_bond_unspecified = n_db_unspec
    except Exception:
        logger.debug("double-bond stereo perception failed", exc_info=True)

    if input_smiles:
        try:
            prepared_canon = _canonical(Chem.MolToSmiles(Chem.RemoveHs(mol)))
            input_canon = _canonical(input_smiles)
            if prepared_canon and input_canon:
                info.matches_input_smiles = prepared_canon == input_canon
        except Exception:
            info.matches_input_smiles = None

    return info


__all__ = ["analyze_stereo"]
