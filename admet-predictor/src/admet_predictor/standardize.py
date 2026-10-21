#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Structure standardization used as an INTERNAL helper for unit conversion.

Protocol (order per user request, lightly corrected):
  1. parse
  2. strip salts/cosolvents/surfactants (exhaustive dict) + keep largest organic fragment
  3. stereo: save -> kekulize (remove aromaticity) -> re-perceive aromaticity -> restore
     -> assign only UNspecified stereo (never overwrite)
  4. Normalizer (functional-group normalization)
  5. canonical/major tautomer
  6. ionization at pH 7.4 (dimorphite-dl)
  7. quaternary/trivalent-N charge sanity check at pH 7.4
Also returns the NEUTRAL desalted parent + its MW (for mass->molar unit conversion).

Results are cached to disk (keyed by the input SMILES) because molecules repeat
heavily across datasets. The curated OUTPUT keeps original SMILES; std_smiles is a
helper column only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "adme_data_curated" / "_cache"
CACHE_PATH = CACHE_DIR / "standardize_cache.parquet"

# --------------------------------------------------------------------------
# exhaustive salt / cosolvent / surfactant dictionary (neutral & ionic forms)
# --------------------------------------------------------------------------
_SALT_SOLVENT_SMILES = [
    # inorganic counterions / acids
    "[Cl-]", "Cl", "[Br-]", "Br", "[I-]", "I", "[F-]", "F",
    "[Na+]", "[K+]", "[Li+]", "[Ca+2]", "[Mg+2]", "[Zn+2]", "[Al+3]", "[NH4+]",
    "O=S(=O)([O-])[O-]", "OS(=O)(=O)O", "O=[N+]([O-])[O-]", "O[N+](=O)[O-]",
    "OP(=O)(O)O", "O=P([O-])([O-])[O-]", "[OH-]", "O=C([O-])[O-]", "OC(=O)O",
    "[Br]", "[Cl]", "[I]",
    # organic acids / counterions
    "CC(=O)O", "CC(=O)[O-]",                         # acetic / acetate
    "OC(=O)C(=O)O", "[O-]C(=O)C(=O)[O-]",            # oxalic
    "OC(=O)/C=C/C(=O)O", "OC(=O)/C=C\\C(=O)O",       # fumaric / maleic
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",                    # citric
    "OC(=O)C(O)C(O)C(=O)O",                          # tartaric
    "OC(=O)CCC(=O)O",                                # succinic
    "OC=O", "[O-]C=O",                               # formic / formate
    "CS(=O)(=O)O", "CS(=O)(=O)[O-]",                 # methanesulfonic / mesylate
    "Cc1ccc(cc1)S(=O)(=O)O", "Cc1ccc(cc1)S(=O)(=O)[O-]",  # tosylate
    "OC(=O)C(F)(F)F", "[O-]C(=O)C(F)(F)F",           # TFA
    "OC(=O)CO",                                      # glycolic
    "OS(=O)(=O)O",                                   # sulfuric
    "OC(=O)c1ccccc1",                                # benzoic
    "O=S(=O)(O)O",
    # solvents
    "O",                                             # water
    "CO", "CCO", "CC(C)O", "CCCO", "OCCO", "OCC(O)CO",  # MeOH/EtOH/iPrOH/PrOH/EG/glycerol
    "CC(=O)C", "CC#N", "CS(C)=O", "CN(C)C=O",        # acetone/MeCN/DMSO/DMF
    "CCOC(C)=O", "CCOCC", "C1CCOC1", "ClCCl", "ClC(Cl)Cl", "ClC(Cl)(Cl)Cl",
    "CCCCCC", "Cc1ccccc1", "C1CCCCC1", "c1ccccc1",   # hexane/toluene/cyclohexane/benzene
    # common surfactants (monomeric where representable)
    "CCCCCCCCCCCCOS(=O)(=O)[O-]", "CCCCCCCCCCCCOS(=O)(=O)O",  # dodecyl sulfate (SDS)
    "CCCCCCCCCCCCCC(=O)[O-]",                        # myristate-ish
]


def _canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


_SALT_SET = {c for c in (_canon(s) for s in _SALT_SOLVENT_SMILES) if c}

_normalizer = rdMolStandardize.Normalizer()
_uncharger = rdMolStandardize.Uncharger()
_tautomer = rdMolStandardize.TautomerEnumerator()
_tautomer.SetMaxTautomers(200)      # bound combinatorial blow-ups (hang guard)
_tautomer.SetMaxTransforms(500)
_largest = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)

# molecules bigger than this skip the expensive tautomer/dimorphite steps (they
# are not modelling targets and can hang the standardizer)
HEAVY_ATOM_LIMIT = 70


def _strip_salts_keep_major(mol: Chem.Mol) -> tuple[Chem.Mol, bool]:
    frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    multi = len(frags) > 1
    if not multi:
        return mol, False
    kept = [f for f in frags if Chem.MolToSmiles(f) not in _SALT_SET]
    pool = kept or frags  # if everything looked like a salt, fall back to all frags
    # largest by heavy-atom count, tie-break by MW
    best = max(pool, key=lambda f: (f.GetNumHeavyAtoms(), Descriptors.MolWt(f)))
    return best, multi


def _preserve_stereo_rearomatize(mol: Chem.Mol) -> Chem.Mol:
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
        Chem.SanitizeMol(mol)  # re-perceive aromaticity
        # assign stereo without overwriting already-specified centers/bonds
        Chem.AssignStereochemistry(mol, cleanIt=False, force=False, flagPossibleStereoCenters=True)
    except Exception:  # noqa: BLE001
        pass
    return mol


def _fix_nitrogen_charges(mol: Chem.Mol) -> bool:
    """Ensure tetravalent (quaternary) N carries +1 at pH 7.4; fix obvious cases."""
    fixed = False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "N":
            val = atom.GetTotalValence()
            chg = atom.GetFormalCharge()
            if val == 4 and chg == 0:
                atom.SetFormalCharge(+1)
                fixed = True
    if fixed:
        try:
            Chem.SanitizeMol(mol)
        except Exception:  # noqa: BLE001
            pass
    return fixed


def standardize_one(smiles: str) -> dict:
    out = {"input_smiles": smiles, "std_smiles": None, "parent_smiles": None,
           "parent_mw": None, "flags": {}, "ok": False, "error": ""}
    flags = {"parse_fail": False, "multi_fragment": False, "tautomer_changed": False,
             "protonation_changed": False, "n_fixed": False}
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            flags["parse_fail"] = True
            out["flags"] = flags
            out["error"] = "parse_fail"
            return out
        # 2. desalt + largest fragment
        mol, multi = _strip_salts_keep_major(mol)
        flags["multi_fragment"] = multi
        # 3. stereo-preserving re-aromatization
        mol = _preserve_stereo_rearomatize(mol)
        # 4. normalize
        mol = _normalizer.normalize(mol)
        heavy = mol.GetNumHeavyAtoms()
        # 5. canonical tautomer (skip for very large molecules -> hang guard)
        if heavy <= HEAVY_ATOM_LIMIT:
            pre_taut = Chem.MolToSmiles(mol)
            try:
                mol = _tautomer.Canonicalize(mol)
                flags["tautomer_changed"] = Chem.MolToSmiles(mol) != pre_taut
            except Exception:  # noqa: BLE001
                flags["tautomer_error"] = True
        else:
            flags["heavy_skipped"] = True
        # neutral desalted parent (for MW used in mass->molar conversion)
        parent = _uncharger.uncharge(Chem.Mol(mol))
        out["parent_smiles"] = Chem.MolToSmiles(parent)
        out["parent_mw"] = float(Descriptors.MolWt(parent))
        # 6. ionization at pH 7.4 (skip for very large molecules)
        neutral_smi = Chem.MolToSmiles(mol)
        prot_smi = neutral_smi
        if heavy <= HEAVY_ATOM_LIMIT:
            try:
                from dimorphite_dl import protonate_smiles
                # precision=0 -> single DOMINANT microstate at pH 7.4 (deterministic);
                # the default precision=1 returns several near-pKa microstates whose order
                # is not stable across processes, so prot[0] would be non-reproducible.
                prot = protonate_smiles(neutral_smi, ph_min=7.4, ph_max=7.4, precision=0.0)
                prot_smi = sorted(prot)[0] if prot else neutral_smi
            except Exception:  # noqa: BLE001
                prot_smi = neutral_smi
        pmol = Chem.MolFromSmiles(prot_smi) or mol
        flags["protonation_changed"] = Chem.MolToSmiles(pmol) != neutral_smi
        # 7. quaternary-N charge check
        flags["n_fixed"] = _fix_nitrogen_charges(pmol)
        out["std_smiles"] = Chem.MolToSmiles(pmol)
        out["ok"] = True
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    out["flags"] = flags
    return out


def standardize_many(smiles_list, use_cache: bool = True, verbose: bool = True) -> dict:
    """Return {input_smiles: result dict} for all unique SMILES, using a disk cache."""
    uniq = [s for s in dict.fromkeys(str(x) for x in smiles_list)]
    cache: dict[str, dict] = {}
    if use_cache and CACHE_PATH.exists():
        cdf = pd.read_parquet(CACHE_PATH)
        for _, r in cdf.iterrows():
            cache[r["input_smiles"]] = {
                "input_smiles": r["input_smiles"], "std_smiles": r["std_smiles"],
                "parent_smiles": r["parent_smiles"], "parent_mw": r["parent_mw"],
                "flags": json.loads(r["flags"]) if isinstance(r["flags"], str) else {},
                "ok": bool(r["ok"]), "error": r["error"],
            }
    missing = [s for s in uniq if s not in cache]
    if verbose and missing:
        print(f"  [standardize] {len(missing)} new / {len(uniq)} unique "
              f"({len(cache)} cached)", flush=True)

    def _flush():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        rows = [{**{k: v for k, v in r.items() if k != "flags"},
                 "flags": json.dumps(r["flags"])} for r in cache.values()]
        tmp = CACHE_PATH.with_suffix(".parquet.tmp")
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        tmp.replace(CACHE_PATH)

    flush_every = 2000
    for i, s in enumerate(missing):
        cache[s] = standardize_one(s)
        if verbose and (i + 1) % flush_every == 0:
            print(f"    standardized {i+1}/{len(missing)}", flush=True)
            if use_cache:
                _flush()               # incremental: persist + make progress visible
    if use_cache and missing:
        _flush()
    return {s: cache[s] for s in uniq}


if __name__ == "__main__":
    import sys
    for smi in sys.argv[1:] or ["CC(=O)O.[Na+]", "c1ccccc1C(=O)O", "C[N+](C)(C)C"]:
        print(smi, "->", standardize_one(smi))
