"""Stage 6 — Protonation and Residue States.

Assigns hydrogen atoms, histidine tautomers (HID/HIE/HIP), and other
protonation states using pdb2pqr3 + PROPKA, pdbfixer_basic, or explicit
user overrides. Output is written using OpenMM's PDBFile.writeFile()
to ensure AMBER-compatible atom names that the interaction-profiler's
own PDBFixer will recognise correctly (preventing double-protonation).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import gemmi

from structure_prep.models import (
    CysState,
    HisState,
    Stage6Profile,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Residue-name sets
# ---------------------------------------------------------------------------

_HIS_NAMES: frozenset[str] = frozenset({"HIS", "HID", "HIE", "HIP"})
_CYS_NAMES: frozenset[str] = frozenset({"CYS", "CYX", "CYM"})
_ACIDIC_NAMES: frozenset[str] = frozenset({"ASP", "GLU", "ASH", "GLH"})


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    structure: gemmi.Structure,
    profile: Stage6Profile,
    structural_metals: list,  # list[RetainedMetal] from RoleMap
    job_dir: Path,
) -> tuple[gemmi.Structure, dict]:
    """Stage 6: Protonation and Residue States.

    Returns (protonated_structure, stage6_report_dict).

    stage6_report_dict keys:
        protonation_method: str
        ph: float
        n_h_atoms_added: int
        histidines: list[HisState]
        acidic_residues: list[dict]
        cysteines: list[CysState]
        metals_coordination: list[dict]
    """
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    method = profile.protonation_method

    if method == "none":
        return structure, _empty_report(profile)

    elif method == "pdb2pqr_propka":
        return _run_pdb2pqr(structure, profile, structural_metals, job_dir)

    elif method == "pdbfixer_basic":
        return _run_pdbfixer_basic(structure, profile, job_dir)

    elif method == "user_explicit":
        protonated, _ = _apply_user_overrides(structure, profile)
        his_states = _collect_his_states_from_structure(protonated, {}, profile)
        cys_states = _collect_cys_states(protonated, profile)
        n_h = _count_h_atoms(protonated)
        return protonated, {
            "protonation_method": "user_explicit",
            "ph": profile.ph,
            "n_h_atoms_added": n_h,
            "histidines": his_states,
            "acidic_residues": [],
            "cysteines": cys_states,
            "metals_coordination": [],
        }

    else:
        logger.warning(
            "Stage 6: unknown protonation_method %r; returning structure unchanged.", method
        )
        return structure, _empty_report(profile)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _empty_report(profile: Stage6Profile) -> dict:
    """Return a report dict with all fields zeroed/empty."""
    return {
        "protonation_method": profile.protonation_method,
        "ph": profile.ph,
        "n_h_atoms_added": 0,
        "histidines": [],
        "acidic_residues": [],
        "cysteines": [],
        "metals_coordination": [],
    }


def _count_h_atoms(structure: gemmi.Structure) -> int:
    """Count hydrogen atoms in model 0."""
    h = gemmi.Element("H")
    return sum(
        1
        for chain in structure[0]
        for res in chain
        for atom in res
        if atom.element == h
    )


# ---------------------------------------------------------------------------
# pdb2pqr + PROPKA implementation
# ---------------------------------------------------------------------------

def _run_pdb2pqr(
    structure: gemmi.Structure,
    profile: Stage6Profile,
    structural_metals: list,
    job_dir: Path,
) -> tuple[gemmi.Structure, dict]:
    """Protonate using pdb2pqr3 + PROPKA at the requested pH."""
    tmp_in = str(job_dir / "pdb2pqr_input.pdb")
    tmp_out = str(job_dir / "pdb2pqr_output.pdb")

    structure.write_pdb(tmp_in)

    success = _invoke_pdb2pqr(tmp_in, tmp_out, profile, job_dir)

    if not success or not Path(tmp_out).exists():
        logger.warning(
            "Stage 6: pdb2pqr_propka could not process this structure "
            "(likely non-standard residues or Fragalysis PDB format incompatibility); "
            "falling back to pdbfixer_basic. His tautomers will use default HIE rule."
        )
        result_st, info = _run_pdbfixer_basic(structure, profile, job_dir)
        info["warnings"] = info.get("warnings", []) + [
            "pdb2pqr_propka requested but fell back to pdbfixer_basic: "
            "pdb2pqr failed to process this structure. "
            "Histidine protonation states are assigned by default rule (HIE), "
            "not by PROPKA pKa calculation."
        ]
        return result_st, info

    # PROPKA file is written to <input_stem>.propka in the job directory.
    propka_stem = Path(tmp_in).stem  # "pdb2pqr_input"
    propka_path = str(job_dir / (propka_stem + ".propka"))
    pka_table = _parse_propka(propka_path)
    if not pka_table:
        # Some versions write to CWD; try a fallback location
        propka_cwd = Path(propka_stem + ".propka")
        if propka_cwd.exists():
            pka_table = _parse_propka(str(propka_cwd))

    protonated = gemmi.read_structure(tmp_out)

    # Apply user overrides (highest priority — after pdb2pqr)
    protonated, _ = _apply_user_overrides(protonated, profile)

    # Apply metal-coordination tautomer forcing for HIS
    his_states, metals_coord = _apply_metal_coordination_overrides(
        protonated, structural_metals, profile, pka_table
    )

    # Collect other residue states
    cys_states = _collect_cys_states(protonated, profile)
    acidic = _collect_acidic_states(protonated, pka_table, profile.ph)

    n_h = _count_h_atoms(protonated)

    return protonated, {
        "protonation_method": "pdb2pqr_propka",
        "ph": profile.ph,
        "n_h_atoms_added": n_h,
        "histidines": his_states,
        "acidic_residues": acidic,
        "cysteines": cys_states,
        "metals_coordination": metals_coord,
    }


def _invoke_pdb2pqr(
    input_path: str,
    output_path: str,
    profile: Stage6Profile,
    job_dir: Path,
) -> bool:
    """Attempt to run pdb2pqr3 via Python API, falling back to subprocess.

    Returns True on apparent success, False on failure.
    """
    import argparse

    # ---- Try the pdb2pqr3 Python API ----
    # pdb2pqr ≥3 writes a PQR (charges+radii) file, not a PDB.
    # We pass --pdb-output to also get a hydrogen-added PDB that gemmi can read.
    pdb_output_path = str(Path(output_path).with_suffix(".pdb2pqr.pdb"))
    try:
        from pdb2pqr.main import main_driver as _pdb2pqr_main  # type: ignore

        args = argparse.Namespace(
            input_path=input_path,
            output_pqr=output_path,         # positional in pdb2pqr ≥3
            ff="AMBER",
            ffout=None,
            with_ph=profile.ph,
            titration_state_method="propka",
            ph_calc_method="propka",        # keep for backward compat
            propka_input=None,
            usernames=None,
            userff=None,
            clean=False,
            nodebump=False,
            noopt=False,
            keep_chain=True,
            assign_only=False,
            drop_water=False,
            include_header=True,
            debump=True,
            opt=True,
            quiet=True,
            verbose=False,
            log_level="WARNING",
            pdb_output=pdb_output_path,     # request PDB alongside PQR
            ligand=None,
            whitespace=False,
            neutraln=False,
            neutralc=False,
            apbs_input=None,
            filenames=None,
            reference=None,
            chains=None,
            titrate_only=None,
            thermophiles=None,
            alignment=None,
            mutations=None,
            parameters=None,
            window=None,
            grid=None,
            mutator=None,
            mutator_options=None,
            show_debug=False,
            oldrun=False,
            do_old_style_multiconf=False,
            protonate_all=False,
        )
        try:
            _pdb2pqr_main(args)
            # Success: return the PDB output (gemmi can't parse PQR)
            if Path(pdb_output_path).exists():
                # swap output_path to point at the PDB file the rest of the
                # caller expects
                import shutil  # noqa: PLC0415
                shutil.copy2(pdb_output_path, output_path)
                return True
            return Path(output_path).exists()
        except SystemExit:
            if Path(pdb_output_path).exists():
                import shutil  # noqa: PLC0415
                shutil.copy2(pdb_output_path, output_path)
                return True
            return Path(output_path).exists()
        except Exception as exc:
            logger.warning("Stage 6: pdb2pqr Python API call failed: %s; trying subprocess.", exc)

    except ImportError:
        logger.warning("Stage 6: pdb2pqr not importable via Python API; trying subprocess.")

    # ---- Subprocess fallback ----
    try:
        result = subprocess.run(
            [
                "pdb2pqr",          # binary name in pdb2pqr ≥3 (was pdb2pqr3)
                "--ff", "AMBER",
                "--titration-state-method", "propka",
                "--with-ph", str(profile.ph),
                "--keep-chain",
                "--include-header",
                "--pdb-output", pdb_output_path,
                input_path,
                output_path,
            ],
            capture_output=True,
            text=True,
            cwd=str(job_dir),
            timeout=600,
        )
        if result.returncode != 0:
            logger.warning(
                "Stage 6: pdb2pqr subprocess exited with code %d: %s",
                result.returncode, result.stderr[:500],
            )
            return False
        # pdb2pqr writes a PQR; copy the PDB sidecar to the expected output path
        if Path(pdb_output_path).exists():
            import shutil  # noqa: PLC0415
            shutil.copy2(pdb_output_path, output_path)
        return Path(output_path).exists()
    except FileNotFoundError:
        logger.warning("Stage 6: pdb2pqr executable not found on PATH.")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Stage 6: pdb2pqr subprocess timed out after 600 s.")
        return False
    except Exception as exc:
        logger.warning("Stage 6: pdb2pqr subprocess failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# PDBFixer basic protonation
# ---------------------------------------------------------------------------

def _run_pdbfixer_basic(
    structure: gemmi.Structure,
    profile: Stage6Profile,
    job_dir: Path,
) -> tuple[gemmi.Structure, dict]:
    """Protonate using PDBFixer's addMissingHydrogens at the requested pH.

    No pKa calculation — PDBFixer applies pH-dependent rules internally.
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    tmp_in = str(job_dir / "pdbfixer_in.pdb")
    tmp_out = str(job_dir / "pdbfixer_out.pdb")

    try:
        structure.write_pdb(tmp_in)

        fixer = PDBFixer(filename=tmp_in)
        fixer.missingResidues = {}  # do not model missing residues
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=profile.ph)

        with open(tmp_out, "w") as fh:
            PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)

        protonated = gemmi.read_structure(tmp_out)

        # Apply user overrides
        protonated, _ = _apply_user_overrides(protonated, profile)

        his_states = _collect_his_states_from_structure(protonated, {}, profile)
        cys_states = _collect_cys_states(protonated, profile)
        n_h = _count_h_atoms(protonated)

        return protonated, {
            "protonation_method": "pdbfixer_basic",
            "ph": profile.ph,
            "n_h_atoms_added": n_h,
            "histidines": his_states,
            "acidic_residues": [],
            "cysteines": cys_states,
            "metals_coordination": [],
        }

    except Exception as exc:
        logger.error("Stage 6 pdbfixer_basic failed: %s", exc, exc_info=True)
        return structure, _empty_report(profile)
    finally:
        for p in (tmp_in, tmp_out):
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# User overrides
# ---------------------------------------------------------------------------

def _apply_user_overrides(
    structure: gemmi.Structure,
    profile: Stage6Profile,
) -> tuple[gemmi.Structure, dict]:
    """Rename residues according to profile.residue_state_overrides (highest priority).

    Example: override with state="HIE" renames the matching HIS residue to HIE.
    Returns (structure, {"applied_overrides": [...]}).
    """
    overrides_map = {
        (o.chain, o.resseq): o
        for o in profile.residue_state_overrides
    }
    if not overrides_map:
        return structure, {}

    applied: list[str] = []
    for chain in structure[0]:
        for res in chain:
            key = (chain.name, res.seqid.num)
            override = overrides_map.get(key)
            if override is None:
                continue
            old_name = res.name
            res.name = override.state  # e.g. rename HIS → HIE
            entry = (
                f"{chain.name}:{res.seqid.num} {old_name} → {override.state} "
                f"({override.reason})"
            )
            applied.append(entry)
            logger.info("Stage 6 [user_override]: %s", entry)

    return structure, {"applied_overrides": applied}


# ---------------------------------------------------------------------------
# Metal-coordination tautomer forcing
# ---------------------------------------------------------------------------

def _apply_metal_coordination_overrides(
    structure: gemmi.Structure,
    structural_metals: list,
    profile: Stage6Profile,
    pka_table: dict,
) -> tuple[list, list]:
    """Force HIS tautomers for residues that coordinate structural metals.

    Convention (Histidine–metal coordination chemistry):
        NE2 coordinates metal  →  ND1 carries the proton  →  HID
        ND1 coordinates metal  →  NE2 carries the proton  →  HIE

    Returns (his_states_list, metals_coord_report_list).
    """
    if not profile.metal_coordination_override or not structural_metals:
        his_states = _collect_his_states_from_structure(structure, pka_table, profile)
        return his_states, []

    # Build {(chain, resseq): position} for every structural metal atom
    metal_positions: dict[tuple[str, int], Any] = {}
    for m in structural_metals:
        for chain in structure[0]:
            if chain.name != m.chain:
                continue
            for res in chain:
                if res.seqid.num == m.resseq and res.name == m.comp_id:
                    for atom in res:
                        metal_positions[(m.chain, m.resseq)] = atom.pos
                        break  # one representative position per metal ion

    if not metal_positions:
        his_states = _collect_his_states_from_structure(structure, pka_table, profile)
        return his_states, []

    cutoff_A = profile.metal_coordination_cutoff_A
    overridden: set[tuple[str, int]] = set()
    metals_coord_report: list[dict] = []

    for chain in structure[0]:
        for res in chain:
            if res.name not in _HIS_NAMES:
                continue

            # Locate the two imidazole nitrogens
            nd1_pos = ne2_pos = None
            for atom in res:
                if atom.name == "ND1":
                    nd1_pos = atom.pos
                elif atom.name == "NE2":
                    ne2_pos = atom.pos

            # Find the closest metal nitrogen interaction within the cutoff
            min_dist = float("inf")
            coord_n: str | None = None
            coord_metal_key: tuple[str, int] | None = None

            for (mchain, mresseq), mpos in metal_positions.items():
                for n_name, n_pos in (("ND1", nd1_pos), ("NE2", ne2_pos)):
                    if n_pos is None:
                        continue
                    d = mpos.dist(n_pos)
                    if d < min_dist and d < cutoff_A:
                        min_dist = d
                        coord_n = n_name
                        coord_metal_key = (mchain, mresseq)

            if coord_n is None or coord_metal_key is None:
                continue  # this HIS is not close to any metal

            prior_name = res.name
            new_name = "HID" if coord_n == "NE2" else "HIE"
            res.name = new_name
            key = (chain.name, res.seqid.num)
            overridden.add(key)

            report_entry = {
                "his_chain": chain.name,
                "his_resseq": res.seqid.num,
                "coordinating_n": coord_n,
                "metal_chain": coord_metal_key[0],
                "metal_resseq": coord_metal_key[1],
                "distance_A": round(min_dist, 3),
                "forced_state": new_name,
                "prior_state": prior_name,
            }
            metals_coord_report.append(report_entry)
            logger.info(
                "Stage 6 [metal_coord]: HIS %s:%d forced to %s "
                "(%s coordinates metal %s:%d, d=%.2f Å)",
                chain.name, res.seqid.num, new_name,
                coord_n, coord_metal_key[0], coord_metal_key[1], min_dist,
            )

    his_states = _collect_his_states_from_structure(
        structure, pka_table, profile, overridden=overridden
    )
    return his_states, metals_coord_report


# ---------------------------------------------------------------------------
# Residue-state collectors
# ---------------------------------------------------------------------------

def _collect_his_states_from_structure(
    structure: gemmi.Structure,
    pka_table: dict,
    profile: Stage6Profile,
    overridden: set[tuple[str, int]] | None = None,
) -> list[HisState]:
    """Build HisState objects for every HIS/HID/HIE/HIP residue in model 0.

    Priority for state assignment:
        1. metal_coordination_override (if residue is in *overridden*)
        2. user_explicit override  (checked via profile.residue_state_overrides)
        3. pdb2pqr/PDBFixer rename  (residue name is already HID/HIE/HIP)
        4. PROPKA pKa table         (for residues still named "HIS")
        5. profile.his_default_tautomer

    For borderline cases the propka_titration_window is applied: if
        |pKa - pH| < window/2, the default tautomer is used.
    """
    if overridden is None:
        overridden = set()

    user_override_keys: set[tuple[str, int]] = {
        (o.chain, o.resseq) for o in profile.residue_state_overrides
    }

    his_states: list[HisState] = []

    for chain in structure[0]:
        for res in chain:
            if res.name not in _HIS_NAMES:
                continue

            resseq = res.seqid.num
            res_key = (chain.name, resseq)

            # Determine state string
            state_raw = res.name
            if state_raw not in ("HID", "HIE", "HIP"):
                # Residue is still "HIS" — assign via pKa or default
                pka_val = pka_table.get(("HIS", chain.name, resseq))
                half_window = profile.propka_titration_window / 2.0

                if pka_val is not None:
                    if pka_val > profile.ph + half_window:
                        # Clearly protonated — doubly protonated HIP
                        state_raw = "HIP"
                    elif abs(pka_val - profile.ph) <= half_window:
                        # Borderline: use default tautomer
                        state_raw = profile.his_default_tautomer
                    else:
                        # Deprotonated: use default tautomer (singly protonated)
                        state_raw = profile.his_default_tautomer
                else:
                    state_raw = profile.his_default_tautomer

            # Final safety: ensure Literal constraint is satisfied
            if state_raw not in ("HID", "HIE", "HIP"):
                state_raw = "HIE"

            state: Any = state_raw  # type: ignore[assignment]

            # pKa from PROPKA table
            pka_val_for_report: float | None = pka_table.get(
                ("HIS", chain.name, resseq)
            )

            # Determine reason string
            if res_key in overridden:
                reason = "metal_coordination_override"
                metal_coord_label: str | None = "coordinated"
            elif res_key in user_override_keys:
                reason = "user_explicit_override"
                metal_coord_label = None
            elif pka_val_for_report is not None:
                reason = f"propka_pka={pka_val_for_report:.2f}_ph={profile.ph:.1f}"
                metal_coord_label = None
            elif res.name != "HIS":
                reason = "pdb2pqr_assigned"
                metal_coord_label = None
            else:
                reason = f"default_{profile.his_default_tautomer}"
                metal_coord_label = None

            his_states.append(
                HisState(
                    chain=chain.name,
                    resseq=resseq,
                    state=state,
                    propka_pka=pka_val_for_report,
                    metal_coordination=metal_coord_label,
                    reason=reason,
                )
            )

    return his_states


def _collect_cys_states(
    structure: gemmi.Structure,
    profile: Stage6Profile,
) -> list[CysState]:
    """Collect CysState entries for all CYS/CYX/CYM residues in model 0.

    If profile.cysteine_disulfide_detection is True, scans structure.connections
    for SSBOND records and marks both partners as CYX.
    """
    # Build disulfide partner map: (chain, resseq) → partner_label
    # Primary: SSBOND records (from original PDB).
    # Fallback: geometric SG···SG distance < 2.5 Å (needed because OpenMM
    # PDBFile.writeFile only emits CONECT records, losing SSBOND records on
    # re-read, so gemmi finds no Disulf connections after pdbfixer round-trip).
    disulfide_partners: dict[tuple[str, int], str] = {}
    if profile.cysteine_disulfide_detection:
        for conn in structure.connections:
            if conn.type == gemmi.ConnectionType.Disulf:
                p1 = conn.partner1
                p2 = conn.partner2
                k1 = (p1.chain_name, p1.res_id.seqid.num)
                k2 = (p2.chain_name, p2.res_id.seqid.num)
                disulfide_partners[k1] = f"{p2.chain_name}:{p2.res_id.seqid.num}"
                disulfide_partners[k2] = f"{p1.chain_name}:{p1.res_id.seqid.num}"

        # Geometric fallback: scan SG–SG pairs when no SSBOND records found
        if not disulfide_partners and len(structure) > 0:
            sg_positions: dict[tuple[str, int], gemmi.Position] = {}
            for chain in structure[0]:
                for res in chain:
                    if res.name in _CYS_NAMES:
                        for atom in res:
                            if atom.name == "SG":
                                sg_positions[(chain.name, res.seqid.num)] = atom.pos
                                break
            sg_items = list(sg_positions.items())
            for i in range(len(sg_items)):
                for j in range(i + 1, len(sg_items)):
                    (c1, r1), pos1 = sg_items[i]
                    (c2, r2), pos2 = sg_items[j]
                    if pos1.dist(pos2) < 2.5:
                        disulfide_partners[(c1, r1)] = f"{c2}:{r2}"
                        disulfide_partners[(c2, r2)] = f"{c1}:{r1}"

    cys_states: list[CysState] = []

    for chain in structure[0]:
        for res in chain:
            if res.name not in _CYS_NAMES:
                continue

            resseq = res.seqid.num
            key = (chain.name, resseq)

            if key in disulfide_partners:
                partner = disulfide_partners[key]
                cys_states.append(
                    CysState(
                        chain=chain.name,
                        resseq=resseq,
                        state="CYX",
                        disulfide_partner=partner,
                        reason=f"disulfide_bond_with_{partner}",
                    )
                )
            elif res.name == "CYX":
                # Residue was already renamed by pdb2pqr (no connection record found)
                cys_states.append(
                    CysState(
                        chain=chain.name,
                        resseq=resseq,
                        state="CYX",
                        disulfide_partner=None,
                        reason="residue_name_CYX_no_connection_record",
                    )
                )
            elif res.name == "CYM":
                cys_states.append(
                    CysState(
                        chain=chain.name,
                        resseq=resseq,
                        state="CYM",
                        disulfide_partner=None,
                        reason="residue_name_CYM_deprotonated_thiolate",
                    )
                )
            else:
                cys_states.append(
                    CysState(
                        chain=chain.name,
                        resseq=resseq,
                        state="CYS",
                        disulfide_partner=None,
                        reason="free_thiol",
                    )
                )

    return cys_states


def _collect_acidic_states(
    structure: gemmi.Structure,
    pka_table: dict,
    ph: float,
) -> list[dict]:
    """Collect protonation state info for ASP and GLU residues in model 0.

    Henderson-Hasselbalch rule: pKa > pH → protonated form (ASH / GLH).
    If the residue was already renamed by pdb2pqr (e.g. to ASH), that takes
    precedence over pKa inference.
    """
    acidic: list[dict] = []

    for chain in structure[0]:
        for res in chain:
            if res.name not in _ACIDIC_NAMES:
                continue

            resseq = res.seqid.num
            # Canonical name for pKa lookup
            canonical = "ASP" if res.name in ("ASP", "ASH") else "GLU"
            pka_val: float | None = pka_table.get((canonical, chain.name, resseq))

            if res.name in ("ASH", "GLH"):
                # Already explicitly protonated (by pdb2pqr or user override)
                protonated = True
                state = res.name
                assigned_by = "pdb2pqr_explicit" if pka_val is not None else "explicit_residue_name"
            elif pka_val is not None:
                protonated = pka_val > ph
                state = ("ASH" if canonical == "ASP" else "GLH") if protonated else canonical
                assigned_by = "propka"
            else:
                # No pKa info: assume standard deprotonated state
                protonated = False
                state = canonical
                assigned_by = "default_deprotonated"

            acidic.append({
                "chain": chain.name,
                "resseq": resseq,
                "resname": canonical,
                "state": state,
                "protonated": protonated,
                "pka": pka_val,
                "assigned_by": assigned_by,
            })

    return acidic


# ---------------------------------------------------------------------------
# PROPKA file parser
# ---------------------------------------------------------------------------

def _parse_propka(propka_path: str) -> dict[tuple[str, str, int], float]:
    """Parse a PROPKA 3.x output file and extract predicted pKa values.

    Handles two formats:

    Format 1 — PROPKA3 summary table (column-fixed after "SUMMARY" header):
        "   ASP A   6                3.58      3.80      --        --"
        "   HIS A  64 HSD            6.60      6.50      --        --"

    Format 2 — Explicit "pKa =" lines (custom logs, older outputs):
        "HIS A  64    pKa =  5.23"

    Returns {(resname, chain, resseq_int): pKa_float}.
    """
    pka: dict[tuple[str, str, int], float] = {}

    # Pattern for PROPKA3 summary table lines.
    # Columns: RESNAME CHAIN RESSEQ [optional-suffix] pKa model_pKa ...
    # The residue name is 3-4 chars, chain is 1 char, resseq is 1-4 digits.
    # The first float after the residue identifier is the predicted pKa.
    summary_re = re.compile(
        r"^\s{1,5}([A-Z]{2,4})\s+([A-Za-z])\s{1,5}(\d+)\s+(?:\S+\s+)?([\d.]+)\s+[\d.]+"
    )

    # Pattern for explicit "pKa =" format
    pka_eq_re = re.compile(
        r"^\s*([A-Z]{2,4})\s+([A-Za-z])\s+(\d+)\b.*?\bpKa\s*=\s*([\d.]+)"
    )

    try:
        with open(propka_path, encoding="utf-8", errors="replace") as fh:
            in_summary = False
            for line in fh:
                # Track summary section so we don't mis-parse other sections
                if "SUMMARY OF THIS PREDICTION" in line:
                    in_summary = True
                    continue

                # Skip divider lines
                if in_summary and re.match(r"^\s*-{3,}", line):
                    continue

                # Try explicit "pKa =" format (works anywhere in the file)
                m2 = pka_eq_re.match(line)
                if m2:
                    resname, chain, resseq_str, pka_str = (
                        m2.group(1), m2.group(2), m2.group(3), m2.group(4)
                    )
                    try:
                        pka[(resname, chain, int(resseq_str))] = float(pka_str)
                    except ValueError:
                        pass
                    continue

                # Try summary table pattern (only while inside summary section)
                if in_summary:
                    m1 = summary_re.match(line)
                    if m1:
                        resname, chain, resseq_str, pka_str = (
                            m1.group(1), m1.group(2), m1.group(3), m1.group(4)
                        )
                        key = (resname, chain, int(resseq_str))
                        try:
                            pka[key] = float(pka_str)
                        except ValueError:
                            pass

    except (FileNotFoundError, OSError):
        logger.debug("Stage 6: PROPKA file not found or unreadable: %s", propka_path)
    except Exception as exc:
        logger.debug("Stage 6: error parsing PROPKA file %s: %s", propka_path, exc)

    if pka:
        logger.debug("Stage 6: parsed %d pKa entries from %s", len(pka), propka_path)

    return pka


__all__ = ["run"]
