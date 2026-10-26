"""Stage 5 — Missing Atoms and Residues.

Reports on missing atoms/residues and optionally models them in
using PDBFixer. All modelled coordinates are annotated as inferred,
never presented as experimentally observed.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import gemmi

from structure_prep.models import Stage5Profile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    structure: gemmi.Structure,
    profile: Stage5Profile,
    seqres_vs_atom: dict,  # from InventoryReport.seqres_vs_atom
    job_dir: Path,
) -> tuple[gemmi.Structure, dict]:
    """Stage 5: Missing Atoms and Residues.

    Returns (updated_structure, stage5_report_dict).

    stage5_report_dict keys:
        missing_policy: str
        missing_residues: list[dict]  # {chain, resseq_start, resseq_end, n_residues, modeled, source}
        missing_atoms: list[dict]     # {chain, resseq, resname, missing_atom_names, modeled}
        n_atoms_modeled: int
        n_residues_modeled: int
    """
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    # Nonstandard residue handling always runs first, before missing-atom modeling.
    structure = _handle_nonstandard_residues(structure, profile)

    policy = profile.missing_policy

    if policy == "report_only":
        return _report_only(structure, profile, seqres_vs_atom, job_dir)
    elif policy == "model_atoms":
        return _model_atoms(structure, profile, seqres_vs_atom, job_dir)
    elif policy == "model_loops":
        return _model_loops(structure, profile, seqres_vs_atom, job_dir)
    elif policy == "fail":
        return _fail_if_missing(structure, profile, seqres_vs_atom, job_dir)
    else:
        logger.warning("Stage 5: unknown missing_policy %r; falling back to report_only", policy)
        return _report_only(structure, profile, seqres_vs_atom, job_dir)


# ---------------------------------------------------------------------------
# Nonstandard residue handling
# ---------------------------------------------------------------------------

def _handle_nonstandard_residues(
    structure: gemmi.Structure,
    profile: Stage5Profile,
) -> gemmi.Structure:
    """Scan model 0 for residues whose names appear in profile.nonstandard_conversions.

    Behaviour depends on profile.nonstandard_residue_policy:
        report_and_skip   — log each occurrence; return structure unchanged.
        convert_to_standard — rename every matching residue in-place.
        fail              — raise ValueError listing all matching positions.
    """
    conversions = profile.nonstandard_conversions
    if not conversions:
        return structure

    # Collect matching residues (chain, seqnum, old_name, proposed_new_name)
    found: list[tuple[str, int, str, str]] = []
    for chain in structure[0]:
        for res in chain:
            new_name = conversions.get(res.name)
            if new_name is not None:
                found.append((chain.name, res.seqid.num, res.name, new_name))

    if not found:
        return structure

    policy = profile.nonstandard_residue_policy

    if policy == "report_and_skip":
        for chain_id, resseq, resname, new_name in found:
            logger.info(
                "Stage 5 [report_and_skip]: nonstandard residue %s at chain %s "
                "resseq %d (would map to %s); leaving unchanged.",
                resname, chain_id, resseq, new_name,
            )
        return structure

    elif policy == "convert_to_standard":
        for chain in structure[0]:
            for res in chain:
                new_name = conversions.get(res.name)
                if new_name is not None:
                    logger.info(
                        "Stage 5 [convert_to_standard]: %s → %s at chain %s resseq %d",
                        res.name, new_name, chain.name, res.seqid.num,
                    )
                    res.name = new_name
        return structure

    elif policy == "fail":
        details = "; ".join(
            f"{resname}@{chain_id}:{resseq}" for chain_id, resseq, resname, _ in found
        )
        raise ValueError(
            f"Stage 5: nonstandard_residue_policy='fail' — "
            f"nonstandard residues detected: {details}"
        )

    else:
        logger.warning(
            "Stage 5: unknown nonstandard_residue_policy %r; skipping.", policy
        )
        return structure


# ---------------------------------------------------------------------------
# PDBFixer helpers
# ---------------------------------------------------------------------------

def _safe_resseq(residue_id: str) -> int | None:
    """Convert a PDB residue-id string (possibly with insertion code) to int.

    Examples: "64" → 64, "64A" → 64, "-3" → -3, "" → None.
    """
    if residue_id is None:
        return None
    s = str(residue_id).strip()
    try:
        return int(s)
    except ValueError:
        m = re.match(r"^(-?\d+)", s)
        if m:
            return int(m.group(1))
        return None


def _extract_gap_info(
    fixer: Any,
    modeled: bool,
) -> tuple[list[dict], list[dict]]:
    """Extract missing-residue gaps and missing-atom entries from a PDBFixer instance.

    Reads fixer.missingResidues (set by findMissingResidues) and
    fixer.missingAtoms (set by findMissingAtoms).

    Returns (missing_residues_list, missing_atoms_list).
    """
    try:
        chains_list = list(fixer.topology.chains())
    except Exception:
        chains_list = []

    # ------------------------------------------------------------------
    # Missing residues
    # ------------------------------------------------------------------
    missing_residues: list[dict] = []
    for key, res_names in (fixer.missingResidues or {}).items():
        try:
            chain_idx, template_pos = key
            if chain_idx >= len(chains_list):
                continue
            chain = chains_list[chain_idx]
            chain_id = chain.id

            # Estimate the residue number range of the gap by examining
            # the observed residues on each side of the template position.
            sorted_chain_res = sorted(
                list(chain),
                key=lambda r: (r.seqid.num if hasattr(r, 'seqid') else 0),
            )
            n_before = min(template_pos, len(sorted_chain_res))
            resseq_before: int | None = None
            if n_before > 0:
                resseq_before = _safe_resseq(sorted_chain_res[n_before - 1].id)

            resseq_start: int | None = (
                resseq_before + 1 if resseq_before is not None else None
            )
            resseq_end: int | None = (
                resseq_start + len(res_names) - 1
                if resseq_start is not None
                else None
            )

            missing_residues.append({
                "chain": chain_id,
                "resseq_start": resseq_start,
                "resseq_end": resseq_end,
                "n_residues": len(res_names),
                "residue_names": list(res_names),
                "modeled": modeled,
                "source": "pdbfixer",
            })
        except Exception as exc:
            logger.debug("Stage 5: could not extract gap info for key %s: %s", key, exc)

    # ------------------------------------------------------------------
    # Missing atoms
    # ------------------------------------------------------------------
    missing_atoms: list[dict] = []
    for residue, atoms in (fixer.missingAtoms or {}).items():
        try:
            chain_id = residue.chain.id
            resseq = _safe_resseq(residue.id)
            missing_atoms.append({
                "chain": chain_id,
                "resseq": resseq,
                "resname": residue.name,
                "missing_atom_names": [a.name for a in atoms],
                "modeled": modeled,
            })
        except Exception as exc:
            logger.debug(
                "Stage 5: could not extract missing-atom info for residue %s: %s",
                residue, exc,
            )

    return missing_residues, missing_atoms


def _build_report_from_seqres(
    seqres_vs_atom: dict,
) -> tuple[list[dict], list[dict]]:
    """Fallback: build a minimal missing-residue report from seqres_vs_atom aggregates.

    Used when PDBFixer detection fails or is unavailable.
    The seqres_vs_atom dict from Stage 0 contains 'n_missing_estimated' per chain;
    if a future inventory update adds 'missing_from_atom' per-residue detail,
    that richer data is used automatically.
    """
    missing_residues: list[dict] = []
    for chain_id, info in seqres_vs_atom.items():
        per_residue: list = info.get("missing_from_atom", [])
        if per_residue:
            for pos in per_residue:
                if isinstance(pos, dict):
                    missing_residues.append({
                        "chain": chain_id,
                        "resseq_start": pos.get("resseq"),
                        "resseq_end": pos.get("resseq"),
                        "n_residues": 1,
                        "residue_names": [pos.get("resname", "UNK")],
                        "modeled": False,
                        "source": "seqres_vs_atom",
                    })
                else:
                    seq = int(pos) if isinstance(pos, (int, float)) else None
                    missing_residues.append({
                        "chain": chain_id,
                        "resseq_start": seq,
                        "resseq_end": seq,
                        "n_residues": 1,
                        "residue_names": ["UNK"],
                        "modeled": False,
                        "source": "seqres_vs_atom",
                    })
        else:
            n_missing = info.get("n_missing_estimated", 0) or 0
            if n_missing > 0:
                missing_residues.append({
                    "chain": chain_id,
                    "resseq_start": None,
                    "resseq_end": None,
                    "n_residues": n_missing,
                    "residue_names": [],
                    "modeled": False,
                    "source": "seqres_vs_atom_aggregate",
                })
    return missing_residues, []


# ---------------------------------------------------------------------------
# Policy implementations
# ---------------------------------------------------------------------------

def _report_only(
    structure: gemmi.Structure,
    profile: Stage5Profile,
    seqres_vs_atom: dict,
    job_dir: Path,
) -> tuple[gemmi.Structure, dict]:
    """Detect and report missing atoms/residues; return structure unchanged."""
    from pdbfixer import PDBFixer  # deferred so pdbfixer is optional for report_only callers

    tmp_pdb = str(job_dir / "_s5_report.pdb")
    missing_residues: list[dict] = []
    missing_atoms: list[dict] = []

    try:
        structure.write_pdb(tmp_pdb)
        fixer = PDBFixer(filename=tmp_pdb)

        try:
            fixer.findMissingResidues()
        except Exception as exc:
            logger.debug("Stage 5 findMissingResidues (report_only) error: %s", exc)

        try:
            fixer.findMissingAtoms()
        except Exception as exc:
            logger.debug("Stage 5 findMissingAtoms (report_only) error: %s", exc)

        missing_residues, missing_atoms = _extract_gap_info(fixer, modeled=False)

    except ImportError:
        logger.warning(
            "Stage 5: pdbfixer not available; using seqres_vs_atom aggregate for report."
        )
        missing_residues, missing_atoms = _build_report_from_seqres(seqres_vs_atom)
    except Exception as exc:
        logger.warning(
            "Stage 5 report_only PDBFixer detection failed (%s); "
            "falling back to seqres_vs_atom aggregate.",
            exc,
        )
        missing_residues, missing_atoms = _build_report_from_seqres(seqres_vs_atom)
    finally:
        try:
            os.unlink(tmp_pdb)
        except OSError:
            pass

    return structure, {
        "missing_policy": "report_only",
        "missing_residues": missing_residues,
        "missing_atoms": missing_atoms,
        "n_atoms_modeled": 0,
        "n_residues_modeled": 0,
    }


def _model_atoms(
    structure: gemmi.Structure,
    profile: Stage5Profile,
    seqres_vs_atom: dict,
    job_dir: Path,
) -> tuple[gemmi.Structure, dict]:
    """Model missing sidechain/backbone atoms; skip missing residues."""
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    tmp_in = str(job_dir / "_s5_atoms_in.pdb")
    tmp_out = str(job_dir / "_s5_atoms_out.pdb")

    try:
        structure.write_pdb(tmp_in)
        fixer = PDBFixer(filename=tmp_in)

        # 1. Detect missing residues for the report, but don't model them.
        missing_res_report: list[dict] = []
        try:
            fixer.findMissingResidues()
            missing_res_report, _ = _extract_gap_info(fixer, modeled=False)
        except Exception as exc:
            logger.debug("Stage 5 findMissingResidues (model_atoms) error: %s", exc)
            missing_res_report = _build_report_from_seqres(seqres_vs_atom)[0]

        # Override: do NOT model missing residues (atoms only)
        fixer.missingResidues = {}

        # 2. Nonstandard residue scan required by some PDBFixer internals
        try:
            fixer.findNonstandardResidues()
        except Exception:
            pass

        # 3. Find missing atoms and record before applying
        fixer.findMissingAtoms()
        _, missing_atoms_report = _extract_gap_info(fixer, modeled=True)
        n_atoms_to_model = sum(len(v) for v in (fixer.missingAtoms or {}).values())

        fixer.addMissingAtoms()

        with open(tmp_out, "w") as fh:
            PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)

        structure_out = gemmi.read_structure(tmp_out)
        logger.info(
            "Stage 5 [model_atoms]: added %d heavy atoms across %d residue positions.",
            n_atoms_to_model, len(missing_atoms_report),
        )

        return structure_out, {
            "missing_policy": "model_atoms",
            "missing_residues": missing_res_report,
            "missing_atoms": missing_atoms_report,
            "n_atoms_modeled": n_atoms_to_model,
            "n_residues_modeled": 0,
        }

    except Exception as exc:
        logger.error("Stage 5 model_atoms failed: %s", exc, exc_info=True)
        return structure, {
            "missing_policy": "model_atoms",
            "missing_residues": _build_report_from_seqres(seqres_vs_atom)[0],
            "missing_atoms": [],
            "n_atoms_modeled": 0,
            "n_residues_modeled": 0,
        }
    finally:
        for p in (tmp_in, tmp_out):
            try:
                os.unlink(p)
            except OSError:
                pass


def _model_loops(
    structure: gemmi.Structure,
    profile: Stage5Profile,
    seqres_vs_atom: dict,
    job_dir: Path,
) -> tuple[gemmi.Structure, dict]:
    """Model missing loop segments (gaps ≤ loop_max_length) and all missing atoms."""
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    tmp_in = str(job_dir / "_s5_loops_in.pdb")
    tmp_out = str(job_dir / "_s5_loops_out.pdb")

    try:
        structure.write_pdb(tmp_in)
        fixer = PDBFixer(filename=tmp_in)

        # 1. Find all missing residues
        try:
            fixer.findMissingResidues()
        except Exception as exc:
            logger.debug("Stage 5 findMissingResidues (model_loops) error: %s", exc)

        # 2. Filter: only model gaps <= loop_max_length residues
        if fixer.missingResidues:
            n_total_gaps = len(fixer.missingResidues)
            fixer.missingResidues = {
                k: v
                for k, v in fixer.missingResidues.items()
                if len(v) <= profile.loop_max_length
            }
            n_filtered = n_total_gaps - len(fixer.missingResidues)
            if n_filtered > 0:
                logger.info(
                    "Stage 5 [model_loops]: skipping %d gap(s) longer than loop_max_length=%d.",
                    n_filtered, profile.loop_max_length,
                )

        # 3. Capture report info before applying
        missing_res_report, _ = _extract_gap_info(fixer, modeled=True)
        n_residues_to_model = sum(len(v) for v in fixer.missingResidues.values())

        # 4. Nonstandard residue scan
        try:
            fixer.findNonstandardResidues()
        except Exception:
            pass

        # 5. Find missing atoms
        fixer.findMissingAtoms()
        _, missing_atoms_report = _extract_gap_info(fixer, modeled=True)
        n_atoms_to_model = sum(len(v) for v in (fixer.missingAtoms or {}).values())

        # 6. Apply modeling
        fixer.addMissingResidues()
        fixer.addMissingAtoms()

        with open(tmp_out, "w") as fh:
            PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)

        structure_out = gemmi.read_structure(tmp_out)
        logger.info(
            "Stage 5 [model_loops]: modelled %d residues and %d heavy atoms.",
            n_residues_to_model, n_atoms_to_model,
        )

        return structure_out, {
            "missing_policy": "model_loops",
            "missing_residues": missing_res_report,
            "missing_atoms": missing_atoms_report,
            "n_atoms_modeled": n_atoms_to_model,
            "n_residues_modeled": n_residues_to_model,
        }

    except Exception as exc:
        logger.error("Stage 5 model_loops failed: %s", exc, exc_info=True)
        return structure, {
            "missing_policy": "model_loops",
            "missing_residues": [],
            "missing_atoms": [],
            "n_atoms_modeled": 0,
            "n_residues_modeled": 0,
        }
    finally:
        for p in (tmp_in, tmp_out):
            try:
                os.unlink(p)
            except OSError:
                pass


def _fail_if_missing(
    structure: gemmi.Structure,
    profile: Stage5Profile,
    seqres_vs_atom: dict,
    job_dir: Path,
) -> tuple[gemmi.Structure, dict]:
    """Raise ValueError if any missing atoms or residues are detected."""
    from pdbfixer import PDBFixer

    tmp_pdb = str(job_dir / "_s5_fail_check.pdb")
    missing_residues: list[dict] = []
    missing_atoms: list[dict] = []

    try:
        structure.write_pdb(tmp_pdb)
        fixer = PDBFixer(filename=tmp_pdb)

        try:
            fixer.findMissingResidues()
        except Exception as exc:
            logger.debug("Stage 5 findMissingResidues (fail mode) error: %s", exc)

        try:
            fixer.findMissingAtoms()
        except Exception as exc:
            logger.debug("Stage 5 findMissingAtoms (fail mode) error: %s", exc)

        missing_residues, missing_atoms = _extract_gap_info(fixer, modeled=False)

    except ImportError:
        logger.warning("Stage 5: pdbfixer not available; using seqres_vs_atom for fail-mode check.")
        missing_residues, _ = _build_report_from_seqres(seqres_vs_atom)
    except Exception as exc:
        logger.warning("Stage 5 fail-mode PDBFixer detection error: %s", exc)
    finally:
        try:
            os.unlink(tmp_pdb)
        except OSError:
            pass

    if missing_residues or missing_atoms:
        lines: list[str] = ["missing_policy='fail' — missing atoms/residues detected:"]
        if missing_residues:
            lines.append("  Missing residues:")
            for r in missing_residues:
                if r["resseq_start"] is not None:
                    rng = f"{r['resseq_start']}–{r['resseq_end']}"
                else:
                    rng = "unknown positions"
                names_str = (
                    ", ".join(r.get("residue_names", []))
                    or f"({r['n_residues']} residue(s))"
                )
                lines.append(f"    chain {r['chain']}: {rng} [{names_str}]")
        if missing_atoms:
            lines.append("  Missing atoms:")
            for a in missing_atoms:
                lines.append(
                    f"    chain {a['chain']} {a['resname']} {a['resseq']}: "
                    f"{', '.join(a['missing_atom_names'])}"
                )
        raise ValueError("Stage 5: " + "\n".join(lines))

    # All clear
    return structure, {
        "missing_policy": "fail",
        "missing_residues": [],
        "missing_atoms": [],
        "n_atoms_modeled": 0,
        "n_residues_modeled": 0,
    }


__all__ = ["run"]
