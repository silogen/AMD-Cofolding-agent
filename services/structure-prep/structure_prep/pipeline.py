"""Structure preparation pipeline orchestrator.

Runs stages 0-9 in sequence. Each stage receives the gemmi.Structure
produced by the previous stage. Collects per-stage reports into a
PreparationReport. All output writes are atomic (tmp file → rename).

Typical call:
    report = run_pipeline(
        input_path=Path("/data/raw/7BV2.pdb"),
        profile=PreparationProfile(),
        job_id="abc123",
        job_dir=Path("/data/jobs/abc123"),
    )
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gemmi

from .additives import get_registry
from .models import (
    AltlocResolved,
    ArtifactPaths,
    CurationEntry,
    CysState,
    HisState,
    InventoryReport,
    LigandParamResult,
    MinimizationResult,
    PreparationProfile,
    PreparationReport,
    RoleMap,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class PrepError(Exception):
    """Raised when a fail-policy is triggered during pipeline execution.

    The message carries the stage prefix, e.g. "[Stage4] <detail>".
    Callers should catch this to mark the job as 'failed' in the job store.
    """


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: Path,
    profile: PreparationProfile,
    job_id: str,
    job_dir: Path,
) -> PreparationReport:
    """Run the full 9-stage preparation pipeline.

    Parameters
    ----------
    input_path:
        Path to the raw PDB or mmCIF input file.
    profile:
        Fully validated ``PreparationProfile`` controlling every stage.
    job_id:
        Opaque job identifier stored in the report and manifest.
    job_dir:
        Working directory for this job.  Output artefacts are written to
        ``job_dir/prepared/`` and ``job_dir/sequences/``.

    Returns
    -------
    PreparationReport
        Fully populated audit report.  All artefacts have been written to
        disk before this function returns.

    Raises
    ------
    PrepError
        When a stage's fail policy is violated (Stage 4: altloc_policy=fail;
        Stage 5: missing_policy=fail).
    """
    input_path = Path(input_path)
    job_dir = Path(job_dir)

    # Ensure output directories exist before any stage runs
    (job_dir / "prepared").mkdir(parents=True, exist_ok=True)
    (job_dir / "sequences").mkdir(parents=True, exist_ok=True)

    # Load additive registry once; all stages that need it receive this object
    registry = get_registry(profile.stage3_curation.additive_registry_path)

    warnings: list[str] = []
    timings: dict[str, float] = {}

    # Initialise the report skeleton (stages fill it section by section)
    report = PreparationReport(
        job_id=job_id,
        profile_hash=profile.profile_hash(),
        input_path=str(input_path),
        received_at=datetime.utcnow().isoformat(),
    )

    # ------------------------------------------------------------------
    # Stage 0 — Inventory (always runs; reads but never mutates)
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    from .stages.inventory import run as run_inventory  # noqa: PLC0415

    structure, inventory = run_inventory(input_path, job_dir, registry)
    timings["stage0"] = time.monotonic() - t0

    report.inventory = inventory
    report.sequences_by_chain = {
        c.id: (c.sequence_from_seqres or c.sequence_from_atom)
        for c in inventory.chains
    }
    report.seqres_vs_atom = inventory.seqres_vs_atom
    report.engineered_residues = inventory.engineered_residues

    logger.info(
        "[Stage 0] Inventory: %d chains, %d HETATM groups, format=%s",
        len(inventory.chains),
        len(inventory.hetatm_groups),
        inventory.format,
    )

    # ------------------------------------------------------------------
    # Stage 1 — Biological assembly / NMR model selection
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    from .stages.assembly import run as run_assembly  # noqa: PLC0415

    structure, s1 = run_assembly(structure, profile.stage1_assembly)
    timings["stage1"] = time.monotonic() - t0

    report.assembly_applied = s1.get("assembly_applied")
    report.chain_rename_map = s1.get("chain_rename_map", {})
    report.model_selected = s1.get("model_selected", "1")

    logger.info(
        "[Stage 1] Assembly: %s  model: %s",
        report.assembly_applied or "asymmetric_unit",
        report.model_selected,
    )

    # ------------------------------------------------------------------
    # Stage 2 — Chain / entity role classification
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    from .stages.roles import run as run_roles  # noqa: PLC0415

    structure, role_map = run_roles(
        structure, profile.stage2_roles, inventory, registry
    )
    timings["stage2"] = time.monotonic() - t0
    report.roles = role_map

    logger.info(
        "[Stage 2] Roles: %d protein, %d binder, %d nucleic-acid chains; "
        "%d structural metals, %d crystallographic ions",
        len(role_map.protein_chains),
        len(role_map.binder_chains),
        len(role_map.nucleic_acid_chains),
        len(role_map.structural_metals),
        len(role_map.crystallographic_ions),
    )

    # ------------------------------------------------------------------
    # Construct form — detect and optionally trim non-native termini
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    from .construct import apply_construct_policy  # noqa: PLC0415

    structure, construct_assessment = apply_construct_policy(
        structure, profile.construct_profile
    )
    timings["construct"] = time.monotonic() - t0
    report.construct_assessment = construct_assessment
    if any(item.trimmed_residues for item in construct_assessment):
        report.construct_form = "trimmed_to_mature"
    elif any(item.form == "immature" for item in construct_assessment):
        report.construct_form = "immature"
    elif construct_assessment and all(
        item.form == "mature" for item in construct_assessment
    ):
        report.construct_form = "mature"
    else:
        report.construct_form = "unknown"

    if any(item.non_native_n_terminal for item in construct_assessment):
        warnings.append(
            "Non-native N-terminal residues detected; inspect "
            "construct_assessment before structure-conditioned prediction."
        )

    # ------------------------------------------------------------------
    # Chain selection — reduce to target chains (extends Stage 2)
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    from .stages.chain_selection import run as run_chain_selection  # noqa: PLC0415

    structure, chain_selection_log = run_chain_selection(
        structure, profile.stage2_roles, role_map
    )
    timings["chain_selection"] = time.monotonic() - t0
    report.chain_selection_log = chain_selection_log

    # ------------------------------------------------------------------
    # Stage 3 — HETATM curation (waters, additives, ligand selection)
    # ------------------------------------------------------------------ (waters, additives, ligand selection)
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    from .stages.curation import run as run_curation  # noqa: PLC0415

    structure, s3 = run_curation(
        structure,
        profile.stage3_curation,
        role_map,
        inventory.hetatm_groups,
        registry,
    )
    timings["stage3"] = time.monotonic() - t0

    _raw_curation = s3.get("curation_log", [])
    report.curation_log = [
        CurationEntry(**e) if isinstance(e, dict) else e for e in _raw_curation
    ]
    report.n_waters_retained = s3.get("n_waters_retained", 0)
    report.removed_additives = s3.get("removed_additives", [])
    # ligand_residues is the master list carried through stages 4-9 and writers
    ligand_residues: list[dict[str, Any]] = s3.get("ligand_residues_retained", [])

    logger.info(
        "[Stage 3] Curation: %d waters retained, %d ligand instances retained, "
        "%d additives removed",
        report.n_waters_retained,
        len(ligand_residues),
        len(report.removed_additives),
    )

    # Collect 3-D positions of all retained ligand heavy atoms for Stage 4
    ligand_positions = _collect_ligand_positions(structure, ligand_residues)

    # ------------------------------------------------------------------
    # Stage 4 — Alternate-location resolution
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    from .stages.altlocs import run as run_altlocs  # noqa: PLC0415

    try:
        structure, s4 = run_altlocs(
            structure, profile.stage4_altlocs, ligand_positions
        )
    except ValueError as exc:
        if profile.stage4_altlocs.altloc_policy == "fail":
            raise PrepError(f"[Stage4] {exc}") from exc
        warnings.append(f"[Stage4] {exc}")
        s4: dict[str, Any] = {}
    timings["stage4"] = time.monotonic() - t0

    report.altloc_policy = profile.stage4_altlocs.altloc_policy
    report.altlocs_resolved = [
        AltlocResolved(**e) if isinstance(e, dict) else e
        for e in s4.get("altlocs_resolved", [])
    ]
    report.binding_site_altlocs = [
        AltlocResolved(**e) if isinstance(e, dict) else e
        for e in s4.get("binding_site_altlocs", [])
    ]

    logger.info(
        "[Stage 4] Altlocs: policy=%s, %d positions resolved, %d near binding site",
        report.altloc_policy,
        len(report.altlocs_resolved),
        len(report.binding_site_altlocs),
    )

    # ------------------------------------------------------------------
    # Stage 5 — Missing atoms and residue loop modeling
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    from .stages.missing import run as run_missing  # noqa: PLC0415

    try:
        structure, s5 = run_missing(
            structure,
            profile.stage5_missing,
            inventory.seqres_vs_atom,
            job_dir,
        )
    except ValueError as exc:
        raise PrepError(f"[Stage5] {exc}") from exc
    timings["stage5"] = time.monotonic() - t0

    report.missing_policy = profile.stage5_missing.missing_policy
    report.missing_residues = s5.get("missing_residues", [])
    report.missing_atoms = s5.get("missing_atoms", [])
    report.n_atoms_modeled = s5.get("n_atoms_modeled", 0)
    report.n_residues_modeled = s5.get("n_residues_modeled", 0)

    logger.info(
        "[Stage 5] Missing: policy=%s, %d residues modeled, %d atoms modeled",
        report.missing_policy,
        report.n_residues_modeled,
        report.n_atoms_modeled,
    )

    # ------------------------------------------------------------------
    # Stage 6 — Protonation state assignment
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    from .stages.protonation import run as run_protonation  # noqa: PLC0415

    structure, s6 = run_protonation(
        structure,
        profile.stage6_protonation,
        role_map.structural_metals,
        job_dir,
    )
    timings["stage6"] = time.monotonic() - t0

    report.protonation_method = s6.get(
        "protonation_method", profile.stage6_protonation.protonation_method
    )
    report.protonation_ph = s6.get("ph", profile.stage6_protonation.ph)
    report.n_h_atoms_added = s6.get("n_h_atoms_added", 0)
    report.histidines = [
        HisState(**e) if isinstance(e, dict) else e
        for e in s6.get("histidines", [])
    ]
    report.acidic_residues = s6.get("acidic_residues", [])
    report.cysteines = [
        CysState(**e) if isinstance(e, dict) else e
        for e in s6.get("cysteines", [])
    ]
    report.metals_coordination = s6.get("metals_coordination", [])

    logger.info(
        "[Stage 6] Protonation: method=%s  pH=%.1f  %d H atoms added",
        report.protonation_method,
        report.protonation_ph,
        report.n_h_atoms_added,
    )

    # ------------------------------------------------------------------
    # Stage 7 — Terminus capping (ACE / NME)
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    from .stages.termini import run as run_termini  # noqa: PLC0415

    structure, s7 = run_termini(
        structure,
        profile.stage7_termini,
        role_map,
        report.missing_residues,
    )
    timings["stage7"] = time.monotonic() - t0

    report.termini_policy = s7.get("termini_policy", profile.stage7_termini.cap_termini)
    report.termini_capped = s7.get("termini_capped", [])

    logger.info(
        "[Stage 7] Termini: policy=%s  %d caps applied",
        report.termini_policy,
        len(report.termini_capped),
    )

    # ------------------------------------------------------------------
    # Stage 8 — Ligand parameterization
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    from .stages.ligand import run as run_ligand  # noqa: PLC0415

    structure, s8 = run_ligand(
        structure, profile.stage8_ligand, ligand_residues, job_dir
    )
    timings["stage8"] = time.monotonic() - t0

    report.ligands = [
        LigandParamResult(**e) if isinstance(e, dict) else e
        for e in s8.get("ligands", [])
    ]

    logger.info(
        "[Stage 8] Ligand parameterization: %d ligand instance(s)",
        len(report.ligands),
    )

    # ------------------------------------------------------------------
    # Stage 9 — Optional geometry minimization
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    from .stages.minimization import run as run_minimization  # noqa: PLC0415

    structure, min_result = run_minimization(
        structure,
        profile.stage9_minimization,
        report.ligands,
        role_map,
        job_dir,
    )
    timings["stage9"] = time.monotonic() - t0

    report.minimization = min_result

    if min_result.performed:
        logger.info(
            "[Stage 9] Minimization: scope=%s  converged=%s  ΔE=%.2f kJ/mol",
            min_result.scope,
            min_result.converged,
            (min_result.energy_delta_kj_mol or 0.0),
        )
    else:
        logger.info("[Stage 9] Minimization: skipped (minimize='none')")

    # The final structure is the only safe source for sequence positions:
    # assembly, construct trimming, and later chain selection may all change it.
    from .numbering import build_residue_numbering  # noqa: PLC0415

    numbering, final_sequences = build_residue_numbering(
        structure,
        profile.construct_profile.uniprot_offsets,
        profile.construct_profile.uniprot_sequence,
    )
    report.residue_numbering = numbering
    report.sequences_by_chain = final_sequences

    # ------------------------------------------------------------------
    # Stage 10 — Optional structural alignment
    # ------------------------------------------------------------------
    from .stages.alignment import run as run_alignment  # noqa: PLC0415

    ref_path = profile.stage10_alignment.reference_path
    if ref_path:
        try:
            reference_structure = gemmi.read_structure(ref_path)
            reference_structure.setup_entities()
        except Exception as exc:
            warnings.append(f"[Stage 10] Cannot read reference structure: {exc}")
            reference_structure = None

        if reference_structure is not None:
            t0 = time.monotonic()
            structure, align_result = run_alignment(
                structure, profile.stage10_alignment, reference_structure
            )
            timings["stage10"] = time.monotonic() - t0
            from .models import AlignmentResult  # noqa: PLC0415

            report.alignment = AlignmentResult(**align_result)
    else:
        report.alignment = report.alignment  # keep default (performed=False)

    # ------------------------------------------------------------------
    # Write all output artefacts
    # ------------------------------------------------------------------
    from .writers import write_all_artefacts, write_manifest, write_report  # noqa: PLC0415

    artifact_paths = write_all_artefacts(
        structure, profile, report, ligand_residues, role_map, job_dir
    )

    # Inventory was written atomically by Stage 0; record its relative path
    inventory_path = job_dir / "inventory.json"
    artifact_paths.inventory_json = _rel(inventory_path, job_dir.parent.parent)

    report.artifacts = artifact_paths
    report.warnings = warnings
    report.stage_timings_s = timings
    report.stage_timings_s["total"] = sum(timings.values())

    # Resolve and record the report path before writing so report.json is
    # self-consistent (contains its own path in artifacts.report_json).
    report_path = job_dir / "report.json"
    report.artifacts.report_json = _rel(report_path, job_dir.parent.parent)

    write_report(report, report_path)
    write_manifest(job_id, profile, report, job_dir)

    logger.info(
        "[Pipeline] Job %s completed in %.1fs total  "
        "(stage timings: %s)",
        job_id,
        report.stage_timings_s["total"],
        "  ".join(
            f"{k}={v:.2f}s"
            for k, v in report.stage_timings_s.items()
            if k != "total"
        ),
    )

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_ligand_positions(
    structure: gemmi.Structure,
    ligand_residues: list[dict[str, Any]],
) -> list[tuple[float, float, float]]:
    """Return (x, y, z) for every heavy atom of every retained ligand.

    Used by Stage 4 to flag altlocs near the binding site.

    Parameters
    ----------
    structure:
        Post-Stage-3 gemmi Structure (first model used).
    ligand_residues:
        List of dicts with keys ``comp_id``, ``chain``, ``resseq``.

    Returns
    -------
    list[tuple[float, float, float]]
        Flat list of heavy-atom Cartesian coordinates.
    """
    positions: list[tuple[float, float, float]] = []
    if not ligand_residues or len(structure) == 0:
        return positions

    model = structure[0]

    # Build a quick index: (comp_id, chain, resseq) → True
    lig_keys: set[tuple[str, str, int]] = {
        (str(l.get("comp_id", "")), str(l.get("chain", "")), int(l.get("resseq", -1)))
        for l in ligand_residues
    }

    for chain in model:
        for res in chain:
            key = (res.name.strip().upper(), chain.name, res.seqid.num)
            if key not in lig_keys:
                continue
            for atom in res:
                # Skip hydrogens
                if atom.element == gemmi.Element("H") or atom.element == gemmi.Element("D"):
                    continue
                positions.append((atom.pos.x, atom.pos.y, atom.pos.z))

    return positions


def _rel(path: Path, base: Path) -> str:
    """Return path relative to base; fall back to absolute string."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


__all__ = ["PrepError", "run_pipeline"]
