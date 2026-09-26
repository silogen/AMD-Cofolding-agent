"""v2 complex profiling pipeline — POST /profile_complex entry point.

Extends the v1 pipeline (core.py) with:
- Explicit binder/target entity model
- Enriched interactions with biological context annotations
- Generalised binder_feature × target_feature × interaction_type triples
- Conformational strain (MMFF94, Lai 2026)
- Efficiency metrics (LE / LLE / BEI) with priority-based affinity selection
- Interface geometry (BSA + shape complementarity)
- Hotspot ensemble occupancy

COMPATIBILITY GUARANTEE
-----------------------
This module does not modify any function in core.py, schemas.py, or any
existing sub-module.  It calls the existing ``_profile_selection()`` internals
by importing from core.py (they are module-level functions, not truly private).
All new output goes into :class:`ComplexResult` which is separate from
:class:`BatchResult` / :class:`MoleculeResult`.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Optional block-10 annotation cache/proxy. Direct APIs remain the fallback.
_ANN_URL: str = os.environ.get("ANNOTATION_SERVICE_URL", "").rstrip("/")


# ---------------------------------------------------------------------------
# Lazy imports — keeps startup fast and avoids hard deps for callers that
# only use the v1 pipeline.
# ---------------------------------------------------------------------------

def _load_schema_models():
    from interaction_profiler.schemas import (
        ComplexResult, ComplexProvenance,
        Entity, EntityRole,
        StrainResult, EfficiencyResult, InterfaceGeometry,
        HotspotOccupancy,
    )
    return (ComplexResult, ComplexProvenance, Entity, EntityRole,
            StrainResult, EfficiencyResult, InterfaceGeometry, HotspotOccupancy)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def profile_complex(
    *,
    structure_path: str | None = None,
    smiles: str | None = None,
    mode: str = "ligand_target",
    binder_chains: list[str] | None = None,
    target_chains: list[str] | None = None,
    binder_role: str | None = None,
    ligand_comp_id: str | None = None,
    ligand_resname: str | None = None,
    ligand_chain: str | None = None,
    annotation_accessions: dict[str, Any] | None = None,
    potency: dict[str, Any] | None = None,
    ph: float = 7.4,
    radius: float = 8.0,
    compute_strain: bool = True,
    compute_efficiency: bool = True,
    compute_interface: bool = True,
    coordinate_source: str = "crystal",
    use_gpu: bool = False,
    max_interactions: int | None = None,
    drop_uncertain: bool = True,
    fail_on_ambiguity: bool = True,
    # ── Extension block 06.v3 — new optional params (all default None = backward compat) ──
    analysis_profile: str | None = None,
    site_profile_id: str | None = None,
    ensemble_profile: str | None = None,
) -> Any:  # ComplexResult
    """Profile a molecular complex with the v2 generalised entity model.

    Args:
        structure_path: Path to a PDB/mmCIF file on the shared volume.
            Required for all modes except ``ligand_only`` with SMILES-only input.
        smiles: Ligand SMILES.  Required for ``ligand_only`` mode.
            Optional for ``ligand_target`` (improves bond-order assignment).
        mode: One of:

            - ``ligand_only`` — 3D→SMILES conversion, fragment taxonomy, strain,
              LE/LLE/BEI.  No receptor geometry.
            - ``ligand_target`` — full small-molecule + receptor pipeline with
              enriched interactions, interface geometry, strain, LE/LLE/BEI.
            - ``binder_target`` — protein/peptide/antibody binder + target chain
              pair with generalised triplets and interface geometry.
            - ``hotspot_ensemble`` — loop over a list of ``structure_path``
              values (supply via ``annotation_accessions["structure_paths"]``)
              and aggregate per-hotspot occupancy.

        binder_chains: Chain IDs belonging to the binder entity.
            ``None`` → auto-detect from structure (may emit warnings).
        target_chains: Chain IDs belonging to the target entity.
            ``None`` → auto-detect.
        binder_role: Explicit entity role for the binder, e.g. ``"nanobody"``,
            ``"peptide"``, ``"antibody"``, ``"protein"``.  When *None*:

            * 2 + binder chains → ``"antibody"`` (Fv/Fab heuristic).
            * 1 binder chain → ``"protein"`` (no length guessing).

            Providing this parameter avoids all heuristics and ambiguity warnings.
        ligand_comp_id: Three-letter residue name of the small-molecule ligand
            (for ``ligand_target`` mode).  ``None`` → auto-detect.
        annotation_accessions: Dict with optional keys:

            - ``uniprot``: UniProt accession for the target protein.
            - ``interpro``: ``True`` to fetch InterPro annotations via the
              UniProt accession.
            - ``annotation_cache_dir``: Path to pre-cached annotation files.
            - ``structure_paths``: List of paths for ``hotspot_ensemble`` mode.

        potency: Dict with optional sub-dicts ``experimental``, ``boltz``,
            ``model``, ``docking`` for LE/LLE/BEI calculation.
        ph: pH for protonation (receptor preparation + ligand).
        radius: Pocket crop radius in Angstroms.
        compute_strain: Compute MMFF94 conformational strain.
        compute_efficiency: Compute LE/LLE/BEI from ``potency``.
        compute_interface: Compute BSA and shape complementarity.
        coordinate_source: Label for where the 3D coordinates came from.
        use_gpu: Use GPU-accelerated distance calculations when available.
        max_interactions: Cap on interaction count (``None`` = adaptive).
        drop_uncertain: Remove uncertain interactions after capping.
        fail_on_ambiguity: In ``binder_target`` mode, return an error immediately
            when both ``binder_chains`` and ``target_chains`` are unspecified.
            Set ``False`` to allow automatic inference (with warning).

    Returns:
        :class:`~interaction_profiler.schemas.ComplexResult`
    """
    (ComplexResult, ComplexProvenance, Entity, EntityRole,
     StrainResult, EfficiencyResult, InterfaceGeometry,
     HotspotOccupancy) = _load_schema_models()

    t0 = time.time()
    warnings: list[str] = []
    prov = ComplexProvenance(mode=mode)

    # Site profile cache: if site_profile_id is given, load the pre-computed
    # SiteProfile from /shared/site_cache/{id}.json to inject pocket context
    # without re-running fpocket.
    cached_site_profile = None
    if site_profile_id:
        import json as _json  # noqa: PLC0415
        _shared = Path(os.environ.get("IP_SHARED_DIR", "/shared"))
        _cache_path = _shared / "site_cache" / f"{site_profile_id}.json"
        if _cache_path.is_file():
            try:
                from interaction_profiler.schemas import SiteProfile  # noqa: PLC0415
                cached_site_profile = SiteProfile.model_validate(
                    _json.loads(_cache_path.read_text())
                )
                prov.annotation_source = "site_cache"
                logger.info("SiteProfile %s loaded from cache.", site_profile_id)
            except Exception as exc:
                warnings.append(
                    f"Could not load cached SiteProfile {site_profile_id!r}: {exc}. "
                    "Proceeding without pocket context."
                )
        else:
            warnings.append(
                f"site_profile_id {site_profile_id!r} not found in cache "
                f"({_cache_path}); will recompute if protein_only mode is used."
            )

    # Strict mode: binder_target requires explicit chain declarations
    if mode == "binder_target" and (not binder_chains and not target_chains):
        msg = (
            "binder_target mode requires explicit binder_chains and/or target_chains. "
            "Auto-inference for protein-protein is ambiguous (cannot distinguish binder from target). "
            "Set mode='ligand_target' for small-molecule ligands, or specify chain groups explicitly."
        )
        result = ComplexResult(mode=mode, provenance=prov, warnings=[msg])
        if fail_on_ambiguity:
            result.warnings.append("Set fail_on_ambiguity=False to allow automatic inference.")
            result.timing_seconds = round(time.time() - t0, 3)
            return result
        else:
            result.warnings.append("Proceeding with auto-inference (fail_on_ambiguity=False).")
            # Fall through to normal dispatch so _profile_binder_target can attempt auto-resolution
            warnings.extend(result.warnings)

    try:
        if mode == "ligand_only":
            result = _profile_ligand_only(
                smiles=smiles,
                structure_path=structure_path,
                potency=potency,
                ph=ph,
                compute_strain=compute_strain,
                compute_efficiency=compute_efficiency,
                coordinate_source=coordinate_source,
                prov=prov,
                warnings=warnings,
            )
        elif mode == "ligand_target":
            result = _profile_ligand_target(
                structure_path=structure_path,
                smiles=smiles,
                binder_chains=binder_chains,
                target_chains=target_chains,
                ligand_comp_id=ligand_comp_id,
                ligand_resname=ligand_resname,
                ligand_chain=ligand_chain,
                annotation_accessions=annotation_accessions or {},
                potency=potency,
                ph=ph,
                radius=radius,
                compute_strain=compute_strain,
                compute_efficiency=compute_efficiency,
                compute_interface=compute_interface,
                coordinate_source=coordinate_source,
                use_gpu=use_gpu,
                max_interactions=max_interactions,
                drop_uncertain=drop_uncertain,
                prov=prov,
                warnings=warnings,
            )
            # Inject cached SiteProfile when provided
            if cached_site_profile is not None and result.site_profile is None:
                result.site_profile = cached_site_profile
        elif mode == "binder_target":
            result = _profile_binder_target(
                structure_path=structure_path,
                binder_chains=binder_chains,
                target_chains=target_chains,
                binder_role=binder_role,
                annotation_accessions=annotation_accessions or {},
                potency=potency,
                ph=ph,
                radius=radius,
                compute_interface=compute_interface,
                use_gpu=use_gpu,
                max_interactions=max_interactions,
                drop_uncertain=drop_uncertain,
                prov=prov,
                warnings=warnings,
            )
            # Inject cached SiteProfile when provided
            if cached_site_profile is not None and result.site_profile is None:
                result.site_profile = cached_site_profile
        elif mode == "hotspot_ensemble":
            result = _profile_hotspot_ensemble(
                annotation_accessions=annotation_accessions or {},
                target_chains=target_chains,
                smiles=smiles,
                ligand_resname=ligand_resname,
                ph=ph,
                radius=radius,
                use_gpu=use_gpu,
                max_interactions=max_interactions,
                drop_uncertain=drop_uncertain,
                prov=prov,
                warnings=warnings,
                ensemble_profile=ensemble_profile,
            )
        elif mode == "protein_only":
            result = _profile_protein_only(
                structure_path=structure_path,
                target_chains=target_chains,
                analysis_profile=analysis_profile or "pocket",
                annotation_accessions=annotation_accessions or {},
                prov=prov,
                warnings=warnings,
            )
        else:
            result = ComplexResult(
                mode=mode,
                provenance=prov,
                warnings=[
                    f"Unknown mode {mode!r}; valid: "
                    "ligand_only | ligand_target | binder_target | "
                    "hotspot_ensemble | protein_only"
                ],
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("profile_complex failed (mode=%s)", mode)
        result = ComplexResult(
            mode=mode,
            provenance=prov,
            warnings=[f"{type(exc).__name__}: {exc}"],
        )

    result.warnings.extend(warnings)
    result.timing_seconds = round(time.time() - t0, 3)
    return result


# ---------------------------------------------------------------------------
# Mode: ligand_only
# ---------------------------------------------------------------------------

def _profile_ligand_only(
    *,
    smiles: str | None,
    structure_path: str | None,
    potency: dict | None,
    ph: float,
    compute_strain: bool,
    compute_efficiency: bool,
    coordinate_source: str,
    prov,
    warnings: list[str],
):
    """3D ligand → SMILES → fragment taxonomy + strain + LE/LLE/BEI."""
    from interaction_profiler.schemas import ComplexResult, StrainResult, EfficiencyResult
    from interaction_profiler.prep import prepare_ligand_from_smiles
    from interaction_profiler.fragmentation import build_composition

    result = ComplexResult(mode="ligand_only", provenance=prov)

    mol = None
    effective_smiles = smiles

    # If a 3D structure file is given: convert 3D → SMILES
    if structure_path and not smiles:
        try:
            from rdkit.Chem import MolFromMolFile, MolFromPDBBlock, MolToSmiles
            path = Path(structure_path)
            if path.suffix.lower() in (".sdf", ".mol"):
                mol = MolFromMolFile(str(path), removeHs=False)
            elif path.suffix.lower() == ".pdb":
                mol = MolFromPDBBlock(path.read_text(), removeHs=False)
            if mol is not None:
                effective_smiles = MolToSmiles(mol, isomericSmiles=True)
            else:
                warnings.append("3D→SMILES: could not parse structure file")
        except Exception as exc:
            warnings.append(f"3D→SMILES conversion failed: {exc}")

    if not effective_smiles:
        result.warnings.append("ligand_only mode requires smiles or structure_path with 3D ligand")
        return result

    # Fragment taxonomy (existing path, unchanged)
    prep = prepare_ligand_from_smiles(effective_smiles, ph)
    if prep.mol is None:
        result.warnings.append(f"Could not prepare ligand SMILES: {effective_smiles!r}")
        return result

    comp = build_composition(prep.mol)
    result.composition = comp

    from interaction_profiler.schemas import PreparationReport
    from interaction_profiler.prep import analyze_stereo
    report = PreparationReport(
        ligand_prepared=True,
        ligand_protonation_ph=ph,
        ligand_smiles_input=effective_smiles,
        ligand_smiles_prepared=prep.smiles_prepared,
        bond_assignment=prep.bond_assignment,
        stereo=analyze_stereo(prep.mol, effective_smiles),
        warnings=prep.warnings,
    )
    result.preparation = report

    # Strain — only if we have 3D coordinates (mol from structure file)
    if compute_strain:
        prov.strain_method = "mmff94_rdkit"
        if mol is not None:
            try:
                from interaction_profiler.metrics.strain import compute_strain_from_mol
                result.strain = compute_strain_from_mol(mol, coordinate_source=coordinate_source)
            except Exception as exc:
                warnings.append(f"Strain calculation failed: {exc}")
                result.strain = StrainResult(error=str(exc), coordinate_source=coordinate_source)
        else:
            result.strain = StrainResult(
                error="no_3d_coordinates — strain requires 3D input, not SMILES-only",
                coordinate_source=coordinate_source,
            )

    # Efficiency
    if compute_efficiency and potency:
        try:
            from interaction_profiler.metrics.efficiency import compute_efficiency
            result.efficiency = compute_efficiency(effective_smiles, potency)
        except Exception as exc:
            warnings.append(f"Efficiency calculation failed: {exc}")
            result.efficiency = EfficiencyResult(error=str(exc))

    return result


# ---------------------------------------------------------------------------
# Mode: ligand_target
# ---------------------------------------------------------------------------

def _profile_ligand_target(
    *,
    structure_path: str | None,
    smiles: str | None,
    binder_chains,
    target_chains,
    ligand_comp_id,
    ligand_resname,
    ligand_chain,
    annotation_accessions: dict,
    potency: dict | None,
    ph: float,
    radius: float,
    compute_strain: bool,
    compute_efficiency: bool,
    compute_interface: bool,
    coordinate_source: str,
    use_gpu: bool,
    max_interactions: int | None,
    drop_uncertain: bool,
    prov,
    warnings: list[str],
):
    """Small molecule + receptor pocket: enriched interactions + metrics."""
    from interaction_profiler.schemas import (
        ComplexResult, Entity, EntityRole,
        StrainResult, EfficiencyResult, InterfaceGeometry,
    )
    from interaction_profiler.io import MoleculeInput
    from interaction_profiler.core import _profile_structure_all  # type: ignore[attr-defined]

    result = ComplexResult(mode="ligand_target", provenance=prov)

    if not structure_path:
        result.warnings.append("ligand_target mode requires structure_path")
        return result

    # ── Run v1 pipeline to get interactions + composition ──────────────────
    inp = MoleculeInput(
        name="complex",
        structure=structure_path,
        smiles=smiles,
        ligand_resname=ligand_resname or ligand_comp_id,
        ligand_chain=ligand_chain,
        ph=ph,
    )
    v1_results = _profile_structure_all(
        inp, ph, radius, use_gpu, max_interactions, drop_uncertain,
        all_ligands=False, t0=time.time(),
    )
    if not v1_results or v1_results[0].error:
        err = v1_results[0].error if v1_results else "no results"
        result.warnings.append(f"v1 pipeline failed: {err}")
        return result

    mol_result = v1_results[0]
    result.composition = mol_result.composition
    result.preparation  = mol_result.preparation

    # ── Build entities ──────────────────────────────────────────────────────
    from interaction_profiler.io.structure import load_structure
    struct = load_structure(structure_path)
    ligs = struct.detect_ligands(
        resname=ligand_resname or ligand_comp_id,
        chain=ligand_chain,
    )
    sel = ligs[0] if ligs else None

    binder_eid = "binder_sm"
    target_eid  = "target_protein"
    binder_entity = Entity(
        entity_id=binder_eid,
        role=EntityRole.small_molecule,
        chain_ids=[sel.chain] if sel else [],
        label=sel.resname if sel else "ligand",
    )
    tgt_chain_types = {c: ct for c, ct in struct.chain_types().items()
                       if ct in ("protein", "rna", "dna")}
    tgt_chains = target_chains or list(tgt_chain_types)
    target_entity = Entity(
        entity_id=target_eid,
        role=EntityRole.protein,
        chain_ids=tgt_chains,
    )
    result.entities = [binder_entity, target_entity]

    # ── Fetch / load annotations ────────────────────────────────────────────
    ss_map: dict = {}
    feature_map: dict = {}
    interpro_map: dict = {}
    uniprot_acc = annotation_accessions.get("uniprot")

    if structure_path:
        try:
            from interaction_profiler.taxonomy.secondary_structure import assign_secondary_structure
            ss_map, ss_method = assign_secondary_structure(structure_path, tgt_chains)
            prov.secondary_structure_method = ss_method
        except Exception as exc:
            warnings.append(f"Secondary structure assignment failed: {exc}")

    if uniprot_acc:
        try:
            _load_uniprot_features(
                uniprot_acc, tgt_chains, structure_path,
                annotation_accessions, feature_map, interpro_map, prov, warnings,
            )
        except Exception as exc:
            warnings.append(f"Annotation loading failed: {exc}")

    # ── Enrich interactions ─────────────────────────────────────────────────
    from interaction_profiler.interactions.enricher import enrich_interactions
    from interaction_profiler.join_generalized import build_generalized_triples

    enriched = enrich_interactions(
        mol_result.interactions,
        ss_map=ss_map,
        feature_map=feature_map,
        interpro_map=interpro_map,
        binder_entity_id=binder_eid,
        target_entity_id=target_eid,
        binder_is_small_molecule=True,
    )
    result.enriched_interactions = enriched
    result.generalized_triples = build_generalized_triples(enriched, binder_is_small_molecule=True)

    # ── Interface geometry ──────────────────────────────────────────────────
    if compute_interface and sel:
        try:
            from interaction_profiler.metrics.interface import compute_interface_geometry
            ligand_chain_ids = [sel.chain] if sel.chain else []
            result.interface = compute_interface_geometry(
                structure_path, ligand_chain_ids, tgt_chains
            )
            prov.bsa_method = result.interface.method_bsa
            prov.sc_method  = result.interface.method_sc
        except Exception as exc:
            warnings.append(f"Interface geometry failed: {exc}")
            result.interface = InterfaceGeometry(error=str(exc))

    # ── Strain ──────────────────────────────────────────────────────────────
    if compute_strain and mol_result.preparation:
        _compute_strain_from_prep(mol_result, coordinate_source, prov, result, warnings)

    # ── Efficiency ──────────────────────────────────────────────────────────
    if compute_efficiency and potency and mol_result.metadata.ligand_smiles:
        try:
            from interaction_profiler.metrics.efficiency import compute_efficiency
            result.efficiency = compute_efficiency(mol_result.metadata.ligand_smiles, potency)
        except Exception as exc:
            warnings.append(f"Efficiency calculation failed: {exc}")
            result.efficiency = EfficiencyResult(error=str(exc))

    return result


# ---------------------------------------------------------------------------
# Mode: binder_target
# ---------------------------------------------------------------------------

def _profile_binder_target(
    *,
    structure_path: str | None,
    binder_chains: list[str] | None,
    target_chains: list[str] | None,
    binder_role: str | None,
    annotation_accessions: dict,
    potency: dict | None,
    ph: float,
    radius: float,
    compute_interface: bool,
    use_gpu: bool,
    max_interactions: int | None,
    drop_uncertain: bool,
    prov,
    warnings: list[str],
):
    """Protein/peptide/antibody binder + protein/nucleic target."""
    from interaction_profiler.schemas import (
        ComplexResult, Entity, EntityRole, InterfaceGeometry,
    )

    result = ComplexResult(mode="binder_target", provenance=prov)

    if not structure_path:
        result.warnings.append("binder_target mode requires structure_path")
        return result
    if not binder_chains and not target_chains:
        result.warnings.append(
            "binder_target mode: at least one of binder_chains or target_chains must be specified; "
            "auto-inference for protein-protein is ambiguous"
        )

    from interaction_profiler.io.structure import load_structure
    from interaction_profiler.entity.resolver import resolve_entities

    struct = load_structure(structure_path)
    uniprot_binder = annotation_accessions.get("uniprot_binder")
    uniprot_target = annotation_accessions.get("uniprot") or annotation_accessions.get("uniprot_target")

    entities, resolve_warnings = resolve_entities(
        struct,
        binder_chains=binder_chains,
        target_chains=target_chains,
        ligand_comp_id=None,
        mode="binder_target",
        binder_uniprot=uniprot_binder,
        target_uniprot=uniprot_target,
        binder_role=binder_role,
    )
    warnings.extend(resolve_warnings)
    result.entities = entities

    binder_ents = [e for e in entities if e.role not in (EntityRole.cofactor, EntityRole.metal, EntityRole.water)]
    target_ents  = [e for e in entities if e.entity_id.startswith("target")]

    binder_eid = binder_ents[0].entity_id if binder_ents else "binder"
    target_eid  = target_ents[0].entity_id if target_ents else "target"

    binder_ch = binder_chains or (binder_ents[0].chain_ids if binder_ents else [])
    target_ch  = target_chains or (target_ents[0].chain_ids if target_ents else [])

    # Profile the binder chain as a peptide-type ligand using the existing _profile_peptide path
    from interaction_profiler.io import MoleculeInput
    from interaction_profiler.core import _profile_peptide  # type: ignore[attr-defined]

    if binder_ch:
        inp = MoleculeInput(
            name="binder_target_complex",
            structure=structure_path,
            peptide_ligand_chain=binder_ch[0],
            ph=ph,
        )
        try:
            # Use profile_molecule_all which is the stable public API for peptide chains
            from interaction_profiler.core import profile_molecule_all
            mol_results = profile_molecule_all(
                inp, ph_default=ph, radius=radius, use_gpu=use_gpu,
                max_interactions=max_interactions, drop_uncertain=drop_uncertain,
            )
            mol_result = mol_results[0] if mol_results else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("binder_target peptide profiling failed: %s", exc)
            mol_result = None

        if mol_result and not mol_result.error:
            result.composition = mol_result.composition
            result.preparation  = mol_result.preparation

            # Annotation maps
            ss_map: dict = {}
            feature_map: dict = {}
            interpro_map: dict = {}
            if uniprot_target:
                try:
                    _load_uniprot_features(
                        uniprot_target, target_ch, structure_path,
                        annotation_accessions, feature_map, interpro_map, prov, warnings,
                    )
                except Exception as exc:
                    warnings.append(f"Target annotation loading failed: {exc}")
            if structure_path:
                try:
                    from interaction_profiler.taxonomy.secondary_structure import assign_secondary_structure
                    ss_map, ss_method = assign_secondary_structure(structure_path, target_ch)
                    prov.secondary_structure_method = ss_method
                except Exception as exc:
                    warnings.append(f"Secondary structure assignment failed: {exc}")

            from interaction_profiler.interactions.enricher import enrich_interactions
            from interaction_profiler.join_generalized import build_generalized_triples

            enriched = enrich_interactions(
                mol_result.interactions,
                ss_map=ss_map,
                feature_map=feature_map,
                interpro_map=interpro_map,
                binder_entity_id=binder_eid,
                target_entity_id=target_eid,
                binder_is_small_molecule=False,
            )
            result.enriched_interactions = enriched
            result.generalized_triples = build_generalized_triples(enriched, binder_is_small_molecule=False)

    # Interface geometry
    if compute_interface and binder_ch and target_ch:
        try:
            from interaction_profiler.metrics.interface import compute_interface_geometry
            result.interface = compute_interface_geometry(structure_path, binder_ch, target_ch)
            prov.bsa_method = result.interface.method_bsa
            prov.sc_method  = result.interface.method_sc
        except Exception as exc:
            warnings.append(f"Interface geometry failed: {exc}")
            result.interface = InterfaceGeometry(error=str(exc))

    return result


# ---------------------------------------------------------------------------
# Mode: hotspot_ensemble
# ---------------------------------------------------------------------------

def _profile_hotspot_ensemble(
    *,
    annotation_accessions: dict,
    target_chains: list[str] | None,
    smiles: str | None,
    ligand_resname: str | None,
    ph: float,
    radius: float,
    use_gpu: bool,
    max_interactions: int | None,
    drop_uncertain: bool,
    prov,
    warnings: list[str],
    ensemble_profile: str | None = None,
):
    """Aggregate hotspot contact occupancy across a list of structure artefacts."""
    from interaction_profiler.schemas import ComplexResult, HotspotOccupancy

    result = ComplexResult(mode="hotspot_ensemble", provenance=prov)

    structure_paths: list[str] = annotation_accessions.get("structure_paths") or []
    if not structure_paths:
        result.warnings.append("hotspot_ensemble mode requires annotation_accessions.structure_paths list")
        return result

    # Per-structure: run ligand_target to get interactions, collect target residues
    # Map: residue_label → {per_structure_contacts}
    hotspot_map: dict[str, list[dict]] = {}
    n_total = len(structure_paths)
    successful = 0
    member_results: list[Any] = []  # collected for extended ensemble analysis

    # F5: parallel member loop using a thread pool.  The GIL allows I/O and
    # gemmi/RDKit calls to run concurrently; most of the time is spent parsing
    # files and computing interactions, not in Python bytecode.
    MAX_WORKERS = int(os.environ.get("IP_ENSEMBLE_WORKERS", "8"))

    def _profile_member(sp: str) -> tuple[str, Any, list[str]]:
        """Profile one ensemble member; return (sp, result_or_None, warnings)."""
        from interaction_profiler.schemas import ComplexProvenance  # noqa: PLC0415
        member_prov = ComplexProvenance(mode="hotspot_ensemble_member")
        member_warnings: list[str] = []
        try:
            r = _profile_ligand_target(
                structure_path=sp,
                smiles=smiles,
                binder_chains=None,
                target_chains=target_chains,
                ligand_comp_id=None,
                ligand_resname=ligand_resname,
                ligand_chain=None,
                annotation_accessions=annotation_accessions,
                potency=None,
                ph=ph, radius=radius,
                compute_strain=False, compute_efficiency=False, compute_interface=False,
                coordinate_source="ensemble_member",
                use_gpu=use_gpu, max_interactions=max_interactions, drop_uncertain=drop_uncertain,
                prov=member_prov, warnings=member_warnings,
            )
            # F2: stamp the source path so fragment families/pairs can carry
            # the structure identifier rather than leaking a Python object id.
            if hasattr(r, "provenance"):
                r.provenance.structure_path = sp
            return sp, r, member_warnings
        except Exception as exc:
            return sp, None, [f"Ensemble member {sp} failed: {exc}"]

    import concurrent.futures  # noqa: PLC0415
    import os as _os  # noqa: PLC0415

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_profile_member, sp): sp for sp in structure_paths}
        ordered: dict[str, tuple] = {}
        for fut in concurrent.futures.as_completed(futures):
            sp, r, member_warnings = fut.result()
            ordered[sp] = (r, member_warnings)

    for sp in structure_paths:
        r, member_warnings = ordered.get(sp, (None, [f"Ensemble member {sp} not processed"]))
        for w in member_warnings:
            warnings.append(w)
        if r is None:
            continue
        successful += 1
        member_results.append(r)
        for ix in r.enriched_interactions:
            rl = ix.target_feature or ix.residue_label
            hotspot_map.setdefault(rl, []).append({
                "source": sp,
                "interaction_type": ix.interaction_type,
                "distance": ix.distance,
                "confidence": ix.confidence,
            })

    warnings.append(f"Hotspot ensemble: {successful}/{n_total} structures processed successfully.")

    # Build HotspotOccupancy records
    for res_label, contacts in hotspot_map.items():
        observed = len({c["source"] for c in contacts})
        itypes = list({c["interaction_type"] for c in contacts})
        missing = [sp for sp in structure_paths if sp not in {c["source"] for c in contacts}]
        result.hotspots.append(HotspotOccupancy(
            target_residue=res_label,
            n_structures_observed=observed,
            n_structures_total=n_total,
            occupancy=round(observed / n_total, 3) if n_total else None,
            interaction_types_observed=itypes,
            per_structure=contacts,
            missing_structures=missing,
        ))

    # Sort by occupancy descending
    result.hotspots.sort(key=lambda h: -(h.occupancy or 0))

    # ── Extended ensemble analysis (ensemble_profile) ─────────────────────
    if ensemble_profile and ensemble_profile != "hotspots" and member_results:
        try:
            from interaction_profiler.site.profiles import ENSEMBLE_PROFILES  # noqa: PLC0415
            from interaction_profiler.report.ensemble import (  # noqa: PLC0415
                compute_fragment_families, compute_compatible_pairs,
            )
            eflags = ENSEMBLE_PROFILES.get(ensemble_profile, {})

            if eflags.get("compute_fragment_families", False):
                result.fragment_families = compute_fragment_families(member_results)

            if eflags.get("compute_compatible_pairs", False):
                pairs = compute_compatible_pairs(
                    member_results,
                    site_profile=getattr(result, "site_profile", None),
                )
                # F4: cap inline pairs and offer a /shared path for the full list.
                INLINE_LIMIT = 50
                result.compatible_pairs = pairs[:INLINE_LIMIT]
                if len(pairs) > INLINE_LIMIT:
                    warnings.append(
                        f"compatible_pairs truncated to {INLINE_LIMIT} of {len(pairs)} "
                        "for inline delivery. Pass ensemble_artifact_dir to write the full "
                        "list to /shared."
                    )
        except Exception as exc:
            warnings.append(f"Extended ensemble analysis failed: {exc}")

    return result


# ---------------------------------------------------------------------------
# Mode: protein_only  (block 06 extension)
# ---------------------------------------------------------------------------

def _profile_protein_only(
    *,
    structure_path: str | None,
    target_chains: list[str] | None,
    analysis_profile: str,
    annotation_accessions: dict,
    prov,
    warnings: list[str],
):
    """Protein-side binding-site characterization (no ligand required).

    Runs fpocket pocket detection, secondary structure assignment, and
    optional UniProt/InterPro annotation overlay; returns a ``ComplexResult``
    with ``site_profile`` populated and no interaction or ligand data.

    The ``SiteProfile`` is cached to ``/shared/site_cache/`` so subsequent
    ``ligand_target`` or ``ensemble`` calls can reuse it without recomputing.
    """
    from interaction_profiler.schemas import ComplexResult  # noqa: PLC0415
    from interaction_profiler.site.profiles import (  # noqa: PLC0415
        PROTEIN_ONLY_PROFILES, resolve_profile_flags,
    )
    from interaction_profiler.site.characterizer import (  # noqa: PLC0415
        characterize_site,
    )

    result = ComplexResult(mode="protein_only", provenance=prov)

    if not structure_path:
        result.warnings.append("protein_only mode requires structure_path")
        return result

    profile_flags = PROTEIN_ONLY_PROFILES.get(analysis_profile, PROTEIN_ONLY_PROFILES["pocket"])

    try:
        site_profile = characterize_site(
            structure_path=structure_path,
            target_chains=target_chains,
            profile_flags=profile_flags,
            annotation_accessions=annotation_accessions,
            warnings=warnings,
        )
        result.site_profile = site_profile
    except Exception as exc:
        logger.exception("protein_only site characterization failed")
        result.warnings.append(f"{type(exc).__name__}: {exc}")

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_strain_from_prep(mol_result, coordinate_source, prov, result, warnings):
    """Try to compute strain from the prepared ligand structure file."""
    from interaction_profiler.schemas import StrainResult
    prov.strain_method = "mmff94_rdkit"
    prepared_path = getattr(mol_result.preparation, "prepared_ligand_path", None) if mol_result.preparation else None
    if not prepared_path:
        # Fall back to reconstructing the mol from the receptor-prepared path
        result.strain = StrainResult(
            error="prepared_ligand_path not available; provide SDF input for strain",
            coordinate_source=coordinate_source,
        )
        return
    try:
        from interaction_profiler.metrics.strain import compute_strain_from_sdf
        result.strain = compute_strain_from_sdf(prepared_path, coordinate_source=coordinate_source)
    except Exception as exc:
        warnings.append(f"Strain calculation failed: {exc}")
        result.strain = StrainResult(error=str(exc), coordinate_source=coordinate_source)


def _load_uniprot_features(
    uniprot_acc, target_chain_ids, structure_path,
    annotation_accessions, feature_map, interpro_map, prov, warnings,
):
    """Load annotations and map them to structure residues.

    Prefer the block-10 annotation service when configured, then a cached
    artifact, and finally the direct UniProt/InterPro APIs.
    """
    import requests as _requests  # noqa: PLC0415

    cache_dir = annotation_accessions.get("annotation_cache_dir")

    if _ANN_URL:
        try:
            response = _requests.post(
                f"{_ANN_URL}/uniprot/features",
                json={"accession": uniprot_acc},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            features = data.get("features", [])
            uniprot_seq = data.get("sequence", "")
            prov.annotation_source = "annotation_service"
        except Exception as exc:
            warnings.append(
                f"Annotation service call failed: {exc}. "
                "Falling back to direct API."
            )
            features = _fetch_uniprot_features_live(uniprot_acc, warnings)
            uniprot_seq = _get_uniprot_sequence(uniprot_acc, warnings)
            prov.annotation_source = "uniprot_api"
    elif cache_dir:
        import json as _json  # noqa: PLC0415
        candidate = Path(cache_dir) / f"uniprot_{uniprot_acc}.json"
        if candidate.is_file():
            try:
                cached = _json.loads(candidate.read_text())
                features = cached.get("features", [])
                uniprot_seq = cached.get("sequence", "")
                prov.annotation_source = "cached_artefact"
            except Exception as exc:
                warnings.append(
                    f"Reading cached annotation failed: {exc}. "
                    "Falling back to direct API."
                )
                features = _fetch_uniprot_features_live(uniprot_acc, warnings)
                uniprot_seq = _get_uniprot_sequence(uniprot_acc, warnings)
                prov.annotation_source = "uniprot_api"
        else:
            features = _fetch_uniprot_features_live(uniprot_acc, warnings)
            uniprot_seq = _get_uniprot_sequence(uniprot_acc, warnings)
            prov.annotation_source = "uniprot_api"
    else:
        features = _fetch_uniprot_features_live(uniprot_acc, warnings)
        uniprot_seq = _get_uniprot_sequence(uniprot_acc, warnings)
        prov.annotation_source = "uniprot_api"

    from interaction_profiler.io.structure import load_structure
    struct = load_structure(structure_path)

    for ch in target_chain_ids:
        struct_seq = _get_chain_sequence(struct, ch)
        if not struct_seq:
            continue
        try:
            from interaction_profiler.taxonomy.annotation_mapper import (
                map_uniprot_features_to_residues,
            )
            if uniprot_seq:
                chain_map = map_uniprot_features_to_residues(
                    features, ch, struct_seq, uniprot_seq
                )
                feature_map.update(chain_map)
        except Exception as exc:
            warnings.append(f"UniProt feature mapping failed for chain {ch}: {exc}")

    if not annotation_accessions.get("interpro", False):
        return

    if _ANN_URL:
        try:
            response = _requests.post(
                f"{_ANN_URL}/interpro/domains",
                json={"uniprot_accession": uniprot_acc},
                timeout=45,
            )
            response.raise_for_status()
            interpro_entries = response.json().get("interpro_entries", [])
            prov.interpro_source = "annotation_service"
        except Exception as exc:
            warnings.append(
                f"Annotation service InterPro call failed: {exc}. "
                "Falling back to direct API."
            )
            raw_entries = _fetch_interpro_live(uniprot_acc, warnings)
            interpro_entries = _normalize_interpro_entries(
                raw_entries, uniprot_acc
            )
            prov.interpro_source = "interpro_api"
    else:
        raw_entries = _fetch_interpro_live(uniprot_acc, warnings)
        interpro_entries = _normalize_interpro_entries(
            raw_entries, uniprot_acc
        )
        prov.interpro_source = "interpro_api"

    try:
        from interaction_profiler.taxonomy.annotation_mapper import (
            map_interpro_to_residues,
        )
        for ch in target_chain_ids:
            struct_seq = _get_chain_sequence(struct, ch)
            if struct_seq and uniprot_seq:
                interpro_map.update(
                    map_interpro_to_residues(
                        interpro_entries, ch, struct_seq, uniprot_seq
                    )
                )
    except Exception as exc:
        warnings.append(f"InterPro annotation mapping failed: {exc}")


def _normalize_interpro_entries(
    raw_results: list[dict], uniprot_acc: str
) -> list[dict]:
    """Convert the raw InterPro API ``results[]`` format to the normalised
    ``interpro_entries[]`` shape used by the annotation service and annotation_mapper.

    Each entry becomes:
        {accession, name, type, residue_ranges[{start,end}], source_databases[]}
    """
    entries = []
    acc_upper = uniprot_acc.upper()
    for item in raw_results or []:
        meta = item.get("metadata") or {}
        ipr_acc  = meta.get("accession", "")
        ipr_name_raw = meta.get("name", "")
        ipr_name = ipr_name_raw.get("name", "") if isinstance(ipr_name_raw, dict) else str(ipr_name_raw)
        ipr_type = meta.get("type", "")

        residue_ranges: list[dict] = []
        for prot in item.get("proteins") or []:
            if (prot.get("accession") or "").upper() != acc_upper:
                continue
            for loc in prot.get("entry_protein_locations") or []:
                for frag in loc.get("fragments") or []:
                    s, e = frag.get("start"), frag.get("end")
                    if s is not None and e is not None:
                        residue_ranges.append({"start": s, "end": e})

        source_dbs: list[dict] = []
        for db_name, db_list in (item.get("entry_integrated") or {}).items():
            for db_entry in db_list or []:
                db_acc = db_entry.get("accession", "")
                if db_acc:
                    source_dbs.append({"db": db_name, "accession": db_acc})

        entries.append({
            "accession":        ipr_acc,
            "name":             ipr_name,
            "type":             ipr_type,
            "residue_ranges":   residue_ranges,
            "source_databases": source_dbs,
        })
    return entries


def _fetch_uniprot_features_live(uniprot_acc: str, warnings: list[str]) -> list[dict]:
    """Fetch the features[] array from the UniProt API."""
    try:
        import requests
        resp = requests.get(
            f"https://rest.uniprot.org/uniprotkb/{uniprot_acc}.json",
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json().get("features", [])
        warnings.append(f"UniProt API returned HTTP {resp.status_code} for {uniprot_acc}")
    except Exception as exc:
        warnings.append(f"UniProt feature fetch failed: {exc}")
    return []


def _fetch_interpro_live(uniprot_acc: str, warnings: list[str]) -> list[dict]:
    """Fetch InterPro domain entries for a UniProt accession.

    InterPro returns HTTP 204 (No Content) when the accession is valid but has
    no InterPro entries — this is normal and is treated as an empty list.
    """
    try:
        import requests
        url = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{uniprot_acc}/?format=json&page_size=100"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("results", [])
        if resp.status_code == 204:
            # Valid accession but no InterPro entries
            return []
        warnings.append(f"InterPro API returned HTTP {resp.status_code} for {uniprot_acc}")
    except Exception as exc:
        warnings.append(f"InterPro fetch failed: {exc}")
    return []


def _get_uniprot_sequence(uniprot_acc: str, warnings: list[str]) -> str | None:
    try:
        import requests
        resp = requests.get(
            f"https://rest.uniprot.org/uniprotkb/{uniprot_acc}.json", timeout=30
        )
        if resp.status_code == 200:
            return resp.json().get("sequence", {}).get("value")
    except Exception as exc:
        warnings.append(f"Could not fetch UniProt sequence for {uniprot_acc}: {exc}")
    return None


def _get_chain_sequence(struct, chain_id: str) -> str:
    """Extract one-letter protein sequence for a chain from the loaded structure."""
    AA3TO1 = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
        "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
        "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
        "HID":"H","HIE":"H","HIP":"H","CYX":"C","MSE":"M","SEC":"C",
    }
    try:
        residues = struct.u.select_atoms(
            f"(chainID {chain_id} or segid {chain_id}) and protein"
        ).residues
        seq = "".join(AA3TO1.get(r.resname.upper(), "X") for r in residues)
        return seq
    except Exception:
        return ""


__all__ = ["profile_complex"]
