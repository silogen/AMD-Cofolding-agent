# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Public API: :func:`profile_batch`, :func:`profile_molecule`, :func:`to_table`.

Pipeline per molecule:

1. No structure -> prepare ligand from SMILES, emit hierarchical taxonomy.
2. Structure  -> prepare receptor (PDBFixer, pH) + ligand (template bond
   assignment + Hs, pH), fragment the ligand, crop the pocket, run the
   geometric interaction detector, attach fragments, group into triples.

One batch in -> one :class:`BatchResult` out.
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Any

from interaction_profiler.fragmentation import build_composition
from interaction_profiler.interactions import detect_interactions
from interaction_profiler.interactions.features import MetalSite, WaterSite
from interaction_profiler.interactions.ranking import rank_and_cap
from interaction_profiler.io import MoleculeInput, load_batch, load_structure
from interaction_profiler.io.structure import (
    METALS,
    WATER,
    LigandSelection,
    parse_covalent_links,
    prefer_formula_match,
)
from interaction_profiler.join import attach_fragments, build_triples
from interaction_profiler.prep import (
    analyze_stereo,
    inchikey_of_mol,
    inchikey_of_smiles,
    inchikey_skeleton,
    prepare_ligand_from_smiles,
    prepare_ligand_from_structure,
    prepare_receptor,
)
from interaction_profiler.prep.ligand import formula_of_smiles, validate_smiles
from interaction_profiler.schemas import (
    BatchProvenance,
    BatchResult,
    MoleculeMetadata,
    MoleculeResult,
    PreparationReport,
)

logger = logging.getLogger(__name__)


def _ligand_selection_str(sel: LigandSelection) -> str:
    base = f"resname {sel.resname} and resid {sel.resnum}"
    if sel.chain:
        base = f"({base}) and (chainID {sel.chain} or segid {sel.chain})"
    return base


def _select_ligand_ag(u: Any, sel: LigandSelection) -> Any:
    ag = u.select_atoms(_ligand_selection_str(sel))
    if ag.n_atoms == 0:
        ag = u.select_atoms(f"resname {sel.resname} and resid {sel.resnum}")
    return ag


def _com(ag: Any) -> list[float] | None:
    try:

        heavy = ag.select_atoms("not name H*")
        target = heavy if heavy.n_atoms else ag
        c = target.positions.mean(axis=0)
        return [round(float(x), 2) for x in c]
    except Exception:
        return None


def _build_pocket(prepared_path: str, sel: LigandSelection, radius: float):
    """Return ``(pocket_atomgroup, metal_sites, water_sites)``.

    The pocket is the protein/nucleic residues within ``radius`` of the
    ligand (kept as an MDAnalysis AtomGroup, with explicit Hs from the
    prepared structure). Metals and waters are returned as explicit
    coordinate sites.
    """
    import MDAnalysis as mda

    u = mda.Universe(prepared_path)
    lig_str = _ligand_selection_str(sel)
    pocket = u.select_atoms(f"(protein or nucleic) and byres (around {radius} ({lig_str}))")
    pocket_ag = pocket if pocket.n_atoms else None

    metals: list[MetalSite] = []
    metal_names = " ".join(sorted(METALS))
    with contextlib.suppress(Exception):
        mag = u.select_atoms(f"(resname {metal_names}) and (around {radius} ({lig_str}))")
        for a in mag.atoms:
            metals.append(MetalSite(xyz=a.position.astype("float64"), meta=_atom_meta_mda(a)))

    waters: list[WaterSite] = []
    water_names = " ".join(sorted(WATER))
    with contextlib.suppress(Exception):
        wag = u.select_atoms(
            f"(resname {water_names}) and (name O or name OW or name OH2 or element O) "
            f"and byres (around {radius} ({lig_str}))"
        )
        for a in wag.atoms:
            waters.append(WaterSite(xyz=a.position.astype("float64"), meta=_atom_meta_mda(a)))

    return pocket_ag, metals, waters


def _atom_meta_mda(a: Any) -> dict:
    def _get(attr, default=""):
        try:
            v = getattr(a, attr)
            return v if v is not None else default
        except Exception:
            return default

    chain = str(_get("chainID") or _get("segid") or "").strip()
    return {
        "name": str(_get("name", "")).strip(),
        "resname": str(_get("resname", "")).strip(),
        "resnum": int(_get("resid", 0)),
        "chain": chain,
    }


def _input_mode(inp: MoleculeInput) -> str:
    """Classify the request: ``smiles_only`` | ``structure_only`` | ``structure+smiles``."""
    if inp.has_structure:
        return "structure+smiles" if inp.smiles else "structure_only"
    return "smiles_only"


def _annotate_inchikey(
    report: PreparationReport, template_smiles: str | None, mol: Any, *, warn_mismatch: bool = True
) -> None:
    """Record template + prepared InChIKeys and whether they match (skeleton).

    ``inchikey_match`` compares the first (connectivity) block, so it is
    robust to protonation / stereo differences between the template SMILES
    and the prepared ligand. ``warn_mismatch`` is False for taxonomy-only
    inputs, where the SMILES *is* the molecule (any skeleton diff comes from
    prep normalization, not a wrong template) so the "wrong ligand" warning
    would be misleading.
    """
    tk = inchikey_of_smiles(template_smiles)
    lk = inchikey_of_mol(mol)
    report.template_inchikey = tk
    report.ligand_inchikey = lk
    if tk and lk:
        report.inchikey_match = inchikey_skeleton(tk) == inchikey_skeleton(lk)
        report.inchikey_exact_match = tk == lk
        if report.inchikey_match is False and warn_mismatch:
            report.warnings.append(
                "prepared ligand InChIKey does not match the SMILES template "
                f"(template={tk}, ligand={lk}); wrong ligand or bond assignment?"
            )


def _effective_cap(max_interactions: int | None, n_heavy: int) -> int:
    """Resolve the interaction cap. ``None`` -> adaptive (~1.2x heavy atoms,
    clamped to [20, 60]); ``0`` or negative -> no cap; else the given value."""
    if max_interactions is None:
        return int(min(60, max(20, round(1.2 * n_heavy))))
    return max_interactions


def profile_molecule(
    inp: MoleculeInput,
    *,
    ph_default: float = 7.4,
    radius: float = 8.0,
    use_gpu: bool = False,
    max_interactions: int | None = None,
    drop_uncertain: bool = True,
    strict_ligand_match: bool = False,
) -> MoleculeResult:
    return profile_molecule_all(
        inp, ph_default=ph_default, radius=radius, use_gpu=use_gpu,
        max_interactions=max_interactions, drop_uncertain=drop_uncertain, all_ligands=False,
        strict_ligand_match=strict_ligand_match,
    )[0]


def _enforce_strict_match(m: MoleculeResult) -> None:
    """Turn an InChIKey template mismatch into a per-molecule error.

    Used when ``strict_ligand_match`` is set: a structure+SMILES input whose
    prepared ligand does not match the template skeleton is rejected (its
    interactions/triples are cleared) so a wrong ligand can't score."""
    prep = m.preparation
    # Only meaningful when a SMILES was used as a template for a structure
    # ligand -- not for taxonomy-only inputs (where the SMILES is the molecule).
    if m.metadata.input_mode != "structure+smiles":
        return
    if m.error or prep is None or prep.inchikey_match is not False:
        return
    m.error = (
        "ligand does not match SMILES template by InChIKey "
        f"(template={prep.template_inchikey}, ligand={prep.ligand_inchikey})"
    )
    m.interactions = []
    m.triples = []
    m.metadata.n_interactions = 0
    m.metadata.n_triples = 0


def profile_molecule_all(
    inp: MoleculeInput,
    *,
    ph_default: float = 7.4,
    radius: float = 8.0,
    use_gpu: bool = False,
    max_interactions: int | None = None,
    drop_uncertain: bool = True,
    all_ligands: bool = False,
    strict_ligand_match: bool = False,
) -> list[MoleculeResult]:
    """Profile a molecule; returns one result per ligand (>1 only when
    ``all_ligands`` and the structure has multiple qualifying ligands)."""
    t0 = time.time()
    ph = inp.ph if inp.ph is not None else ph_default
    md = MoleculeMetadata(name=inp.name, ph=ph, source=inp.source, has_structure=inp.has_structure,
                          input_mode=_input_mode(inp))
    # Fast-fail on an unparseable SMILES (parse + InChIKey in one pass) rather
    # than silently degrading to geometry-perceived bond orders.
    if inp.smiles and not validate_smiles(inp.smiles)[0]:
        md.timing_seconds = round(time.time() - t0, 3)
        return [MoleculeResult(metadata=md, error=f"invalid SMILES: {inp.smiles!r}")]
    try:
        if not inp.has_structure:
            results = [_taxonomy_only(inp, md, ph, t0)]
        elif inp.peptide_ligand_chain:
            results = [_profile_peptide(inp, md, ph, radius, use_gpu,
                                        max_interactions, drop_uncertain, t0)]
        else:
            results = _profile_structure_all(inp, ph, radius, use_gpu, max_interactions,
                                             drop_uncertain, all_ligands, t0)
    except Exception as exc:  # noqa: BLE001 - report per-molecule failure
        logger.exception("profiling failed for %s", inp.name)
        md.timing_seconds = round(time.time() - t0, 3)
        return [MoleculeResult(metadata=md, error=f"{type(exc).__name__}: {exc}")]
    if strict_ligand_match:
        for m in results:
            _enforce_strict_match(m)
    return results


def _taxonomy_only(inp: MoleculeInput, md: MoleculeMetadata, ph: float, t0: float) -> MoleculeResult:
    if not inp.smiles:
        raise ValueError("molecule has neither structure nor SMILES")
    prep = prepare_ligand_from_smiles(inp.smiles, ph)
    if prep.mol is None:
        raise ValueError(f"could not parse SMILES: {inp.smiles}")
    comp = build_composition(prep.mol)
    md.receptor = "none"
    md.ligand_smiles = prep.smiles_prepared or inp.smiles
    report = PreparationReport(
        ligand_prepared=True,
        ligand_protonation_ph=ph,
        ligand_smiles_input=inp.smiles,
        ligand_smiles_protonated=prep.smiles_protonated,
        ligand_smiles_prepared=prep.smiles_prepared,
        bond_assignment=prep.bond_assignment,
        stereo=analyze_stereo(prep.mol, inp.smiles),
        warnings=prep.warnings,
    )
    _annotate_inchikey(report, inp.smiles, prep.mol, warn_mismatch=False)
    md.timing_seconds = round(time.time() - t0, 3)
    return MoleculeResult(metadata=md, composition=comp, preparation=report)


def _profile_structure_all(
    inp: MoleculeInput,
    ph: float,
    radius: float,
    use_gpu: bool,
    max_interactions: int | None,
    drop_uncertain: bool,
    all_ligands: bool,
    t0: float,
) -> list[MoleculeResult]:
    """Profile one structure: the best ligand (default) or every qualifying
    ligand (``all_ligands``). Receptor is prepared once and reused."""
    struct = load_structure(inp.structure)
    rkind = struct.receptor_kind() or "none"
    chains = struct.chains
    chain_types = struct.chain_types()

    mode = _input_mode(inp)
    ligs = struct.detect_ligands(
        resname=inp.ligand_resname, chain=inp.ligand_chain, keyword=inp.ligand_keyword
    )
    # When a SMILES template is supplied, prefer the ligand copy/type whose
    # heavy-atom formula matches it (identifies the intended molecule).
    if inp.smiles and not all_ligands:
        ligs = prefer_formula_match(struct.u, ligs, formula_of_smiles(inp.smiles))
    if not ligs:
        md = MoleculeMetadata(name=inp.name, ph=ph, source=inp.source, has_structure=True,
                              input_mode=mode, receptor=rkind, chains=chains, chain_types=chain_types,
                              timing_seconds=round(time.time() - t0, 3))
        return [MoleculeResult(metadata=md, error="ValueError: no small-molecule ligand detected "
                               "(pass ligand_resname/ligand_chain)")]

    recp = prepare_receptor(inp.structure, ph=ph)
    prepared_path = recp.prepared_path or inp.structure

    targets = ligs if all_ligands else ligs[:1]
    results: list[MoleculeResult] = []
    for sel in targets:
        name = inp.name if len(targets) == 1 else f"{inp.name}:{sel.resname}{sel.resnum}.{sel.chain}"
        md = MoleculeMetadata(name=name, ph=ph, source=inp.source, has_structure=True,
                              input_mode=mode, receptor=rkind, chains=chains, chain_types=chain_types)
        try:
            results.append(_profile_selection(inp, md, struct, sel, recp, prepared_path,
                                               radius, use_gpu, max_interactions, drop_uncertain, t0))
        except Exception as exc:  # noqa: BLE001
            logger.exception("profiling failed for %s", name)
            md.timing_seconds = round(time.time() - t0, 3)
            results.append(MoleculeResult(metadata=md, error=f"{type(exc).__name__}: {exc}"))
    return results


def _profile_selection(
    inp: MoleculeInput,
    md: MoleculeMetadata,
    struct,
    sel,
    recp,
    prepared_path: str,
    radius: float,
    use_gpu: bool,
    max_interactions: int | None,
    drop_uncertain: bool,
    t0: float,
) -> MoleculeResult:
    ph = md.ph
    lig_ag = _select_ligand_ag(struct.u, sel)
    lig_prep = prepare_ligand_from_structure(lig_ag, template_smiles=inp.smiles, ph=ph)
    if lig_prep.mol is None:
        raise ValueError("ligand preparation failed")
    if not inp.smiles:
        lig_prep.warnings.append(
            "no SMILES template supplied: ligand bond orders were perceived from geometry "
            f"(bond_assignment={lig_prep.bond_assignment}); pass 'smiles' for reliable chemistry"
        )

    comp = build_composition(lig_prep.mol)
    md.ligand_smiles = lig_prep.smiles_prepared or inp.smiles
    md.ligand_com = _com(lig_ag)

    pocket_ag, metals, waters = _build_pocket(prepared_path, sel, radius)
    covalent_links = parse_covalent_links(inp.structure, sel.resname)

    interactions = []
    if pocket_ag is not None:
        interactions = detect_interactions(
            lig_prep.mol, lig_prep.atom_names, pocket_ag,
            metals=metals, waters=waters, covalent_links=covalent_links, use_gpu=use_gpu,
        )
    else:
        lig_prep.warnings.append("pocket could not be built; no interactions computed")

    n_detected = len(interactions)
    cap = _effective_cap(max_interactions, comp.n_heavy_atoms)
    interactions = rank_and_cap(interactions, cap)
    # Optionally drop uncertain contacts AFTER the cap (so the cap still keeps
    # the highest-priority contacts, then the user can strip borderline ones).
    if drop_uncertain:
        interactions = [i for i in interactions if i.confidence == "confident"]
    attach_fragments(interactions, comp)
    triples = build_triples(interactions)

    md.n_interactions = len(interactions)
    md.n_interactions_detected = n_detected
    md.n_dropped = max(0, n_detected - len(interactions))
    md.interaction_cap = cap if cap and cap > 0 else None
    md.n_triples = len(triples)
    md.n_confident = sum(t.n_confident for t in triples)
    md.n_uncertain = sum(t.n_uncertain for t in triples)
    md.timing_seconds = round(time.time() - t0, 3)

    report = PreparationReport(
        ligand_prepared=True,
        ligand_protonation_ph=ph,
        ligand_smiles_input=inp.smiles,
        ligand_smiles_protonated=lig_prep.smiles_protonated,
        ligand_smiles_prepared=lig_prep.smiles_prepared,
        ligand_n_hydrogens=lig_prep.n_hydrogens,
        bond_assignment=lig_prep.bond_assignment,
        receptor_prepared=recp.prepared_path is not None,
        receptor_ph=ph,
        receptor_atoms_added=recp.atoms_added,
        receptor_hydrogens_added=recp.hydrogens_added,
        prepared_receptor_path=recp.prepared_path,
        stereo=analyze_stereo(lig_prep.mol, inp.smiles),
        warnings=list(lig_prep.warnings) + list(recp.warnings),
    )
    _annotate_inchikey(report, inp.smiles, lig_prep.mol)
    return MoleculeResult(
        metadata=md, composition=comp, interactions=interactions,
        triples=triples, preparation=report,
    )


def _profile_peptide(
    inp: MoleculeInput,
    md: MoleculeMetadata,
    ph: float,
    radius: float,
    use_gpu: bool,
    max_interactions: int | None,
    drop_uncertain: bool,
    t0: float,
) -> MoleculeResult:
    """Profile a whole chain (e.g. a cyclic-peptide macrocycle like
    cyclosporin) as the ligand, with the rest of the structure as receptor."""
    ch = inp.peptide_ligand_chain
    struct = load_structure(inp.structure)
    md.receptor = struct.receptor_kind() or "none"
    md.chains = struct.chains
    md.chain_types = struct.chain_types()

    chain_sel = f"(chainID {ch} or segid {ch})"
    lig_ag = struct.u.select_atoms(chain_sel)
    if lig_ag.n_atoms == 0:
        raise ValueError(f"peptide-ligand chain {ch!r} has no atoms")

    lig_prep = prepare_ligand_from_structure(lig_ag, template_smiles=inp.smiles, ph=ph)
    if lig_prep.mol is None:
        raise ValueError("peptide-ligand preparation failed")
    if not inp.smiles:
        lig_prep.warnings.append(
            "no SMILES template supplied: ligand bond orders were perceived from geometry "
            f"(bond_assignment={lig_prep.bond_assignment}); pass 'smiles' for reliable chemistry"
        )
    comp = build_composition(lig_prep.mol)
    md.ligand_smiles = lig_prep.smiles_prepared or inp.smiles
    md.ligand_com = _com(lig_ag)

    recp = prepare_receptor(inp.structure, ph=ph)
    prepared_path = recp.prepared_path or inp.structure

    import MDAnalysis as mda

    u = mda.Universe(prepared_path)
    pocket_ag = u.select_atoms(
        f"(protein or nucleic) and not {chain_sel} and byres (around {radius} ({chain_sel}))"
    )
    interactions = []
    if pocket_ag.n_atoms:
        interactions = detect_interactions(
            lig_prep.mol, lig_prep.atom_names, pocket_ag, use_gpu=use_gpu
        )
    else:
        lig_prep.warnings.append("no receptor within radius of the peptide ligand")

    n_detected = len(interactions)
    cap = _effective_cap(max_interactions, comp.n_heavy_atoms)
    interactions = rank_and_cap(interactions, cap)
    if drop_uncertain:
        interactions = [i for i in interactions if i.confidence == "confident"]
    attach_fragments(interactions, comp)
    triples = build_triples(interactions)

    md.n_interactions = len(interactions)
    md.n_interactions_detected = n_detected
    md.n_dropped = max(0, n_detected - len(interactions))
    md.interaction_cap = cap if cap and cap > 0 else None
    md.n_triples = len(triples)
    md.n_confident = sum(t.n_confident for t in triples)
    md.n_uncertain = sum(t.n_uncertain for t in triples)
    md.timing_seconds = round(time.time() - t0, 3)

    report = PreparationReport(
        ligand_prepared=True, ligand_protonation_ph=ph, ligand_smiles_input=inp.smiles,
        ligand_smiles_prepared=lig_prep.smiles_prepared, ligand_n_hydrogens=lig_prep.n_hydrogens,
        bond_assignment=lig_prep.bond_assignment, receptor_prepared=recp.prepared_path is not None,
        receptor_ph=ph, prepared_receptor_path=recp.prepared_path,
        stereo=analyze_stereo(lig_prep.mol, inp.smiles),
        warnings=list(lig_prep.warnings) + list(recp.warnings),
    )
    _annotate_inchikey(report, inp.smiles, lig_prep.mol)
    return MoleculeResult(metadata=md, composition=comp, interactions=interactions,
                          triples=triples, preparation=report)


def _profile_one(args) -> list[MoleculeResult]:
    """Top-level worker for the parallel batch (must be importable for spawn)."""
    inp, kwargs = args
    return profile_molecule_all(inp, **kwargs)


def profile_batch(
    batch_path: str | Path,
    *,
    ph: float = 7.4,
    radius: float = 8.0,
    use_gpu: bool = False,
    max_interactions: int | None = None,
    drop_uncertain: bool = True,
    all_ligands: bool = False,
    strict_ligand_match: bool = False,
    workers: int = 1,
    out_path: str | Path | None = None,
) -> BatchResult:
    batch_id, molecules = load_batch(batch_path)
    from interaction_profiler.interactions.geometry import gpu_available

    prov = BatchProvenance(
        interaction_radius=radius,
        max_interactions=max_interactions,
        drop_uncertain=drop_uncertain,
        gpu=bool(use_gpu and gpu_available()),
        # version + tool_versions are filled by BatchProvenance defaults.
    )
    result = BatchResult(batch_id=batch_id, n_molecules=len(molecules), provenance=prov)

    kwargs = dict(ph_default=ph, radius=radius, use_gpu=use_gpu,
                  max_interactions=max_interactions, drop_uncertain=drop_uncertain,
                  all_ligands=all_ligands, strict_ligand_match=strict_ligand_match)
    if workers and workers > 1 and not use_gpu and len(molecules) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        # 'spawn' avoids deadlocks from forking after native libs (OpenMM/RDKit/
        # MDAnalysis) are already imported in the parent.
        ctx = mp.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                for res_list in ex.map(_profile_one, [(m, kwargs) for m in molecules]):
                    result.molecules.extend(res_list)
        except Exception:
            logger.exception("parallel batch failed; falling back to serial")
            result.molecules = []
            for m in molecules:
                result.molecules.extend(profile_molecule_all(m, **kwargs))
    else:
        for m in molecules:
            result.molecules.extend(profile_molecule_all(m, **kwargs))

    result.n_molecules = len(result.molecules)
    if out_path is not None:
        Path(out_path).write_text(result.model_dump_json(indent=2, exclude_none=False))
    return result


def to_table(result: BatchResult):
    """Return a pandas DataFrame (one row per interaction)."""
    import pandas as pd

    return pd.DataFrame(result.to_rows())


__all__ = ["profile_batch", "profile_molecule", "to_table"]
