"""Protein binding-site characterization pipeline.

Combines fpocket pocket geometry, secondary structure, and UniProt/InterPro
annotations into a ``SiteProfile`` object.

This module is the implementation of ``profile_complex(mode='protein_only')``.
It is intentionally independent of the ligand-facing profiling pipeline so
the SiteProfile can be computed once and reused by subsequent ligand_target
or ensemble calls.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Standard amino-acid 3→1 letter code used by _get_chain_sequence
_AA3TO1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "MSE": "M",
}


def characterize_site(
    structure_path: str,
    target_chains: list[str] | None,
    profile_flags: dict[str, Any],
    annotation_accessions: dict[str, Any],
    warnings: list[str],
) -> "SiteProfile":  # noqa: F821  (forward ref resolved at runtime)
    """Run the site characterization pipeline and return a SiteProfile.

    Parameters
    ----------
    structure_path:
        Absolute path to the PDB / mmCIF file to characterise.
    target_chains:
        Protein chain IDs to focus on.  None = all protein chains.
    profile_flags:
        Resolved compute flags from ``resolve_profile_flags()``.
        Keys: run_fpocket, assign_ss, fetch_uniprot, fetch_interpro,
              classify_residue_roles, name_subpockets.
    annotation_accessions:
        Same dict as used by profile_complex — may contain 'uniprot',
        'interpro', 'annotation_cache_dir'.
    warnings:
        Mutable list; non-fatal issues are appended here.

    Returns
    -------
    SiteProfile
    """
    from interaction_profiler.schemas import (  # noqa: PLC0415
        SiteProfile, PocketSubpocket,
    )

    t0 = time.time()
    structure_path = str(Path(structure_path).resolve())

    # SHA-256 of the structure file
    sha256 = _sha256_file(structure_path)

    # Detect chains
    from interaction_profiler.io.structure import load_structure  # noqa: PLC0415
    struct = load_structure(structure_path)
    try:
        ct = struct.chain_types()
    except Exception:
        ct = {}

    if target_chains:
        chains = list(target_chains)
    else:
        chains = [ch for ch, tp in ct.items() if tp == "protein"]

    # ── 1. Secondary structure ────────────────────────────────────────────
    ss_map: dict = {}
    ss_method: str | None = None
    ss_summary: dict[str, float] = {}

    if profile_flags.get("assign_ss", True):
        try:
            from interaction_profiler.taxonomy.secondary_structure import (  # noqa: PLC0415
                assign_secondary_structure,
            )
            ss_map, ss_method = assign_secondary_structure(structure_path, chains)
            ss_summary = _compute_ss_summary(ss_map)
        except Exception as exc:
            warnings.append(f"Secondary structure assignment failed: {exc}")

    # ── 2. UniProt features ────────────────────────────────────────────────
    feature_map: dict = {}
    active_site_residues: list[str] = []
    binding_site_residues: list[str] = []
    uniprot_acc = annotation_accessions.get("uniprot")

    if profile_flags.get("fetch_uniprot", False) and uniprot_acc:
        try:
            feature_map, active_site_residues, binding_site_residues = _fetch_uniprot_overlay(
                uniprot_acc, chains, struct, annotation_accessions, warnings,
            )
        except Exception as exc:
            warnings.append(f"UniProt annotation failed: {exc}")

    # ── 3. InterPro domains ────────────────────────────────────────────────
    interpro_map: dict = {}
    domain_context: dict[str, str] = {}

    if profile_flags.get("fetch_interpro", False) and uniprot_acc:
        try:
            interpro_map, domain_context = _fetch_interpro_overlay(
                uniprot_acc, chains, struct, annotation_accessions, warnings,
            )
        except Exception as exc:
            warnings.append(f"InterPro annotation failed: {exc}")

    # ── 4. fpocket ────────────────────────────────────────────────────────
    fpocket_version: str | None = None
    pocket_volume: float | None = None
    pocket_druggability: float | None = None
    polar_fraction: float | None = None
    pocket_character: str | None = None
    subpockets: list[PocketSubpocket] = []

    if profile_flags.get("run_fpocket", False):
        try:
            from interaction_profiler.site.fpocket import run_fpocket  # noqa: PLC0415
            fp_result = run_fpocket(structure_path, chain_ids=chains)
            fpocket_version = fp_result.fpocket_version
            if fp_result.error:
                warnings.append(f"fpocket: {fp_result.error}")
            elif fp_result.pockets:
                top = fp_result.pockets[0]
                pocket_volume       = top.volume_A3
                pocket_druggability = top.druggability_score
                polar_fraction      = top.polarity_score
                pocket_character    = _classify_pocket_character(top.polarity_score)
                if profile_flags.get("name_subpockets", False):
                    subpockets = _build_subpockets(
                        fp_result.pockets, feature_map, interpro_map
                    )
        except Exception as exc:
            warnings.append(f"fpocket pipeline failed: {exc}")

    # ── 5. Residue role classification ────────────────────────────────────
    residue_roles: dict[str, list[str]] = {}

    if profile_flags.get("classify_residue_roles", False):
        try:
            residue_roles = _classify_residue_roles(
                chains, struct, ss_map, active_site_residues, binding_site_residues,
            )
        except Exception as exc:
            warnings.append(f"Residue role classification failed: {exc}")

    # ── 6. Site type classification ───────────────────────────────────────
    site_type, site_type_confidence = _classify_site_type(
        active_site_residues, binding_site_residues, interpro_map, pocket_druggability,
    )

    elapsed = time.time() - t0

    return SiteProfile(
        structure_path=structure_path,
        structure_sha256=sha256,
        target_chains=chains,
        analysis_profile=_flags_to_profile_name(profile_flags),
        site_type=site_type,
        site_type_confidence=site_type_confidence,
        pocket_volume_A3=pocket_volume,
        pocket_druggability=pocket_druggability,
        pocket_character=pocket_character,
        polar_fraction=polar_fraction,
        subpockets=subpockets,
        residue_roles=residue_roles,
        active_site_residues=active_site_residues,
        binding_site_residues=binding_site_residues,
        ss_summary=ss_summary,
        domain_context=domain_context,
        fpocket_version=fpocket_version,
        warnings=warnings[:],
        timing_seconds=round(elapsed, 3),
    )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def site_profile_cache_key(
    structure_path: str, target_chains: list[str] | None
) -> str:
    """Build a stable cache key from file SHA-256 + sorted chains."""
    sha = _sha256_file(structure_path)
    chains_str = ",".join(sorted(target_chains or []))
    return f"{sha}_{chains_str}"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_ss_summary(ss_map: dict) -> dict[str, float]:
    """Return fraction of each SS state across the ss_map."""
    if not ss_map:
        return {}
    counts: dict[str, int] = {}
    for state in ss_map.values():
        counts[state] = counts.get(state, 0) + 1
    total = sum(counts.values()) or 1
    return {k: round(v / total, 3) for k, v in counts.items()}


def _classify_pocket_character(polarity_score: float | None) -> str | None:
    if polarity_score is None:
        return None
    if polarity_score >= 0.6:
        return "polar"
    if polarity_score <= 0.35:
        return "hydrophobic"
    return "mixed"


def _classify_site_type(
    active_site: list[str],
    binding_site: list[str],
    interpro_map: dict,
    druggability: float | None,
) -> tuple[str, str]:
    """Deterministic site-type classification with confidence."""
    # PPI domains (known large-flat interfaces)
    _PPI_DOMAINS = frozenset({
        "SH2 domain", "SH3 domain", "PDZ domain", "Bromodomain",
        "WD40 repeat", "Leucine-rich repeat",
    })

    if active_site:
        return "orthosteric", "high"

    # Check InterPro domain names for known PPI domains
    for domain_name in interpro_map.values():
        if any(ppi in str(domain_name) for ppi in _PPI_DOMAINS):
            return "ppi", "moderate"

    if binding_site and not active_site:
        return "allosteric", "moderate"

    if druggability is not None and druggability > 0.5:
        return "unknown", "low"

    return "unknown", "low"


def _classify_residue_roles(
    chains: list[str],
    struct: Any,
    ss_map: dict,
    active_site_residues: list[str],
    binding_site_residues: list[str],
) -> dict[str, list[str]]:
    """Assign a role to each residue based on annotations and SS state."""
    roles: dict[str, list[str]] = {
        "catalytic": [],
        "gatekeeper": [],
        "anchor": [],
        "solvent_exposed": [],
        "hinge": [],
    }

    active_set  = set(active_site_residues)
    binding_set = set(binding_site_residues)

    # Use SS map to find hinge residues (loop between helix blocks)
    prev_state = None
    for res_label, state in ss_map.items():
        if prev_state in ("helix", "sheet") and state == "loop":
            roles["hinge"].append(res_label)
        prev_state = state

    # Active site → catalytic
    roles["catalytic"].extend(sorted(active_set))

    # Binding site (not active) → anchor
    roles["anchor"].extend(sorted(binding_set - active_set))

    return {k: v for k, v in roles.items() if v}


def _build_subpockets(
    pockets: list[Any],
    feature_map: dict,
    interpro_map: dict,
) -> list[Any]:
    """Build named PocketSubpocket objects from fpocket results."""
    from interaction_profiler.schemas import PocketSubpocket  # noqa: PLC0415

    subpockets = []
    for i, pocket in enumerate(pockets[:5], start=1):  # top 5 only
        pocket_id = f"S{i}"
        # Try to find a UniProt/InterPro name for this subpocket
        uniprot_feature = None
        if feature_map:
            # Check if any feature residue overlaps with pocket residues
            pocket_res_set = set(pocket.residues)
            for res_key, feat in feature_map.items():
                if res_key in pocket_res_set:
                    uniprot_feature = feat.get("feature_type")
                    break

        subpockets.append(PocketSubpocket(
            id=pocket_id,
            volume_A3=pocket.volume_A3,
            character=_classify_pocket_character(pocket.polarity_score),
            residues=pocket.residues,
            uniprot_feature=uniprot_feature,
            druggability_score=pocket.druggability_score,
        ))

    return subpockets


def _fetch_uniprot_overlay(
    uniprot_acc: str,
    chains: list[str],
    struct: Any,
    annotation_accessions: dict,
    warnings: list[str],
) -> tuple[dict, list[str], list[str]]:
    """Fetch UniProt features and map to structure residues."""
    from interaction_profiler.core_complex import (  # noqa: PLC0415
        _fetch_uniprot_features_live, _get_uniprot_sequence,
    )
    from interaction_profiler.taxonomy.annotation_mapper import (  # noqa: PLC0415
        map_uniprot_features_to_residues,
    )

    features = _fetch_uniprot_features_live(uniprot_acc, warnings)
    uniprot_seq = _get_uniprot_sequence(uniprot_acc, warnings)

    feature_map: dict = {}
    idx_to_label: dict[int, str] = {}
    active_site_residues: list[str] = []
    binding_site_residues: list[str] = []

    # Single pass so feature_map keys and idx_to_label share ONE global index
    # space.  map_uniprot_features_to_residues() returns keys that are 0-based
    # indices into *that chain's* residue list; previously they were merged with
    # feature_map.update() (chain A res 5 and chain B res 5 collided) while
    # idx_to_label used a global running offset — so the two schemes disagreed for
    # every chain after the first and labels came out wrong.  Here each chain's
    # feature keys are shifted by the same running offset used to build
    # idx_to_label, so keys never collide and every key resolves to its label.
    offset = 0
    for ch in chains:
        try:
            residues = struct.u.select_atoms(
                f"(chainID {ch} or segid {ch}) and protein"
            ).residues
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not select residues for chain {ch}: {exc}")
            continue

        # Label every residue of this chain at its global index.
        for i, r in enumerate(residues):
            chain_id = getattr(r, "chainID", None) or getattr(r, "segid", ch)
            idx_to_label[offset + i] = f"{r.resname}:{r.resid}.{chain_id}"

        # Map UniProt features for this chain and shift keys into the same space.
        try:
            ch_seq = _get_chain_sequence(struct, ch)
            if ch_seq and uniprot_seq:
                ch_map = map_uniprot_features_to_residues(
                    features, ch, ch_seq, uniprot_seq,
                )
                for local_idx, feat in ch_map.items():
                    key = offset + local_idx if isinstance(local_idx, int) else local_idx
                    feature_map[key] = feat
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"UniProt feature mapping chain {ch}: {exc}")

        offset += len(residues)

    # Extract active/binding site residue labels using the shared index→label map
    for res_key, feat in feature_map.items():
        ftype = feat.get("feature_type", "")
        # Convert integer index to label string; fall back to str(index) if not found
        label = idx_to_label.get(res_key, str(res_key)) if isinstance(res_key, int) else str(res_key)
        if "Active site" in ftype:
            active_site_residues.append(label)
        elif "Binding site" in ftype:
            binding_site_residues.append(label)

    return feature_map, active_site_residues, binding_site_residues


def _fetch_interpro_overlay(
    uniprot_acc: str,
    chains: list[str],
    struct: Any,
    annotation_accessions: dict,
    warnings: list[str],
) -> tuple[dict, dict]:
    """Fetch InterPro domains and map to structure residues."""
    from interaction_profiler.core_complex import _fetch_interpro_live  # noqa: PLC0415
    from interaction_profiler.taxonomy.annotation_mapper import (  # noqa: PLC0415
        map_interpro_to_residues,
    )
    from interaction_profiler.core_complex import (  # noqa: PLC0415
        _get_uniprot_sequence, _normalize_interpro_entries,
    )

    raw_entries = _fetch_interpro_live(uniprot_acc, warnings)
    interpro_entries = _normalize_interpro_entries(raw_entries, uniprot_acc)
    uniprot_seq = _get_uniprot_sequence(uniprot_acc, warnings)

    interpro_map: dict = {}
    domain_context: dict[str, str] = {}

    for ch in chains:
        try:
            ch_seq = _get_chain_sequence(struct, ch)
            if ch_seq and uniprot_seq:
                ch_map = map_interpro_to_residues(
                    interpro_entries, ch, ch_seq, uniprot_seq,
                )
                interpro_map.update(ch_map)
                # Pick the first domain name as the chain's domain context
                for feat in ch_map.values():
                    dn = feat.get("domain_name")
                    if dn and ch not in domain_context:
                        domain_context[ch] = dn
                        break
        except Exception as exc:
            warnings.append(f"InterPro mapping chain {ch}: {exc}")

    return interpro_map, domain_context


def _get_chain_sequence(struct: Any, chain_id: str) -> str:
    """Extract one-letter protein sequence for a chain."""
    try:
        residues = struct.u.select_atoms(
            f"(chainID {chain_id} or segid {chain_id}) and protein"
        ).residues
        return "".join(_AA3TO1.get(r.resname.upper(), "X") for r in residues)
    except Exception:
        return ""


def _flags_to_profile_name(flags: dict[str, Any]) -> str:
    """Reverse-map resolved flags to the closest profile name (best-effort)."""
    if flags.get("name_subpockets"):
        return "full"
    if flags.get("fetch_uniprot"):
        return "annotated"
    if flags.get("run_fpocket"):
        return "pocket"
    return "structure"
