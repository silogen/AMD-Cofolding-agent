"""MCP tool for rich complex interaction profiling (v2).

Exposes the interaction-profiler's ``POST /profile_complex`` endpoint as an
MCP tool, providing:

- Typed entity model (binder / target roles)
- Enriched interactions with secondary structure, UniProt features, and
  InterPro domain context at each target residue
- Generalised ``binder_feature × target_feature × interaction_type`` triples
- Conformational strain (MMFF94, Lai 2026 method)
- Ligand efficiency metrics (LE / LLE / BEI) with priority-based affinity
- Interface geometry (BSA + shape complementarity)
- Hotspot occupancy across structure ensembles

Coexists with the existing ``profile_interactions()`` helper in
``reinvent4_mcp/tools/external.py``, which calls the v1 ``/analyze`` endpoint.
Both can be used simultaneously; this tool adds the richer v2 capability.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Annotated

import httpx
from pydantic import Field

logger = logging.getLogger(__name__)

def _ip_url() -> str:
    """Read INTERACTION_PROFILER_URL at call time so it can be set after import."""
    return os.environ.get("INTERACTION_PROFILER_URL", "http://interaction-profiler:8080")

_TIMEOUT = 120.0
_TIMEOUT_ENSEMBLE = 1800.0  # hotspot_ensemble over 100s of structures can take minutes


def _summarise_hotspot(data: dict) -> str:
    """Compact summary of a hotspot_ensemble result for the LLM context window.

    The raw response can be 2M+ tokens. Returns a compact summary with correct
    field names from the service schema:
      target_residue, occupancy, n_structures_observed, interaction_types_observed
    """
    hotspots: list[dict] = data.get("hotspots") or []
    warnings: list[str] = data.get("warnings") or []
    timing: float | None = data.get("timing_seconds")

    # parse "Hotspot ensemble: N/M structures processed successfully" from warnings
    n_total, n_ok = 0, 0
    for w in warnings:
        if "structures processed successfully" in w:
            try:
                part = w.split(":")[-1].strip()
                n_ok_str, rest = part.split("/", 1)
                n_total_str = rest.split()[0]
                n_ok = int(n_ok_str.strip())
                n_total = int(n_total_str.strip())
            except Exception:
                pass

    # sort by occupancy descending — actual service field name is "occupancy"
    sorted_hs = sorted(hotspots, key=lambda r: r.get("occupancy", 0.0), reverse=True)
    top50 = sorted_hs[:50]

    summary = {
        "mode": "hotspot_ensemble",
        "n_structures_ok": n_ok,
        "n_structures_total": n_total,
        "n_hotspot_residues": len(hotspots),
        "timing_seconds": timing,
        "compatible_pairs_count": len(data.get("compatible_pairs") or []),
        "fragment_families_count": len(data.get("fragment_families") or []),
        "top_50_residues_by_occupancy": [
            {
                "target_residue": h.get("target_residue"),
                "occupancy": round(h.get("occupancy", 0.0), 4),
                "n_structures_observed": h.get("n_structures_observed"),
                "interaction_types": h.get("interaction_types_observed"),
            }
            for h in top50
        ],
        "warnings": warnings[:20],
        "errors": [w for w in warnings if "failed" in w.lower()][:10],
    }
    return json.dumps(summary, indent=2)


def _client(timeout: float = _TIMEOUT) -> httpx.Client:
    return httpx.Client(base_url=_ip_url(), timeout=timeout)


def _resolve_shared_path(path_str: str | None) -> str | None:
    """Strip leading /shared/ prefix for service normalisation (same as external.py)."""
    if path_str is None:
        return None
    shared_root = os.environ.get("IP_SHARED_DIR", "/shared")
    p = path_str.strip()
    if p.startswith(shared_root + "/"):
        return p[len(shared_root) + 1:]
    if p.startswith("/"):
        # Absolute path outside shared root — pass through; service will validate
        return p
    return p


def register_tools(mcp) -> None:
    """Register interaction-profiler v2 tools with the MCP server."""

    @mcp.tool()
    def profile_complex_interaction(
        mode: Annotated[
            str,
            Field(
                default="ligand_target",
                description=(
                    "Profiling mode. One of:\n"
                    "  ligand_only     — 3D ligand → SMILES → fragments + strain + LE/LLE/BEI\n"
                    "  ligand_target   — small molecule in protein/RNA/DNA pocket (default)\n"
                    "  binder_target   — protein/peptide/antibody binder + target chain pair\n"
                    "  hotspot_ensemble — aggregate contact occupancy across multiple structures"
                ),
            ),
        ] = "ligand_target",
        structure: Annotated[
            str | None,
            Field(
                default=None,
                description=(
                    "Shared-volume path to the PDB or mmCIF structure file. "
                    "Required for ligand_target, binder_target, hotspot_ensemble. "
                    "For ligand_only: supply a 3D SDF/PDB file to enable strain calculation."
                ),
            ),
        ] = None,
        smiles: Annotated[
            str | None,
            Field(
                default=None,
                description=(
                    "Ligand SMILES. Required for ligand_only (if no 3D structure). "
                    "Optional for ligand_target (improves bond-order assignment if supplied)."
                ),
            ),
        ] = None,
        binder_chains: Annotated[
            list[str] | None,
            Field(
                default=None,
                description=(
                    "Chain IDs belonging to the binder entity (e.g. ['H','L'] for an antibody Fab). "
                    "None = auto-detect from structure; may emit ambiguity warnings for protein-protein."
                ),
            ),
        ] = None,
        target_chains: Annotated[
            list[str] | None,
            Field(
                default=None,
                description="Chain IDs of the target entity. None = auto-detect all protein/nucleic chains.",
            ),
        ] = None,
        binder_role: Annotated[
            str | None,
            Field(
                default=None,
                description=(
                    "Explicit entity role for the binder. One of: protein | peptide | nanobody | antibody | dna | rna. "
                    "When None: 2+ binder chains → antibody (Fv/Fab), 1 chain → protein (no length guessing). "
                    "Always provide this to avoid ambiguity warnings — e.g. binder_role='nanobody' for a VHH."
                ),
            ),
        ] = None,
        ligand_resname: Annotated[
            str | None,
            Field(default=None, description="Three-letter residue name of the small-molecule ligand."),
        ] = None,
        ligand_chain: Annotated[
            str | None,
            Field(default=None, description="Chain ID of the small-molecule ligand."),
        ] = None,
        annotation_accessions: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Optional annotation sources. Keys:\n"
                    "  uniprot: 'P24941'    — fetch UniProt active sites / binding sites / domains\n"
                    "  interpro: true       — fetch InterPro domain classifications\n"
                    "  annotation_cache_dir: '/shared/annotations'  — cache dir for fetched files\n"
                    "  structure_paths: [...]  — list of structures for hotspot_ensemble mode"
                ),
            ),
        ] = None,
        potency: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Potency values for LE / LLE / BEI calculation. "
                    "Priority: experimental > Boltz > model > docking. "
                    "Sub-keys: experimental={value, unit, type, relation, source}, "
                    "boltz={affinity_pred_value, confidence_score, ligand_iptm, posebusters_all_pass}, "
                    "model={value, unit, type, model_name}, "
                    "docking={score, unit, docking_engine, conversion_record}."
                ),
            ),
        ] = None,
        ph: Annotated[float, Field(default=7.4, description="Protonation pH.")] = 7.4,
        radius: Annotated[float, Field(default=8.0, description="Pocket crop radius in Angstroms.")] = 8.0,
        compute_strain: Annotated[bool, Field(default=True, description="Compute MMFF94 conformational strain.")] = True,
        compute_efficiency: Annotated[bool, Field(default=True, description="Compute LE / LLE / BEI.")] = True,
        compute_interface: Annotated[bool, Field(default=True, description="Compute BSA and shape complementarity.")] = True,
        fail_on_ambiguity: Annotated[
            bool,
            Field(
                default=True,
                description=(
                    "When True (default), fail with a clear error in binder_target mode "
                    "if binder_chains and target_chains are both unspecified. "
                    "This prevents silent misclassification of target chains as binders. "
                    "Set False only if auto-inference is acceptable and you understand the risk."
                ),
            ),
        ] = True,
        coordinate_source: Annotated[
            str,
            Field(default="crystal", description="crystal | boltz | gnina | generated — provenance of 3D coordinates."),
        ] = "crystal",
        analysis_profile: Annotated[
            str | None,
            Field(
                default=None,
                description=(
                    "Preset compute profile — sets all compute flags at once. "
                    "fast: interactions only (no strain/efficiency/interface). "
                    "standard: + strain + efficiency + BSA/Sc (current default behavior). "
                    "enriched: + exit/growth vector classification + unsatisfied polar detection. "
                    "full: enriched + UniProt/InterPro annotation overlay. "
                    "None = use individual compute flags (fully backward-compatible)."
                ),
            ),
        ] = None,
        site_profile_id: Annotated[
            str | None,
            Field(
                default=None,
                description=(
                    "SHA-256 cache key of a previously computed SiteProfile "
                    "(from characterize_site_profile). When supplied the service reads "
                    "the cached pocket characterization from /shared/site_cache/{id}.json "
                    "and injects pocket context (subpocket IDs, residue roles) into the "
                    "interaction analysis without recomputing fpocket."
                ),
            ),
        ] = None,
        ligand_sdf: Annotated[
            str | None,
            Field(
                default=None,
                description=(
                    "Shared-volume relative path to a pre-extracted ligand SDF "
                    "(e.g. the 'sdf' value from get_preparation_status artifacts.ligands). "
                    "When supplied, MMFF94 conformational strain is computed from this file "
                    "instead of requiring a prepared_ligand_path on the server. "
                    "Pass this whenever you have run prepare_structure first — it enables "
                    "strain and ligand efficiency in the output."
                ),
            ),
        ] = None,
        ensemble_profile: Annotated[
            str | None,
            Field(
                default=None,
                description=(
                    "Ensemble analysis preset — only for hotspot_ensemble mode. "
                    "hotspots: per-residue contact occupancy (current default). "
                    "families: + fragment families by dominant binding mode. "
                    "full: + compatible fragment pairs (requires exit vector data). "
                    "None = existing hotspot occupancy behavior."
                ),
            ),
        ] = None,
    ) -> str:
        """Profile a molecular complex with enriched biological context (v2).

        Returns enriched interactions annotated with secondary structure,
        UniProt active/binding site, and InterPro domain at each target residue,
        generalised triples, and metric bundles (strain, efficiency, interface).

        Existing ``profile_interactions()`` continues to work for simple use
        cases; use this tool when you need the richer v2 output.

        Artefact routing (from the interaction profiler result):
        - ``enriched_interactions``: typed contacts with SS + UniProt + InterPro context
        - ``generalized_triples``: binder_feature × target_feature × interaction_type rows
        - ``strain.strain_per_heavy_atom``: conformational strain in kcal/mol/HA
        - ``efficiency.le``: ligand efficiency; ``.lle``: lipophilic LE; ``.bei``: BEI
        - ``interface.bsa_A2``: buried surface area; ``.shape_complementarity``: Sc score
        - ``hotspots``: per-residue contact occupancy (hotspot_ensemble mode only)
        """
        payload: dict[str, Any] = {
            "mode": mode,
            "structure": _resolve_shared_path(structure),
            "smiles": smiles,
            "binder_chains": binder_chains,
            "target_chains": target_chains,
            "binder_role": binder_role,
            "ligand_comp_id": ligand_resname,   # service uses ligand_comp_id
            "ligand_resname": ligand_resname,   # keep both for compatibility
            "ligand_chain": ligand_chain,
            "annotation_accessions": annotation_accessions,
            "potency": potency,
            "ph": ph,
            "radius": radius,
            "compute_strain": compute_strain,
            "compute_efficiency": compute_efficiency,
            "compute_interface": compute_interface,
            "fail_on_ambiguity": fail_on_ambiguity,
            "coordinate_source": coordinate_source,
            "analysis_profile": analysis_profile,
            "site_profile_id": site_profile_id,
            "ensemble_profile": ensemble_profile,
            "ligand_sdf": ligand_sdf,
        }
        # Drop explicit None values so the service uses its own defaults.
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            # Use a longer timeout for ensemble mode — 366 structures can take minutes
            timeout = _TIMEOUT_ENSEMBLE if mode == "hotspot_ensemble" else _TIMEOUT
            with _client(timeout) as client:
                resp = client.post("/profile_complex", json=payload)
                resp.raise_for_status()
                data = resp.json()
                # hotspot_ensemble returns per-residue contact data across hundreds of
                # structures — the raw JSON can exceed the LLM context window (2M+ tokens).
                # Return a compact summary instead of the full payload.
                if mode == "hotspot_ensemble":
                    return _summarise_hotspot(data)
                return json.dumps(data, indent=2)
        except httpx.HTTPStatusError as e:
            return json.dumps({"error": str(e), "detail": e.response.text}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

    @mcp.tool()
    def get_interaction_profiler_schema() -> str:
        """Return the JSON schema for the interaction profiler v2 ComplexResult.

        Use this to understand the full structure of the rich output before
        calling profile_complex_interaction().
        """
        try:
            with _client() as client:
                resp = client.get("/schema")
                resp.raise_for_status()
                return json.dumps(resp.json(), indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

    @mcp.tool()
    def characterize_site_profile(
        structure: Annotated[
            str,
            Field(description=(
                "Path to the protein structure file (PDB or mmCIF) under /shared. "
                "No ligand is required — this runs on the apo protein."
            )),
        ],
        target_chains: Annotated[
            list[str] | None,
            Field(default=None, description=(
                "Protein chains to characterise. None = all protein chains."
            )),
        ] = None,
        analysis_profile: Annotated[
            str,
            Field(default="pocket", description=(
                "Characterization depth:\n"
                "  structure  — secondary structure + chain info only (~1 s, no network)\n"
                "  pocket     — + fpocket geometry: volume, druggability, character (~3 s)\n"
                "  annotated  — + UniProt active/binding sites + InterPro domains (~8 s, network)\n"
                "  full       — + named subpockets + residue role classification (~10 s, network)\n"
                "Default: pocket."
            )),
        ] = "pocket",
        uniprot_accession: Annotated[
            str | None,
            Field(default=None, description=(
                "UniProt accession for the target protein (e.g. 'P00533' for EGFR). "
                "Required for analysis_profile='annotated' or 'full'. "
                "Enables active/binding site mapping and InterPro domain overlay."
            )),
        ] = None,
    ) -> str:
        """Characterize a protein binding site — pocket geometry, residue roles, annotations.

        Run this BEFORE profile_complex_interaction to understand the target pocket.
        The result (SiteProfile) is cached by protein SHA-256 under /shared/site_cache/.
        Pass the returned ``site_profile_id`` to profile_complex_interaction to inject
        pocket context (subpocket IDs, residue roles, site type) into the interaction analysis.

        Returns JSON with:
        - ``site_profile.site_type``: orthosteric | allosteric | ppi | unknown
        - ``site_profile.pocket_volume_A3``: pocket volume (if fpocket ran)
        - ``site_profile.pocket_druggability``: fpocket druggability score 0–1
        - ``site_profile.pocket_character``: polar | hydrophobic | mixed
        - ``site_profile.subpockets``: named sub-regions with residue lists
        - ``site_profile.residue_roles``: catalytic / gatekeeper / anchor residues
        - ``site_profile.active_site_residues``: UniProt 'Active site' residues
        - ``site_profile.structure_sha256``: cache key for site_profile_id
        """
        annotation_accessions: dict[str, Any] = {}
        if uniprot_accession:
            annotation_accessions["uniprot"] = uniprot_accession
            annotation_accessions["interpro"] = True

        payload: dict[str, Any] = {
            "structure": _resolve_shared_path(structure),
            "target_chains": target_chains,
            "analysis_profile": analysis_profile,
            "annotation_accessions": annotation_accessions or None,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            with _client() as client:
                resp = client.post("/characterize_site", json=payload)
                resp.raise_for_status()
                return json.dumps(resp.json(), indent=2)
        except httpx.HTTPStatusError as e:
            return json.dumps({"error": str(e), "detail": e.response.text}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
