"""MCP tools for structure preparation.

Three tools:
  validate_structure(path, profile=None)  — synchronous inventory + profile_preview
  prepare_structure(path, profile)         — async job submission; returns job_id
  get_preparation_status(job_id)           — poll status; returns artefact paths on completion

Calls the structure-prep FastAPI service (SP_SERVICE_URL env var, default http://structure-prep:8080).
Path confinement is enforced by the service; we trust it.
"""

from __future__ import annotations

import json
import os
import logging
from typing import Any, Annotated

import httpx
from pydantic import Field

logger = logging.getLogger(__name__)

_SP_URL = os.environ.get("SP_SERVICE_URL", "http://structure-prep:8080")
_TIMEOUT = 30.0  # seconds for sync calls; /jobs/{id} polling is short


def _client() -> httpx.Client:
    return httpx.Client(base_url=_SP_URL, timeout=_TIMEOUT)


def _extract_ligand_sdf_local(pdb_path: str, comp_id: str, smiles: str, out_path: str) -> dict:
    """Extract a ligand residue from PDB and write an SDF with correct bond orders.

    Uses gemmi to isolate the residue and rdkit to assign bond orders from a
    SMILES template so the SDF has correct aromaticity/bond order for strain
    calculation.
    """
    try:
        import gemmi
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from pathlib import Path
        import tempfile

        # Read structure and find the ligand residue
        st = gemmi.read_structure(pdb_path)
        model = st[0]

        mini_st = gemmi.Structure()
        mini_model = gemmi.Model("1")
        found = False
        for chain in model:
            for res in chain:
                if res.name.strip() == comp_id.strip():
                    mini_chain = gemmi.Chain("A")
                    mini_chain.add_residue(res)
                    mini_model.add_chain(mini_chain)
                    found = True
                    break
            if found:
                break

        if not found:
            return {"error": f"Residue {comp_id!r} not found in {pdb_path}"}

        mini_st.add_model(mini_model)
        tmp_pdb = tempfile.mktemp(suffix=".pdb")
        mini_st.write_pdb(tmp_pdb)

        # Load via rdkit and assign bond orders from SMILES template
        template = Chem.MolFromSmiles(smiles)
        if template is None:
            return {"error": f"Invalid SMILES: {smiles!r}"}
        mol_raw = Chem.MolFromPDBFile(tmp_pdb, removeHs=False, sanitize=False)
        if mol_raw is None:
            return {"error": "Could not parse extracted residue PDB"}

        mol_fixed = AllChem.AssignBondOrdersFromTemplate(template, mol_raw)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(out_path)
        writer.write(mol_fixed)
        writer.close()

        return {
            "sdf_path": out_path,
            "atoms": mol_fixed.GetNumAtoms(),
            "has_3d": mol_fixed.GetNumConformers() > 0,
        }
    except Exception as e:
        return {"error": str(e)}


def register_tools(mcp) -> None:
    """Register structure preparation tools with the MCP server."""

    @mcp.tool()
    def validate_structure(
        path: Annotated[str, Field(description="Shared-volume path to PDB or CIF file (e.g. 'rcsb_structures/3PXY.cif')")],
        profile: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Optional partial PreparationProfile JSON to generate a tailored "
                    "profile_preview. Omit to get the default inventory."
                ),
            ),
        ] = None,
    ) -> str:
        """Inspect a raw PDB/CIF file without modifying it.

        Returns a complete inventory (chains, HETATM groups, altloc positions,
        missing residues, disulfide bonds) and a profile_preview (proposed chain
        roles, detected ligands, detected additives, altloc candidates).

        Call this BEFORE prepare_structure to review ambiguous chain roles and
        ligand/additive classifications. The agent can then construct a targeted
        PreparationProfile addressing any flagged ambiguities.
        """
        payload: dict[str, Any] = {"path": path}
        if profile:
            payload["profile"] = profile
        try:
            with _client() as client:
                resp = client.post("/validate", json=payload)
                resp.raise_for_status()
                return json.dumps(resp.json(), indent=2)
        except httpx.HTTPStatusError as e:
            return json.dumps({"error": str(e), "detail": e.response.text}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

    @mcp.tool()
    def prepare_structure(
        path: Annotated[str, Field(description="Shared-volume path to PDB or CIF file")],
        profile: Annotated[
            dict[str, Any],
            Field(
                description=(
                    "Complete PreparationProfile JSON. Must include '\"version\": \"1.0\"'. "
                    "Build this after reviewing the validate_structure output. "
                    "Key fields: stage3_curation.ligand_policy, stage3_curation.keep_metals, "
                    "stage3_curation.keep_waters, stage4_altlocs.altloc_policy, "
                    "stage5_missing.missing_policy, stage6_protonation.protonation_method (default pdb2pqr_propka), "
                    "stage8_ligand.ligand_ff (default openff_2.1), output.emit_fasta (true for Boltz)."
                ),
            ),
        ],
    ) -> str:
        """Submit an asynchronous structure preparation job.

        Returns immediately with a job_id. Poll with get_preparation_status(job_id)
        until status='completed'. On completion, the response includes artifact paths
        for all output files.

        Default profile (safe for X-ray drug-target, single ligand):
          - Assembly 1, highest-occupancy altlocs, model_atoms for missing
          - pdb2pqr_propka at pH 7.4, no termini capping
          - OpenFF 2.1 + NAGL charges, emit SDF for GNINA
          - No minimization

        For NMR structures: set stage1_assembly.model_selection='1' (keep representative model).
        For simulation: set stage7_termini.cap_termini='ace_nme' and stage9_minimization.minimize='ligand'.
        """
        payload = {"path": path, "profile": profile}
        try:
            with httpx.Client(base_url=_SP_URL, timeout=10.0) as client:
                resp = client.post("/jobs", json=payload)
                resp.raise_for_status()
                return json.dumps(resp.json(), indent=2)
        except httpx.HTTPStatusError as e:
            return json.dumps({"error": str(e), "detail": e.response.text}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

    @mcp.tool()
    def get_preparation_status(
        job_id: Annotated[str, Field(description="Job ID returned by prepare_structure")],
    ) -> str:
        """Poll a structure preparation job.

        Returns status ('queued'|'running'|'completed'|'failed'|'cancelled'),
        and on completion:
          - artifacts: {complex, receptor, receptor_noh, ligands, sequences_fasta, ...}
          - report_summary: {histidines (with HID/HIE/HIP state), removed_additives, warnings, timings}

        Artifact paths are relative to /shared — pass them directly to
        profile_interactions (complex path) or use receptor_noh for GNINA.
        The sequences_fasta path → read it for Boltz target.protein_sequence.
        """
        try:
            with _client() as client:
                resp = client.get(f"/jobs/{job_id}")
                resp.raise_for_status()
                return json.dumps(resp.json(), indent=2)
        except httpx.HTTPStatusError as e:
            return json.dumps({"error": str(e), "detail": e.response.text}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

    @mcp.tool()
    def extract_ligand_sdf(
        pdb_path: Annotated[str, Field(description="Absolute or /shared-relative path to the prepared PDB.")],
        ligand_comp_id: Annotated[str, Field(description="Three-letter CCD residue name (e.g. 'MI1', '35R').")],
        smiles: Annotated[str, Field(description="Canonical SMILES — used as bond-order template for correct aromaticity in the SDF.")],
        out_path: Annotated[str | None, Field(default=None, description="Output SDF path. Defaults to <pdb_stem>_<comp_id>.sdf.")],
    ) -> str:
        """Extract a ligand residue from a prepared PDB and write a 3D SDF.

        The SDF uses crystallographic 3D coordinates with correct bond orders from
        the SMILES template. Pass the resulting SDF path to profile_complex_interaction
        to enable MMFF94 conformational strain calculation.

        Returns: {sdf_path, sdf_relative_path, atoms, has_3d} on success, {error} on failure.
        ``sdf_relative_path`` is relative to the shared volume root — pass this value
        directly as the ``ligand_sdf`` argument to ``profile_complex_interaction``.
        """
        import os
        from pathlib import Path

        shared_root = os.environ.get("IP_SHARED_DIR", "/shared")
        p = pdb_path.strip()
        if not os.path.isabs(p):
            p = os.path.join(shared_root, p)
        if not out_path:
            out_path = str(Path(p).parent / f"{Path(p).stem}_{ligand_comp_id}.sdf")

        result = _extract_ligand_sdf_local(p, ligand_comp_id, smiles, out_path)
        if "sdf_path" in result:
            # Add the relative path (what the profiler service sees as /shared/...)
            out = Path(result["sdf_path"])
            shared = Path(shared_root)
            try:
                result["sdf_relative_path"] = str(out.relative_to(shared))
            except ValueError:
                result["sdf_relative_path"] = out.name  # fallback to filename only
        return json.dumps(result, indent=2)
