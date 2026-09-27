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
                    "Key fields:\n"
                    "  stage3_curation.ligand_policy     — 'all'|'none'|'primary'|'listed'\n"
                    "  stage3_curation.keep_metals        — 'structural'|'all'|'none'\n"
                    "  stage3_curation.keep_waters        — 'none'|'pocket'|'structural'|'all'\n"
                    "  stage3_curation.water_pocket_radius_A — float (default 4.0)\n"
                    "  stage2_roles.chain_policy          — 'keep_all'|'keep_target'|'keep_listed'\n"
                    "  stage2_roles.keep_chains           — list[str] (used when chain_policy='keep_listed')\n"
                    "  stage4_altlocs.altloc_policy       — 'highest_occupancy'|'all'|'none'\n"
                    "  stage5_missing.missing_policy      — 'model_atoms'|'none'\n"
                    "  stage6_protonation.protonation_method — 'pdb2pqr_propka'|'pdbfixer_basic'\n"
                    "  stage8_ligand.ligand_ff             — 'openff_2.1'|'none'\n"
                    "  output.emit_fasta                  — true for Boltz input\n"
                    "Do NOT send any other top-level or nested keys — extra fields are rejected (HTTP 422)."
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

    @mcp.tool()
    def resolve_author_residue(
        numbering_json_path: Annotated[
            str,
            Field(
                description=(
                    "Absolute or /shared-relative path to the residue_numbering.json "
                    "artifact produced by prepare_structure."
                )
            ),
        ],
        chain: Annotated[str, Field(description="PDB chain ID, e.g. 'A'.")],
        author_resseq: Annotated[
            int,
            Field(description="Author (PDB) residue sequence number, e.g. 145."),
        ],
        expected_resname: Annotated[
            str | None,
            Field(
                default=None,
                description=(
                    "Optional 3-letter residue name to assert, e.g. 'CYS'. "
                    "Raises an error if the found residue does not match — "
                    "this is the primary guard against the +3-offset silent mismatch."
                ),
            ),
        ] = None,
        author_icode: Annotated[
            str,
            Field(default="", description="Insertion code, usually empty."),
        ] = "",
    ) -> str:
        """Translate a PDB author residue number into a 1-based Boltz sequence position.

        Boltz constraint tokens are indexed by 1-based position in the sequence string
        supplied at prediction time, NOT by PDB author number. Structures with non-native
        N-terminal extensions (e.g. immature 7KVL starting at author -2) shift every
        canonical residue by the extension length.

        This tool reads the residue_numbering.json artifact written by prepare_structure,
        resolves the author residue, optionally asserts its identity, and returns the
        sequence position safe to use in Boltz pocket/contact/bond constraints.

        Returns {chain, author_resseq, sequence_position, resname, uniprot_position} on
        success or {error} on failure.
        """
        import json as _json
        from pathlib import Path

        shared_root = os.environ.get("IP_SHARED_DIR", "/shared")
        p = numbering_json_path.strip()
        if not os.path.isabs(p):
            p = os.path.join(shared_root, p)

        try:
            data = _json.loads(Path(p).read_text())
        except Exception as e:
            return _json.dumps({"error": f"Cannot read numbering map: {e}"}, indent=2)

        from reinvent4_mcp.tools._numbering import (  # noqa: PLC0415
            resolve_author_resseq,
        )

        try:
            result = resolve_author_resseq(
                data["entries"],
                chain=chain,
                author_resseq=author_resseq,
                expected_resname=expected_resname,
                author_icode=author_icode,
            )
            return _json.dumps(result, indent=2)
        except ValueError as e:
            return _json.dumps({"error": str(e)}, indent=2)

    @mcp.tool()
    def validate_pose(
        smiles: Annotated[str, Field(description="Ligand SMILES.")],
        structure_path: Annotated[
            str,
            Field(description="Path to the predicted complex (from a Boltz prediction)."),
        ],
        protein_pdb_path: Annotated[
            str | None,
            Field(default=None, description="Receptor PDB path (required for 'dock' mode)."),
        ] = None,
        ligand_sdf_path: Annotated[
            str | None,
            Field(
                default=None,
                description=(
                    "Pre-extracted ligand SDF with correct bond orders. "
                    "If absent, the service attempts extraction from the complex."
                ),
            ),
        ] = None,
        mode: Annotated[
            str, Field(default="dock", description="PoseBusters mode (dock, mol, redock, gen).")
        ] = "dock",
    ) -> str:
        """Validate a predicted pose with PoseBusters (K3).

        Wraps the PoseBusters service so it is callable ad hoc after cofolding,
        not only inside a TTP pipeline. Returns {score, checks, pose_status} on
        success or {error} on failure.

        `pose_status` is one of: passed | failed | declined | unavailable | service_error.

        When `ligand_sdf_path` is not supplied, PoseBusters will attempt to
        extract the ligand from the complex PDB using the fixed residue name 'LIG'.
        For Boltz outputs the dedicated `ligand.sdf` artifact in the prediction
        directory is more reliable.
        """
        import json as _json  # noqa: PLC0415

        pb_url = os.environ.get("POSEBUSTERS_SERVICE_URL", "http://posebusters:8080")
        shared_root = os.environ.get("IP_SHARED_DIR", "/shared")

        def _abs(p: str) -> str:
            return p if os.path.isabs(p) else os.path.join(shared_root, p)

        payload: dict = {
            "smiles": [smiles],
            "params": {
                "mode": mode,
                "structure_paths": [_abs(structure_path)],
            },
        }
        if protein_pdb_path:
            payload["params"]["protein_pdb_path"] = _abs(protein_pdb_path)
        if ligand_sdf_path:
            payload["params"]["structure_paths"] = [_abs(ligand_sdf_path)]

        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(f"{pb_url}/score", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            return _json.dumps({"error": str(exc), "pose_status": "service_error"}, indent=2)

        scores = data.get("scores", [None])
        score = scores[0] if scores else None
        checks = {k: v[0] if v else None for k, v in (data.get("metadata") or {}).items()}
        pose_status = checks.get("pose_status", "unavailable" if score is None else ("passed" if score == 1.0 else "failed"))
        return _json.dumps({
            "score": score,
            "pose_status": pose_status,
            "checks": checks,
        }, indent=2)
