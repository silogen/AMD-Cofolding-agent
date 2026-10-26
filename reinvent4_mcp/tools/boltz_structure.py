"""MCP tool — structure-conditioned Boltz prediction (block 09).

Exposes ``POST /predict_structure`` from the Boltz2 service as an MCP tool.
The agent provides a prepared protein structure (from block 03) and a ligand
SMILES; Boltz conditions the diffusion on the protein backbone template so
the ligand is placed in the constrained pocket.

Returns all diffusion samples ranked by confidence, the best structure path,
affinity metadata, and an optional PoseBusters plausibility result.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Annotated, Any

import httpx
from pydantic import Field

logger = logging.getLogger(__name__)

_BOLTZ_URL = os.environ.get("BOLTZ_SERVICE_URL", "http://boltz:8080")
_TIMEOUT    = 600.0   # ~20 s per molecule on MI300X; 600 s covers batches of ~30


def _client() -> httpx.Client:
    # This is an internal service URL (usually host-loopback under Slurm).
    # Ignore ambient HTTP(S)_PROXY settings so localhost traffic is never
    # sent to, or resolved through, a corporate proxy.
    return httpx.Client(
        base_url=_BOLTZ_URL,
        timeout=_TIMEOUT,
        trust_env=False,
    )


def _resolve_shared_path(path_str: str | None) -> str | None:
    if path_str is None:
        return None
    shared_root = os.environ.get("BOLTZ_SHARED_DIR", "/shared")
    p = path_str.strip()
    if p.startswith(shared_root + "/"):
        return p[len(shared_root) + 1:]
    return p


def register_tools(mcp) -> None:
    """Register the structure-conditioned Boltz prediction tool."""

    @mcp.tool()
    def predict_structure_conditioned(
        smiles: Annotated[
            str,
            Field(description="Ligand SMILES string to place in the protein pocket."),
        ],
        protein_sequence: Annotated[
            str,
            Field(description=(
                "Amino-acid sequence of the receptor chain (one-letter code). "
                "Must match the sequence in the template structure."
            )),
        ],
        structure_path: Annotated[
            str,
            Field(description=(
                "Path under /shared to the prepared PDB or mmCIF structure file. "
                "Obtain this from the structure preparator (block 03) or RCSB "
                "fetcher (block 01).  Example: '/shared/prepared/6m0j_prepared.pdb'."
            )),
        ],
        chain_ids: Annotated[
            list[str],
            Field(description=(
                "Protein chain IDs in the template file to use for conditioning. "
                "E.g. ['A'] for a single-chain target, ['H','L'] for a Fab."
            )),
        ],
        template_ids: Annotated[
            list[str] | None,
            Field(
                default=None,
                description=(
                    "Boltz template chain IDs corresponding to chain_ids.  "
                    "None = preserve mmCIF chain IDs or auto-assign the PDB "
                    "A1/B1 convention. Only provide this after inspecting the "
                    "template when a custom mapping is required."
                ),
            ),
        ] = None,
        force: Annotated[
            bool,
            Field(
                default=False,
                description=(
                    "When True, enforce the template via a harmonic potential — "
                    "the backbone cannot deviate more than 'threshold' Å. "
                    "When False (default), the template is used as a soft prior."
                ),
            ),
        ] = False,
        threshold: Annotated[
            float,
            Field(
                default=2.0,
                description=(
                    "Maximum backbone deviation in Ångström allowed when force=True. "
                    "Ignored when force=False."
                ),
            ),
        ] = 2.0,
        expected_sha256: Annotated[
            str | None,
            Field(
                default=None,
                description=(
                    "Optional SHA-256 hex digest of the structure file. "
                    "When provided, the service verifies the file before running. "
                    "Recommended for reproducibility."
                ),
            ),
        ] = None,
        diffusion_samples: Annotated[
            int,
            Field(
                default=3,
                description=(
                    "Number of independent ligand poses to generate. "
                    "Returned sorted by Boltz confidence score (highest first). "
                    "Higher values improve diversity but increase runtime."
                ),
            ),
        ] = 3,
        run_posebusters: Annotated[
            bool,
            Field(
                default=True,
                description=(
                    "Run PoseBusters on the best predicted pose to check "
                    "chemical plausibility (clashes, bond lengths, valence). "
                    "Requires POSEBUSTERS_SERVICE_URL to be reachable."
                ),
            ),
        ] = True,
        msa_path: Annotated[
            str | None,
            Field(
                default=None,
                description=(
                    "Path to a pre-computed .a3m MSA file for the receptor. "
                    "Improves confidence vs. single-sequence mode. "
                    "Use list_shared_assets to browse available MSA files."
                ),
            ),
        ] = None,
        pocket_constraints: Annotated[
            list[dict] | None,
            Field(
                default=None,
                description=(
                    "Boltz pocket constraints (E1). Each entry: "
                    "{binder: 'B', contacts: [['A', 145], ['A', 41]], "
                    "max_distance: 6.0, force: true}. "
                    "Use resolve_author_residue to translate PDB author numbers "
                    "to Boltz sequence positions before setting contacts. "
                    "Force automatically enables --use_potentials (E4)."
                ),
            ),
        ] = None,
        contact_constraints: Annotated[
            list[dict] | None,
            Field(
                default=None,
                description=(
                    "Boltz contact constraints (E1). Each entry: "
                    "{token1: ['A', 145], token2: ['B', 'O20'], max_distance: 3.5, force: true}. "
                    "Ligand atom names can be computed with the boltz_atom_names helper."
                ),
            ),
        ] = None,
        bond_constraints: Annotated[
            list[dict] | None,
            Field(
                default=None,
                description=(
                    "Boltz bond constraints for covalent warheads (L1). Each entry: "
                    "{atom1: ['A', 145, 'SG'], atom2: ['B', 1, 'C7']}. "
                    "Affinity prediction is automatically suppressed (L2)."
                ),
            ),
        ] = None,
        use_msa_server: Annotated[
            bool,
            Field(
                default=False,
                description=(
                    "Query the ColabFold MSA server for a fresh alignment. "
                    "The result is cached so subsequent calls for the same "
                    "sequence are free.  Requires internet access."
                ),
            ),
        ] = True,
        msa_free_fallback: Annotated[
            bool,
            Field(
                default=False,
                description=(
                    "If no MSA source is available, run single-sequence mode "
                    "(msa: empty).  Confidence will be lower.  Set True to "
                    "prefer a fast result over a hard failure."
                ),
            ),
        ] = False,
        recycling_steps: Annotated[
            int,
            Field(default=3, description="Number of Boltz recycling steps."),
        ] = 3,
        sampling_steps: Annotated[
            int,
            Field(default=200, description="Diffusion sampling steps for the structure."),
        ] = 200,
        affinity_mode: Annotated[
            str,
            Field(
                default="pred_value",
                description=(
                    "Boltz affinity output field to return as the primary score. "
                    "Options: pred_value | probability_binary."
                ),
            ),
        ] = "pred_value",
        output_format: Annotated[
            str,
            Field(default="pdb", description="Output structure format: pdb | mmcif."),
        ] = "pdb",
    ) -> str:
        """Predict a ligand binding pose conditioned on a prepared protein structure.

        The protein backbone is used as a structural template — Boltz places
        the ligand in the constrained pocket and returns ranked poses with
        affinity and confidence metadata.

        Typical workflow:
        1. Fetch structure: ``fetch_pdb_structure(pdb_id='6M0J')``
        2. Prepare: ``prepare_structure(structure_path=..., ph=7.4)``
        3. Predict: ``predict_structure_conditioned(smiles=..., structure_path=..., chain_ids=['A'])``
        4. Profile: ``profile_complex_interaction(structure_path=best_sample.structure_path, ...)``

        Returns JSON with:
        - ``samples``: all poses sorted by ``confidence_score`` descending
        - ``best_sample``: highest-confidence pose with optional PoseBusters result
        - ``input_structure_path``, ``input_sha256``: provenance
        - ``template_chain_ids``, ``template_ids_used``
        - ``warnings``: any non-fatal issues
        """
        payload: dict[str, Any] = {
            "smiles": smiles,
            "params": {
                # Strip FASTA headers (>...) and whitespace so raw FASTA text
                # from read_shared_file doesn't cause a Boltz 422 rejection.
                "protein_sequence": "".join(
                    line for line in protein_sequence.splitlines()
                    if not line.startswith(">")
                ).replace(" ", "").strip(),
                "diffusion_samples": diffusion_samples,
                "run_posebusters": run_posebusters,
                "recycling_steps": recycling_steps,
                "sampling_steps": sampling_steps,
                "affinity_mode": affinity_mode,
                "output_format": output_format,
                "msa_free_fallback": msa_free_fallback,
                "use_msa_server": use_msa_server,
                "template": {
                    "structure_path": _resolve_shared_path(structure_path),
                    "chain_ids": chain_ids,
                },
            },
        }
        # Only include force/threshold when force=True (mirrors build_template_dict behaviour)
        if force:
            payload["params"]["template"]["force"] = True
            payload["params"]["template"]["threshold"] = threshold
        if template_ids is not None:
            payload["params"]["template"]["template_ids"] = template_ids
        if expected_sha256 is not None:
            payload["params"]["template"]["expected_sha256"] = expected_sha256
        if msa_path is not None:
            payload["params"]["msa_path"] = _resolve_shared_path(msa_path)

        # E1-E4: pass constraints through to the service
        if pocket_constraints:
            payload["params"]["pocket_constraints"] = pocket_constraints
        if contact_constraints:
            payload["params"]["contact_constraints"] = contact_constraints
        if bond_constraints:
            payload["params"]["bond_constraints"] = bond_constraints

        try:
            with _client() as client:
                resp = client.post("/predict_structure", json=payload)
                resp.raise_for_status()
                return json.dumps(resp.json(), indent=2)
        except httpx.HTTPStatusError as exc:
            return json.dumps(
                {"error": str(exc), "detail": exc.response.text}, indent=2
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)}, indent=2)
