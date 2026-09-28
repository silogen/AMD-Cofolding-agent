"""MCP tools for REINVENT generation campaign management — block 08.

Provides four tools:

    list_generation_models          Discover available priors with manifest info
    validate_generation_campaign    Validate a campaign TTP before launch
    create_generation_campaign      Create and persist a generation-capable TTP
    run_generation_campaign         Validate then submit a generation job

Generation modes
----------------
reinvent   — De novo molecule generation with the Reinvent prior.
libinvent  — Scaffold-conditioned R-group decoration (LibInvent prior).
linkinvent — Fragment-to-fragment linker generation (LinkInvent prior).
mol2mol    — Seed-conditioned molecule-to-molecule lead optimization.

All tools return plain dicts.  No scientific service or root Compose file is
touched.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_KNOWN_PRIORS: dict[str, list[str]] = {
    ".reinvent": ["reinvent"],
    ".libinvent": ["libinvent"],
    ".linkinvent": ["linkinvent"],
    ".libinvent_transformer": ["libinvent"],
    ".linkinvent_transformer": ["linkinvent"],
    ".m2m_medium": ["mol2mol"],
    ".m2m_high": ["mol2mol"],
    ".m2m_similarity": ["mol2mol"],
    ".m2m_scaffold": ["mol2mol"],
    ".m2m_scaffold_generic": ["mol2mol"],
    ".m2m_mmp": ["mol2mol"],
}

_DEFAULT_MODEL_BY_MODE = {
    "reinvent": ".reinvent",
    "libinvent": ".libinvent",
    "linkinvent": ".linkinvent",
    "mol2mol": ".m2m_medium",
}


def register_tools(mcp: Any) -> None:
    """Register generation MCP tools with the server."""

    # ── 1. list_generation_models ──────────────────────────────────────────

    @mcp.tool()
    def list_generation_models(
        mode_filter: Optional[str] = None,
    ) -> dict[str, Any]:
        """List available REINVENT prior models and their manifest metadata.

        Args:
            mode_filter: Optional generation mode to filter by
                (``"reinvent"``, ``"libinvent"``, ``"linkinvent"``,
                or ``"mol2mol"``).
                None returns all available priors.

        Returns:
            ``{models: [{registry_key, mode, manifest_path, files_present,
              sha256_verified, version, notes}], mode_filter}``
        """
        from reinvent4_mcp.tools.discovery import resolve_model_manifest  # noqa: PLC0415

        results: list[dict[str, Any]] = []
        for key, modes in _KNOWN_PRIORS.items():
            if mode_filter and mode_filter not in modes:
                continue
            try:
                manifest = resolve_model_manifest(key)
                manifest["registry_key"] = key
                manifest["supported_modes"] = modes
                results.append(manifest)
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "registry_key": key,
                    "supported_modes": modes,
                    "available": False,
                    "error": str(exc),
                })

        return {"models": results, "mode_filter": mode_filter}

    # ── 2. validate_generation_campaign ───────────────────────────────────

    @mcp.tool()
    def validate_generation_campaign(
        ttp_id: Optional[str] = None,
        ttp: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Validate a generation campaign TTP without submitting a job.

        Checks (in order):
        1. TTP is parseable by the Pydantic schema.
        2. ``generation`` section is present.
        3. Mode-specific required inputs are present
           (linkinvent → fragments, libinvent → scaffold, mol2mol → seeds).
        4. Fragment / scaffold / seed SMILES are valid.
        5. Model registry key is known.
        6. Scoring objectives have no broken ``depends_on`` references.

        Args:
            ttp_id: Retrieve the TTP from the persistent store by ID.
            ttp:    Inline TTP dict (used when ttp_id is None).

        Returns:
            ``{valid: bool, errors: [...], warnings: [...]}``

        Raises:
            Nothing — all errors are returned in ``errors`` so the agent can
            surface them without crashing.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # ── Resolve TTP ──────────────────────────────────────────────────
        raw: dict[str, Any] | None = None
        if ttp_id is not None:
            try:
                from reinvent4_mcp.ttp.store import TTPStore  # noqa: PLC0415
                from reinvent4_mcp.config import get_config     # noqa: PLC0415
                store = TTPStore(get_config().ttp.storage_dir)
                ttp_obj = store.get(ttp_id)
                raw = ttp_obj.model_dump()
            except FileNotFoundError:
                errors.append(f"TTP '{ttp_id}' not found in the store.")
                return {"valid": False, "errors": errors, "warnings": warnings}
            except Exception as exc:
                errors.append(f"Could not load TTP '{ttp_id}': {exc}")
                return {"valid": False, "errors": errors, "warnings": warnings}
        elif ttp is not None:
            raw = ttp
        else:
            errors.append("Provide either ttp_id or ttp (inline dict).")
            return {"valid": False, "errors": errors, "warnings": warnings}

        # ── Parse through Pydantic ──────────────────────────────────────
        try:
            from reinvent4_mcp.ttp.schema import TTP  # noqa: PLC0415
            ttp_parsed = TTP.model_validate(raw)
        except Exception as exc:
            errors.append(f"TTP schema validation failed: {exc}")
            return {"valid": False, "errors": errors, "warnings": warnings}

        # ── Generation section present? ──────────────────────────────────
        if ttp_parsed.generation is None:
            warnings.append(
                "No 'generation' section in TTP. "
                "run_generation_campaign will use run_optimization defaults."
            )
            return {"valid": True, "errors": errors, "warnings": warnings}

        g = ttp_parsed.generation

        # ── Mode-specific SMILES validation ──────────────────────────────
        from reinvent4_mcp.ttp.schema import (  # noqa: PLC0415
            GenerationMode,
            _validate_linkinvent_fragment,
            _validate_libinvent_scaffold,
        )

        if g.mode == GenerationMode.linkinvent:
            if g.linkinvent_input is None:
                errors.append(
                    "linkinvent mode requires linkinvent_input with fragment SMILES."
                )
            else:
                frag_errors: list[str] = []
                map_nums_seen: set[int] = set()
                for i, smi in enumerate(g.linkinvent_input.fragments):
                    frag_errors.extend(_validate_linkinvent_fragment(smi, i))
                    try:
                        from rdkit import Chem  # noqa: PLC0415
                        mol = Chem.MolFromSmiles(smi)
                        if mol:
                            for atom in mol.GetAtoms():
                                if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum():
                                    mn = atom.GetAtomMapNum()
                                    if mn in map_nums_seen:
                                        frag_errors.append(
                                            f"Fragment {i}: map number :{mn} "
                                            "already used in another fragment"
                                        )
                                    map_nums_seen.add(mn)
                    except ImportError:
                        warnings.append("RDKit not available — skipping SMILES structural validation.")
                errors.extend(frag_errors)

        elif g.mode == GenerationMode.libinvent:
            if g.libinvent_input is None:
                errors.append(
                    "libinvent mode requires libinvent_input with scaffold SMILES."
                )
            else:
                scaffold_errors = _validate_libinvent_scaffold(g.libinvent_input.scaffold)
                errors.extend(scaffold_errors)
        elif g.mode == GenerationMode.mol2mol and g.mol2mol_input is None:
            errors.append(
                "mol2mol mode requires mol2mol_input with one or more seed SMILES."
            )

        # ── Model registry check ─────────────────────────────────────────
        try:
            from reinvent4_mcp.tools.discovery import resolve_model_manifest  # noqa: PLC0415
            resolve_model_manifest(g.model)
            supported_modes = _KNOWN_PRIORS.get(g.model)
            if supported_modes is not None and g.mode.value not in supported_modes:
                errors.append(
                    f"Model {g.model!r} does not support mode {g.mode.value!r}; "
                    f"supported modes: {supported_modes}."
                )
        except KeyError:
            errors.append(
                f"Model registry key '{g.model}' is not known. "
                "Use list_generation_models to see available keys."
            )
        except RuntimeError as exc:
            warnings.append(f"Model manifest check warning: {exc}")
        except Exception:  # noqa: BLE001
            warnings.append(
                f"Could not verify model '{g.model}' — "
                "manifest check will be repeated at launch."
            )

        # ── Mapper validation ─────────────────────────────────────────────
        try:
            from reinvent4_mcp.ttp.mapper import ttp_to_generation_config  # noqa: PLC0415
            ttp_to_generation_config(ttp_parsed)
        except ValueError as exc:
            errors.append(f"Generation config mapping failed: {exc}")

        # ── Covalent / affinity conflict check ───────────────────────────
        # bond_constraints in any objective's service_params suppress Boltz
        # affinity prediction (L1 and L2 are mutually exclusive).  The
        # TTP schema accepts this combination silently, but the resulting
        # run will score 0.0 on every molecule.
        has_bond_constraints = bool(
            ttp_parsed.target and ttp_parsed.target.covalent_bond
        ) or any(
            "bond_constraints" in (obj.service_params or {})
            for obj in (ttp_parsed.scoring.objectives if ttp_parsed.scoring else [])
        )
        has_boltz_affinity = any(
            "boltz" in (obj.component or "").lower() or "affinity" in (obj.name or "").lower()
            for obj in (ttp_parsed.scoring.objectives if ttp_parsed.scoring else [])
        )
        has_pose_gate = any(
            "posebusters" in str(obj.service_url or "").lower() or
            "pose_gate" in (obj.name or "").lower()
            for obj in (ttp_parsed.scoring.objectives if ttp_parsed.scoring else [])
        )
        if has_bond_constraints and has_boltz_affinity:
            warnings.append(
                "COVALENT CONFLICT: bond_constraints are present alongside a "
                "Boltz affinity objective. Boltz suppresses its affinity head "
                "whenever bond_constraints (L1 constraints) are active — every "
                "molecule will receive NaN affinity, coerced to 0.0. Remove the "
                "BoltzAffinity objective or remove bond_constraints to avoid a "
                "silent all-zero scoring run."
            )
        if has_bond_constraints and has_pose_gate:
            warnings.append(
                "COVALENT CONFLICT: bond_constraints are present alongside a "
                "PoseBusters pose gate. PoseBusters is not covalent-aware and "
                "will flag every covalent pose as 'failed' on "
                "minimum_distance_to_protein, zeroing out the gate score. "
                "Remove the PoseBusters gate or use mode='redock' for covalent poses."
            )

        valid = len(errors) == 0
        return {"valid": valid, "errors": errors, "warnings": warnings}

    # ── 3. create_generation_campaign ─────────────────────────────────────

    @mcp.tool()
    def create_generation_campaign(
        campaign_name: str,
        campaign_id: str,
        mode: str = "reinvent",
        model: Optional[str] = None,
        batch_size: int = 128,
        n_steps: int = 100,
        seed: Optional[int] = None,
        device: str = "cuda",
        objectives: Optional[list[dict[str, Any]]] = None,
        fragments: Optional[list[str]] = None,
        scaffold: Optional[str] = None,
        decoration_positions: Optional[list[int]] = None,
        seed_smiles: Optional[list[str]] = None,
        target_artifact_ids: Optional[list[str]] = None,
        prepared_structure_artifact_ids: Optional[list[str]] = None,
        site_artifact_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Create and persist a generation-capable TTP, then validate it.

        Constructs a TTP with the correct ``generation`` section for the
        chosen mode and optional scoring objectives, persists it via the
        TTP store, and runs ``validate_generation_campaign`` automatically.

        Args:
            campaign_name: Human-readable name for the campaign.
            campaign_id:   Unique TTP identifier (lowercase alphanum/hyphens).
            mode:          Generation mode: ``"reinvent"`` | ``"libinvent"``
                           | ``"linkinvent"`` | ``"mol2mol"``.
            model:         Prior registry key. None selects the mode default
                           (Mol2Mol defaults to ``".m2m_medium"``).
            batch_size:    Molecules generated per RL step.
            n_steps:       Total RL steps.
            seed:          Optional random seed for reproducibility.
            device:        Compute device (``"cuda"`` or ``"cpu"``).
            objectives:    List of scoring objective dicts
                           (type/service/weight/transform …).  None = no
                           external scoring (native descriptors only).
            fragments:     LinkInvent fragment SMILES (required for
                           ``mode="linkinvent"``).
            scaffold:      LibInvent scaffold SMILES (required for
                           ``mode="libinvent"``).
            seed_smiles:   Mol2Mol seed SMILES (required for
                           ``mode="mol2mol"``).
            decoration_positions:
                           Map numbers to decorate (LibInvent, optional).
            target_artifact_ids:
                           Provenance: target record artifact IDs.
            prepared_structure_artifact_ids:
                           Provenance: prepared structure artifact IDs.
            site_artifact_ids:
                           Provenance: binding-site artifact IDs.

        Returns:
            ``{ttp_id, validation_result}``
        """
        from reinvent4_mcp.ttp.schema import (  # noqa: PLC0415
            TTP, ScoringSection, ScoringObjective,
            GenerationMode, GenerationSection,
            LinkInventInput, LibInventInput, Mol2MolInput,
        )
        from reinvent4_mcp.ttp.store import TTPStore   # noqa: PLC0415
        from reinvent4_mcp.config import get_config     # noqa: PLC0415

        errors_pre: list[str] = []

        # ── Build mode-specific inputs ────────────────────────────────────
        linkinvent_input = None
        libinvent_input  = None
        mol2mol_input = None

        try:
            gmode = GenerationMode(mode)
        except ValueError:
            return {
                "ttp_id": None,
                "validation_result": {
                    "valid": False,
                    "errors": [
                        f"Unknown generation mode {mode!r}; expected one of "
                        f"{[item.value for item in GenerationMode]}."
                    ],
                    "warnings": [],
                },
            }

        selected_model = model or _DEFAULT_MODEL_BY_MODE[gmode.value]

        if gmode == GenerationMode.linkinvent:
            if not fragments:
                errors_pre.append(
                    "linkinvent mode requires 'fragments' (list of SMILES)."
                )
            else:
                try:
                    linkinvent_input = LinkInventInput(fragments=fragments)
                except Exception as exc:
                    errors_pre.append(f"Invalid linkinvent fragments: {exc}")

        elif gmode == GenerationMode.libinvent:
            if not scaffold:
                errors_pre.append(
                    "libinvent mode requires 'scaffold' (SMILES with [*:n] points)."
                )
            else:
                try:
                    libinvent_input = LibInventInput(
                        scaffold=scaffold,
                        decoration_positions=decoration_positions,
                    )
                except Exception as exc:
                    errors_pre.append(f"Invalid libinvent scaffold: {exc}")
        elif gmode == GenerationMode.mol2mol:
            if not seed_smiles:
                errors_pre.append(
                    "mol2mol mode requires 'seed_smiles' (one or more SMILES)."
                )
            else:
                try:
                    mol2mol_input = Mol2MolInput(seed_smiles=seed_smiles)
                except Exception as exc:
                    errors_pre.append(f"Invalid Mol2Mol seed SMILES: {exc}")

        if errors_pre:
            return {
                "ttp_id": None,
                "validation_result": {
                    "valid": False,
                    "errors": errors_pre,
                    "warnings": [],
                },
            }

        generation_section = GenerationSection(
            mode=gmode,
            model=selected_model,
            batch_size=batch_size,
            n_steps=n_steps,
            seed=seed,
            device=device,
            linkinvent_input=linkinvent_input,
            libinvent_input=libinvent_input,
            mol2mol_input=mol2mol_input,
            target_artifact_ids=target_artifact_ids or [],
            prepared_structure_artifact_ids=prepared_structure_artifact_ids or [],
            site_artifact_ids=site_artifact_ids or [],
        )

        # ── Build scoring section ─────────────────────────────────────────
        obj_models = []
        for obj_dict in objectives or []:
            try:
                obj_models.append(ScoringObjective.model_validate(obj_dict))
            except Exception as exc:
                return {
                    "ttp_id": None,
                    "validation_result": {
                        "valid": False,
                        "errors": [f"Invalid objective: {exc}"],
                        "warnings": [],
                    },
                }

        scoring_section = ScoringSection(objectives=obj_models)

        # ── Assemble and persist TTP ──────────────────────────────────────
        try:
            ttp = TTP(
                id=campaign_id,
                name=campaign_name,
                scoring=scoring_section,
                generation=generation_section,
            )
        except Exception as exc:
            return {
                "ttp_id": None,
                "validation_result": {
                    "valid": False,
                    "errors": [f"TTP construction failed: {exc}"],
                    "warnings": [],
                },
            }

        try:
            store = TTPStore(get_config().ttp.storage_dir)
            stored = store.create(ttp)
        except Exception as exc:
            return {
                "ttp_id": None,
                "validation_result": {
                    "valid": False,
                    "errors": [f"TTP persistence failed: {exc}"],
                    "warnings": [],
                },
            }

        # ── Run validation ────────────────────────────────────────────────
        validation = validate_generation_campaign(ttp_id=stored.id)
        return {"ttp_id": stored.id, "validation_result": validation}

    # ── 4. run_generation_campaign ─────────────────────────────────────────

    @mcp.tool()
    def run_generation_campaign(
        ttp_id: str,
        n_steps: Optional[int] = None,
        device: Optional[str] = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Validate then submit a generation campaign job.

        Validates the TTP before any GPU work starts.  If validation fails
        the job is NOT submitted and a structured error is returned.

        Args:
            ttp_id:   ID of the TTP to run (must have a ``generation`` section).
            n_steps:  Override the TTP's ``generation.n_steps``.
            device:   Override the TTP's ``generation.device``.
            dry_run:  If True, validate and build the config but do NOT submit.
                      Useful for inspecting the generated TOML before launch.

        Returns:
            ``{job_id, campaign_manifest, toml_preview (if dry_run),
              artifact_locations, errors (if any)}``
        """
        from reinvent4_mcp.ttp.store import TTPStore                  # noqa: PLC0415
        from reinvent4_mcp.ttp.mapper import (                        # noqa: PLC0415
            ttp_to_scorer_config, ttp_to_generation_config,
        )
        from reinvent4_mcp.jobs.toml_generator import (               # noqa: PLC0415
            generate_generation_toml, resolve_model_path,
        )
        from reinvent4_mcp.config import get_config                    # noqa: PLC0415

        # ── Validate first ────────────────────────────────────────────────
        validation = validate_generation_campaign(ttp_id=ttp_id)
        if not validation["valid"]:
            return {
                "job_id": None,
                "submitted": False,
                "validation": validation,
                "errors": validation["errors"],
            }

        # ── Load and apply overrides ──────────────────────────────────────
        store = TTPStore(get_config().ttp.storage_dir)
        ttp = store.get(ttp_id)

        if ttp.generation is not None:
            if n_steps is not None:
                ttp.generation.n_steps = n_steps
            if device is not None:
                ttp.generation.device = device

        # ── Map to configs ────────────────────────────────────────────────
        try:
            scorer_config    = ttp_to_scorer_config(ttp)
            generation_config = ttp_to_generation_config(ttp)
        except Exception as exc:
            return {
                "job_id": None,
                "submitted": False,
                "errors": [f"Config mapping failed: {exc}"],
            }

        # ── Resolve prior path ────────────────────────────────────────────
        model_key = generation_config.get("model", ".reinvent")
        try:
            prior_file = resolve_model_path(model_key)
        except ValueError as exc:
            return {
                "job_id": None,
                "submitted": False,
                "errors": [str(exc)],
            }

        # ── Generate TOML ─────────────────────────────────────────────────
        try:
            toml_str = generate_generation_toml(
                scorer_config=scorer_config,
                generation_config=generation_config,
                prior_file=prior_file,
            )
        except Exception as exc:
            return {
                "job_id": None,
                "submitted": False,
                "errors": [f"TOML generation failed: {exc}"],
            }

        if dry_run:
            auxiliary_files = []
            if generation_config.get("mode") == "mol2mol":
                auxiliary_files.append("seed.smi")
            return {
                "job_id": None,
                "submitted": False,
                "dry_run": True,
                "toml_preview": toml_str,
                "validation": validation,
                "auxiliary_files": auxiliary_files,
            }

        # ── Submit job ────────────────────────────────────────────────────
        try:
            from reinvent4_mcp.jobs.manager import get_job_manager  # noqa: PLC0415

            extra_files: dict[str, str] | None = None
            if generation_config.get("mode") == "mol2mol":
                mol2mol_input = generation_config.get("mol2mol_input") or {}
                seed_smiles = mol2mol_input.get("seed_smiles") or []
                extra_files = {"seed.smi": "\n".join(seed_smiles) + "\n"}

            mgr = get_job_manager()
            submission = mgr.submit_toml(
                toml_str,
                job_name=f"generation-{ttp_id}",
                extra_files=extra_files,
                max_steps=int(generation_config.get("n_steps", 0)),
                model=model_key,
            )
            job_id = submission["job_id"]
        except Exception as exc:
            return {
                "job_id": None,
                "submitted": False,
                "errors": [f"Job submission failed: {exc}"],
            }

        return {
            "job_id": job_id,
            "submitted": True,
            "ttp_id": ttp_id,
            "mode": generation_config.get("mode", "reinvent"),
            "model": model_key,
            "state": submission["state"],
            "validation": validation,
            "campaign_manifest": {
                "ttp_id": ttp_id,
                "mode": generation_config.get("mode", "reinvent"),
                "model": model_key,
                "n_steps": generation_config.get("n_steps"),
                "n_seed_smiles": len(
                    (generation_config.get("mol2mol_input") or {}).get(
                        "seed_smiles", []
                    )
                ),
            },
            "artifact_locations": {
                "working_dir": submission["working_dir"],
                "results_csv": submission["csv_path"],
                "checkpoint": submission["checkpoint_path"],
                "tb_logs": f"{submission['working_dir']}/tb_logs/",
            },
            "stopped_previous_jobs": submission.get("stopped_previous_jobs", []),
        }
