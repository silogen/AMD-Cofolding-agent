"""Runtime-contract tests for unified generation campaigns."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from reinvent4_mcp.tools import generation
from reinvent4_mcp.ttp.schema import (
    GenerationSection,
    Mol2MolInput,
    ScoringSection,
    TTP,
)
from reinvent4_mcp.ttp.store import TTPStore


def _registered_tools() -> dict:
    mcp = MagicMock()
    registered: dict = {}

    def tool():
        def decorator(function):
            registered[function.__name__] = function
            return function

        return decorator

    mcp.tool = tool
    generation.register_tools(mcp)
    return registered


def _config(tmp_path: Path):
    return SimpleNamespace(ttp=SimpleNamespace(storage_dir=str(tmp_path / "ttps")))


def _mol2mol_ttp() -> TTP:
    return TTP(
        id="mol2mol-campaign",
        name="Mol2Mol campaign",
        scoring=ScoringSection(objectives=[]),
        generation=GenerationSection(
            mode="mol2mol",
            model=".m2m_medium",
            n_steps=25,
            device="cpu",
            mol2mol_input=Mol2MolInput(seed_smiles=["CCO", "CCN"]),
        ),
    )


def _manifest():
    return {
        "registry_key": ".m2m_medium",
        "available": True,
        "supported_modes": ["mol2mol"],
        "files": [],
    }


def test_create_mol2mol_campaign_selects_default_model(tmp_path) -> None:
    tools = _registered_tools()
    config = _config(tmp_path)

    with (
        patch("reinvent4_mcp.config.get_config", return_value=config),
        patch(
            "reinvent4_mcp.tools.discovery.resolve_model_manifest",
            return_value=_manifest(),
        ),
    ):
        result = tools["create_generation_campaign"](
            campaign_name="Mol2Mol",
            campaign_id="mol2mol-created",
            mode="mol2mol",
            seed_smiles=["CCO"],
        )

    assert result["validation_result"]["valid"] is True
    stored = TTPStore(config.ttp.storage_dir).get("mol2mol-created")
    assert stored.generation.model == ".m2m_medium"
    assert stored.generation.mol2mol_input.seed_smiles == ["CCO"]


def test_mol2mol_dry_run_references_seed_file(tmp_path) -> None:
    tools = _registered_tools()
    config = _config(tmp_path)
    ttp = _mol2mol_ttp()

    with (
        patch("reinvent4_mcp.config.get_config", return_value=config),
        patch.object(TTPStore, "get", return_value=ttp),
        patch(
            "reinvent4_mcp.tools.discovery.resolve_model_manifest",
            return_value=_manifest(),
        ),
        patch(
            "reinvent4_mcp.jobs.toml_generator.resolve_model_path",
            return_value="/models/mol2mol.prior",
        ),
    ):
        result = tools["run_generation_campaign"](
            ttp_id=ttp.id,
            dry_run=True,
        )

    assert result["dry_run"] is True
    assert result["auxiliary_files"] == ["seed.smi"]
    assert 'smiles_file = "seed.smi"' in result["toml_preview"]
    assert "max_steps = 25" in result["toml_preview"]


def test_mol2mol_submission_uses_singleton_and_real_paths(tmp_path) -> None:
    tools = _registered_tools()
    config = _config(tmp_path)
    ttp = _mol2mol_ttp()
    job_dir = tmp_path / "jobs" / "job_abc123"
    manager = MagicMock()
    manager.submit_toml.return_value = {
        "job_id": "abc123",
        "state": "running",
        "working_dir": str(job_dir),
        "csv_path": str(job_dir / "results_1.csv"),
        "checkpoint_path": str(job_dir / "stage.chkpt"),
        "stopped_previous_jobs": [{"job_id": "old-job"}],
    }

    with (
        patch("reinvent4_mcp.config.get_config", return_value=config),
        patch.object(TTPStore, "get", return_value=ttp),
        patch(
            "reinvent4_mcp.tools.discovery.resolve_model_manifest",
            return_value=_manifest(),
        ),
        patch(
            "reinvent4_mcp.jobs.toml_generator.resolve_model_path",
            return_value="/models/mol2mol.prior",
        ),
        patch(
            "reinvent4_mcp.jobs.manager.get_job_manager",
            return_value=manager,
        ),
    ):
        result = tools["run_generation_campaign"](
            ttp_id=ttp.id,
            dry_run=False,
        )

    assert result["submitted"] is True
    assert result["campaign_manifest"]["mode"] == "mol2mol"
    assert result["campaign_manifest"]["n_seed_smiles"] == 2
    assert result["artifact_locations"]["results_csv"] == str(
        job_dir / "results_1.csv"
    )
    assert result["stopped_previous_jobs"] == [{"job_id": "old-job"}]

    kwargs = manager.submit_toml.call_args.kwargs
    assert kwargs["max_steps"] == 25
    assert kwargs["model"] == ".m2m_medium"
    assert kwargs["extra_files"] == {"seed.smi": "CCO\nCCN\n"}
