"""Contract tests for the FastAPI service (SMILES-only paths, no OpenMM).

Structure-based profiling is exercised by the integration tests and the
service benchmark; here we only pin the HTTP contract so refactors keep
the request/response shape stable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

# The FastAPI app now lives in the deployment wrapper (services/<name>/app.py), the single
# app per the boltz2/posebusters convention -- it is no longer a package submodule. Load it
# by path so this contract test still exercises the real service.
_APP_PATH = Path(__file__).resolve().parents[3] / "services" / "interaction-profiler" / "app.py"
if not _APP_PATH.exists():  # pragma: no cover
    pytest.skip(f"service app not found at {_APP_PATH}", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("ip_service_app", _APP_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # noqa: E402
create_app = _mod.create_app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["tool"] == "interaction-profiler"
    assert "shared_dir" in body


def test_schema(client: TestClient) -> None:
    r = client.get("/schema")
    assert r.status_code == 200
    assert "properties" in r.json()


def test_profile_smiles_only(client: TestClient) -> None:
    r = client.post("/profile", json={"molecule": {"name": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"}})
    assert r.status_code == 200
    payload = r.json()
    assert len(payload["results"]) == 1
    res = payload["results"][0]
    assert res.get("error") is None
    assert len(res["composition"]["fragments"]) > 0
    # Mode is classified and echoed on the metadata.
    assert res["metadata"]["input_mode"] == "smiles_only"
    # InChIKeys are reported; smiles-only matches its own template skeleton.
    prep = res["preparation"]
    assert prep["ligand_inchikey"]
    assert prep["inchikey_match"] is True


def test_strict_ligand_match_smiles_only_unaffected(client: TestClient) -> None:
    # strict mode only rejects structure+SMILES mismatches; smiles_only passes.
    r = client.post("/profile", json={
        "molecule": {"name": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
        "options": {"strict_ligand_match": True},
    })
    assert r.status_code == 200
    assert r.json()["results"][0].get("error") is None


def test_analyze_list_and_missing_structure(client: TestClient) -> None:
    r = client.post("/analyze", json={
        "batch_id": "t",
        "molecules": [
            {"name": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
            {"name": "nope", "structure": "does_not_exist.pdb"},
        ],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["n_molecules"] == 2
    by_name = {m["metadata"]["name"]: m for m in body["molecules"]}
    assert by_name["aspirin"].get("error") is None
    # A provided-but-missing structure yields a per-molecule error, not a 500.
    assert "not found" in (by_name["nope"].get("error") or "")


def test_invalid_smiles_rejected(client: TestClient) -> None:
    # A malformed SMILES fails fast with a clear per-molecule error (no 500).
    r = client.post("/analyze", json={
        "molecules": [
            {"name": "good", "smiles": "CCO"},
            {"name": "bad", "smiles": "C1CC"},
        ],
    })
    assert r.status_code == 200
    by_name = {m["metadata"]["name"]: m for m in r.json()["molecules"]}
    assert by_name["good"].get("error") is None
    assert "invalid SMILES" in (by_name["bad"].get("error") or "")


def test_structure_path_traversal_rejected(client: TestClient) -> None:
    # A path escaping the shared dir is refused before any file access.
    r = client.post("/analyze", json={
        "molecules": [{"name": "evil", "structure": "../../etc/passwd"}],
    })
    assert r.status_code == 200
    err = r.json()["molecules"][0].get("error") or ""
    assert "escapes the shared directory" in err


def test_bad_request_rejected(client: TestClient) -> None:
    # Neither structure nor smiles -> per-molecule error via /analyze.
    r = client.post("/analyze", json={"molecules": [{"name": "empty"}]})
    assert r.status_code == 200
    assert "neither" in (r.json()["molecules"][0].get("error") or "")
    # Malformed body (missing required 'molecules') -> 422 from validation.
    r2 = client.post("/analyze", json={"batch_id": "x"})
    assert r2.status_code == 422
