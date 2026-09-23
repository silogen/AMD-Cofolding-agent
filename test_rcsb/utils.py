"""Shared test utilities for the RCSB fetcher test harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Import SCHEMA_VERSION lazily to avoid circular import at module level
def _get_schema_version() -> str:
    import sys
    _ROOT = Path(__file__).parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from rcsb import SCHEMA_VERSION as _SV
    return _SV

SCHEMA_VERSION = _get_schema_version()

REQUIRED_ARTIFACT_FIELDS = {
    "artifact_id",
    "kind",
    "entry_id",
    "format",
    "assembly_id",
    "uri",
    "sha256",
    "size_bytes",
    "source_url",
    "retrieved_at",
    "schema_version",
}

REQUIRED_NORMALIZED_FIELDS = {
    "entry_id",
    "source",
    "schema_version",
    "retrieved_at",
}


def load_fixture(category: str, name: str) -> dict:
    """Load a JSON fixture; raises FileNotFoundError with a clear message."""
    path = FIXTURE_DIR / category / name
    if not path.exists():
        raise FileNotFoundError(
            f"Fixture {path} not found. "
            f"Available in {(FIXTURE_DIR / category).name}/: "
            f"{sorted(p.name for p in (FIXTURE_DIR / category).iterdir()) if (FIXTURE_DIR / category).exists() else []}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def assert_artifact_valid(artifact: dict) -> None:
    """Assert that an artifact dict contains all required fields."""
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact.keys())
    assert not missing, f"Artifact missing required fields: {sorted(missing)}\nArtifact: {artifact}"
    assert artifact["kind"] == "raw_structure", f"Expected kind=raw_structure, got {artifact['kind']!r}"
    assert artifact["schema_version"] == SCHEMA_VERSION, f"Wrong schema_version: {artifact['schema_version']!r}"
    if artifact.get("error") is None:
        assert artifact["sha256"], "sha256 must be non-empty for successful artifact"
        assert artifact["size_bytes"] >= 0, "size_bytes must be >= 0"
        assert artifact["uri"], "uri must be non-empty"


def assert_normalized_entry_valid(entry: dict) -> None:
    """Assert that a normalized entry dict has the minimum required structure."""
    missing = REQUIRED_NORMALIZED_FIELDS - set(entry.keys())
    assert not missing, f"Normalized entry missing: {sorted(missing)}"
    assert entry["source"] == "rcsb"
    assert entry["schema_version"] == SCHEMA_VERSION
    assert entry["entry_id"]


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def make_search_response(
    total_count: int = 5,
    result_ids: list[str] | None = None,
    facets: list[dict] | None = None,
) -> dict:
    """Build a minimal RCSB Search API response dict for use in tests."""
    if result_ids is None:
        result_ids = [f"MOCK{i}" for i in range(min(total_count, 25))]
    return {
        "query_id": "test-query-id",
        "result_type": "entry",
        "total_count": total_count,
        "result_list": [
            {"identifier": rid, "score": 1.0, "services": None}
            for rid in result_ids
        ],
        "facets": facets or [],
    }


def make_entries_response(entries: list[dict]) -> dict:
    """Build a minimal Data API response dict for use in tests."""
    return {"data": {"entries": entries}}


def extract_post_payload(mock_http_client: Any) -> dict:
    """Get the payload from the last POST call on a mock HTTP client."""
    return mock_http_client.last_post_payload()
