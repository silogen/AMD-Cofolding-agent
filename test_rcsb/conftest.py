"""Shared test fixtures for the RCSB fetcher test harness.

sys.path is patched so that `import rcsb` resolves to
01-rcsb-pdb-fetcher/rcsb.py — no agent stack needed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── path setup ───────────────────────────────────────────────────────────────
# Insert the parent of test_rcsb/ (i.e. 01-rcsb-pdb-fetcher/) so that
# `import rcsb` and `import registry` work without any package install.
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ── fixture helpers ───────────────────────────────────────────────────────────

def load_fixture(category: str, name: str) -> dict:
    """Load a JSON fixture file from fixtures/{category}/{name}."""
    path = FIXTURE_DIR / category / name
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ── mock_http fixture ─────────────────────────────────────────────────────────

class _MockHTTPClient:
    """Replaces RCSBHTTPClient for tests; returns fixture data by fixture key."""

    def __init__(self) -> None:
        self._responses: dict[str, Any] = {}
        self._calls: list[dict] = []
        self.files_base = "https://files.rcsb.org/download"
        self.search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        self.timeout = (10, 60)
        self.max_retries = 4

    def set_response(self, key: str, data: Any) -> None:
        """Register a fixture response for a given key."""
        self._responses[key] = data

    def post_json(self, url: str, payload: dict) -> dict:
        self._calls.append({"method": "POST", "url": url, "payload": payload})
        # Key by return_type and first result of query for test matching
        key = payload.get("_test_key", "default")
        if key in self._responses:
            return self._responses[key]
        # Fall back: return the last registered response
        if self._responses:
            return next(reversed(self._responses.values()))
        raise ValueError(f"No mock response registered for key={key!r}")

    def download_stream(self, url: str, dest: Path) -> tuple[str, int]:
        self._calls.append({"method": "GET", "url": url, "dest": str(dest)})
        # Write minimal synthetic CIF content
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = f"# Synthetic CIF for {url}\ndata_MOCK\n_rcsb_id {dest.stem}\n".encode()
        dest.write_bytes(content)
        sha256 = hashlib.sha256(content).hexdigest()
        return sha256, len(content)

    @property
    def calls(self) -> list[dict]:
        return self._calls

    def last_post_payload(self) -> dict:
        posts = [c for c in self._calls if c["method"] == "POST"]
        return posts[-1]["payload"] if posts else {}


@pytest.fixture()
def mock_http_client():
    """Return a _MockHTTPClient instance; tests set responses before calling."""
    return _MockHTTPClient()


@pytest.fixture()
def mock_http(monkeypatch, mock_http_client):
    """Monkeypatch RCSBHTTPClient so tests don't hit the network."""
    monkeypatch.setattr("rcsb.RCSBHTTPClient", lambda **_kw: mock_http_client)
    return mock_http_client


# ── mock_data_api fixture ─────────────────────────────────────────────────────

class _MockDataQuery:
    """Mock for rcsbapi.data.DataQuery. Returns fixture JSON by (input_type, ids)."""

    _registry: dict[tuple, dict] = {}

    @classmethod
    def register(cls, input_type: str, ids_key: str, data: dict) -> None:
        cls._registry[(input_type, ids_key)] = data

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()

    def __init__(
        self,
        input_type: str,
        input_ids: list,
        return_data_list: list[str] | None = None,
    ) -> None:
        self.input_type = input_type
        self.input_ids = input_ids
        self.return_data_list = return_data_list or []

    def exec(self) -> dict:
        ids_key = ",".join(str(i) for i in sorted(self.input_ids))
        key = (self.input_type, ids_key)
        if key in self._registry:
            return self._registry[key]
        # Fallback: look for any key with matching input_type
        for (itype, _), val in self._registry.items():
            if itype == self.input_type:
                return val
        return {"data": {self.input_type: []}}


@pytest.fixture()
def mock_data_api(monkeypatch):
    """Monkeypatch rcsbapi.data.DataQuery; returns a helper for registration."""
    _MockDataQuery.clear()
    monkeypatch.setattr("rcsb.DataQuery", _MockDataQuery, raising=False)

    # Also patch the import inside MetadataClient's methods
    import rcsb as _rcsb_mod
    original_get_entries = _rcsb_mod.MetadataClient.get_entries

    def _patched_get_entries(self, entry_ids, extra_fields=None):
        ids_key = ",".join(sorted(entry_ids))
        key = ("entries", ids_key)
        data = _MockDataQuery._registry.get(key, {"data": {"entries": []}})
        entries = data.get("data", {}).get("entries", [])
        by_id = {eid: None for eid in entry_ids}
        for e in entries:
            eid = e.get("rcsb_id") or e.get("entry_id")
            if eid in by_id:
                by_id[eid] = e
        return by_id

    def _patched_get_summary(self, identifiers, return_data_list, input_type="entries"):
        ids_key = ",".join(sorted(str(i) for i in identifiers))
        key = (input_type, ids_key)
        data = _MockDataQuery._registry.get(key, {"data": {input_type: []}})
        items = data.get("data", {}).get(input_type, [])
        out = {}
        for item in items:
            iid = item.get("rcsb_id") or item.get("entry_id")
            if iid:
                out[iid] = item
        return out

    def _patched_get_chem(self, comp_ids):
        ids_key = ",".join(sorted(comp_ids))
        key = ("chem_comps", ids_key)
        data = _MockDataQuery._registry.get(key, {"data": {"chem_comps": []}})
        items = data.get("data", {}).get("chem_comps", [])
        by_id = {cid: None for cid in comp_ids}
        for item in items:
            cid = item.get("rcsb_id")
            if cid in by_id:
                by_id[cid] = item
        return by_id

    def _patched_get_pubmed(self, pubmed_ids):
        ids_key = ",".join(str(p) for p in sorted(pubmed_ids))
        key = ("pubmed", ids_key)
        data = _MockDataQuery._registry.get(key, {"data": {"pubmed": []}})
        items = data.get("data", {}).get("pubmed", [])
        by_id = {pid: None for pid in pubmed_ids}
        for item in items:
            pid = item.get("rcsb_id")
            if pid is not None and int(pid) in by_id:
                by_id[int(pid)] = item
        return by_id

    monkeypatch.setattr(_rcsb_mod.MetadataClient, "get_entries", _patched_get_entries)
    monkeypatch.setattr(_rcsb_mod.MetadataClient, "get_summary_fields", _patched_get_summary)
    monkeypatch.setattr(_rcsb_mod.MetadataClient, "get_chem_comps", _patched_get_chem)
    monkeypatch.setattr(_rcsb_mod.MetadataClient, "get_pubmed", _patched_get_pubmed)

    return _MockDataQuery


@pytest.fixture()
def tmp_output_dir(tmp_path):
    """Fresh temporary directory for file-download tests."""
    d = tmp_path / "rcsb_downloads"
    d.mkdir()
    return d


# ── convenience fixtures that load common fixtures ───────────────────────────

@pytest.fixture()
def fixture_3pxy():
    fx = load_fixture("data", "3pxy_entry.json")
    return fx["data"]["entries"][0]


@pytest.fixture()
def fixture_4hhb():
    fx = load_fixture("data", "4hhb_entry.json")
    return fx["data"]["entries"][0]


@pytest.fixture()
def fixture_1fqv():
    fx = load_fixture("data", "1fqv_entry.json")
    return fx["data"]["entries"][0]


@pytest.fixture()
def fixture_5fmb():
    fx = load_fixture("data", "5fmb_entry.json")
    return fx["data"]["entries"][0]


@pytest.fixture()
def fixture_1hsa():
    fx = load_fixture("data", "1hsa_entry.json")
    return fx["data"]["entries"][0]


@pytest.fixture()
def fixture_af():
    fx = load_fixture("data", "af_afp68871f1.json")
    return fx["data"]["entries"][0]


@pytest.fixture()
def fixture_7nss():
    fx = load_fixture("data", "7nss_ribosome.json")
    return fx["data"]["entries"][0]


@pytest.fixture()
def fixture_cdk2_p1():
    return load_fixture("search", "cdk2_text_p1.json")


@pytest.fixture()
def fixture_no_results():
    return load_fixture("search", "no_results.json")
