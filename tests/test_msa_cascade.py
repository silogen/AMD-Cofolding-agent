"""Tests for the MSA cascade in both the mapper and the Boltz service.

All tests are unit tests: no network calls, no Docker, no GPU.
Fixtures are generic (synthetic protein sequences, tmp_path directories)
— nothing is specific to any particular target.

The cascade being tested:
  1. Explicit .a3m file supplied  -> use file, skip server.
  2. No file, cache miss          -> try MSA server.
     a. Server succeeds           -> write to cache, use file.
     b. Server fails              -> msa_error set.
        i.  msa_free_fallback True  -> single-sequence prediction with warning.
        ii. msa_free_fallback False -> hard abort with actionable message.
  3. No file, cache hit           -> use cached file, skip server.
  4. auto_msa_fallback False      -> mapper returns {} (no MSA params).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Make reinvent4_mcp importable from the repo root.
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# Generic 20-aa sequence used in all tests — not target-specific.
_SEQ = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_boltz_config(**kwargs) -> SimpleNamespace:
    """Return a BoltzConfig-like namespace with sensible defaults."""
    defaults = dict(
        use_msa_server=True,
        msa_server_url="https://api.colabfold.com",
        auto_msa_fallback=True,
        msa_free_fallback=True,
        msa_cache_dir="/nonexistent/msa_cache",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _mock_full_config(boltz_ns: SimpleNamespace) -> MagicMock:
    config = MagicMock()
    config.scoring.boltz = boltz_ns
    return config


# ---------------------------------------------------------------------------
# Mapper-layer tests  (_resolve_boltz_msa)
# ---------------------------------------------------------------------------

class TestMapperMsaCascade:
    """Tests for reinvent4_mcp.ttp.mapper._resolve_boltz_msa."""

    def _call(self, boltz_cfg, seq: str = _SEQ):
        from reinvent4_mcp.ttp.mapper import _resolve_boltz_msa
        cfg = _mock_full_config(boltz_cfg)
        # get_config is imported lazily inside _resolve_boltz_msa, so patch
        # the canonical location rather than the mapper module's namespace.
        with patch("reinvent4_mcp.config.get_config", return_value=cfg):
            return _resolve_boltz_msa(seq)

    # -- 2a. No file, cache miss -> mapper enables server (auto_msa_fallback=True)

    def test_no_file_no_cache_enables_server(self):
        """With no local file and no cache, mapper sets use_msa_server=True."""
        bcfg = _mock_boltz_config(auto_msa_fallback=True, msa_free_fallback=True)
        result = self._call(bcfg)
        assert result.get("use_msa_server") is True
        assert result.get("msa_free_fallback") is True
        assert "msa_server_url" in result
        assert "msa_path" not in result

    # -- 3. Cache hit -> mapper returns file path, suppresses server

    def test_cache_hit_returns_file_path(self, tmp_path):
        """When a cached .a3m exists, mapper returns msa_path and disables server."""
        from reinvent4_mcp.ttp.mapper import sequence_msa_hash
        h = sequence_msa_hash(_SEQ)
        (tmp_path / f"{h}.a3m").write_text(">seq\n" + _SEQ + "\n")

        bcfg = _mock_boltz_config(msa_cache_dir=str(tmp_path))
        result = self._call(bcfg)
        assert "msa_path" in result
        assert result["msa_path"].endswith(f"{h}.a3m")
        assert result.get("use_msa_server") is False

    # -- 4. auto_msa_fallback=False -> mapper returns empty dict

    def test_auto_fallback_disabled_returns_empty(self):
        """With auto_msa_fallback=False and no cache, mapper emits no MSA params."""
        bcfg = _mock_boltz_config(auto_msa_fallback=False)
        result = self._call(bcfg)
        assert result == {}

    # -- Ordering: cached file must win even when auto_msa_fallback=False

    def test_cache_hit_wins_over_disabled_server(self, tmp_path):
        """Cache hit must be used even when auto_msa_fallback is False."""
        from reinvent4_mcp.ttp.mapper import sequence_msa_hash
        h = sequence_msa_hash(_SEQ)
        (tmp_path / f"{h}.a3m").write_text(">seq\n" + _SEQ + "\n")

        bcfg = _mock_boltz_config(
            msa_cache_dir=str(tmp_path),
            auto_msa_fallback=False,
        )
        result = self._call(bcfg)
        assert "msa_path" in result
        assert result.get("use_msa_server") is False


# ---------------------------------------------------------------------------
# Service-layer tests  (_resolve_msa in services/boltz2/app.py)
# ---------------------------------------------------------------------------

_SERVICE_DIR = _ROOT / "services" / "boltz2"
sys.path.insert(0, str(_SERVICE_DIR))


def _make_score_params(**kwargs):
    """Build a minimal ScoreParams-like namespace for _resolve_msa."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "boltz_app", str(_SERVICE_DIR / "app.py")
    )
    defaults = dict(
        protein_sequence=_SEQ,
        msa_path=None,
        use_msa_server=False,
        msa_server_url="https://api.colabfold.com",
        msa_free_fallback=True,
        derive_sequence_from_msa=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestServiceMsaCascade:
    """Tests for _resolve_msa in services/boltz2/app.py."""

    @pytest.fixture(autouse=True)
    def _import_service(self):
        """Import app.py functions lazily so the test doesn't require boltz itself."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "boltz_app", str(_SERVICE_DIR / "app.py")
        )
        self.mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(self.mod)
        except Exception:
            pytest.skip("boltz service not importable (dependencies missing)")

    # -- 1. Explicit file -> returned immediately, server not called

    def test_explicit_file_wins(self, tmp_path):
        """An explicit msa_path is returned without touching the server."""
        a3m = tmp_path / "target.a3m"
        a3m.write_text(">seq\n" + _SEQ + "\n")

        params = _make_score_params(msa_path=str(a3m))
        with patch.object(self.mod, "_ensure_cached_msa", side_effect=AssertionError("should not be called")):
            with patch.object(self.mod, "SHARED_DIR", tmp_path):
                path, use_server, err = self.mod._resolve_msa(params)
        assert path == str(a3m)
        assert use_server is False
        assert err is None

    # -- 2b-i. Server fails + msa_free_fallback=True -> msa_error set, no exception

    def test_server_fails_msa_error_set(self):
        """When the MSA server raises, _resolve_msa returns an msa_error string."""
        params = _make_score_params(use_msa_server=True, msa_free_fallback=True)
        with patch.object(
            self.mod, "_ensure_cached_msa", side_effect=RuntimeError("404 Not Found")
        ):
            path, use_server, err = self.mod._resolve_msa(params)
        assert path is None
        assert use_server is False
        assert err is not None
        assert "404" in err or "MSA" in err.lower()

    # -- 2a. Server succeeds -> path returned

    def test_server_success_returns_path(self, tmp_path):
        """When the MSA server succeeds, _resolve_msa returns the cached path."""
        a3m = tmp_path / "cached.a3m"
        a3m.write_text(">seq\n" + _SEQ + "\n")

        params = _make_score_params(use_msa_server=True, msa_free_fallback=True)
        with patch.object(self.mod, "_ensure_cached_msa", return_value=str(a3m)):
            path, use_server, err = self.mod._resolve_msa(params)
        assert path == str(a3m)
        assert use_server is False
        assert err is None

    # -- No MSA source at all -> (None, False, None)

    def test_no_msa_source_returns_none_triple(self):
        """When neither msa_path nor use_msa_server is set, all three values are None/False."""
        params = _make_score_params(use_msa_server=False, msa_path=None)
        path, use_server, err = self.mod._resolve_msa(params)
        assert path is None
        assert use_server is False
        assert err is None
