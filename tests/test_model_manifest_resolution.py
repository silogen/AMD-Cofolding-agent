"""Tests for block 08 model manifest resolution (discovery.py additions).

Covers:
- resolve_model_manifest: known/unknown keys, file presence, SHA-256 checks
- Returns available=False gracefully when no manifest is on disk
- Raises KeyError for unknown registry keys
- Raises RuntimeError for corrupt SHA-256 or missing file
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from reinvent4_mcp.tools.discovery import resolve_model_manifest, _sha256_file


# ---------------------------------------------------------------------------
# Helper: build a minimal manifest.json + dummy model files
# ---------------------------------------------------------------------------

def _make_manifest(
    tmp_path: Path,
    files: list[dict],
    version: str = "1.0",
    sha256_correct: bool = True,
) -> Path:
    """Write manifest.json and dummy asset files to tmp_path."""
    for entry in files:
        rel = entry["file"]
        asset_path = tmp_path / rel
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_bytes(b"mock_model_data_" + rel.encode())

    # Compute actual SHA-256 for each file
    file_entries = []
    for entry in files:
        rel = entry["file"]
        asset_path = tmp_path / rel
        if sha256_correct:
            sha = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        else:
            sha = "a" * 64  # wrong SHA
        file_entries.append({"file": rel, "sha256": sha})

    manifest = {
        "version": version,
        "supported_modes": ["reinvent"],
        "files": file_entries,
    }
    m_path = tmp_path / "reinvent_manifest.json"
    m_path.write_text(json.dumps(manifest))
    return tmp_path


# ===========================================================================
# Category 1: Known registry keys and no manifest on disk
# ===========================================================================

class TestManifestNotFound:
    def test_known_key_returns_available_false_when_no_manifest(self, tmp_path, monkeypatch):
        """No manifest on disk → returns stub with available=False."""
        # Point search dirs to empty directory
        monkeypatch.setattr(
            "reinvent4_mcp.tools.discovery._DEFAULT_MODEL_DIRS",
            [tmp_path],
        )
        monkeypatch.delenv("REINVENT_MODELS_DIR", raising=False)

        result = resolve_model_manifest(".reinvent")
        assert result["available"] is False
        assert result["registry_key"] == ".reinvent"
        assert result["manifest_path"] is None

    def test_all_known_keys_return_gracefully_when_no_manifest(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "reinvent4_mcp.tools.discovery._DEFAULT_MODEL_DIRS",
            [tmp_path],
        )
        monkeypatch.delenv("REINVENT_MODELS_DIR", raising=False)

        for key in (".reinvent", ".libinvent", ".linkinvent"):
            result = resolve_model_manifest(key)
            assert result["available"] is False, f"Expected unavailable for {key}"
            assert result["registry_key"] == key


# ===========================================================================
# Category 2: Unknown registry key
# ===========================================================================

class TestUnknownKey:
    def test_unknown_key_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown model registry key"):
            resolve_model_manifest(".unknown_prior")

    def test_absolute_path_raises_key_error(self):
        """Non-registry keys (abs paths, custom names) raise KeyError."""
        with pytest.raises(KeyError):
            resolve_model_manifest("/path/to/custom/model.prior")

    def test_empty_string_raises_key_error(self):
        with pytest.raises(KeyError):
            resolve_model_manifest("")


# ===========================================================================
# Category 3: Valid manifest on disk
# ===========================================================================

class TestValidManifest:
    def test_correct_manifest_returns_available_true(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "reinvent4_mcp.tools.discovery._DEFAULT_MODEL_DIRS",
            [tmp_path],
        )
        monkeypatch.delenv("REINVENT_MODELS_DIR", raising=False)
        _make_manifest(tmp_path, [{"file": "weights/prior.pt"}], sha256_correct=True)

        result = resolve_model_manifest(".reinvent")
        assert result["available"] is True
        assert result["registry_key"] == ".reinvent"
        assert result["version"] == "1.0"
        assert result["sha256_verified"] is True

    def test_manifest_reports_file_entries(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "reinvent4_mcp.tools.discovery._DEFAULT_MODEL_DIRS",
            [tmp_path],
        )
        monkeypatch.delenv("REINVENT_MODELS_DIR", raising=False)
        _make_manifest(tmp_path, [{"file": "prior.pt"}, {"file": "vocab.txt"}])

        result = resolve_model_manifest(".reinvent")
        assert len(result["files"]) == 2
        assert all(f["exists"] for f in result["files"])

    def test_manifest_path_reported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "reinvent4_mcp.tools.discovery._DEFAULT_MODEL_DIRS",
            [tmp_path],
        )
        monkeypatch.delenv("REINVENT_MODELS_DIR", raising=False)
        _make_manifest(tmp_path, [{"file": "prior.pt"}])

        result = resolve_model_manifest(".reinvent")
        assert result["manifest_path"] is not None
        assert "reinvent_manifest.json" in result["manifest_path"]


# ===========================================================================
# Category 4: SHA-256 mismatch
# ===========================================================================

class TestSHA256Mismatch:
    def test_wrong_sha_raises_runtime_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "reinvent4_mcp.tools.discovery._DEFAULT_MODEL_DIRS",
            [tmp_path],
        )
        monkeypatch.delenv("REINVENT_MODELS_DIR", raising=False)
        _make_manifest(tmp_path, [{"file": "prior.pt"}], sha256_correct=False)

        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            resolve_model_manifest(".reinvent")

    def test_placeholder_sha_skips_check(self, tmp_path, monkeypatch):
        """PLACEHOLDER sha256 skips the integrity check (dev mode)."""
        monkeypatch.setattr(
            "reinvent4_mcp.tools.discovery._DEFAULT_MODEL_DIRS",
            [tmp_path],
        )
        monkeypatch.delenv("REINVENT_MODELS_DIR", raising=False)

        # Write manifest with PLACEHOLDER sha
        asset_path = tmp_path / "prior.pt"
        asset_path.write_bytes(b"mock")
        manifest = {
            "version": "1.0",
            "supported_modes": ["reinvent"],
            "files": [{"file": "prior.pt", "sha256": "PLACEHOLDER"}],
        }
        (tmp_path / "reinvent_manifest.json").write_text(json.dumps(manifest))

        result = resolve_model_manifest(".reinvent")
        assert result["available"] is True
        # sha256_verified is False when any file uses PLACEHOLDER
        assert result["sha256_verified"] is False


# ===========================================================================
# Category 5: Missing file
# ===========================================================================

class TestMissingFile:
    def test_missing_file_raises_runtime_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "reinvent4_mcp.tools.discovery._DEFAULT_MODEL_DIRS",
            [tmp_path],
        )
        monkeypatch.delenv("REINVENT_MODELS_DIR", raising=False)

        # Write manifest that references a file that doesn't exist
        manifest = {
            "version": "1.0",
            "supported_modes": ["reinvent"],
            "files": [{"file": "missing_weights.pt", "sha256": "PLACEHOLDER"}],
        }
        (tmp_path / "reinvent_manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(RuntimeError, match="not found"):
            resolve_model_manifest(".reinvent")

    def test_error_message_includes_file_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "reinvent4_mcp.tools.discovery._DEFAULT_MODEL_DIRS",
            [tmp_path],
        )
        monkeypatch.delenv("REINVENT_MODELS_DIR", raising=False)

        manifest = {
            "version": "1.0", "supported_modes": ["reinvent"],
            "files": [{"file": "very_specific_name.pt", "sha256": "PLACEHOLDER"}],
        }
        (tmp_path / "reinvent_manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(RuntimeError) as exc_info:
            resolve_model_manifest(".reinvent")
        assert "very_specific_name.pt" in str(exc_info.value)


# ===========================================================================
# Category 6: _sha256_file helper
# ===========================================================================

class TestSha256Helper:
    def test_sha256_matches_hashlib(self, tmp_path):
        p = tmp_path / "test.bin"
        content = b"hello world"
        p.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert _sha256_file(p) == expected

    def test_different_files_different_digests(self, tmp_path):
        p1 = tmp_path / "a.bin"
        p2 = tmp_path / "b.bin"
        p1.write_bytes(b"file_a")
        p2.write_bytes(b"file_b")
        assert _sha256_file(p1) != _sha256_file(p2)
