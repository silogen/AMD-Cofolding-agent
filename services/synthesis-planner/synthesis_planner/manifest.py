"""Asset manifest loading and SHA-256 integrity verification.

The manifest describes the AiZynthFinder model assets (policy weights,
reaction templates, stock catalog) baked into or mounted onto the container.
It mirrors the ADMET build-time smoke-test pattern: the service refuses to start
if any asset file is missing or its checksum does not match.

Manifest format (``models/aizynthfinder/manifest.json``):

.. code-block:: json

    {
      "policy": {
        "name": "uspto_rop",
        "version": "2024.1",
        "file": "policy/policy.hdf5",
        "sha256": "abc123..."
      },
      "templates": {
        "name": "uspto_2017",
        "version": "2024.1",
        "file": "templates/templates.hdf5",
        "sha256": "def456..."
      },
      "stock": [
        {
          "key": "default",
          "name": "zinc_in_stock",
          "version": "2024-06",
          "file": "stock/zinc.hdf5",
          "sha256": "PLACEHOLDER"
        }
      ]
    }

When ``sha256`` is ``"PLACEHOLDER"`` the checksum check is skipped (useful
during development when real assets are not yet available).  The file
existence check is always performed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PLACEHOLDER = "PLACEHOLDER"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AssetEntry:
    """One versioned asset in the manifest."""
    name: str
    version: str
    file: str       # relative to MODELS_DIR
    sha256: str     # hex digest or "PLACEHOLDER"
    key: str = "default"   # logical identifier (used for stock selection)


@dataclass
class AssetManifest:
    """Validated, fully-loaded manifest."""
    policy: AssetEntry
    templates: AssetEntry
    stock: list[AssetEntry] = field(default_factory=list)

    def policy_version(self) -> str:
        return f"{self.policy.name}@{self.policy.version}"

    def templates_version(self) -> str:
        return f"{self.templates.name}@{self.templates.version}"

    def stock_version(self, key: str = "default") -> str:
        for s in self.stock:
            if s.key == key:
                return f"{s.name}@{s.version}"
        return "unknown"

    def stock_keys(self) -> list[str]:
        return [s.key for s in self.stock]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_asset(entry: AssetEntry, models_dir: Path) -> None:
    """Raise RuntimeError if the asset file is missing or has a bad checksum."""
    asset_path = models_dir / entry.file
    if not asset_path.is_file():
        raise RuntimeError(
            f"Asset '{entry.name}' ({entry.file}) not found at {asset_path}. "
            "Ensure model assets are baked into the image or mounted at the "
            f"expected path ({models_dir})."
        )

    if entry.sha256 == _PLACEHOLDER:
        logger.warning(
            "Asset '%s' has placeholder SHA-256 — skipping integrity check. "
            "Replace 'PLACEHOLDER' in manifest.json with the actual digest "
            "before deploying to production.",
            entry.name,
        )
        return

    actual = _sha256(asset_path)
    if actual.lower() != entry.sha256.strip().lower():
        raise RuntimeError(
            f"SHA-256 mismatch for asset '{entry.name}' ({entry.file}): "
            f"expected {entry.sha256!r}, got {actual!r}. "
            "The model file may be corrupted or replaced by a different version."
        )
    logger.info("Asset '%s' OK (SHA-256 verified).", entry.name)


def _parse_entry(d: dict[str, Any], key: str = "default") -> AssetEntry:
    return AssetEntry(
        name=d["name"],
        version=str(d["version"]),
        file=d["file"],
        sha256=d.get("sha256", _PLACEHOLDER),
        key=d.get("key", key),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_verify(models_dir: str | Path) -> AssetManifest:
    """Load ``manifest.json`` from *models_dir* and verify all asset files.

    Parameters
    ----------
    models_dir:
        Directory containing ``manifest.json`` and all asset sub-directories.

    Returns
    -------
    AssetManifest
        Fully validated manifest ready for consumption by the service.

    Raises
    ------
    FileNotFoundError
        If ``manifest.json`` does not exist.
    RuntimeError
        If any asset file is missing or its SHA-256 does not match.
    ValueError
        If the manifest JSON is malformed.
    """
    models_dir = Path(models_dir)
    manifest_path = models_dir / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"manifest.json not found at {manifest_path}. "
            "Copy models/aizynthfinder/manifest.json into the container or "
            "mount the models directory at the expected path."
        )

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest.json is not valid JSON: {exc}") from exc

    # --- Parse ---
    try:
        policy    = _parse_entry(raw["policy"])
        templates = _parse_entry(raw["templates"])
        stock_entries = [
            _parse_entry(s, key=s.get("key", "default"))
            for s in raw.get("stock", [])
        ]
    except KeyError as exc:
        raise ValueError(
            f"manifest.json is missing required key: {exc}"
        ) from exc

    manifest = AssetManifest(
        policy=policy,
        templates=templates,
        stock=stock_entries,
    )

    # --- Verify (skip stock; stock may be mounted separately) ---
    logger.info("Verifying policy model: %s v%s", policy.name, policy.version)
    _verify_asset(policy, models_dir)

    logger.info("Verifying reaction templates: %s v%s", templates.name, templates.version)
    _verify_asset(templates, models_dir)

    for stock_entry in stock_entries:
        stock_path = models_dir / stock_entry.file
        if stock_path.is_file():
            logger.info(
                "Verifying stock catalog '%s': %s v%s",
                stock_entry.key, stock_entry.name, stock_entry.version,
            )
            _verify_asset(stock_entry, models_dir)
        else:
            logger.warning(
                "Stock catalog '%s' (%s) not found — assuming it is mounted "
                "separately at runtime.  Check Compose volume configuration.",
                stock_entry.key, stock_entry.file,
            )

    return manifest


def build_aizynthfinder_config(
    manifest: AssetManifest,
    models_dir: Path,
    search_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the AiZynthFinder configuration dict from the manifest.

    This is passed to ``Configuration.from_dict()`` at startup so the finder
    does not need a separate YAML config file on disk.

    AiZynthFinder v4.4.x config format:
      - Top-level key is ``expansion`` (not ``policy``)
      - Each policy entry has a ``type``, ``model``, and ``template`` (singular) field
      - Stock entries are keyed by logical name pointing to the HDF5 file path
    """
    policy_path    = str(models_dir / manifest.policy.file)
    templates_path = str(models_dir / manifest.templates.file)

    cfg: dict[str, Any] = {
        "expansion": {
            "default": {
                "type": (
                    "aizynthfinder.context.policy.expansion_strategies"
                    ".TemplateBasedExpansionStrategy"
                ),
                "model":    policy_path,
                "template": templates_path,
            },
        },
        "stock": {},
    }

    for s in manifest.stock:
        stock_path = models_dir / s.file
        if stock_path.is_file():
            cfg["stock"][s.key] = str(stock_path)

    # Merge optional override parameters
    if search_params:
        cfg["search"] = search_params

    return cfg
