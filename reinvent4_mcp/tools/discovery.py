"""Discovery tools: list available scoring components, models, and external services."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

from reinvent4_mcp.utils import ensure_reinvent4_importable

logger = logging.getLogger(__name__)


def _type_name(field_type: Any) -> str:
    """Extract a human-readable type name from a dataclass field type annotation."""

    if hasattr(field_type, "__name__"):
        return field_type.__name__
    if hasattr(field_type, "__origin__"):
        return str(field_type)
    return str(field_type)


def _has_default(f: dataclasses.Field) -> bool:
    return (
        f.default is not dataclasses.MISSING
        or f.default_factory is not dataclasses.MISSING
    )


def _safe_default(val: Any) -> Any:
    """Convert a default value to a JSON-safe representation."""
    if val is dataclasses.MISSING or val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, (list, tuple)):
        return [_safe_default(v) for v in val]
    return repr(val)


def _params_schema_from_dataclass(param_cls: type) -> list[dict[str, Any]]:
    """Build a parameter schema list from a dataclass."""

    schema: list[dict[str, Any]] = []
    for f in dataclasses.fields(param_cls):
        entry: dict[str, Any] = {
            "name": f.name,
            "type": _type_name(f.type),
            "required": not _has_default(f),
        }
        if _has_default(f):
            entry["default"] = _safe_default(f.default)
        schema.append(entry)
    return schema


def _component_kind(component_cls: type) -> str:
    """Classify a component as scorer, filter, or penalty."""

    tag = getattr(component_cls, "__component", True)
    if isinstance(tag, str) and tag in ("filter", "penalty"):
        return tag
    return "scorer"


def list_scoring_components() -> list[dict[str, Any]]:
    """Return metadata for all registered native scoring components.

    Each entry contains the component name, its type (scorer/filter/penalty),
    and the parameter schema (field names + types) if available.
    """

    ensure_reinvent4_importable()
    from reinvent.scoring.importer import get_registry

    registry = get_registry()
    components: list[dict[str, Any]] = []

    for name, (component_cls, param_cls) in sorted(registry.items()):
        entry: dict[str, Any] = {
            "name": name,
            "class": component_cls.__name__,
            "component_type": _component_kind(component_cls),
        }

        if param_cls and dataclasses.is_dataclass(param_cls):
            entry["parameters"] = _params_schema_from_dataclass(param_cls)
        else:
            entry["parameters"] = None

        components.append(entry)

    return components


def list_available_models() -> list[dict[str, Any]]:
    """Return the prior model registry with availability status.

    Each entry contains the registry key (e.g. ``.reinvent``), the expected
    file path, and whether the file exists on disk.
    """

    ensure_reinvent4_importable()
    from reinvent.utils.prior_registry import prior_registry

    models: list[dict[str, Any]] = []

    for key, path in sorted(prior_registry.items()):
        path = Path(path)

        model_type = key.lstrip(".").split("_")[0]
        type_labels = {
            "reinvent": "Reinvent (de novo generation)",
            "libinvent": "LibInvent (R-group decoration)",
            "linkinvent": "LinkInvent (fragment linking)",
            "m2m": "Mol2Mol (molecule-to-molecule translation)",
        }

        models.append({
            "key": key,
            "path": str(path),
            "exists": path.exists(),
            "filename": path.name,
            "model_type": type_labels.get(model_type, model_type),
        })

    return models


def list_transforms() -> list[dict[str, Any]]:
    """Return the available score transform types and their parameter schemas."""

    ensure_reinvent4_importable()
    from reinvent.scoring.transforms.transform import registry as transform_registry

    transforms: list[dict[str, Any]] = []

    for name, (transform_cls, param_cls) in sorted(transform_registry.items()):
        entry: dict[str, Any] = {
            "name": name,
            "class": transform_cls.__name__,
        }

        if param_cls and dataclasses.is_dataclass(param_cls):
            entry["parameters"] = _params_schema_from_dataclass(param_cls)
        else:
            entry["parameters"] = None

        transforms.append(entry)

    return transforms


# ---------------------------------------------------------------------------
# Model manifest resolution — block 08 addition
# ---------------------------------------------------------------------------

import hashlib as _hashlib  # noqa: E402 (after existing imports)
import json as _json         # noqa: E402


# Registry of known prior manifest metadata.
# In production these paths come from the host/container environment.
# ``REINVENT_MODELS_DIR`` (env var) or the defaults below are used.
_DEFAULT_MODEL_DIRS: list[Path] = [
    Path("/app/models/reinvent"),
    Path("/shared/models"),
    Path.home() / ".reinvent" / "models",
]

# Known prior types and their canonical manifest file names
_PRIOR_MANIFEST_NAMES: dict[str, str] = {
    ".reinvent":              "reinvent_manifest.json",
    ".libinvent":             "libinvent_manifest.json",
    ".linkinvent":            "linkinvent_manifest.json",
    # Mol2Mol variants (lead-hopping / molecular translation)
    # Transformer variants (better scaffold generalization than RNN)
    ".libinvent_transformer":  "libinvent_transformer_manifest.json",
    ".linkinvent_transformer": "linkinvent_transformer_manifest.json",
    # Mol2Mol variants — canonical REINVENT4 prior-registry keys
    ".m2m_medium":             "mol2mol_medium_manifest.json",
    ".m2m_high":               "mol2mol_high_manifest.json",
    ".m2m_similarity":         "mol2mol_similarity_manifest.json",
    ".m2m_scaffold":           "mol2mol_scaffold_manifest.json",
    ".m2m_scaffold_generic":   "mol2mol_scaffold_generic_manifest.json",
    ".m2m_mmp":                "mol2mol_mmp_manifest.json",
    # Backward-compatible long-form aliases
    ".mol2mol":               "mol2mol_medium_manifest.json",   # alias for medium similarity
    ".mol2mol_medium":        "mol2mol_medium_manifest.json",
    ".mol2mol_high":          "mol2mol_high_manifest.json",
    ".mol2mol_similarity":    "mol2mol_similarity_manifest.json",
    ".mol2mol_scaffold":      "mol2mol_scaffold_manifest.json",
    ".mol2mol_mmp":           "mol2mol_mmp_manifest.json",
    # Pepinvent (peptide generation)
    ".pepinvent":             "pepinvent_manifest.json",
}


def _locate_manifest(registry_key: str) -> Path | None:
    """Search standard model directories for a prior manifest JSON."""
    import os  # noqa: PLC0415

    manifest_name = _PRIOR_MANIFEST_NAMES.get(registry_key)
    if manifest_name is None:
        return None

    # Allow override via env var
    env_dir = os.environ.get("REINVENT_MODELS_DIR")
    search_dirs = ([Path(env_dir)] if env_dir else []) + _DEFAULT_MODEL_DIRS

    for d in search_dirs:
        candidate = d / manifest_name
        if candidate.is_file():
            return candidate
    return None


def _sha256_file(path: Path) -> str:
    """Compute the lowercase hex SHA-256 digest of a file."""
    h = _hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_model_manifest(registry_key: str) -> dict[str, Any]:
    """Resolve a prior registry key to its manifest and verify integrity.

    Searches ``REINVENT_MODELS_DIR`` (env var) and standard model directories
    for a manifest JSON describing the prior files, versions, and SHA-256
    checksums.  Missing files and checksum mismatches raise immediately so
    the agent catches problems before any GPU work starts.

    Parameters
    ----------
    registry_key:
        A prior registry key such as ``".reinvent"``, ``".libinvent"``, or
        ``".linkinvent"``.

    Returns
    -------
    dict
        ``{registry_key, manifest_path, version, supported_modes, files,
          sha256_verified, available}``

    Raises
    ------
    KeyError
        If *registry_key* is not a known prior type.
    RuntimeError
        If a listed asset file is missing or its SHA-256 does not match.
    """
    if registry_key not in _PRIOR_MANIFEST_NAMES:
        valid = ", ".join(sorted(_PRIOR_MANIFEST_NAMES))
        raise KeyError(
            f"Unknown model registry key {registry_key!r}. "
            f"Valid keys: {valid}. "
            "Pass an absolute file path for custom priors."
        )

    manifest_path = _locate_manifest(registry_key)

    # If no manifest file is found on disk (e.g. in CI/dev without models),
    # return a stub with available=False rather than crashing.
    if manifest_path is None:
        return {
            "registry_key":  registry_key,
            "manifest_path": None,
            "version":       None,
            "supported_modes": [registry_key.lstrip(".")],
            "files":         [],
            "sha256_verified": False,
            "available":     False,
            "note": (
                f"Manifest not found in any of the search paths. "
                "Set REINVENT_MODELS_DIR or place the manifest under "
                "/app/models/reinvent/, /shared/models/, or ~/.reinvent/models/."
            ),
        }

    # Load manifest
    try:
        raw = _json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Manifest at {manifest_path} is not valid JSON: {exc}"
        ) from exc

    model_dir = manifest_path.parent
    file_entries: list[dict[str, Any]] = []
    sha256_ok = True

    for entry in raw.get("files", []):
        rel_path = entry.get("file", "")
        expected_sha = entry.get("sha256", "PLACEHOLDER")
        asset_path = model_dir / rel_path

        if not asset_path.is_file():
            raise RuntimeError(
                f"Asset file '{rel_path}' listed in manifest {manifest_path.name} "
                f"not found at {asset_path}. "
                "Ensure model weights are present or re-download from the source."
            )

        if expected_sha not in ("PLACEHOLDER", "", None):
            actual_sha = _sha256_file(asset_path)
            if actual_sha.lower() != expected_sha.strip().lower():
                raise RuntimeError(
                    f"SHA-256 mismatch for '{rel_path}': "
                    f"expected {expected_sha!r}, got {actual_sha!r}. "
                    "The file may be corrupt or replaced by a different version."
                )
            sha256_checked = True
        else:
            sha256_checked = False   # PLACEHOLDER → skip
            logger.warning(
                "Asset '%s' has PLACEHOLDER sha256 in manifest — "
                "skipping integrity check.",
                rel_path,
            )

        file_entries.append({
            "file":           rel_path,
            "exists":         True,
            "sha256_verified": sha256_checked,
        })
        sha256_ok = sha256_ok and sha256_checked

    return {
        "registry_key":    registry_key,
        "manifest_path":   str(manifest_path),
        "version":         raw.get("version"),
        "supported_modes": raw.get("supported_modes", [registry_key.lstrip(".")]),
        "files":           file_entries,
        "sha256_verified": sha256_ok,
        "available":       True,
        "notes":           raw.get("notes"),
    }
