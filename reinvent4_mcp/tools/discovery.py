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
