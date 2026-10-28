# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TTP management tools: create, update, get, list, delete.

Provides the implementation behind the MCP ``create_ttp``,
``update_ttp``, ``get_ttp``, ``list_ttps``, and ``delete_ttp`` tools.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from reinvent4_mcp.config import get_config
from reinvent4_mcp.ttp.mapper import target_sequence
from reinvent4_mcp.ttp.schema import TTP
from reinvent4_mcp.ttp.store import TTPStore

logger = logging.getLogger(__name__)

_store: TTPStore | None = None


def _get_store() -> TTPStore:
    global _store
    if _store is None:
        config = get_config()
        _store = TTPStore(config.ttp.storage_dir)
    return _store


def reset_store() -> None:
    """Discard the cached store (for tests)."""
    global _store
    _store = None


def _structure_problem(ttp: TTP) -> str | None:
    """Why the target's structure file cannot give a sequence, if it cannot."""
    if not (ttp.target and ttp.target.structure_path):
        return None
    try:
        target_sequence(ttp.target)
    except ValueError as exc:
        return str(exc)
    return None


def _exclusive_target_fields(updates: dict[str, Any]) -> dict[str, Any]:
    """Setting one of protein_sequence / structure_path clears the other."""
    target = updates.get("target")
    if not isinstance(target, dict):
        return updates
    target = dict(target)
    if target.get("structure_path") and "protein_sequence" not in target:
        target["protein_sequence"] = None
    if target.get("protein_sequence") and "structure_path" not in target:
        target["structure_path"] = None
        target.setdefault("structure_chain", None)
    return {**updates, "target": target}


def create_ttp(ttp_data: dict[str, Any]) -> dict[str, Any]:
    """Validate and persist a new TTP.

    Returns a summary on success or an ``{"error": ...}`` dict on failure.
    """
    try:
        ttp = TTP.model_validate(ttp_data)
    except Exception as exc:
        return {"error": f"Invalid TTP: {exc}"}
    problem = _structure_problem(ttp)
    if problem:
        return {"error": f"Invalid TTP target: {problem}"}

    try:
        created = _get_store().create(ttp)
    except FileExistsError as exc:
        return {"error": str(exc)}

    return {
        "status": "created",
        "id": created.id,
        "name": created.name,
        "version": created.version,
        "ttp": created.model_dump(mode="json"),
    }


def update_ttp(ttp_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Apply a partial update to an existing TTP.

    Version is auto-incremented.  The returned dict contains the full
    updated TTP so the agent can inspect the result.
    """
    try:
        updated = _get_store().update(ttp_id, _exclusive_target_fields(updates))
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"Invalid update: {exc}"}

    result = {
        "status": "updated",
        "id": updated.id,
        "name": updated.name,
        "version": updated.version,
        "ttp": updated.model_dump(mode="json"),
    }
    problem = _structure_problem(updated)
    if problem:
        result["warning"] = f"The target structure cannot be read yet: {problem}"
    return result


def get_ttp(
    ttp_id: str,
    version: Optional[int] = None,
) -> dict[str, Any]:
    """Retrieve a TTP by ID, optionally at a specific historical version."""
    try:
        ttp = _get_store().get(ttp_id, version=version)
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    return ttp.model_dump(mode="json")


def list_ttps() -> list[dict[str, Any]]:
    """List all stored TTPs (latest versions only, summary metadata)."""
    return _get_store().list_all()


def delete_ttp(ttp_id: str) -> dict[str, Any]:
    """Delete a TTP and its version history."""
    try:
        _get_store().delete(ttp_id)
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    return {"status": "deleted", "id": ttp_id}


def resolve_ttp(
    ttp_id: Optional[str] = None,
    ttp_data: Optional[dict[str, Any]] = None,
) -> TTP | dict[str, Any]:
    """Resolve a TTP from either an ID (stored) or inline data.

    Returns a validated ``TTP`` instance on success or an
    ``{"error": ...}`` dict on failure.
    """
    if ttp_id and ttp_data:
        return {"error": "Provide either ttp_id or ttp (inline), not both."}
    if ttp_id:
        try:
            return _get_store().get(ttp_id)
        except FileNotFoundError as exc:
            return {"error": str(exc)}
    if ttp_data:
        try:
            return TTP.model_validate(ttp_data)
        except Exception as exc:
            return {"error": f"Invalid inline TTP: {exc}"}
    return {"error": "Provide either ttp_id or ttp (inline TTP definition)."}
