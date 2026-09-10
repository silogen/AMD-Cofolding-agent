"""TTP persistence layer -- versioned JSON file storage.

Each TTP is stored as ``{id}.json`` (current version) alongside
historical snapshots ``{id}_v{N}.json``.  On update the version
auto-increments and both files are written atomically.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from reinvent4_mcp.ttp.schema import TTP

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, updates: dict) -> None:
    """Recursively merge *updates* into *base* in-place.

    Leaf values in *updates* overwrite those in *base*.  Nested dicts
    are merged recursively; lists and scalars are replaced outright.
    """
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value


class TTPStore:
    """File-backed TTP persistence with version history.

    Parameters
    ----------
    storage_dir:
        Directory in which TTP JSON files are written.  Created
        automatically if it does not exist.
    """

    def __init__(self, storage_dir: str | Path) -> None:
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # -- path helpers -----------------------------------------------------

    def _current_path(self, ttp_id: str) -> Path:
        return self._dir / f"{ttp_id}.json"

    def _version_path(self, ttp_id: str, version: int) -> Path:
        return self._dir / f"{ttp_id}_v{version}.json"

    # -- public API -------------------------------------------------------

    def create(self, ttp: TTP) -> TTP:
        """Persist a new TTP.  Raises ``FileExistsError`` if the ID is taken."""
        path = self._current_path(ttp.id)
        if path.exists():
            raise FileExistsError(
                f"TTP '{ttp.id}' already exists. Use update() to modify it."
            )
        ttp = ttp.model_copy(update={"version": 1})
        self._write(ttp)
        return ttp

    def update(self, ttp_id: str, updates: dict[str, Any]) -> TTP:
        """Apply a partial update to an existing TTP.

        The ``version`` is auto-incremented.  The original ``id`` is
        preserved regardless of what *updates* contains.

        Returns the validated, updated TTP.
        """
        current = self.get(ttp_id)
        merged = current.model_dump(mode="json")
        _deep_merge(merged, updates)
        merged["version"] = current.version + 1
        merged["id"] = ttp_id
        updated = TTP.model_validate(merged)
        self._write(updated)
        return updated

    def get(self, ttp_id: str, version: Optional[int] = None) -> TTP:
        """Load a TTP by ID, optionally at a specific historical version."""
        path = (
            self._version_path(ttp_id, version)
            if version is not None
            else self._current_path(ttp_id)
        )
        if not path.exists():
            detail = f" version {version}" if version is not None else ""
            raise FileNotFoundError(f"TTP '{ttp_id}'{detail} not found")

        data = json.loads(path.read_text())
        return TTP.model_validate(data)

    def list_all(self) -> list[dict[str, Any]]:
        """Return summary metadata for every stored TTP (latest versions only)."""
        summaries: list[dict[str, Any]] = []
        for path in sorted(self._dir.glob("*.json")):
            if "_v" in path.stem:
                continue
            try:
                data = json.loads(path.read_text())
                summaries.append(
                    {
                        "id": data["id"],
                        "name": data.get("name", ""),
                        "version": data.get("version", 1),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                logger.warning("Skipping invalid TTP file: %s", path)
        return summaries

    def delete(self, ttp_id: str) -> None:
        """Remove a TTP and all its version snapshots."""
        path = self._current_path(ttp_id)
        if not path.exists():
            raise FileNotFoundError(f"TTP '{ttp_id}' not found")
        path.unlink()
        for vpath in self._dir.glob(f"{ttp_id}_v*.json"):
            vpath.unlink()
        logger.info("Deleted TTP '%s' and its version history", ttp_id)

    def exists(self, ttp_id: str) -> bool:
        return self._current_path(ttp_id).exists()

    # -- internal ---------------------------------------------------------

    def _write(self, ttp: TTP) -> None:
        data = json.dumps(ttp.model_dump(mode="json"), indent=2)
        self._current_path(ttp.id).write_text(data)
        self._version_path(ttp.id, ttp.version).write_text(data)
        logger.info("Saved TTP '%s' v%d", ttp.id, ttp.version)
