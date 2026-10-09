# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Adapter around ``accfg.AccFG`` for the unified fragmenter interface."""

from __future__ import annotations

import logging
from typing import Any

from interaction_profiler.fragmentation.base import Fragmenter
from interaction_profiler.schemas import Fragment
from interaction_profiler.vocabulary import canonical_fragment

logger = logging.getLogger(__name__)


class AccFGAdapter(Fragmenter):
    """Wrap :class:`accfg.AccFG` so it speaks the unified Fragmenter contract.

    Uses ``AccFG.run_mol`` (available since AccFG >= 0.0.3) which takes an
    RDKit ``Mol`` directly, preserving atom indices.
    """

    name = "accfg"

    def __init__(
        self,
        *,
        lite: bool = False,
        user_defined_fgs: dict[str, str] | None = None,
    ) -> None:
        try:
            from accfg import AccFG
        except ImportError as exc:  # pragma: no cover - import error path
            raise ImportError(
                "AccFG is required for the 'accfg' fragmentation backend. "
                "Install with: pip install accfg"
            ) from exc

        kwargs: dict[str, Any] = {"print_load_info": False}
        if lite:
            kwargs["lite"] = True
        if user_defined_fgs:
            kwargs["user_defined_fgs"] = user_defined_fgs

        self._impl = AccFG(**kwargs)
        self._version: str | None = None
        try:
            import accfg

            self._version = getattr(accfg, "__version__", None)
        except Exception:  # pragma: no cover - best-effort
            pass

    def fragment(self, mol: Any) -> list[Fragment]:
        # ``AccFG.run_mol`` returns ``(fgs_dict, fg_graph)``.
        # fgs_dict: dict[str, list[tuple[int, ...]]]
        # Each tuple is one occurrence; multiple occurrences of the same
        # functional-group kind become multiple Fragment entries.
        try:
            fgs, _graph = self._impl.run_mol(mol, show_atoms=True, show_graph=True)
        except Exception as exc:
            logger.exception("AccFG fragmentation failed: %s", exc)
            return []

        results: list[Fragment] = []
        for fg_name, matches in (fgs or {}).items():
            for match in matches or ():
                # Defensive: ensure indices are plain ints.
                atom_indices = [int(i) for i in match]
                if not atom_indices:
                    continue
                results.append(
                    Fragment(
                        name=fg_name,
                        canonical_name=canonical_fragment(fg_name),
                        level="functional_group",
                        atom_indices=atom_indices,
                    )
                )
        return results

    def tool_version(self) -> str | None:
        return self._version


__all__ = ["AccFGAdapter"]
