"""Abstract interface for fragmentation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from interaction_profiler.schemas import Fragment


class Fragmenter(ABC):
    """Abstract interface for an in-process fragmentation backend.

    Implementations MUST preserve the atom indices of the input RDKit Mol
    in their output ``Fragment.atom_indices``: the downstream join layer
    pairs these indices with interaction atom indices to produce triples.
    """

    name: str

    @abstractmethod
    def fragment(self, mol: Any) -> list[Fragment]:
        """Return all fragments found in ``mol`` with their atom indices.

        Parameters
        ----------
        mol
            An RDKit ``Mol`` (or duck-typed equivalent the backend accepts).
        """

    def tool_version(self) -> str | None:
        """Best-effort version string of the underlying tool, if available."""
        return None


__all__ = ["Fragmenter"]
