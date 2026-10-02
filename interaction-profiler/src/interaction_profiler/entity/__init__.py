"""Entity resolution package.

Resolves chain information from a loaded structure and entity declarations
from a profiling request into a canonical list of :class:`~interaction_profiler.schemas.Entity`
objects, one per biological participant in the complex.
"""

from __future__ import annotations

from interaction_profiler.entity.resolver import resolve_entities

__all__ = ["resolve_entities"]
