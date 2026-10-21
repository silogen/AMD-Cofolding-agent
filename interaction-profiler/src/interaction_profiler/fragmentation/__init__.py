# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fragmenter registry.

Add a backend by registering it under a new ``method`` key in
:data:`_REGISTRY`. The factory :func:`get_fragmenter` is used everywhere
else so the rest of the codebase never imports a backend directly.
"""

from __future__ import annotations

from typing import Any

from interaction_profiler.fragmentation.accfg_adapter import AccFGAdapter
from interaction_profiler.fragmentation.base import Fragmenter
from interaction_profiler.fragmentation.hierarchy import build_composition

_REGISTRY: dict[str, type[Fragmenter]] = {
    "accfg": AccFGAdapter,
}


def get_fragmenter(method: str = "accfg", **kwargs: Any) -> Fragmenter:
    """Return an instantiated Fragmenter for the requested ``method``."""
    if method not in _REGISTRY:
        raise ValueError(
            f"Unknown fragmentation method: {method!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[method](**kwargs)


def available_fragmenters() -> list[str]:
    """List of registered fragmentation method names."""
    return sorted(_REGISTRY)


__all__ = [
    "Fragmenter",
    "AccFGAdapter",
    "get_fragmenter",
    "available_fragmenters",
    "build_composition",
]
