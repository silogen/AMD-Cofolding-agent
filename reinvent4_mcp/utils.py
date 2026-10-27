# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Utility helpers: device detection, temp directories, REINVENT4 path discovery."""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REINVENT4_ROOT: Path | None = None


def find_reinvent4_root() -> Path:
    """Locate the REINVENT4 project directory.

    Resolution order:
    1. ``REINVENT4_ROOT`` environment variable
    2. ``../REINVENT4`` relative to this package (workspace layout)
    """

    global REINVENT4_ROOT

    if REINVENT4_ROOT is not None:
        return REINVENT4_ROOT

    env_root = os.environ.get("REINVENT4_ROOT")
    if env_root:
        REINVENT4_ROOT = Path(env_root).resolve()
        return REINVENT4_ROOT

    workspace_candidate = Path(__file__).resolve().parent.parent / "REINVENT4"
    if workspace_candidate.is_dir() and (workspace_candidate / "reinvent").is_dir():
        REINVENT4_ROOT = workspace_candidate
        return REINVENT4_ROOT

    raise RuntimeError(
        "Cannot locate REINVENT4 project directory. "
        "Set REINVENT4_ROOT environment variable or ensure the workspace layout "
        "(reinvent4_mcp/ alongside REINVENT4/)."
    )


def custom_plugins_path() -> str:
    """Return the directory that holds our custom REINVENT4 plugins.

    The directory is laid out as ``<here>/_plugins/reinvent_plugins/components/``
    (PEP 420 namespace: no ``__init__.py`` in ``reinvent_plugins/`` or
    ``components/``). Putting this directory on ``sys.path`` merges our
    components into REINVENT4's ``reinvent_plugins.components`` namespace
    natively.
    """

    return str(Path(__file__).resolve().parent / "_plugins")


def ensure_reinvent4_importable() -> None:
    """Add REINVENT4 project to ``sys.path`` so ``import reinvent`` works, and
    put our custom-plugins dir on ``sys.path`` so ``ExternalRESTScorer`` is
    discoverable by REINVENT4's plugin importer.
    """

    root = find_reinvent4_root()
    root_str = str(root)

    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        logger.info(f"Added {root_str} to sys.path")

    _register_custom_components()


def _register_custom_components() -> None:
    """Make our custom scoring components discoverable by REINVENT4.

    REINVENT4's ``reinvent_plugins.components`` is a PEP 420 namespace package;
    its importer scans every ``reinvent_plugins/components/`` directory found via
    ``sys.path``. We ship ours under ``_plugins/reinvent_plugins/components/`` and
    simply put ``_plugins`` on ``sys.path``. Unlike appending to the namespace
    ``__path__`` at runtime (which Python's ``_NamespacePath`` silently drops when
    ``sys.path`` later changes -- the bug that broke the RL worker), a real
    ``sys.path`` entry is re-read on every namespace recalculation, so the
    components stay discoverable in every process.
    """

    plugins_dir = custom_plugins_path()
    if plugins_dir not in sys.path:
        sys.path.insert(0, plugins_dir)
        logger.info("Added custom plugins path %s", plugins_dir)


def detect_device() -> str:
    """Return the best available torch device string."""

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"
