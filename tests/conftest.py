"""Shared test fixtures / import shims.

The custom REINVENT4 scoring components live in a PEP-420 namespace-package
layout under ``reinvent4_mcp/_plugins/reinvent_plugins/components/`` -- the exact
directory REINVENT4's plugin importer scans at runtime (see
``reinvent4_mcp.utils.custom_plugins_path``). Tests import the scorer from that
canonical location so they exercise the *same* module that runs in production,
rather than a divergent copy.

REINVENT4 itself is usually not installed in the test environment, so the two
leaf modules the scorer imports (``add_tag`` and ``component_results``) are
stubbed here. We only stub those leaves -- ``reinvent_plugins`` and
``reinvent_plugins.components`` are left to resolve as real namespace packages
that include our ``_plugins`` dir, so ``comp_external_rest`` loads from disk.
"""

from __future__ import annotations

import sys
import types

from reinvent4_mcp.utils import custom_plugins_path

# Make the namespace package (which includes comp_external_rest.py) importable.
_plugins_dir = custom_plugins_path()
if _plugins_dir not in sys.path:
    sys.path.insert(0, _plugins_dir)

# Resolve reinvent_plugins / reinvent_plugins.components as real namespace
# packages (do not stub them, or comp_external_rest would not be found).
import reinvent_plugins  # noqa: E402
import reinvent_plugins.components  # noqa: E402

# Stub only the two leaf modules REINVENT4 would normally provide.
if "reinvent_plugins.components.add_tag" not in sys.modules:
    _at = types.ModuleType("reinvent_plugins.components.add_tag")

    def _add_tag(tag):  # noqa: ANN001, ANN202
        def _decorator(cls):  # noqa: ANN001, ANN202
            return cls

        return _decorator

    _at.add_tag = _add_tag
    sys.modules["reinvent_plugins.components.add_tag"] = _at
    reinvent_plugins.components.add_tag = _at

if "reinvent_plugins.components.component_results" not in sys.modules:
    _cr = types.ModuleType("reinvent_plugins.components.component_results")

    class _ComponentResults:
        def __init__(self, scores, metadata=None):  # noqa: ANN001
            self.scores = scores
            self.metadata = metadata

    _cr.ComponentResults = _ComponentResults
    sys.modules["reinvent_plugins.components.component_results"] = _cr
    reinvent_plugins.components.component_results = _cr
