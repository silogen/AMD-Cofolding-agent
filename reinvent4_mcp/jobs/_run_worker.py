#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Worker process: registers custom scoring components then runs REINVENT4.

Launched as a subprocess by the job manager. It puts REINVENT4 and the custom
plugins dir on ``sys.path`` so the ``ExternalRESTScorer`` plugin is discoverable
via the ``reinvent_plugins.components`` namespace package, then runs REINVENT4.

Usage::

    python _run_worker.py <toml_config_path> [log_path]

Environment variables:

    REINVENT4_ROOT          Path to REINVENT4 project directory
    DD_MCP_PLUGINS_DIR      Path to the custom plugins dir (reinvent_plugins
                            namespace layout); optional fallback
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python _run_worker.py <toml_path> [log_path]", file=sys.stderr)
        sys.exit(1)

    toml_path = sys.argv[1]
    log_path = sys.argv[2] if len(sys.argv) > 2 else "reinvent.log"

    # Put REINVENT4 AND the custom plugins dir on sys.path so REINVENT4's
    # reinvent_plugins.components namespace scan discovers ExternalRESTScorer
    # natively. Without this the RL worker's registry has no 'externalrestscorer'
    # and every TTP with external objectives fails with KeyError. A real sys.path
    # entry (not a runtime __path__ append) survives REINVENT4's later sys.path
    # mutations, which is what broke the previous approach.
    try:
        from reinvent4_mcp.utils import ensure_reinvent4_importable

        ensure_reinvent4_importable()
    except Exception:  # noqa: BLE001 - fall back to env-based discovery
        reinvent4_root = os.environ.get("REINVENT4_ROOT", "")
        if reinvent4_root and reinvent4_root not in sys.path:
            sys.path.insert(0, reinvent4_root)
        plugins_dir = os.environ.get("DD_MCP_PLUGINS_DIR", "")
        if plugins_dir and plugins_dir not in sys.path:
            sys.path.insert(0, plugins_dir)

    sys.argv = ["reinvent", "-l", log_path, toml_path]

    from reinvent.Reinvent import main_script as reinvent_main

    reinvent_main()


if __name__ == "__main__":
    main()
