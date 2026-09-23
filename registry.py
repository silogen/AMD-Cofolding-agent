"""registry.py — Extension-registration seam.

One import in server.py calls load_extensions().
Every subsequent feature branch adds its module name to EXTENSION_MODULES.
server.py is never edited again for extension loading.
"""

from __future__ import annotations

import importlib
import logging

log = logging.getLogger(__name__)

EXTENSION_MODULES: list[str] = [
    "reinvent4_mcp.tools.rcsb",   # 01 — RCSB/PDB fetcher
    # 02 — ChEMBL fetcher  (add here when implemented)
    # 03 — structure prep   (add here when implemented)
]


def load_extensions(mcp: object) -> None:
    """Import each extension module and call its register_tools(mcp).

    A failure in one extension is logged and isolated;
    it does not prevent other extensions or existing tools from loading.
    """
    for mod_path in EXTENSION_MODULES:
        try:
            mod = importlib.import_module(mod_path)
            mod.register_tools(mcp)  # type: ignore[attr-defined]
            log.info("Loaded RCSB extension: %s", mod_path)
        except Exception as exc:
            log.error("Failed to load extension %s: %s", mod_path, exc, exc_info=True)
