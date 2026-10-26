"""Extension module registry for the DrugDiscovery MCP server.

Each string in EXTENSION_MODULES must resolve to a Python module that
exports ``register_tools(mcp: FastMCP) -> None``.  The server iterates
this list at startup and calls ``register_tools`` on each module so that
all extension tools are registered with the same ``mcp`` instance.

Ordering follows the block numbering in the project plan:
  01 — RCSB/PDB fetcher
  02 — ChEMBL fetcher
  03 — structure preparator
  06 — interaction profiler v2
  07 — AiZynthFinder synthesis planner
  08 — REINVENT generation contracts (LibInvent / LinkInvent)
  09 — Boltz structure conditioning
  10 — UniProt extended features + InterPro domain annotations
"""

import importlib
import logging

log = logging.getLogger(__name__)

EXTENSION_MODULES: list[str] = [
    "reinvent4_mcp.tools.rcsb",                 # 01 — RCSB/PDB fetcher
    "reinvent4_mcp.tools.chembl",               # 02 — ChEMBL fetcher
    "reinvent4_mcp.tools.structure_prep",        # 03 — structure preparator
    "reinvent4_mcp.tools.interaction_profiler",  # 06 — interaction profiler v2
    "reinvent4_mcp.tools.synthesis",             # 07 — AiZynthFinder synthesis planner
    "reinvent4_mcp.tools.generation",            # 08 — REINVENT generation contracts
    "reinvent4_mcp.tools.boltz_structure",       # 09 — Boltz structure conditioning
    "reinvent4_mcp.tools.uniprot_features",      # 10 — UniProt extended features
    "reinvent4_mcp.tools.interpro",              # 10 — InterPro domain annotations
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
            log.info("Loaded extension: %s", mod_path)
        except Exception as exc:
            log.error(
                "Failed to load extension %s: %s", mod_path, exc, exc_info=True
            )
