"""PubChem compound lookup: resolve a molecule NAME to a real SMILES.

The agent must never recall a SMILES from memory (it hallucinates invalid or
wrong structures - e.g. bad JQ1 variants that RDKit cannot parse or 3D-embed).
This tool looks up a named compound on PubChem and returns its canonical
SMILES so downstream cofolding/scoring uses a real structure.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_TIMEOUT = 20
# Property names in the current PubChem API: SMILES is the isomeric (stereo)
# form; ConnectivitySMILES is the flat/canonical form.
_PROPS = "SMILES,ConnectivitySMILES,MolecularFormula,MolecularWeight,IUPACName,Title"


def fetch_compound(name: str) -> dict[str, Any]:
    """Resolve a compound name (or synonym) to a real SMILES via PubChem.

    Args:
        name: Common name, synonym, or identifier (e.g. "JQ1", "aspirin",
            "acetazolamide", "imatinib").

    Returns:
        Dict with ``cid``, ``smiles`` (isomeric, use this for cofolding/scoring),
        ``connectivity_smiles`` (flat), ``molecular_formula``, ``molecular_weight``,
        ``iupac_name`` and ``title``. On failure returns ``{"error": ...}`` and
        does NOT invent a structure.
    """
    name = (name or "").strip()
    if not name:
        return {"error": "Empty name. Provide a compound name or synonym."}

    try:
        resp = requests.get(
            f"{_BASE}/compound/name/{requests.utils.quote(name)}/property/{_PROPS}/JSON",
            timeout=_TIMEOUT,
        )
        if resp.status_code == 404:
            return {"error": f"No PubChem compound found for name {name!r}."}
        resp.raise_for_status()
        props = (resp.json().get("PropertyTable") or {}).get("Properties") or []
        if not props:
            return {"error": f"No properties returned by PubChem for {name!r}."}
        p = props[0]
        smiles = p.get("SMILES") or p.get("ConnectivitySMILES")
        if not smiles:
            return {
                "error": (
                    f"PubChem returned no SMILES for {name!r} (CID {p.get('CID')})."
                )
            }
        return {
            "query": name,
            "cid": p.get("CID"),
            "smiles": smiles,
            "connectivity_smiles": p.get("ConnectivitySMILES"),
            "molecular_formula": p.get("MolecularFormula"),
            "molecular_weight": p.get("MolecularWeight"),
            "iupac_name": p.get("IUPACName"),
            "title": p.get("Title"),
        }
    except requests.RequestException as exc:
        logger.warning("PubChem request failed: %s", exc)
        return {
            "error": (
                f"Could not reach PubChem ({exc}). The SMILES was NOT retrieved; "
                "do not fabricate one - ask the user to paste the SMILES."
            )
        }
