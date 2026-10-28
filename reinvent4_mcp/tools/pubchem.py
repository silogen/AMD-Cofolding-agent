# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PubChem compound lookup: resolve a molecule NAME to a real SMILES.

The agent must never recall a SMILES from memory (it hallucinates invalid or
wrong structures - e.g. bad JQ1 variants that RDKit cannot parse or 3D-embed).
This tool looks up a named compound on PubChem and returns its canonical
SMILES so downstream cofolding/scoring uses a real structure.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_TIMEOUT = 20
# Property names in the current PubChem API: SMILES is the isomeric (stereo)
# form; ConnectivitySMILES is the flat/canonical form.
_PROPS = "SMILES,ConnectivitySMILES,MolecularFormula,MolecularWeight,IUPACName,Title"

# PubChem throttles rather than failing: 503 PUGREST.ServerBusy and 429, both
# of which clear within seconds. One attempt turns that into a dead end -- a
# live run stalled on a 503 and asked the operator to paste a SMILES by hand,
# which is the one thing this module exists to prevent.
_RETRY_STATUS = frozenset({429, 503})
# Two retries: enough for a busy window, bounded so an interactive turn is not
# held for a minute by an endpoint that is actually down.
_RETRY_DELAYS = (1.0, 3.0)
# PubChem's own Retry-After is honoured, but not unboundedly.
_MAX_RETRY_WAIT = 5.0


def _retry_wait(response: Optional[requests.Response], fallback: float) -> float:
    """How long to wait, preferring what the server asked for."""

    header = response.headers.get("Retry-After") if response is not None else None
    try:
        asked = float(header) if header else fallback
    except (TypeError, ValueError):
        asked = fallback
    return min(max(asked, 0.0), _MAX_RETRY_WAIT)


def _request_properties(name: str) -> requests.Response:
    """GET the property table, riding out throttling.

    Raises the last transport error if every attempt failed, so the caller's
    handler still reports an unreachable PubChem honestly.
    """

    url = (
        f"{_BASE}/compound/name/{requests.utils.quote(name)}"
        f"/property/{_PROPS}/JSON"
    )
    last_error: Optional[requests.RequestException] = None
    last_response: Optional[requests.Response] = None

    for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
        try:
            response = requests.get(url, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            last_error, response = exc, None
        else:
            if response.status_code not in _RETRY_STATUS:
                return response
            last_error, last_response = None, response

        if delay is None:
            break
        logger.info(
            "PubChem is throttling %r (attempt %d); retrying in %.0fs",
            name, attempt + 1, _retry_wait(last_response, delay),
        )
        time.sleep(_retry_wait(last_response, delay))

    if last_response is not None:
        return last_response
    raise last_error  # type: ignore[misc]


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
        resp = _request_properties(name)
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
