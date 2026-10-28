# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the Boltz-2 service is cofolding right now, for the UI's run bar.

The browser cannot reach the services on the compute node, so the gateway
reads Boltz's ``/health`` (workers, queue and the requests in progress) and
hands back a compact summary.  The UI polls this only while it believes a
cofold is running, through a slow tunnel, so a slow or absent service answers
``{"available": false}`` quickly instead of holding the request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

TIMEOUT_S = 2.0


def boltz_health_url() -> str | None:
    """``/health`` of the Boltz service the MCP tools score with, if enabled."""
    try:
        from reinvent4_mcp.config import get_config

        url = get_config().enabled_external_services.get("boltz")
    except Exception as exc:  # noqa: BLE001 - no MCP config is "no service"
        logger.debug("No Boltz service configured: %s", exc)
        return None
    if not url:
        return None
    return f"{url.rsplit('/', 1)[0]}/health"


def _read_health(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:  # noqa: S310 - configured service URL
        body = json.loads(response.read(1_000_000))
    if not isinstance(body, dict):
        raise ValueError("Boltz /health did not answer an object")
    return body


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def summarize(health: dict[str, Any]) -> dict[str, Any]:
    """The run bar's view of one Boltz ``/health`` answer."""
    workers = [w for w in health.get("workers") or [] if isinstance(w, dict)]
    states = [w.get("state") for w in workers]
    busy_devices = {w.get("device") for w in workers if w.get("state") == "busy"}
    requests = []
    for entry in health.get("requests") or []:
        if not isinstance(entry, dict):
            continue
        elapsed = entry.get("elapsed_s")
        requests.append(
            {
                "id": str(entry.get("id", "")),
                "molecules": _int(entry.get("molecules")),
                "done": _int(entry.get("done")),
                "queued": _int(entry.get("queued")),
                "running": _int(entry.get("running")),
                "state": entry.get("state") if entry.get("state") in ("preparing", "queued", "running") else "running",
                "elapsedSeconds": round(float(elapsed), 1) if isinstance(elapsed, (int, float)) else None,
            }
        )
    queue_length = _int(health.get("queue_length"))
    busy = states.count("busy")
    return {
        "available": True,
        "status": str(health.get("status", "unknown")),
        "mode": str(health.get("mode") or ("pool" if workers else "subprocess")),
        "workers": {
            "total": len(workers),
            "busy": busy,
            "loading": states.count("loading"),
            "failed": states.count("failed"),
        },
        "gpus": {"total": len({w.get("device") for w in workers}), "busy": len(busy_devices)},
        "queueLength": queue_length,
        "requests": requests,
        "active": bool(requests or busy or queue_length),
    }


async def cofold_status() -> dict[str, Any]:
    url = boltz_health_url()
    if url is None:
        return {"available": False}
    try:
        health = await asyncio.wait_for(asyncio.to_thread(_read_health, url), TIMEOUT_S + 0.5)
    except Exception as exc:  # noqa: BLE001 - any failure means "not available now"
        logger.debug("Boltz health at %s unavailable: %s", url, exc)
        return {"available": False}
    return summarize(health)
