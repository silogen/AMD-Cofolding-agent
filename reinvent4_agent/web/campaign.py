# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A conversation's campaign log: what the agent is after and what it learned.

Working on its own, the agent is woken by events hours apart, and replaying
the whole conversation each time is both slow and the reason it forgets what
it already tried.  The log is the part worth carrying: the goal, what each
experiment showed, and the next step with the result that argues for it.  The
agent keeps it with the ``campaign_log`` tool, every wake shows it back, and
the browser shows it beside the chat.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

#: Findings kept per conversation; the oldest go first.
MAX_FINDINGS = 40

#: Findings a wake shows the agent, newest last.
FINDINGS_SHOWN = 12

#: Longest goal, finding or next step, in characters: one or two sentences.
MAX_TEXT_CHARS = 600

TOOL_NAME = "campaign_log"

TOOL_DEFINITION: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Keep this conversation's campaign log: the goal you are pursuing, what "
        "each experiment or analysis showed, and the next step with the result "
        "that justifies it. The user sees it beside the chat, and it is shown "
        "back to you whenever you continue on your own, so keep every line short "
        "and specific (numbers, job ids, molecules). Call it after a result "
        "changes your view and whenever you decide the next step. All fields are "
        "optional; returns the log."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "The campaign objective in one sentence; replaces the current one.",
            },
            "finding": {
                "type": "string",
                "description": "What a finished experiment or analysis showed; appended.",
            },
            "next": {
                "type": "string",
                "description": (
                    "The next step and the result that argues for it; replaces the "
                    "current one. An empty string clears it."
                ),
            },
        },
    },
}


def empty() -> dict[str, Any]:
    return {"goal": None, "findings": [], "next": None, "updated": None}


def normalise(raw: Any) -> dict[str, Any]:
    """A stored log in the current shape, whatever was on disk."""
    log = empty()
    if not isinstance(raw, dict):
        return log
    if isinstance(raw.get("goal"), str) and raw["goal"].strip():
        log["goal"] = raw["goal"]
    if isinstance(raw.get("next"), str) and raw["next"].strip():
        log["next"] = raw["next"]
    if isinstance(raw.get("updated"), str):
        log["updated"] = raw["updated"]
    findings = raw.get("findings")
    if isinstance(findings, list):
        log["findings"] = [
            {"ts": str(item.get("ts") or ""), "text": item["text"]}
            for item in findings
            if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip()
        ][-MAX_FINDINGS:]
    return log


def is_empty(log: dict[str, Any]) -> bool:
    return not (log.get("goal") or log.get("findings") or log.get("next"))


def apply(log: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """The log after one ``campaign_log`` call.  Raises ValueError on a call that changes nothing."""
    updated = normalise(log)
    changed = False
    goal = _text(arguments.get("goal"))
    if goal:
        updated["goal"] = goal
        changed = True
    finding = _text(arguments.get("finding"))
    if finding:
        updated["findings"] = [*updated["findings"], {"ts": _now(), "text": finding}][-MAX_FINDINGS:]
        changed = True
    if "next" in arguments and isinstance(arguments["next"], str):
        updated["next"] = _text(arguments["next"]) or None
        changed = True
    if not changed:
        raise ValueError("Give at least one of 'goal', 'finding' or 'next'.")
    updated["updated"] = _now()
    return updated


def as_tool_result(log: dict[str, Any]) -> str:
    return json.dumps(log)


def render(log: dict[str, Any]) -> str:
    """The log as a wake shows it to the agent; empty when there is nothing in it."""
    if is_empty(log):
        return ""
    lines = ["Campaign log (yours, kept with campaign_log):"]
    lines.append(f"Goal: {log['goal'] or '(not set -- set it)'}")
    findings = log["findings"][-FINDINGS_SHOWN:]
    hidden = len(log["findings"]) - len(findings)
    if findings:
        lines.append("Findings, oldest first" + (f" ({hidden} earlier not shown)" if hidden else "") + ":")
        lines.extend(f"- {item['text']}" for item in findings)
    lines.append(f"Next: {log['next'] or '(not decided)'}")
    return "\n".join(lines)


def _text(value: Any) -> str:
    return value.strip()[:MAX_TEXT_CHARS] if isinstance(value, str) else ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
