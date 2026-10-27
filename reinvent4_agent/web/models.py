# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The chat models the UI may choose between, and the swap between them.

On LUMI a single vLLM owns all four GCDs of a node, so there is one chat
model at a time and choosing another means restarting that server.  The
restart is not ours: a shell supervisor watches a control directory and does
the work, which takes three to six minutes.  This module is the gateway's
half of that handshake, plus the catalogue of what may be asked for.

The control directory is named by ``REINVENT4_WEB_MODEL_CONTROL`` and holds
two files::

    model.want     the key of the wanted model, written by the gateway
    model.state    JSON written by the supervisor, e.g.
                   {"model": "gemma", "state": "loading",
                    "job": "23208674", "at": 1793102522}

Neither file exists on a laptop, where nothing supervises anything.  Then the
catalogue is still reported and switching is simply unavailable, so the UI can
show the model it is talking to without offering to change it.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Directory shared with the supervisor that restarts vLLM.
CONTROL_DIR_ENV = "REINVENT4_WEB_MODEL_CONTROL"

WANT_FILE = "model.want"
STATE_FILE = "model.state"

#: States the supervisor reports.  Anything else is read as ``failed``, since
#: an unknown state is not one to send a turn into.
STATES = frozenset({"loading", "ready", "failed"})


@dataclass(frozen=True)
class ModelSpec:
    """One servable chat model, as the UI and the backend need to see it."""

    key: str
    served_id: str
    label: str
    efforts: tuple[str, ...] = ()
    default_effort: str | None = None
    #: Extra request body vLLM needs for this model.  Gemma's thinking is
    #: always on, which this build only honours through the chat template,
    #: and it rejects ``reasoning_effort`` alongside it.
    extra_body: dict[str, Any] = field(default_factory=dict)

    @property
    def supports_effort(self) -> bool:
        return bool(self.efforts)

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "id": self.served_id,
            "label": self.label,
            "efforts": list(self.efforts),
            "defaultEffort": self.default_effort,
        }


#: Both models the LUMI deployment can serve, in the order the UI lists them.
CATALOGUE: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="qwen",
        served_id="Qwen/Qwen3.8-27B",
        label="Qwen3.8 27B",
        efforts=("low", "medium", "high"),
        default_effort="low",
    ),
    ModelSpec(
        key="gemma",
        served_id="google/gemma-4-31B-it",
        label="Gemma 4 31B",
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    ),
)

#: The model assumed when nothing says otherwise.
DEFAULT_KEY = CATALOGUE[0].key


def by_key(key: str | None) -> ModelSpec | None:
    return next((spec for spec in CATALOGUE if spec.key == key), None)


def key_for_served_id(served_id: str | None) -> str | None:
    """The key of the model an endpoint serves under *served_id*."""
    if not served_id:
        return None
    wanted = served_id.strip().lower()
    return next((spec.key for spec in CATALOGUE if spec.served_id.lower() == wanted), None)


class ModelControl:
    """Reads and writes the control directory; answers which model is live.

    Every read goes to disk, because the supervisor writes the state file from
    another process and a turn starting now must go to the model serving now.
    The files are tiny and a turn is minutes long, so there is nothing here
    worth caching.
    """

    def __init__(self, directory: str | None = None, fallback_model: str | None = None) -> None:
        raw = directory if directory is not None else os.environ.get(CONTROL_DIR_ENV)
        self.directory = Path(raw).expanduser() if raw else None
        #: What the endpoint serves when no supervisor says otherwise: the
        #: model ``dd-agent`` itself was configured with.
        self.fallback_model = fallback_model
        self.fallback_key = key_for_served_id(fallback_model) or DEFAULT_KEY

    @property
    def switchable(self) -> bool:
        """Whether a swap can be asked for at all.

        A directory that is configured but absent means no supervisor is
        running, and a want nobody reads would leave the UI loading for ever.
        """
        return self.directory is not None and self.directory.is_dir()

    def status(self) -> tuple[str, str]:
        """The live model's key and its state.

        Falls back to the configured model in the ``ready`` state whenever the
        supervisor has said nothing yet, which covers both the laptop and the
        first minutes of a session, before anything has been swapped.
        """
        state = self._read_state()
        spec = self._reported(state)
        if spec is None:
            return self.fallback_key, "ready"
        reported = state.get("state")
        return spec.key, reported if reported in STATES else "failed"

    def active(self) -> ModelSpec | None:
        """The catalogued model requests must go to now.

        ``None`` when the endpoint serves something outside the catalogue
        altogether — a hosted Claude, say — in which case the caller keeps the
        model it was configured with and nothing here applies to it.
        """
        return self._reported(self._read_state()) or by_key(key_for_served_id(self.fallback_model))

    def request(self, key: str) -> None:
        """Ask the supervisor for *key*; raises OSError if it cannot be written."""
        if self.directory is None:
            raise OSError("No model control directory is configured")
        target = self.directory / WANT_FILE
        tmp = target.with_name(f".{WANT_FILE}.tmp")
        tmp.write_text(f"{key}\n", encoding="utf-8")
        tmp.replace(target)
        logger.info("Asked the supervisor for model %s", key)

    def snapshot(self, current: str | None = None, state: str | None = None) -> dict[str, Any]:
        """The payload ``/api/models`` answers with.

        *current* and *state* override what the control directory says, so a
        swap that has just been asked for is reported as loading before the
        supervisor has had a chance to write anything.
        """
        live_key, live_state = self.status()
        return {
            "current": current or live_key,
            "state": state or live_state,
            "switchable": self.switchable,
            "models": [spec.describe() for spec in CATALOGUE],
        }

    @staticmethod
    def _reported(state: dict[str, Any]) -> ModelSpec | None:
        """The model the supervisor says is up, if it says anything sensible."""
        key = state.get("model")
        return by_key(key) if isinstance(key, str) else None

    def _read_state(self) -> dict[str, Any]:
        if self.directory is None:
            return {}
        try:
            raw = (self.directory / STATE_FILE).read_text(encoding="utf-8")
        except OSError:
            return {}
        try:
            value = json.loads(raw)
        except ValueError:
            logger.warning("Ignoring unreadable %s in %s", STATE_FILE, self.directory)
            return {}
        return value if isinstance(value, dict) else {}
