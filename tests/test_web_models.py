# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the chat model catalogue and the control directory.

Only the filesystem handshake with the supervisor is exercised here, so these
run anywhere::

    python -m pytest tests/test_web_models.py
"""

from __future__ import annotations

import json

import pytest
from reinvent4_agent.web import models


@pytest.fixture
def control(tmp_path, monkeypatch):
    """A control directory as the supervisor would leave it."""
    directory = tmp_path / "model-control"
    directory.mkdir()
    monkeypatch.setenv(models.CONTROL_DIR_ENV, str(directory))
    return directory


def write_state(directory, model, state, **extra):
    payload = {"model": model, "state": state, "job": "23208674", "at": 1793102522, **extra}
    (directory / models.STATE_FILE).write_text(json.dumps(payload), encoding="utf-8")


# -- the catalogue ---------------------------------------------------------


def test_the_catalogue_describes_both_models():
    described = [spec.describe() for spec in models.CATALOGUE]
    assert described == [
        {
            "key": "qwen",
            "id": "Qwen/Qwen3.8-27B",
            "label": "Qwen3.8 27B",
            "efforts": ["low", "medium", "high"],
            "defaultEffort": "low",
        },
        {
            "key": "gemma",
            "id": "google/gemma-4-31B-it",
            "label": "Gemma 4 31B",
            "efforts": [],
            "defaultEffort": None,
        },
    ]


def test_only_gemma_carries_a_chat_template_option():
    assert models.by_key("qwen").extra_body == {}
    assert models.by_key("gemma").extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
    assert models.by_key("qwen").supports_effort
    assert not models.by_key("gemma").supports_effort


def test_a_served_id_names_its_key():
    assert models.key_for_served_id("google/gemma-4-31B-it") == "gemma"
    assert models.key_for_served_id("qwen/qwen3.8-27b") == "qwen"
    assert models.key_for_served_id("claude-sonnet-4") is None
    assert models.key_for_served_id(None) is None


# -- without a supervisor --------------------------------------------------


def test_without_a_control_directory_nothing_can_be_switched():
    control = models.ModelControl(fallback_model="Qwen/Qwen3.8-27B")
    assert not control.switchable
    assert control.snapshot() == {
        "current": "qwen",
        "state": "ready",
        "switchable": False,
        "models": [spec.describe() for spec in models.CATALOGUE],
    }


def test_the_configured_model_decides_which_one_is_current():
    assert models.ModelControl(directory=None, fallback_model="google/gemma-4-31B-it").status() == ("gemma", "ready")
    assert models.ModelControl(directory=None, fallback_model="something else").status() == ("qwen", "ready")


def test_a_model_outside_the_catalogue_is_left_alone():
    """A hosted Claude is not ours to swap, so nothing rewrites it."""
    assert models.ModelControl(directory=None, fallback_model="claude-sonnet-4").active() is None
    assert models.ModelControl(directory=None, fallback_model="Qwen/Qwen3.8-27B").active().key == "qwen"


def test_a_configured_but_absent_directory_is_not_switchable(tmp_path, monkeypatch):
    monkeypatch.setenv(models.CONTROL_DIR_ENV, str(tmp_path / "gone"))
    control = models.ModelControl(fallback_model="Qwen/Qwen3.8-27B")
    assert not control.switchable
    assert control.status() == ("qwen", "ready")


# -- with a supervisor -----------------------------------------------------


def test_the_state_file_says_which_model_serves(control):
    write_state(control, "gemma", "loading")
    live = models.ModelControl(fallback_model="Qwen/Qwen3.8-27B")
    assert live.switchable
    assert live.status() == ("gemma", "loading")
    assert live.active().served_id == "google/gemma-4-31B-it"
    assert live.snapshot()["current"] == "gemma"


@pytest.mark.parametrize("state", ["loading", "ready", "failed"])
def test_every_reported_state_is_passed_on(control, state):
    write_state(control, "qwen", state)
    assert models.ModelControl().status() == ("qwen", state)


@pytest.mark.parametrize(
    "raw",
    ["", "not json", "[]", '{"model": "llama", "state": "ready"}', '{"state": "ready"}'],
)
def test_an_unusable_state_file_falls_back_to_the_configured_model(control, raw):
    (control / models.STATE_FILE).write_text(raw, encoding="utf-8")
    assert models.ModelControl(fallback_model="google/gemma-4-31B-it").status() == ("gemma", "ready")


def test_an_unknown_state_is_read_as_failed(control):
    write_state(control, "gemma", "exploded")
    assert models.ModelControl().status() == ("gemma", "failed")


def test_asking_for_a_model_writes_the_want_file(control):
    live = models.ModelControl()
    live.request("gemma")
    assert (control / models.WANT_FILE).read_text(encoding="utf-8").strip() == "gemma"
    live.request("qwen")
    assert (control / models.WANT_FILE).read_text(encoding="utf-8").strip() == "qwen"
    # The partial file is renamed over the want, never left beside it.
    assert sorted(p.name for p in control.iterdir()) == [models.WANT_FILE]


def test_asking_without_a_directory_fails_rather_than_pretending():
    with pytest.raises(OSError):
        models.ModelControl(directory=None).request("gemma")


def test_a_snapshot_can_report_a_swap_before_the_supervisor_does(control):
    write_state(control, "qwen", "ready")
    snapshot = models.ModelControl().snapshot(current="gemma", state="loading")
    assert snapshot["current"] == "gemma"
    assert snapshot["state"] == "loading"
    assert snapshot["switchable"] is True
