# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the campaign log's shape, updates and rendering."""

from __future__ import annotations

import pytest

from reinvent4_agent.web import campaign


def test_an_empty_log_renders_as_nothing():
    assert campaign.render(campaign.empty()) == ""
    assert campaign.is_empty(campaign.empty())


def test_anything_unreadable_on_disk_is_an_empty_log():
    for raw in (None, "text", [], {"findings": "nope"}, {"goal": "   "}):
        assert campaign.normalise(raw) == campaign.empty()


def test_goal_and_next_replace_findings_append():
    log = campaign.apply(campaign.empty(), {"goal": "A", "finding": "first", "next": "N1"})
    log = campaign.apply(log, {"goal": "B", "finding": "second", "next": "N2"})

    assert log["goal"] == "B"
    assert [f["text"] for f in log["findings"]] == ["first", "second"]
    assert log["next"] == "N2"
    assert log["updated"]


def test_an_empty_next_clears_it():
    log = campaign.apply(campaign.empty(), {"next": "Cofold the top 5"})
    log = campaign.apply(log, {"next": ""})
    assert log["next"] is None


def test_a_call_with_nothing_in_it_is_refused():
    with pytest.raises(ValueError):
        campaign.apply(campaign.empty(), {"goal": "  ", "finding": ""})


def test_long_text_is_cut_and_old_findings_drop_off():
    log = campaign.empty()
    for i in range(campaign.MAX_FINDINGS + 5):
        log = campaign.apply(log, {"finding": f"finding {i}"})
    log = campaign.apply(log, {"goal": "x" * 5000})

    assert len(log["findings"]) == campaign.MAX_FINDINGS
    assert log["findings"][0]["text"] == "finding 5"
    assert len(log["goal"]) == campaign.MAX_TEXT_CHARS


def test_a_wake_shows_the_recent_findings_and_says_how_many_are_hidden():
    log = campaign.empty()
    for i in range(campaign.FINDINGS_SHOWN + 3):
        log = campaign.apply(log, {"finding": f"finding {i}"})

    text = campaign.render(log)

    assert "Goal: (not set -- set it)" in text
    assert "(3 earlier not shown)" in text
    assert "- finding 3" in text
    assert "- finding 2" not in text
    assert "Next: (not decided)" in text
