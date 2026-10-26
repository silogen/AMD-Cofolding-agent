# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Saving and restoring a backend's conversation, as the web gateway does."""

from __future__ import annotations

import json

from reinvent4_agent.backends.anthropic_backend import AnthropicBackend
from reinvent4_agent.backends.openai_backend import OpenAIBackend


class _Block:
    """Stands in for an Anthropic SDK content block."""

    def __init__(self, **fields):
        self.fields = fields

    def model_dump(self, mode="json", exclude_none=True):
        return dict(self.fields)


def test_openai_history_round_trips_under_the_current_system_prompt():
    old = OpenAIBackend(model="x", api_key="dummy", system_prompt="old prompt")
    old.add_user_message("hi")
    old._messages.append({"role": "assistant", "content": "hello", "reasoning_content": "think"})
    saved = json.loads(json.dumps(old.export_history()))

    new = OpenAIBackend(model="x", api_key="dummy", system_prompt="new prompt")
    new.import_history(saved)

    assert new._messages[0] == {"role": "system", "content": "new prompt"}
    assert [m["role"] for m in new._messages] == ["system", "user", "assistant"]


def test_anthropic_sdk_blocks_become_plain_data_and_can_be_repaired():
    old = AnthropicBackend(model="x", api_key="dummy", system_prompt="sp")
    old.add_user_message("design")
    old._messages.append(
        {"role": "assistant", "content": [_Block(type="tool_use", id="toolu_1", name="get_ttp", input={})]}
    )
    saved = json.loads(json.dumps(old.export_history()))
    assert saved[1]["content"][0]["id"] == "toolu_1"

    new = AnthropicBackend(model="x", api_key="dummy", system_prompt="sp")
    new.import_history(saved)
    new.repair_dangling_tool_use()
    assert new._messages[-1]["content"][0]["tool_use_id"] == "toolu_1"
