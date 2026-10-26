"""How a tool result is summarised back to the agent.

This one line is the agent's whole view of whether a call worked, so getting it
wrong does not merely misinform a human reading the transcript -- the agent
acts on it. A conductor told its successful enqueue had failed opened three
more campaigns trying to recover.
"""

from __future__ import annotations

from reinvent4_agent.display import _summarize_tool_result


class TestAnErrorFieldThatIsEmpty:
    """Rows carry a nullable ``error`` column; null means nothing is wrong.

    The store's task and job records always include ``error``, so testing for
    the key rather than a message turned every successful enqueue into
    "Error: None".
    """

    TASK_ROW = {
        "task_id": "a1b2c3d4",
        "campaign_id": "feed0001",
        "state": "queued",
        "attempts": 0,
        "error": None,
    }

    def test_a_null_error_is_not_reported_as_a_failure(self):
        summary = _summarize_tool_result("enqueue_task", self.TASK_ROW)

        assert "Error" not in summary

    def test_an_empty_string_error_is_not_either(self):
        summary = _summarize_tool_result("enqueue_task", {**self.TASK_ROW, "error": ""})

        assert "Error" not in summary

    def test_a_real_error_is_still_reported(self):
        summary = _summarize_tool_result(
            "enqueue_task", {"error": "Campaign feed0001 is draining"}
        )

        assert summary == "Error: Campaign feed0001 is draining"

    def test_a_real_error_beside_other_fields_is_still_reported(self):
        summary = _summarize_tool_result(
            "run_optimization",
            {"job_id": None, "error": "Maximum concurrent jobs (6) already active"},
        )

        assert "Maximum concurrent jobs" in summary
