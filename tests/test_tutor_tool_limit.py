# ABOUTME: Ensures tutor routing avoids infinite tool loops in a single turn.
# ABOUTME: Blocks repeated tool requests after a tool has already run.

from __future__ import annotations

import unittest


class _DummyToolRegistry:
    def call(self, *_args, **_kwargs):
        return None


class TestTutorToolLimit(unittest.TestCase):
    def test_prevents_second_tool_call_in_same_turn(self) -> None:
        from medical_tutor.config import Config
        from medical_tutor.contracts import AgentState, ToolCall, ToolResult
        from medical_tutor.runtime import nodes
        from medical_tutor.tutoring import TutorAction

        state = AgentState(
            messages=[{"role": "user", "content": "Case stem"}],
            student_profile="novice",
            tool_calls=[ToolCall(id="call-1", name="zoom", arguments={"bbox_2d": [0, 0, 1, 1]})],
            tool_results=[ToolResult(tool_name="zoom", ok=True, output={"summary": "Zoomed."})],
        )

        def policy(_state: AgentState) -> TutorAction:
            return TutorAction(
                type="REQUEST_TOOL", arguments={"name": "enhance", "arguments": {"factor": 1.2}}
            )

        result = nodes.tutor_decide(
            state.model_dump(),
            Config(),
            _DummyToolRegistry(),
            "",
            None,
            policy=policy,
        )
        updated = AgentState.model_validate(result)

        self.assertNotEqual(updated.next_action, "tool")
        self.assertEqual(len(updated.tool_calls), 1)


if __name__ == "__main__":
    unittest.main()
