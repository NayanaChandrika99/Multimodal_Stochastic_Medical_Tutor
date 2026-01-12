# ABOUTME: Tests tutor-action routing behavior in the LangGraph node implementation.
# ABOUTME: Ensures TutorAction is converted into tool calls, assistant messages, and next_action values.
# pyright: reportMissingImports=false

from __future__ import annotations

import unittest


class _FakeToolRegistry:
    def call(self, *_: object, **__: object) -> None:  # pragma: no cover - not used in this test
        raise AssertionError("tool execution should not be invoked by tutor_decide")


class TestTutorRouting(unittest.TestCase):
    def test_tutor_decide_routes_request_tool_to_tool_exec(self) -> None:
        from medical_tutor.config import Config
        from medical_tutor.runtime.nodes import tutor_decide
        from medical_tutor.tutoring import TutorAction

        def policy(_: object) -> TutorAction:
            return TutorAction(
                type="REQUEST_TOOL",
                arguments={
                    "name": "zoom",
                    "arguments": {"bbox_2d": [0.0, 0.0, 1.0, 1.0], "padding": 0.1},
                },
            )

        out = tutor_decide(
            {
                "messages": [{"role": "user", "content": "Case stem"}],
                "run_id": "run_test",
                "student_profile": "novice",
                "student_attempt_count": 0,
            },
            Config(),
            _FakeToolRegistry(),
            playbook="",
            tracer=None,
            policy=policy,
        )

        self.assertEqual(out.get("next_action"), "tool")
        self.assertTrue(out.get("tool_calls"), "expected a tool call to be enqueued")

    def test_tutor_decide_applies_reveal_gate(self) -> None:
        from medical_tutor.config import Config
        from medical_tutor.runtime.nodes import tutor_decide
        from medical_tutor.tutoring import TutorAction

        def policy(_: object) -> TutorAction:
            return TutorAction(type="REVEAL_ANSWER", arguments={})

        out = tutor_decide(
            {
                "messages": [{"role": "user", "content": "Case stem"}],
                "run_id": "run_test",
                "student_profile": "novice",
                "student_attempt_count": 0,
            },
            Config(),
            _FakeToolRegistry(),
            playbook="",
            tracer=None,
            policy=policy,
        )

        self.assertEqual(out.get("next_action"), "respond")
        messages = out.get("messages") or []
        self.assertTrue(any(m.get("role") == "assistant" for m in messages))

    def test_tutor_decide_reveals_when_student_requests_reveal(self) -> None:
        from medical_tutor.config import Config
        from medical_tutor.runtime.nodes import tutor_decide
        from medical_tutor.tutoring import TutorAction

        def policy(_: object) -> TutorAction:
            return TutorAction(type="ASK_PROBE", arguments={"content": "What stands out to you?"})

        out = tutor_decide(
            {
                "messages": [{"role": "user", "content": "Case stem"}],
                "run_id": "run_test",
                "student_profile": "novice",
                "student_attempt_count": 1,
                "student_reply_text": "Please reveal the answer now.",
            },
            Config(),
            _FakeToolRegistry(),
            playbook="",
            tracer=None,
            policy=policy,
        )

        self.assertEqual(out.get("next_action"), "answer")
        self.assertEqual((out.get("tutor_action") or {}).get("type"), "REVEAL_ANSWER")

    def test_tutor_decide_requests_image_findings_when_student_asks_for_description(self) -> None:
        from medical_tutor.config import Config
        from medical_tutor.runtime.nodes import tutor_decide
        from medical_tutor.tutoring import TutorAction

        def policy(_: object) -> TutorAction:
            return TutorAction(type="ASK_PROBE", arguments={"content": "What stands out to you?"})

        out = tutor_decide(
            {
                "messages": [{"role": "user", "content": "Case stem"}],
                "run_id": "run_test",
                "student_profile": "novice",
                "student_attempt_count": 1,
                "student_reply_text": "Can you describe what looks abnormal in the image?",
                "image": "/tmp/example.png",
            },
            Config(),
            _FakeToolRegistry(),
            playbook="",
            tracer=None,
            policy=policy,
        )

        self.assertEqual(out.get("next_action"), "tool")
        tool_calls = out.get("tool_calls") or []
        self.assertEqual(tool_calls[0]["name"], "image_findings")

    def test_tutor_decide_does_not_request_second_tool_after_other_tools(self) -> None:
        from medical_tutor.config import Config
        from medical_tutor.runtime.nodes import tutor_decide
        from medical_tutor.tutoring import TutorAction

        def policy(_: object) -> TutorAction:
            return TutorAction(
                type="REQUEST_TOOL", arguments={"name": "enhance", "arguments": {"factor": 1.2}}
            )

        out = tutor_decide(
            {
                "messages": [{"role": "user", "content": "Case stem"}],
                "run_id": "run_test",
                "student_profile": "novice",
                "student_attempt_count": 1,
                "student_reply_text": "Can you describe what looks abnormal in the image?",
                "image": "/tmp/example.png",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "zoom",
                        "arguments": {"bbox_2d": [0.0, 0.0, 1.0, 1.0], "padding": 0.1},
                        "target_artifact": None,
                    }
                ],
                "tool_results": [
                    {
                        "tool_name": "zoom",
                        "ok": True,
                        "output": {"summary": "Zoom done."},
                        "error": None,
                        "artifact_refs": [],
                    }
                ],
            },
            Config(),
            _FakeToolRegistry(),
            playbook="",
            tracer=None,
            policy=policy,
        )

        self.assertNotEqual(out.get("next_action"), "tool")
        tool_calls = out.get("tool_calls") or []
        self.assertEqual(len(tool_calls), 1)
        messages = out.get("messages") or []
        assistant_messages = [
            m.get("content", "") for m in messages if m.get("role") == "assistant"
        ]
        self.assertTrue(
            any("already ran one tool" in str(content) for content in assistant_messages)
        )

    def test_tutor_decide_responds_with_image_findings_summary_after_tool_runs(self) -> None:
        from medical_tutor.config import Config
        from medical_tutor.runtime.nodes import tutor_decide

        def policy(_: object):  # pragma: no cover - should not be invoked
            raise AssertionError("policy should not be invoked when image findings are available")

        out = tutor_decide(
            {
                "messages": [{"role": "user", "content": "Case stem"}],
                "run_id": "run_test",
                "student_profile": "novice",
                "student_attempt_count": 1,
                "student_reply_text": "Can you describe what looks abnormal in the image?",
                "image": "/tmp/example.png",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "image_findings",
                        "arguments": {"bbox_2d": [0.2, 0.2, 0.8, 0.8]},
                        "target_artifact": None,
                    }
                ],
                "tool_results": [
                    {
                        "tool_name": "image_findings",
                        "ok": True,
                        "output": {"summary": "Test finding summary."},
                        "error": None,
                        "artifact_refs": [],
                    }
                ],
            },
            Config(),
            _FakeToolRegistry(),
            playbook="",
            tracer=None,
            policy=policy,
        )

        self.assertEqual(out.get("next_action"), "respond")
        messages = out.get("messages") or []
        assistant_messages = [
            m.get("content", "") for m in messages if m.get("role") == "assistant"
        ]
        self.assertTrue(
            any("Test finding summary." in str(content) for content in assistant_messages)
        )


if __name__ == "__main__":
    unittest.main()
