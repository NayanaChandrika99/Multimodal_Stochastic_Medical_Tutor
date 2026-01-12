# ABOUTME: Ensures tutor policy prompts include assessment and tool summaries.
# ABOUTME: Prevents silent loss of tutoring context for controller decisions.

from __future__ import annotations

import unittest


class TestTutorPromptContext(unittest.TestCase):
    def test_prompt_includes_assessment_and_tool_summaries(self) -> None:
        from medical_tutor.contracts import AgentState, ToolResult
        from medical_tutor.runtime import tutor_policy

        state = AgentState(
            messages=[{"role": "user", "content": "Case stem"}],
            run_id="run_test",
            student_profile="novice",
            student_attempt_count=1,
            case_prompt="Case stem\n\nAnswer Choices:\n(A) Foo\n(B) Bar",
            student_reply_text="I think the answer is A",
            student_reply_grade="wrong",
            student_reply_misconception="Missed key radiograph finding",
            tool_results=[
                ToolResult(
                    tool_name="zoom",
                    ok=True,
                    output={"summary": "Zoomed into RUQ.", "bounds": [0, 0, 10, 10]},
                )
            ],
        )

        prompt = tutor_policy._build_tutor_prompt(state, playbook="", tool_registry=None)

        self.assertIn("Student assessment", prompt)
        self.assertIn("wrong", prompt)
        self.assertIn("Missed key radiograph finding", prompt)
        self.assertIn("Tool summaries", prompt)
        self.assertIn("zoom", prompt)
        self.assertIn("Answer Choices", prompt)

    def test_prompt_biases_tool_use_when_image_present(self) -> None:
        from medical_tutor.contracts import AgentState
        from medical_tutor.runtime import tutor_policy

        state = AgentState(
            messages=[{"role": "user", "content": "Case stem"}],
            run_id="run_test",
            student_profile="novice",
            student_attempt_count=0,
            case_prompt="Case stem\n\nAnswer Choices:\n(A) Foo\n(B) Bar",
            image="image.png",
        )

        prompt = tutor_policy._build_tutor_prompt(state, playbook="", tool_registry=None)

        self.assertIn("request one vision tool", prompt)


if __name__ == "__main__":
    unittest.main()
