# ABOUTME: Tests answer prompt construction for tutor reveal runs.
# ABOUTME: Ensures the case prompt is used instead of the student's reveal request.

from __future__ import annotations

import unittest


class TestAnswerPrompt(unittest.TestCase):
    def test_uses_case_prompt_when_available(self) -> None:
        from medical_tutor.contracts import AgentState
        from medical_tutor.runtime.answer import build_answer_prompt_text

        state = AgentState(
            messages=[
                {"role": "user", "content": "Case stem"},
                {"role": "assistant", "content": "Tutor question"},
                {"role": "user", "content": "Please reveal the answer now."},
            ],
            case_prompt="ACTUAL CASE QUESTION + OPTIONS",
            retrieval_summary="some retrieval",
        )

        prompt = build_answer_prompt_text(state)

        self.assertIn("Question: ACTUAL CASE QUESTION + OPTIONS", prompt)

    def test_reveal_uses_case_label(self) -> None:
        from medical_tutor.config import Config
        from medical_tutor.contracts import AgentState
        from medical_tutor.runtime.answer import answer_question

        state = AgentState(
            messages=[
                {"role": "user", "content": "Case stem"},
                {"role": "assistant", "content": "Tutor question"},
                {"role": "user", "content": "Please reveal the answer now."},
            ],
            case_prompt=(
                "Question stem here.\n\n"
                "Answer Choices:\n"
                "(A) First option\n"
                "(B) Second option\n"
                "(C) Third option"
            ),
            case_label="B",
            student_profile="novice",
            tutor_action={"type": "REVEAL_ANSWER", "arguments": {}},
        )

        answer, metadata = answer_question(state, Config())

        self.assertEqual(answer, "Correct answer: (B) Second option.")
        self.assertEqual(metadata.get("source"), "case_label")


if __name__ == "__main__":
    unittest.main()
