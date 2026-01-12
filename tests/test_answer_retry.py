# ABOUTME: Tests answer-node retry behavior when the answer model returns insufficient evidence.
# ABOUTME: Ensures model errors do not trigger pointless retries and duplicate error records.

from __future__ import annotations

import unittest
from unittest.mock import patch


class TestAnswerRetry(unittest.TestCase):
    def test_does_not_retry_when_answer_model_errors(self) -> None:
        from medical_tutor.config import Config
        from medical_tutor.contracts import AgentState
        from medical_tutor.runtime import nodes

        state = AgentState(
            messages=[{"role": "user", "content": "Case stem"}],
            image="/tmp/example.png",
            run_id="run_test",
        )

        with patch(
            "medical_tutor.runtime.nodes.answer_question",
            return_value=("Insufficient evidence to answer confidently.", {"error": "boom"}),
        ):
            out = nodes.answer(state.model_dump(), Config(), tracer=None)

        updated = AgentState.model_validate(out)
        self.assertEqual(updated.next_action, "respond")
        self.assertEqual(updated.retry_count, 0)
        self.assertTrue(any(err.code == "answer_error" for err in updated.errors))


if __name__ == "__main__":
    unittest.main()
