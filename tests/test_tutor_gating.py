# ABOUTME: Tests the Socratic tutor gating and student reply assessment utilities.
# ABOUTME: Ensures we do not reveal answers before a student attempt and can grade MCQ replies.
# pyright: reportMissingImports=false

from __future__ import annotations

import unittest


class TestTutorGating(unittest.TestCase):
    def test_extract_mcq_choice_ignores_free_text(self) -> None:
        from medical_tutor.tutoring import extract_mcq_choice

        self.assertIsNone(extract_mcq_choice("A 62-year-old woman presents with abdominal pain."))
        self.assertIsNone(extract_mcq_choice("The type 2 diabetes?"))
        self.assertEqual(extract_mcq_choice("I think (D) because ..."), "D")
        self.assertEqual(extract_mcq_choice("Answer: A"), "A")

    def test_reveal_gate_blocks_reveal_before_first_attempt(self) -> None:
        from medical_tutor.contracts import AgentState
        from medical_tutor.tutoring import TutorAction, apply_reveal_gate

        state = AgentState(
            messages=[{"role": "user", "content": "Case question here"}],
            run_id="run_test",
            student_profile="novice",
            student_attempt_count=0,
        )
        action = TutorAction(type="REVEAL_ANSWER", arguments={})
        gated = apply_reveal_gate(state, action)

        self.assertNotEqual(gated.type, "REVEAL_ANSWER")
        self.assertEqual(gated.type, "ASK_PROBE")

    def test_assess_student_reply_marks_correct_choice(self) -> None:
        from medical_tutor.tutoring import assess_student_mcq_reply

        assessment = assess_student_mcq_reply(
            correct_label="B", student_reply="I think the answer is B."
        )
        self.assertEqual(assessment.grade, "correct")

    def test_assess_student_reply_marks_wrong_choice(self) -> None:
        from medical_tutor.tutoring import assess_student_mcq_reply

        assessment = assess_student_mcq_reply(correct_label="B", student_reply="Answer: A")
        self.assertEqual(assessment.grade, "wrong")

    def test_update_tutor_state_escalates_after_wrong_reply(self) -> None:
        from medical_tutor.contracts import AgentState
        from medical_tutor.tutoring import update_tutor_state_from_assessment

        state = AgentState(
            messages=[{"role": "user", "content": "Case question here"}],
            run_id="run_test",
            student_profile="novice",
            student_attempt_count=1,
            student_reply_grade="wrong",
            student_reply_misconception="missed key finding",
            tutor_hint_level=1,
            tutor_consecutive_wrong=0,
        )
        update_tutor_state_from_assessment(state)

        self.assertEqual(state.tutor_hint_level, 2)
        self.assertEqual(state.tutor_consecutive_wrong, 1)
        self.assertEqual(state.tutor_last_grade, "wrong")
        self.assertEqual(state.tutor_last_misconception, "missed key finding")

    def test_update_tutor_state_resets_after_correct_reply(self) -> None:
        from medical_tutor.contracts import AgentState
        from medical_tutor.tutoring import update_tutor_state_from_assessment

        state = AgentState(
            messages=[{"role": "user", "content": "Case question here"}],
            run_id="run_test",
            student_profile="novice",
            student_attempt_count=2,
            student_reply_grade="correct",
            student_reply_misconception="stale misconception",
            tutor_hint_level=3,
            tutor_consecutive_wrong=2,
            tutor_last_misconception="stale misconception",
        )
        update_tutor_state_from_assessment(state)

        self.assertEqual(state.tutor_hint_level, 1)
        self.assertEqual(state.tutor_consecutive_wrong, 0)
        self.assertEqual(state.tutor_last_grade, "correct")
        self.assertIsNone(state.tutor_last_misconception)

    def test_assessment_gate_forces_hint_after_wrong_reply(self) -> None:
        from medical_tutor.contracts import AgentState
        from medical_tutor.tutoring import TutorAction, apply_assessment_gate

        state = AgentState(
            messages=[{"role": "user", "content": "Case question here"}],
            run_id="run_test",
            student_profile="novice",
            student_attempt_count=1,
            student_reply_grade="wrong",
            tutor_hint_level=2,
        )
        action = TutorAction(type="ASK_PROBE", arguments={"content": "Probe."})
        gated = apply_assessment_gate(state, action)

        self.assertEqual(gated.type, "ASK_PROBE")
        self.assertEqual(gated.arguments.get("content"), "Probe.")

    def test_assessment_gate_blocks_reveal_after_wrong_reply(self) -> None:
        from medical_tutor.contracts import AgentState
        from medical_tutor.tutoring import TutorAction, apply_assessment_gate

        state = AgentState(
            messages=[{"role": "user", "content": "Case question here"}],
            run_id="run_test",
            student_profile="novice",
            student_attempt_count=1,
            student_reply_grade="wrong",
            tutor_hint_level=2,
        )
        action = TutorAction(type="REVEAL_ANSWER", arguments={})
        gated = apply_assessment_gate(state, action)

        self.assertEqual(gated.type, "HINT")
        self.assertEqual(gated.arguments.get("level"), 2)

    def test_assessment_gate_allows_quiz_after_wrong_reply(self) -> None:
        from medical_tutor.contracts import AgentState
        from medical_tutor.tutoring import TutorAction, apply_assessment_gate

        state = AgentState(
            messages=[{"role": "user", "content": "Case question here"}],
            run_id="run_test",
            student_profile="novice",
            student_attempt_count=1,
            student_reply_grade="wrong",
            tutor_hint_level=2,
        )
        action = TutorAction(type="QUIZ", arguments={"content": "Quiz."})
        gated = apply_assessment_gate(state, action)

        self.assertEqual(gated.type, "QUIZ")
        self.assertEqual(gated.arguments.get("content"), "Quiz.")


if __name__ == "__main__":
    unittest.main()
