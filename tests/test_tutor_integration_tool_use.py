# ABOUTME: Runs a minimal end-to-end tutor flow against OpenAI to validate wiring.
# ABOUTME: Confirms tutor actions, trace creation, and tool usage when a playbook is set.

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path


class TestTutorIntegrationToolUse(unittest.TestCase):
    def _workspace_root(self) -> Path:
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "references").exists():
                return parent
        raise FileNotFoundError("Could not locate workspace root containing references/ directory.")

    def _first_existing(self, candidates: list[Path]) -> Path | None:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def test_tutor_uses_tool_with_playbook(self) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY is required for integration test.")

        from medical_tutor.config import load_config
        from medical_tutor.medxpert import build_case_inputs, resolve_medxpert_case
        from medical_tutor.runtime.runner import GraphRunner
        from medical_tutor.tutoring import assess_student_mcq_reply

        repo_root = self._workspace_root()
        case_path = (
            repo_root
            / "references"
            / "MedTutor-R1"
            / "code"
            / "Patient_simulate"
            / "MedXpert_patient_script_MM_dev.json"
        )
        image_root = self._first_existing(
            [
                repo_root / "cache" / "medxpert_images_mm_dev",
                repo_root
                / "med-visual-tutor"
                / "Medical_Tutor"
                / ".cache"
                / "medxpert_images_mm_dev",
                repo_root / "Medical_Tutor" / ".cache" / "medxpert_images_mm_dev",
            ]
        )
        playbook_path = self._first_existing(
            [
                repo_root / "Medical_Tutor" / "playbooks" / "phase1_tool_use.txt",
                repo_root
                / "med-visual-tutor"
                / "Medical_Tutor"
                / "playbooks"
                / "phase1_tool_use.txt",
            ]
        )

        if not case_path.exists():
            self.skipTest(f"Missing case file: {case_path}")
        if image_root is None:
            self.skipTest("Missing image root directory for MM_dev images.")
        if playbook_path is None:
            self.skipTest("Missing phase1_tool_use.txt playbook file.")

        case = resolve_medxpert_case(case_file=case_path, case_id="MM-2000")
        case_prompt, image_path = build_case_inputs(case, image_root=image_root)
        if not image_path or not Path(image_path).exists():
            self.skipTest(f"Missing image for case: {image_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            old_output_dir = os.environ.get("MEDTUTOR_OUTPUT_DIR")
            old_playbook = os.environ.get("MEDTUTOR_PLAYBOOK_PATH")
            os.environ["MEDTUTOR_OUTPUT_DIR"] = tmpdir
            os.environ["MEDTUTOR_PLAYBOOK_PATH"] = str(playbook_path)
            try:
                runner = GraphRunner(config=load_config())
                run_id = "tutor_integration"
                result1 = runner.run(
                    question=case_prompt,
                    image_path=image_path,
                    conversation_history=[],
                    run_id=run_id,
                    student_profile="novice",
                    student_attempt_count=0,
                    case_prompt=case_prompt,
                )
                self.assertIsNotNone(result1.tutor_action)
                assert result1.tutor_action is not None
                self.assertNotEqual(result1.tutor_action.get("type"), "REVEAL_ANSWER")

                trace_path = Path(tmpdir) / run_id / "trace.jsonl"
                self.assertTrue(trace_path.exists(), f"Expected trace at {trace_path}")

                conversation = list(result1.messages)
                reply = "A"
                assessment = assess_student_mcq_reply(correct_label=case.label, student_reply=reply)
                result2 = runner.run(
                    question=reply,
                    image_path=image_path,
                    conversation_history=conversation,
                    run_id=run_id,
                    student_profile="novice",
                    student_attempt_count=0,
                    case_prompt=case_prompt,
                    student_reply_text=reply,
                    student_reply_grade=assessment.grade,
                    student_reply_misconception=assessment.misconception,
                )
                self.assertIsNotNone(result2.tutor_action)
                assert result2.tutor_action is not None
                self.assertNotEqual(result2.tutor_action.get("type"), "REVEAL_ANSWER")

                tool_calls = list(result1.tool_calls) + list(result2.tool_calls)
                tool_results = list(result1.tool_results) + list(result2.tool_results)
                self.assertTrue(
                    tool_calls or tool_results,
                    "Expected at least one tool call or result with playbook enabled.",
                )
            finally:
                if old_output_dir is None:
                    os.environ.pop("MEDTUTOR_OUTPUT_DIR", None)
                else:
                    os.environ["MEDTUTOR_OUTPUT_DIR"] = old_output_dir
                if old_playbook is None:
                    os.environ.pop("MEDTUTOR_PLAYBOOK_PATH", None)
                else:
                    os.environ["MEDTUTOR_PLAYBOOK_PATH"] = old_playbook


if __name__ == "__main__":
    unittest.main()
