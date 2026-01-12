# ABOUTME: Tests prompt formatting and dataset splitting for action controller training.
# ABOUTME: Ensures deterministic splits and valid target serialization.

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TestActionControllerTraining(unittest.TestCase):
    def test_format_action_prompt_includes_fields(self) -> None:
        from medical_tutor.ops.train_action_controller import format_action_prompt

        example = {
            "case_id": "MM-1",
            "student_profile": "novice",
            "kc_tags": {
                "body_system": "Neuro",
                "medical_task": "Diagnosis",
                "question_type": "Reasoning",
            },
            "question": "Test question?",
            "options": {"A": "One", "B": "Two"},
            "history": [{"role": "student", "content": "Answer: A"}],
            "tool_summaries": [{"tool": "zoom", "summary": "Zoomed in"}],
            "student_reply_assessment": {"grade": "wrong", "misconception": "Missed key"},
        }

        prompt = format_action_prompt(example)

        self.assertIn("Student profile: novice", prompt)
        self.assertIn("body_system=Neuro", prompt)
        self.assertIn("Question: Test question?", prompt)
        self.assertIn("(A) One", prompt)
        self.assertIn("History:", prompt)
        self.assertIn("zoom", prompt)
        self.assertIn("grade=wrong", prompt)

    def test_build_training_examples_serializes_targets(self) -> None:
        from medical_tutor.ops.train_action_controller import build_training_examples

        examples = [
            {
                "case_id": "MM-1",
                "student_profile": "novice",
                "kc_tags": {
                    "body_system": "Neuro",
                    "medical_task": "Diagnosis",
                    "question_type": "Reasoning",
                },
                "question": "Test question?",
                "options": {"A": "One", "B": "Two"},
                "history": [],
                "tool_summaries": [],
                "student_reply_assessment": {"grade": "wrong", "misconception": None},
                "target_action": {"type": "HINT", "arguments": {"content": "Focus", "level": 1}},
            }
        ]

        training = build_training_examples(examples)

        self.assertEqual(len(training), 1)
        target = json.loads(training[0].target_json)
        self.assertEqual(target["type"], "HINT")

    def test_load_action_dataset_hydrates_case_context(self) -> None:
        from medical_tutor.ops.train_action_controller import load_action_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "action_sft.jsonl"
            cases_path = Path(tmpdir) / "cases.jsonl"

            dataset_path.write_text(
                json.dumps(
                    {
                        "case_id": "MM-1",
                        "student_profile": "novice",
                        "kc_tags": {},
                        "image_ids": [],
                        "turn_index": 0,
                        "history": [],
                        "tool_summaries": [],
                        "student_reply_assessment": {"grade": "wrong", "misconception": None},
                        "target_action": {"type": "ASK_PROBE", "arguments": {"content": "?"}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cases_path.write_text(
                json.dumps(
                    {
                        "case_id": "MM-1",
                        "question": "Case stem?",
                        "options": {"A": "One", "B": "Two"},
                        "images": ["MM-1.png"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            examples = load_action_dataset(dataset_path, cases_path=cases_path)

            self.assertEqual(examples[0]["question"], "Case stem?")
            self.assertIn("A", examples[0]["options"])
            self.assertEqual(examples[0]["image_ids"], ["MM-1.png"])

    def test_split_examples_is_deterministic(self) -> None:
        from medical_tutor.ops.train_action_controller import split_examples

        examples = [{"id": idx} for idx in range(10)]
        split_a = split_examples(examples, seed=7, val_ratio=0.2)
        split_b = split_examples(examples, seed=7, val_ratio=0.2)

        self.assertEqual(split_a, split_b)


if __name__ == "__main__":
    unittest.main()
