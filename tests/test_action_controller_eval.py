# ABOUTME: Tests evaluation helpers for action controller outputs.
# ABOUTME: Verifies JSON parsing and metric calculations.

from __future__ import annotations

import unittest
from typing import Any


class TestActionControllerEval(unittest.TestCase):
    def test_parse_action_output_handles_embedded_json(self) -> None:
        from medical_tutor.ops.eval_action_controller import parse_action_output

        raw = 'Model output: {"type": "HINT", "arguments": {"content": "Focus"}}'
        parsed = parse_action_output(raw)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["type"], "HINT")

    def test_compute_metrics_counts_validity(self) -> None:
        from medical_tutor.ops.eval_action_controller import compute_metrics

        examples = [
            {"turn_index": 0, "image_ids": [], "student_profile": "novice"},
            {"turn_index": 0, "image_ids": ["img.png"], "student_profile": "expert"},
        ]
        predictions: list[dict[str, Any] | None] = [
            {"type": "ASK_PROBE", "arguments": {"content": "Q?"}},
            None,
        ]

        metrics = compute_metrics(predictions, examples)
        self.assertEqual(metrics["json_validity"], 0.5)

    def test_compute_metrics_profiles_action_distribution(self) -> None:
        from medical_tutor.ops.eval_action_controller import compute_metrics

        examples = [
            {"turn_index": 0, "image_ids": [], "student_profile": "novice"},
            {"turn_index": 1, "image_ids": [], "student_profile": "novice"},
            {"turn_index": 0, "image_ids": [], "student_profile": "expert"},
        ]
        predictions: list[dict[str, Any] | None] = [
            {"type": "HINT", "arguments": {"content": "hint"}},
            {"type": "ASK_PROBE", "arguments": {"content": "probe"}},
            {"type": "HINT", "arguments": {"content": "hint"}},
        ]

        metrics = compute_metrics(predictions, examples)
        distributions = metrics["action_distribution_by_profile"]

        self.assertAlmostEqual(distributions["novice"]["HINT"], 0.5)
        self.assertAlmostEqual(distributions["novice"]["ASK_PROBE"], 0.5)
        self.assertAlmostEqual(distributions["expert"]["HINT"], 1.0)


if __name__ == "__main__":
    unittest.main()
