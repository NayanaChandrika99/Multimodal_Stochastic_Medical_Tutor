# ABOUTME: Tests ToolCall argument normalization for stringified literals.
# ABOUTME: Ensures structured arguments are parsed into lists when needed.

from __future__ import annotations

import unittest

from medical_tutor.contracts import ToolCall


class TestToolCallNormalization(unittest.TestCase):
    def test_parses_bbox_2d_string(self) -> None:
        call = ToolCall(
            id="call-1",
            name="zoom",
            arguments={"bbox_2d": "[0.1, 0.2, 0.3, 0.4]", "padding": 0.1},
        )

        self.assertEqual(call.arguments["bbox_2d"], [0.1, 0.2, 0.3, 0.4])

    def test_parses_point_string(self) -> None:
        call = ToolCall(
            id="call-2",
            name="segment",
            arguments={"point": "(10, 20)"},
        )

        self.assertEqual(call.arguments["point"], [10, 20])


if __name__ == "__main__":
    unittest.main()
