# ABOUTME: Verifies tutor policy response parsing handles fenced JSON output.
# ABOUTME: Prevents controller JSON formatting from breaking Socratic routing.

from __future__ import annotations

import unittest


class TestTutorPolicyParsing(unittest.TestCase):
    def test_parse_fenced_json(self) -> None:
        from medical_tutor.runtime import tutor_policy

        fenced = "```json\n" '{ "type": "ASK_PROBE", "arguments": { "content": "Why?" } }\n' "```"

        payload = tutor_policy._parse_tutor_action_response(fenced)

        self.assertEqual(payload["type"], "ASK_PROBE")
        self.assertEqual(payload["arguments"]["content"], "Why?")


if __name__ == "__main__":
    unittest.main()
