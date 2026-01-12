# ABOUTME: Tests the standalone ACE runner for Medical_Tutor using unittest.
# ABOUTME: Validates that an ACE run writes per-sample logs and a playbook file.

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path


class TestAceRunner(unittest.TestCase):
    def test_ace_run_writes_artifacts(self) -> None:
        from medical_tutor.ace.runner import AceRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            old_output_dir = os.environ.get("MEDTUTOR_OUTPUT_DIR")
            os.environ["MEDTUTOR_OUTPUT_DIR"] = tmpdir
            os.environ.pop("OPENAI_API_KEY", None)
            try:
                runner = AceRunner()
                result = runner.run(
                    samples=[{"id": "s1", "question": "What is shown?", "image_path": None}],
                    playbook_path=None,
                )

                run_id = result["run_id"]
                run_dir = Path(tmpdir) / run_id
                per_sample = run_dir / "per_sample.jsonl"
                playbook = run_dir / "playbook.txt"

                self.assertTrue(per_sample.exists(), f"Missing {per_sample}")
                self.assertTrue(playbook.exists(), f"Missing {playbook}")

                lines = per_sample.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), 1)
                payload = json.loads(lines[0])
                self.assertEqual(payload.get("sample_id"), "s1")
                self.assertIn("answer", payload)
            finally:
                if old_output_dir is None:
                    os.environ.pop("MEDTUTOR_OUTPUT_DIR", None)
                else:
                    os.environ["MEDTUTOR_OUTPUT_DIR"] = old_output_dir


if __name__ == "__main__":
    unittest.main()
