# ABOUTME: Tests the standalone evaluation harness for Medical_Tutor using unittest.
# ABOUTME: Validates that eval runs produce summary artifacts deterministically.

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path


class TestEvalRunner(unittest.TestCase):
    def test_eval_run_writes_summary(self) -> None:
        from medical_tutor.ops.eval import EvalRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            old_output_dir = os.environ.get("MEDTUTOR_OUTPUT_DIR")
            os.environ["MEDTUTOR_OUTPUT_DIR"] = tmpdir
            os.environ.pop("OPENAI_API_KEY", None)
            try:
                dataset_path = Path(tmpdir) / "dataset.jsonl"
                dataset_path.write_text(
                    json.dumps(
                        {
                            "id": "ex1",
                            "question": "What is shown?",
                            "image_path": None,
                            "answer": "Insufficient evidence to answer confidently.",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

                runner = EvalRunner()
                result = runner.run(dataset_path=str(dataset_path))

                run_dir = Path(result["output_dir"])
                summary_path = run_dir / "eval_summary.json"
                self.assertTrue(summary_path.exists(), f"Missing {summary_path}")

                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                self.assertIn("accuracy", summary)
            finally:
                if old_output_dir is None:
                    os.environ.pop("MEDTUTOR_OUTPUT_DIR", None)
                else:
                    os.environ["MEDTUTOR_OUTPUT_DIR"] = old_output_dir


if __name__ == "__main__":
    unittest.main()
