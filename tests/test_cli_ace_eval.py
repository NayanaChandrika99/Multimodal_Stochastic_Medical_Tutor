# ABOUTME: Validates `medical-tutor ace` and `medical-tutor eval` CLI entrypoints.
# ABOUTME: Ensures both commands write expected artifacts without network access.

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


class TestMedicalTutorCliAceEval(unittest.TestCase):
    def test_cli_eval_writes_eval_summary(self) -> None:
        from medical_tutor import cli

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

                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = cli.main(["eval", "--dataset", str(dataset_path)])
                self.assertEqual(exit_code, 0)

                output = buffer.getvalue()
                self.assertIn("eval_run_id=", output)

                run_id = None
                for line in output.splitlines():
                    if line.startswith("eval_run_id="):
                        run_id = line.split("=", 1)[1].strip()
                        break
                self.assertTrue(run_id)

                summary_path = Path(tmpdir) / str(run_id) / "eval_summary.json"
                self.assertTrue(summary_path.exists(), f"Expected {summary_path}")
            finally:
                if old_output_dir is None:
                    os.environ.pop("MEDTUTOR_OUTPUT_DIR", None)
                else:
                    os.environ["MEDTUTOR_OUTPUT_DIR"] = old_output_dir

    def test_cli_ace_writes_playbook_and_per_sample(self) -> None:
        from medical_tutor import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            old_output_dir = os.environ.get("MEDTUTOR_OUTPUT_DIR")
            os.environ["MEDTUTOR_OUTPUT_DIR"] = tmpdir
            os.environ.pop("OPENAI_API_KEY", None)
            try:
                dataset_path = Path(tmpdir) / "ace_dataset.jsonl"
                dataset_path.write_text(
                    json.dumps(
                        {
                            "id": "s1",
                            "question": "What is shown?",
                            "image_path": None,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = cli.main(["ace", "--dataset", str(dataset_path)])
                self.assertEqual(exit_code, 0)

                output = buffer.getvalue()
                self.assertIn("ace_run_id=", output)

                run_id = None
                for line in output.splitlines():
                    if line.startswith("ace_run_id="):
                        run_id = line.split("=", 1)[1].strip()
                        break
                self.assertTrue(run_id)

                run_dir = Path(tmpdir) / str(run_id)
                self.assertTrue((run_dir / "per_sample.jsonl").exists())
                self.assertTrue((run_dir / "playbook.txt").exists())
            finally:
                if old_output_dir is None:
                    os.environ.pop("MEDTUTOR_OUTPUT_DIR", None)
                else:
                    os.environ["MEDTUTOR_OUTPUT_DIR"] = old_output_dir


if __name__ == "__main__":
    unittest.main()
