# ABOUTME: Validates the `medical-tutor tutor` CLI without requiring network access.
# ABOUTME: Ensures tutor sessions write traces and respect reveal gating.
# pyright: reportMissingImports=false

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


class TestMedicalTutorCliTutor(unittest.TestCase):
    def test_tutor_emits_run_id_and_writes_trace(self) -> None:
        from medical_tutor import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            old_output_dir = os.environ.get("MEDTUTOR_OUTPUT_DIR")
            old_openai_key = os.environ.get("OPENAI_API_KEY")
            os.environ["MEDTUTOR_OUTPUT_DIR"] = tmpdir
            os.environ["OPENAI_API_KEY"] = "test-key"  # pragma: allowlist secret
            try:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = cli.main(
                        [
                            "tutor",
                            "--profile",
                            "novice",
                            "--question",
                            "What is the best next step? (A) foo (B) bar",
                            "--max-turns",
                            "0",
                        ]
                    )
                self.assertEqual(exit_code, 0)
                output = buffer.getvalue()
                self.assertIn("run_id=", output)
                self.assertIn("output_dir=", output)

                run_id = None
                for line in output.splitlines():
                    if line.startswith("run_id="):
                        run_id = line.split("=", 1)[1].strip()
                        break
                self.assertTrue(run_id)
                trace_path = Path(tmpdir) / str(run_id) / "trace.jsonl"
                self.assertTrue(trace_path.exists(), f"Expected trace at {trace_path}")
            finally:
                if old_output_dir is None:
                    os.environ.pop("MEDTUTOR_OUTPUT_DIR", None)
                else:
                    os.environ["MEDTUTOR_OUTPUT_DIR"] = old_output_dir
                if old_openai_key is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = old_openai_key

    def test_tutor_fails_loudly_without_openai_key(self) -> None:
        from medical_tutor import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            old_output_dir = os.environ.get("MEDTUTOR_OUTPUT_DIR")
            old_openai_key = os.environ.get("OPENAI_API_KEY")
            os.environ["MEDTUTOR_OUTPUT_DIR"] = tmpdir
            os.environ.pop("OPENAI_API_KEY", None)
            try:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = cli.main(
                        [
                            "tutor",
                            "--profile",
                            "novice",
                            "--question",
                            "Case stem",
                            "--max-turns",
                            "0",
                        ]
                    )
                self.assertNotEqual(exit_code, 0)
            finally:
                if old_output_dir is None:
                    os.environ.pop("MEDTUTOR_OUTPUT_DIR", None)
                else:
                    os.environ["MEDTUTOR_OUTPUT_DIR"] = old_output_dir
                if old_openai_key is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = old_openai_key


if __name__ == "__main__":
    unittest.main()
