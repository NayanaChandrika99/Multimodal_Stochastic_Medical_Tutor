# ABOUTME: Validates the standalone Medical_Tutor CLI behaviors using unittest.
# ABOUTME: Ensures `medical-tutor run` produces a trace directory deterministically.

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


class TestMedicalTutorCliRun(unittest.TestCase):
    def test_run_emits_run_id_and_writes_trace(self) -> None:
        from medical_tutor import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            old_output_dir = os.environ.get("MEDTUTOR_OUTPUT_DIR")
            old_openai_key = os.environ.get("OPENAI_API_KEY")
            old_playbook_path = os.environ.get("MEDTUTOR_PLAYBOOK_PATH")
            os.environ["MEDTUTOR_OUTPUT_DIR"] = tmpdir
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("MEDTUTOR_PLAYBOOK_PATH", None)
            try:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = cli.main(
                        [
                            "run",
                            "--question",
                            "What is shown?",
                        ]
                    )
                self.assertEqual(exit_code, 0)

                output = buffer.getvalue()
                self.assertIn("run_id=", output)

                run_id = None
                for line in output.splitlines():
                    if line.startswith("run_id="):
                        run_id = line.split("=", 1)[1].strip()
                        break
                self.assertTrue(run_id)

                trace_path = Path(tmpdir) / str(run_id) / "trace.jsonl"
                self.assertTrue(trace_path.exists(), f"Expected trace at {trace_path}")

                run_config_path = Path(tmpdir) / str(run_id) / "run_config.json"
                self.assertTrue(
                    run_config_path.exists(),
                    f"Expected run metadata at {run_config_path}",
                )
            finally:
                if old_output_dir is None:
                    os.environ.pop("MEDTUTOR_OUTPUT_DIR", None)
                else:
                    os.environ["MEDTUTOR_OUTPUT_DIR"] = old_output_dir
                if old_openai_key is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = old_openai_key
                if old_playbook_path is None:
                    os.environ.pop("MEDTUTOR_PLAYBOOK_PATH", None)
                else:
                    os.environ["MEDTUTOR_PLAYBOOK_PATH"] = old_playbook_path

    def test_run_includes_playbook_in_trace_when_configured(self) -> None:
        from medical_tutor import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            old_output_dir = os.environ.get("MEDTUTOR_OUTPUT_DIR")
            old_openai_key = os.environ.get("OPENAI_API_KEY")
            old_playbook_path = os.environ.get("MEDTUTOR_PLAYBOOK_PATH")
            os.environ["MEDTUTOR_OUTPUT_DIR"] = tmpdir
            os.environ.pop("OPENAI_API_KEY", None)
            try:
                playbook_path = Path(tmpdir) / "active_playbook.txt"
                playbook_marker = "PLAYBOOK_MARKER_ACE_123"
                playbook_path.write_text(
                    f"## OTHERS\n[oth-00001] helpful=0 harmful=0 :: {playbook_marker}\n",
                    encoding="utf-8",
                )
                os.environ["MEDTUTOR_PLAYBOOK_PATH"] = str(playbook_path)

                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = cli.main(["run", "--question", "What is shown?"])
                self.assertEqual(exit_code, 0)

                output = buffer.getvalue()
                run_id = None
                for line in output.splitlines():
                    if line.startswith("run_id="):
                        run_id = line.split("=", 1)[1].strip()
                        break
                self.assertTrue(run_id)

                trace_path = Path(tmpdir) / str(run_id) / "trace.jsonl"
                trace_text = trace_path.read_text(encoding="utf-8")
                self.assertIn(playbook_marker, trace_text)
            finally:
                if old_output_dir is None:
                    os.environ.pop("MEDTUTOR_OUTPUT_DIR", None)
                else:
                    os.environ["MEDTUTOR_OUTPUT_DIR"] = old_output_dir
                if old_openai_key is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = old_openai_key
                if old_playbook_path is None:
                    os.environ.pop("MEDTUTOR_PLAYBOOK_PATH", None)
                else:
                    os.environ["MEDTUTOR_PLAYBOOK_PATH"] = old_playbook_path


if __name__ == "__main__":
    unittest.main()
