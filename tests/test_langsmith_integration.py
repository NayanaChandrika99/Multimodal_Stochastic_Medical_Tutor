# ABOUTME: Tests LangSmith configuration and integration behavior for Medical_Tutor.
# ABOUTME: Ensures LangSmith enablement is graceful when API keys are missing (no network).

from __future__ import annotations

import os
import tempfile
import unittest


class TestLangSmithIntegration(unittest.TestCase):
    def test_langsmith_enabled_without_api_key_is_graceful(self) -> None:
        from medical_tutor.config import load_config
        from medical_tutor.runtime.runner import GraphRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            old_output_dir = os.environ.get("MEDTUTOR_OUTPUT_DIR")
            old_enabled = os.environ.get("MEDTUTOR_LANGSMITH_ENABLED")
            old_api_key = os.environ.get("LANGSMITH_API_KEY")

            os.environ["MEDTUTOR_OUTPUT_DIR"] = tmpdir
            os.environ["MEDTUTOR_LANGSMITH_ENABLED"] = "1"
            os.environ.pop("LANGSMITH_API_KEY", None)
            os.environ.pop("MEDTUTOR_LANGSMITH_API_KEY", None)
            try:
                config = load_config()
                self.assertTrue(config.langsmith_enabled)

                runner = GraphRunner(config=config)
                state = runner.run(question="What is shown?", image_path=None)
                self.assertTrue(state.run_id)
                self.assertTrue(
                    any(err.code == "langsmith" for err in state.errors),
                    "Expected a langsmith-related warning when enabled but key missing.",
                )
            finally:
                if old_output_dir is None:
                    os.environ.pop("MEDTUTOR_OUTPUT_DIR", None)
                else:
                    os.environ["MEDTUTOR_OUTPUT_DIR"] = old_output_dir
                if old_enabled is None:
                    os.environ.pop("MEDTUTOR_LANGSMITH_ENABLED", None)
                else:
                    os.environ["MEDTUTOR_LANGSMITH_ENABLED"] = old_enabled
                if old_api_key is None:
                    os.environ.pop("LANGSMITH_API_KEY", None)
                else:
                    os.environ["LANGSMITH_API_KEY"] = old_api_key


if __name__ == "__main__":
    unittest.main()
