# ABOUTME: Validates the full ACE loop (generator→reflector→curator→playbook ops).
# ABOUTME: Uses deterministic LLM stubs to avoid network and heavy model loads.

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any


def _fake_call_llm(
    _client: object,
    _api_provider: str,
    _model: str,
    prompt: str,
    *,
    role: str,
    call_id: str,
    max_tokens: int,
    log_dir: str | None,
    use_json_mode: bool,
) -> tuple[str, dict[str, Any]]:
    del _client, _api_provider, _model, max_tokens, log_dir, use_json_mode

    if role == "reflector":
        payload = {
            "error_identification": "Incorrect answer",
            "root_cause_analysis": "Missing relevant reasoning",
            "correct_approach": "Use retrieval when needed",
            "key_insights": ["Prefer retrieve before web_browser when unsure."],
            "failure_modes": ["insufficient_visual_evidence"],
            "bullet_tags": [{"id": "ctx-00001", "tag": "helpful"}],
        }
        return json.dumps(payload), {"role": role, "call_id": call_id, "prompt": prompt}

    if role == "curator":
        payload = {
            "reasoning": "Add a missing heuristic.",
            "operations": [
                {
                    "type": "ADD",
                    "section": "problem_solving_heuristics",
                    "content": "When unsure, use retrieve before web_browser to stay grounded.",
                }
            ],
        }
        return json.dumps(payload), {"role": role, "call_id": call_id, "prompt": prompt}

    return json.dumps({}), {"role": role, "call_id": call_id, "prompt": prompt}


class TestAceRunnerFullLoop(unittest.TestCase):
    def test_full_loop_writes_curator_trace_and_final_playbook(self) -> None:
        from medical_tutor.ace.runner import AceRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            old_output_dir = os.environ.get("MEDTUTOR_OUTPUT_DIR")
            os.environ["MEDTUTOR_OUTPUT_DIR"] = tmpdir
            os.environ.pop("OPENAI_API_KEY", None)
            try:
                playbook_path = Path(tmpdir) / "initial_playbook.txt"
                playbook_path.write_text(
                    "\n".join(
                        [
                            "## PROBLEM SOLVING HEURISTICS",
                            "[ctx-00001] helpful=0 harmful=0 :: Use retrieve before web_browser when unsure.",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )

                runner = AceRunner(call_llm=_fake_call_llm)
                result = runner.run(
                    samples=[
                        {
                            "id": "s1",
                            "question": "When unsure, should I retrieve before web browser?",
                            "image_path": None,
                            "answer": "Some other answer",
                        }
                    ],
                    playbook_path=str(playbook_path),
                )

                run_dir = Path(tmpdir) / str(result["run_id"])
                self.assertTrue((run_dir / "per_sample.jsonl").exists())
                self.assertTrue((run_dir / "reflector_trace.jsonl").exists())
                self.assertTrue((run_dir / "curator_trace.jsonl").exists())
                self.assertTrue((run_dir / "final_playbook.txt").exists())

                final_playbook = (run_dir / "final_playbook.txt").read_text(encoding="utf-8")
                self.assertIn("helpful=1", final_playbook)
                self.assertIn("When unsure, use retrieve before web_browser", final_playbook)
            finally:
                if old_output_dir is None:
                    os.environ.pop("MEDTUTOR_OUTPUT_DIR", None)
                else:
                    os.environ["MEDTUTOR_OUTPUT_DIR"] = old_output_dir


if __name__ == "__main__":
    unittest.main()
