# ABOUTME: Tests the Socratic dataset builder for deterministic case/episode outputs.
# ABOUTME: Validates JSONL outputs and manifest structure.

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TestSocraticDatasetBuilder(unittest.TestCase):
    def _workspace_root(self) -> Path:
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "references").exists():
                return parent
        raise FileNotFoundError("Could not locate workspace root containing references/ directory.")

    def test_build_case_index_includes_text_and_mm(self) -> None:
        from medical_tutor.ops.socratic_dataset import build_case_index

        repo_root = self._workspace_root()
        text_path = (
            repo_root
            / "references"
            / "MedTutor-R1"
            / "code"
            / "Patient_simulate"
            / "MedXpert_patient_script_Text_for_test.json"
        )
        mm_path = (
            repo_root
            / "references"
            / "MedTutor-R1"
            / "code"
            / "Patient_simulate"
            / "MedXpert_patient_script_MM_for_test.json"
        )
        if not text_path.exists() or not mm_path.exists():
            self.skipTest("Missing MedTutor-R1 reference JSONs required for this test.")

        cases = build_case_index(text_path=text_path, mm_path=mm_path)

        modalities = {case["modality"] for case in cases}
        self.assertIn("text", modalities)
        self.assertIn("mm", modalities)
        self.assertTrue(any(case["socratic_steps"] for case in cases))

    def test_build_action_examples_is_deterministic(self) -> None:
        from medical_tutor.ops.socratic_dataset import build_action_examples, build_case_index

        repo_root = self._workspace_root()
        text_path = (
            repo_root
            / "references"
            / "MedTutor-R1"
            / "code"
            / "Patient_simulate"
            / "MedXpert_patient_script_Text_for_test.json"
        )
        mm_path = (
            repo_root
            / "references"
            / "MedTutor-R1"
            / "code"
            / "Patient_simulate"
            / "MedXpert_patient_script_MM_for_test.json"
        )
        if not text_path.exists() or not mm_path.exists():
            self.skipTest("Missing MedTutor-R1 reference JSONs required for this test.")

        cases = build_case_index(text_path=text_path, mm_path=mm_path)
        subset = cases[:3]

        examples_a = build_action_examples(subset, seed=1234)
        examples_b = build_action_examples(subset, seed=1234)

        self.assertEqual(examples_a, examples_b)
        self.assertTrue(examples_a)

    def test_write_dataset_outputs_manifest(self) -> None:
        from medical_tutor.ops.socratic_dataset import write_dataset

        repo_root = self._workspace_root()
        text_path = (
            repo_root
            / "references"
            / "MedTutor-R1"
            / "code"
            / "Patient_simulate"
            / "MedXpert_patient_script_Text_for_test.json"
        )
        mm_path = (
            repo_root
            / "references"
            / "MedTutor-R1"
            / "code"
            / "Patient_simulate"
            / "MedXpert_patient_script_MM_for_test.json"
        )
        if not text_path.exists() or not mm_path.exists():
            self.skipTest("Missing MedTutor-R1 reference JSONs required for this test.")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "dataset"
            result = write_dataset(
                text_path=text_path,
                mm_path=mm_path,
                output_dir=out_dir,
                seed=99,
            )

            cases_file = out_dir / "cases.jsonl"
            actions_file = out_dir / "action_sft.jsonl"
            manifest_file = out_dir / "manifest.json"

            self.assertTrue(cases_file.exists())
            self.assertTrue(actions_file.exists())
            self.assertTrue(manifest_file.exists())

            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_version"], result["dataset_version"])
            self.assertGreater(manifest["num_examples"], 0)

    def test_write_dataset_falls_back_without_openai_key(self) -> None:
        import os

        from medical_tutor.ops.socratic_dataset import write_dataset

        repo_root = self._workspace_root()
        text_path = (
            repo_root
            / "references"
            / "MedTutor-R1"
            / "code"
            / "Patient_simulate"
            / "MedXpert_patient_script_Text_for_test.json"
        )
        mm_path = (
            repo_root
            / "references"
            / "MedTutor-R1"
            / "code"
            / "Patient_simulate"
            / "MedXpert_patient_script_MM_for_test.json"
        )
        if not text_path.exists() or not mm_path.exists():
            self.skipTest("Missing MedTutor-R1 reference JSONs required for this test.")

        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_dir = Path(tmpdir) / "dataset"
                result = write_dataset(
                    text_path=text_path,
                    mm_path=mm_path,
                    output_dir=out_dir,
                    seed=42,
                    use_controller=True,
                )
                manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["action_labeling"], "heuristic_fallback")
                self.assertEqual(result["labeling_mode"], "heuristic_fallback")
                logs_path = out_dir / "logs" / "labeling.jsonl"
                self.assertTrue(logs_path.exists())
                lines = [
                    json.loads(line)
                    for line in logs_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertTrue(lines)
                self.assertEqual(lines[0].get("event"), "controller_unavailable")
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key


if __name__ == "__main__":
    unittest.main()
