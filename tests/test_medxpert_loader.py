# ABOUTME: Tests MedXpert case loading and question formatting utilities.
# ABOUTME: Ensures MM_dev cases load with options and image metadata for tutoring.

from __future__ import annotations

import unittest
from pathlib import Path


class TestMedXpertLoader(unittest.TestCase):
    def _workspace_root(self) -> Path:
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "references").exists():
                return parent
        raise FileNotFoundError("Could not locate workspace root containing references/ directory.")

    def test_loads_mm_dev_case_and_formats_question(self) -> None:
        from medical_tutor.medxpert import (
            build_case_inputs,
            load_medxpert_cases,
            resolve_case_image_path,
        )

        repo_root = self._workspace_root()
        source_path = (
            repo_root
            / "references"
            / "MedTutor-R1"
            / "code"
            / "Patient_simulate"
            / "MedXpert_patient_script_MM_dev.json"
        )
        if not source_path.exists():
            self.skipTest(f"Missing MedTutor-R1 reference file: {source_path}")
        cases = load_medxpert_cases(source_path)
        self.assertIn("MM-2000", cases)

        case = cases["MM-2000"]
        formatted = case.format_question()
        self.assertIn("Answer Choices:", formatted)
        self.assertIn("(A)", formatted)
        self.assertIn(case.options.get("A", ""), formatted)
        self.assertTrue(case.images, "Expected MM_dev case to include images")

        image_path = resolve_case_image_path(case, image_root=Path("/tmp/images"))
        self.assertTrue(str(image_path).endswith(case.images[0]))

        question_text, image_path_built = build_case_inputs(case, image_root=Path("/tmp/images"))
        self.assertIn("Answer Choices", question_text)
        self.assertTrue(str(image_path_built).endswith(case.images[0]))


if __name__ == "__main__":
    unittest.main()
