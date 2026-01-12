# ABOUTME: Makes the standalone `medical_tutor` package importable during pytest runs.
# ABOUTME: Ensures tests can run without requiring an editable install step.

from __future__ import annotations

import sys
from pathlib import Path


def _add_medical_tutor_src_to_sys_path() -> None:
    tests_dir = Path(__file__).resolve().parent
    medical_tutor_src = tests_dir.parent / "src"
    sys.path.insert(0, str(medical_tutor_src))


_add_medical_tutor_src_to_sys_path()
