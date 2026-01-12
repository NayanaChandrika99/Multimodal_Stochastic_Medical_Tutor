# ABOUTME: Tests the answer runtime hardware selection helpers for MedGemma inference.
# ABOUTME: Ensures we choose GPU-friendly defaults without importing heavy model weights.

from __future__ import annotations

import unittest


class TestAnswerRuntime(unittest.TestCase):
    def test_select_device_prefers_cuda_when_available(self) -> None:
        from medical_tutor.runtime.answer import select_device

        self.assertEqual(select_device(cuda_available=True, mps_available=False), "cuda")

    def test_select_device_falls_back_to_mps(self) -> None:
        from medical_tutor.runtime.answer import select_device

        self.assertEqual(select_device(cuda_available=False, mps_available=True), "mps")

    def test_select_device_falls_back_to_cpu(self) -> None:
        from medical_tutor.runtime.answer import select_device

        self.assertEqual(select_device(cuda_available=False, mps_available=False), "cpu")

    def test_select_dtype_name_prefers_bfloat16_on_cuda_when_supported(self) -> None:
        from medical_tutor.runtime.answer import select_dtype_name

        self.assertEqual(
            select_dtype_name(device="cuda", cuda_bf16_supported=True),
            "bfloat16",
        )

    def test_select_dtype_name_falls_back_to_float16_on_cuda_when_bf16_unsupported(self) -> None:
        from medical_tutor.runtime.answer import select_dtype_name

        self.assertEqual(
            select_dtype_name(device="cuda", cuda_bf16_supported=False),
            "float16",
        )

    def test_select_dtype_name_uses_float16_on_mps(self) -> None:
        from medical_tutor.runtime.answer import select_dtype_name

        self.assertEqual(
            select_dtype_name(device="mps", cuda_bf16_supported=False),
            "float16",
        )

    def test_select_dtype_name_uses_float32_on_cpu(self) -> None:
        from medical_tutor.runtime.answer import select_dtype_name

        self.assertEqual(
            select_dtype_name(device="cpu", cuda_bf16_supported=False),
            "float32",
        )


if __name__ == "__main__":
    unittest.main()
