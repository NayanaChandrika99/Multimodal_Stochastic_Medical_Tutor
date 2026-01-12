# ABOUTME: Tests MedXpert image download helpers without hitting the network.
# ABOUTME: Validates image ID resolution and image payload saving behavior.

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image


class TestMedXpertDownloaderHelpers(unittest.TestCase):
    def test_resolve_image_id_prefers_explicit_field(self) -> None:
        from medical_tutor.medxpert import resolve_image_id

        record = {"image_id": "MM-1.jpg", "image": {"path": "/tmp/other.jpg"}}
        resolved = resolve_image_id(record, image_field="image")
        self.assertEqual(resolved, "MM-1.jpg")

    def test_resolve_image_id_falls_back_to_image_path(self) -> None:
        from medical_tutor.medxpert import resolve_image_id

        record = {"image": {"path": "/tmp/MM-2.png"}}
        resolved = resolve_image_id(record, image_field="image")
        self.assertEqual(resolved, "MM-2.png")

    def test_save_image_payload_writes_file(self) -> None:
        from medical_tutor.medxpert import save_image_payload

        with tempfile.TemporaryDirectory() as tmpdir:
            image = Image.new("RGB", (4, 4), color=(255, 0, 0))
            output_path = Path(tmpdir) / "out.jpg"
            saved = save_image_payload(image, output_path)
            self.assertTrue(saved.exists())


if __name__ == "__main__":
    unittest.main()
