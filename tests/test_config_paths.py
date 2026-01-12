# ABOUTME: Tests config path resolution for retrieval indices across different working directories.
# ABOUTME: Ensures default paths can resolve when indices live under med-visual-tutor/data.

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path


class TestConfigPaths(unittest.TestCase):
    def test_resolves_retrieval_paths_under_med_visual_tutor(self) -> None:
        from medical_tutor.config import load_config

        old_cwd = Path.cwd()
        old_env = dict(os.environ)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                bm25_path = root / "med-visual-tutor" / "data" / "retrieval" / "v3" / "bm25.pkl"
                text_index = (
                    root / "med-visual-tutor" / "data" / "retrieval" / "v3" / "text_enhanced_new"
                )
                image_index = (
                    root / "med-visual-tutor" / "data" / "retrieval" / "v3" / "images_pmc_vqa1"
                )
                bm25_path.parent.mkdir(parents=True, exist_ok=True)
                text_index.mkdir(parents=True, exist_ok=True)
                image_index.mkdir(parents=True, exist_ok=True)
                bm25_path.write_bytes(b"stub")

                os.chdir(root)
                os.environ.pop("MEDTUTOR_BM25_PATH", None)
                os.environ.pop("MEDTUTOR_TEXT_INDEX_PATH", None)
                os.environ.pop("MEDTUTOR_IMAGE_INDEX_PATH", None)

                config = load_config()

                self.assertEqual(config.bm25_path, str(bm25_path.relative_to(root)))
                self.assertEqual(config.text_index_path, str(text_index.relative_to(root)))
                self.assertEqual(config.image_index_path, str(image_index.relative_to(root)))
        finally:
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_env)


if __name__ == "__main__":
    unittest.main()
