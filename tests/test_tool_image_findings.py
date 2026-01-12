# ABOUTME: Tests the image_findings tool registration and basic argument handling.
# ABOUTME: Ensures the tool fails fast without an image and is discoverable by the controller.

from __future__ import annotations

import unittest


class TestImageFindingsTool(unittest.TestCase):
    def test_default_registry_includes_image_findings(self) -> None:
        from medical_tutor.tools.registry import build_default_registry

        registry = build_default_registry()
        tool_names = {spec.get("name") for spec in registry.describe()}

        self.assertIn("image_findings", tool_names)

    def test_image_findings_requires_image(self) -> None:
        from medical_tutor.contracts import AgentState
        from medical_tutor.tools.registry import build_default_registry

        registry = build_default_registry()
        registry.set_context(artifact_store=None, retriever=None)
        result = registry.call("image_findings", {}, AgentState())

        self.assertFalse(result.ok)
        self.assertIn("No image", str(result.error))


if __name__ == "__main__":
    unittest.main()
