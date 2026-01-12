# ABOUTME: Tests UI formatting helpers for tool cards, retrieval items, and traces.
# ABOUTME: Ensures Gradio app construction returns a Blocks instance.

from __future__ import annotations

import json
import tempfile
import unittest


class TestUiFormatting(unittest.TestCase):
    def test_format_chat_history_extracts_text_from_multimodal_messages(self) -> None:
        from medical_tutor.ui.app import format_chat_history

        messages: list[dict[str, object]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/tmp/example.png"},
                    {"type": "text", "text": "Case prompt"},
                ],
            },
            {"role": "assistant", "content": "What do you think?"},
            {"role": "user", "content": "A"},
        ]

        chat = format_chat_history(messages)

        self.assertEqual([entry["role"] for entry in chat], ["user", "assistant", "user"])
        self.assertIn("Case prompt", chat[0]["content"])
        self.assertIn("What do you think?", chat[1]["content"])
        self.assertEqual(chat[2]["content"], "A")

    def test_format_tool_cards_pairs_calls_with_results(self) -> None:
        from medical_tutor.contracts import ToolCall, ToolResult
        from medical_tutor.ui.app import format_tool_cards

        calls = [
            ToolCall(id="call-1", name="zoom", arguments={"bbox_2d": [0, 0, 1, 1]}),
            ToolCall(id="call-2", name="enhance", arguments={"factor": 1.2}),
        ]
        results = [
            ToolResult(tool_name="zoom", ok=True, output={"summary": "Zoomed in"}),
            ToolResult(tool_name="enhance", ok=False, error="bad factor"),
        ]

        cards = format_tool_cards(calls, results)

        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0]["name"], "zoom")
        self.assertTrue(cards[0]["ok"])
        self.assertEqual(cards[0]["summary"], "Zoomed in")
        self.assertEqual(cards[1]["name"], "enhance")
        self.assertFalse(cards[1]["ok"])
        self.assertEqual(cards[1]["error"], "bad factor")

    def test_format_tool_images_returns_gallery_payload(self) -> None:
        from medical_tutor.contracts import ArtifactRef
        from medical_tutor.ui.app import format_tool_images

        artifacts = [
            ArtifactRef(id="image_1", path="/tmp/a.png", kind="image", summary="Zoomed in"),
            ArtifactRef(id="text_1", path="/tmp/a.txt", kind="text", summary="OCR output"),
            ArtifactRef(id="mask_1", path="/tmp/m.png", kind="mask", summary="Segmentation mask"),
        ]

        gallery = format_tool_images(artifacts)

        self.assertEqual(
            gallery, [("/tmp/a.png", "Zoomed in"), ("/tmp/m.png", "Segmentation mask")]
        )

    def test_format_retrieval_hits_serializes_items(self) -> None:
        from medical_tutor.contracts import RetrievalItem
        from medical_tutor.ui.app import format_retrieval_hits

        hits = [
            RetrievalItem(
                doc_id="doc-1",
                modality="text",
                score=0.12,
                provenance="bm25",
                snippet="Test snippet",
                uri=None,
            )
        ]

        rendered = format_retrieval_hits(hits)
        self.assertEqual(rendered[0]["doc_id"], "doc-1")
        self.assertEqual(rendered[0]["modality"], "text")

    def test_load_trace_events_reads_jsonl(self) -> None:
        from medical_tutor.ui.app import load_trace_events

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = f"{tmpdir}/trace.jsonl"
            with open(trace_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"node": "decide"}) + "\n")
                handle.write(json.dumps({"node": "answer"}) + "\n")

            events = load_trace_events(trace_path)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["node"], "decide")

    def test_build_app_returns_blocks(self) -> None:
        try:
            import gradio as gr
        except ModuleNotFoundError:
            self.skipTest("gradio is not installed")

        from medical_tutor.ui.app import build_app

        app = build_app()
        self.assertIsInstance(app, gr.Blocks)


if __name__ == "__main__":
    unittest.main()
