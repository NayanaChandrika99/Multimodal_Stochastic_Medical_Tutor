# ABOUTME: Registers runtime tools and exposes their schemas to the controller models.
# ABOUTME: Routes tool calls to adapter functions and injects retriever/artifact contexts.
"""Tool registry for Medical_Tutor."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from medical_tutor.contracts import ToolResult
from medical_tutor.tools import adapters


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., ToolResult]] = {}
        self._descriptions: dict[str, dict[str, Any]] = {}
        self.artifact_store: Any | None = None
        self.retriever: Any | None = None

    def register(
        self,
        name: str,
        handler: Callable[..., ToolResult],
        description: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> None:
        self._tools[name] = handler
        self._descriptions[name] = {
            "name": name,
            "description": description,
            "arguments": arguments or {},
        }

    def set_context(
        self, *, artifact_store: Any | None = None, retriever: Any | None = None
    ) -> None:
        self.artifact_store = artifact_store
        self.retriever = retriever

    def call(self, name: str, arguments: dict[str, Any], state: Any) -> ToolResult:
        handler = self._tools.get(name)
        if handler is None:
            return ToolResult(tool_name=name, ok=False, error=f"Tool '{name}' is not registered")
        return handler(arguments=arguments, state=state, registry=self)

    def describe(self) -> list[dict[str, Any]]:
        return list(self._descriptions.values())


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        "zoom",
        adapters.zoom_adapter,
        description="Zoom into a region of the image.",
        arguments={"bbox_2d": "[x1, y1, x2, y2] normalized 0-1", "padding": "float"},
    )
    registry.register(
        "enhance",
        adapters.enhance_adapter,
        description="Enhance image contrast.",
        arguments={"factor": "float"},
    )
    registry.register(
        "segment",
        adapters.segment_adapter,
        description="Segment an image region.",
        arguments={"point": "(x, y) optional", "bbox_2d": "optional [x1, y1, x2, y2]"},
    )
    registry.register(
        "retrieve",
        adapters.retrieve_adapter,
        description="Retrieve supporting passages (text or image).",
        arguments={
            "query": "string",
            "modality": "optional text|image",
            "top_k": "optional integer",
        },
    )
    registry.register(
        "web_browser",
        adapters.web_browser_adapter,
        description="Search the web (Google CSE) or fetch a URL.",
        arguments={
            "query": "search query (requires GOOGLE_SEARCH_API_KEY + GOOGLE_SEARCH_ENGINE_ID)",
            "url": "http(s) URL to visit",
            "max_content_length": "optional integer (default 5000)",
            "max_links": "optional integer (default 5)",
        },
    )
    registry.register(
        "image_findings",
        adapters.image_findings_adapter,
        description="Describe objective findings visible in the image (no diagnosis).",
        arguments={
            "bbox_2d": "optional [x1, y1, x2, y2] normalized 0-1",
            "padding": "optional float",
            "prompt": "optional string",
            "max_new_tokens": "optional integer",
        },
    )

    return registry
