"""Orchestrator prompt and decision parsing for Medical_Tutor."""

from __future__ import annotations

import json
from typing import Any, cast

from medical_tutor import prompts as prompt_templates
from medical_tutor.config import Config
from medical_tutor.contracts import AgentState

DEFAULT_TOOL_SPECS = [
    {
        "name": "zoom",
        "description": "Zoom into a region of the image.",
        "arguments": {
            "bbox_2d": "[x1, y1, x2, y2] normalized 0-1",
            "padding": "float",
        },
    },
    {
        "name": "segment",
        "description": "Segment an image region.",
        "arguments": {
            "point": "(x, y) optional",
            "bbox_2d": "optional [x1, y1, x2, y2]",
        },
    },
    {
        "name": "enhance",
        "description": "Enhance image contrast.",
        "arguments": {
            "factor": "float",
        },
    },
    {
        "name": "retrieve",
        "description": "Retrieve supporting passages (text or image).",
        "arguments": {
            "query": "string",
            "modality": "optional: text or image",
            "top_k": "optional integer",
        },
    },
    {
        "name": "web_browser",
        "description": "Search the web (Google CSE) or fetch a URL.",
        "arguments": {
            "query": "search query (requires GOOGLE_SEARCH_API_KEY + GOOGLE_SEARCH_ENGINE_ID)",
            "url": "http(s) URL to visit",
            "max_content_length": "optional integer",
            "max_links": "optional integer",
        },
    },
]


def _tool_specs(tool_registry: Any | None) -> list[dict[str, Any]]:
    if tool_registry is None:
        return DEFAULT_TOOL_SPECS
    if hasattr(tool_registry, "describe"):
        try:
            described = tool_registry.describe()
        except Exception:
            return DEFAULT_TOOL_SPECS
        if isinstance(described, list):
            return cast(list[dict[str, Any]], described)
        return DEFAULT_TOOL_SPECS
    return DEFAULT_TOOL_SPECS


def build_orchestrator_prompt(
    state: AgentState,
    config: Config,
    playbook: str,
    tool_registry: Any | None,
) -> str:
    tool_specs = _tool_specs(tool_registry)
    tool_lines = prompt_templates.format_tool_specs(tool_specs)

    retrieval_summary = state.retrieval_summary or "(no retrieval summary)"
    playbook_section = prompt_templates.format_playbook(playbook)

    prompt = (
        f"{prompt_templates.ORCHESTRATOR_INSTRUCTIONS}"
        f"{prompt_templates.TOOL_POLICY}"
        f"{playbook_section}\n\n"
        "Available tools:\n"
        f"{tool_lines}\n\n"
        "Retrieved context summary:\n"
        f"{retrieval_summary}\n\n"
        "User question:\n"
        f"{_latest_user_question(state)}\n"
    )
    return prompt


def _latest_user_question(state: AgentState) -> str:
    for message in reversed(state.messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # content may include image blocks; extract text parts
                texts = [
                    c.get("text")
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                ]
                return " ".join([t for t in texts if t])
    return ""


def parse_orchestrator_response(text: str) -> dict[str, Any]:
    if not text:
        return {"type": "answer"}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Attempt to extract JSON substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {"type": "answer"}
        else:
            return {"type": "answer"}

    if not isinstance(data, dict):
        return {"type": "answer"}

    decision_type = data.get("type")
    if decision_type == "tool":
        name = data.get("name")
        arguments = data.get("arguments")
        if isinstance(name, str) and isinstance(arguments, dict):
            return {"type": "tool", "name": name, "arguments": arguments}
        return {"type": "answer"}

    if decision_type == "answer":
        return {"type": "answer", "answer": data.get("answer")}

    return {"type": "answer"}


def run_orchestrator(
    state: AgentState,
    config: Config,
    playbook: str,
    tool_registry: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = build_orchestrator_prompt(state, config, playbook, tool_registry)
    metadata: dict[str, Any] = {"prompt": prompt}

    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=config.orchestrator_model,
            messages=[
                {"role": "system", "content": "You are a routing model. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        metadata["raw_response"] = content
        decision = parse_orchestrator_response(content)
        return decision, metadata
    except Exception as exc:  # pragma: no cover - network/credential errors
        metadata["error"] = str(exc)
        return {"type": "answer", "answer": ""}, metadata
