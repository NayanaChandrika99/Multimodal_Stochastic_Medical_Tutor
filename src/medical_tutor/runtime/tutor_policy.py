# ABOUTME: Calls a multimodal controller model to choose the next Socratic tutoring action.
# ABOUTME: Produces a structured TutorAction that the runtime can gate and route deterministically.

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

from medical_tutor.config import Config
from medical_tutor.contracts import AgentState
from medical_tutor.runtime.orchestrator import _tool_specs
from medical_tutor.tutoring import TutorAction


def _encode_image_data_url(image_path: str) -> str:
    path = Path(image_path)
    suffix = path.suffix.lower().lstrip(".") or "png"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:image/{suffix};base64,{b64}"


def _format_tool_summaries(state: AgentState) -> str:
    if not state.tool_results:
        return "(none)"
    lines = []
    for result in state.tool_results:
        output = result.output or {}
        summary = None
        if isinstance(output, dict):
            summary = output.get("summary")
            if summary is None:
                summary = {
                    key: value for key, value in output.items() if key not in {"hits", "image"}
                }
        if summary is None:
            summary = output
        lines.append(f"- {result.tool_name}: {summary}")
    return "\n".join(lines)


def _format_assessment(state: AgentState) -> str:
    if not state.student_reply_grade:
        return "(none)"
    parts = [f"grade={state.student_reply_grade}"]
    if state.student_reply_misconception:
        parts.append(f"misconception={state.student_reply_misconception}")
    return "; ".join(parts)


def _format_tutoring_state(state: AgentState) -> str:
    parts = [
        f"hint_level={state.tutor_hint_level}",
        f"consecutive_wrong={state.tutor_consecutive_wrong}",
    ]
    if state.tutor_last_grade:
        parts.append(f"last_grade={state.tutor_last_grade}")
    if state.tutor_last_misconception:
        parts.append(f"last_misconception={state.tutor_last_misconception}")
    return "; ".join(parts)


def _format_case_prompt(state: AgentState) -> str:
    if state.case_prompt and state.case_prompt.strip():
        return state.case_prompt.strip()
    for message in state.messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = [
                item.get("text")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            joined = " ".join(text for text in texts if text)
            if joined.strip():
                return joined.strip()
    return "(no case prompt)"


def _parse_tutor_action_response(response_text: str) -> dict[str, Any]:
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        data = None

    if data is None:
        match = re.search(r"```json\s*(\{.*\}|\[.*\])\s*```", response_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                data = None

    if data is None:
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", response_text)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None

    if not isinstance(data, dict):
        raise ValueError("Tutor policy response did not contain a JSON object.")
    return data


def _build_tutor_prompt(state: AgentState, playbook: str, tool_registry: Any | None) -> str:
    tool_specs = _tool_specs(tool_registry)
    tool_lines = "\n".join(
        [
            f"- {spec.get('name')}: {spec.get('description')} args={spec.get('arguments')}"
            for spec in tool_specs
        ]
    )
    retrieval_summary = state.retrieval_summary or "(no retrieval summary)"
    profile = state.student_profile or "(unset)"
    assessment = _format_assessment(state)
    tutoring_state = _format_tutoring_state(state)
    tool_summaries = _format_tool_summaries(state)
    case_prompt = _format_case_prompt(state)
    tool_guidance = ""
    if state.image and not state.tool_results and not state.tool_calls:
        tool_guidance = (
            "Tool guidance:\n"
            "An image is provided and no vision tools have been used yet; "
            "request one vision tool (zoom or enhance) before asking another question.\n\n"
        )

    return (
        "You are a Socratic medical tutor. Your job is to teach, not to dump answers.\n"
        "Choose the next tutoring action as JSON only.\n\n"
        "Output schema:\n"
        '{ "type": "ASK_PROBE|HINT|MICROLESSON|QUIZ|REQUEST_TOOL|REVEAL_ANSWER|SAFETY_REFUSE", "arguments": { ... } }\n\n'
        "Action requirements:\n"
        '- ASK_PROBE: arguments={ "content": "question to ask the student" }\n'
        '- HINT: arguments={ "content": "hint text", "level": 1..3 }\n'
        '- MICROLESSON: arguments={ "content": "short teaching content" }\n'
        '- QUIZ: arguments={ "content": "a short quiz question" }\n'
        '- REQUEST_TOOL: arguments={ "name": "<tool name>", "arguments": { ...tool args... } }\n'
        "- REVEAL_ANSWER: arguments={}\n"
        '- SAFETY_REFUSE: arguments={ "content": "brief refusal" }\n\n'
        f"{tool_guidance}"
        f"Case prompt (question + options):\n{case_prompt}\n\n"
        f"Student profile: {profile}\n"
        f"Student attempt count: {state.student_attempt_count}\n\n"
        f"Student assessment: {assessment}\n"
        f"Tutoring state: {tutoring_state}\n"
        f"Student reply: {state.student_reply_text or '(none)'}\n\n"
        f"Playbook:\n{playbook}\n\n"
        f"Available tools:\n{tool_lines}\n\n"
        f"Tool summaries:\n{tool_summaries}\n\n"
        f"Retrieved context summary:\n{retrieval_summary}\n\n"
        "Conversation (most recent last):\n"
        f"{state.messages}\n"
    )


def run_tutor_policy(
    state: AgentState,
    config: Config,
    playbook: str,
    tool_registry: Any | None,
) -> tuple[TutorAction, dict[str, Any]]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; cannot run tutor policy.")

    prompt = _build_tutor_prompt(state, playbook, tool_registry)
    metadata: dict[str, Any] = {"prompt": prompt}

    try:
        from openai import OpenAI  # pyright: ignore[reportMissingImports]

        client = OpenAI()
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        if state.image:
            content.append(
                {
                    "type": "input_image",
                    "image_url": _encode_image_data_url(state.image),
                }
            )

        response = client.responses.create(
            model=config.orchestrator_model,
            input=[{"role": "user", "content": content}],
        )
        text = response.output_text or ""
        metadata["raw_response"] = text
        data = _parse_tutor_action_response(text)
        action = TutorAction.model_validate(data)
        return action, metadata
    except Exception as exc:
        metadata["error"] = str(exc)
        raise
