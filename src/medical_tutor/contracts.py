# ABOUTME: Defines state and data contracts shared across the Medical_Tutor runtime.
# ABOUTME: Keeps the LangGraph state payload stable and easily serializable for tracing.
"""Shared contracts for Medical_Tutor runtime components."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ArtifactRef(BaseModel):
    id: str
    path: str
    kind: str  # "image", "mask", "text", "json"
    summary: str


class RetrievalItem(BaseModel):
    doc_id: str
    modality: str  # "text" or "image"
    score: float
    provenance: str
    snippet: str
    uri: str | None = None


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]
    target_artifact: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_arguments(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        arguments = data.get("arguments")
        if not isinstance(arguments, dict):
            return data
        data["arguments"] = _normalize_tool_arguments(arguments)
        return data


def _normalize_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    for key in ("bbox_2d", "point"):
        if key in normalized:
            normalized[key] = _normalize_argument_value(normalized[key])
    return normalized


def _normalize_argument_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in ("[", "(", "{"):
        return value
    try:
        parsed = ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return value
    if isinstance(parsed, tuple):
        return list(parsed)
    return parsed


class ToolResult(BaseModel):
    tool_name: str
    ok: bool = True
    output: dict[str, Any] | None = None
    error: str | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)


class Citation(BaseModel):
    source_id: str
    kind: str  # "retrieval" or "artifact"
    note: str | None = None


class ErrorRecord(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class AgentState(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)
    image: str | None = None
    student_profile: str | None = None
    student_attempt_count: int = 0
    case_prompt: str | None = None
    case_label: str | None = None
    student_reply_text: str | None = None
    student_reply_grade: str | None = None
    student_reply_misconception: str | None = None
    tutor_hint_level: int = 1
    tutor_consecutive_wrong: int = 0
    tutor_last_grade: str | None = None
    tutor_last_misconception: str | None = None
    tutor_action: dict[str, Any] | None = None
    retrieval_hits: list[RetrievalItem] = Field(default_factory=list)
    retrieval_summary: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    errors: list[ErrorRecord] = Field(default_factory=list)
    next_action: str | None = None
    retry_count: int = 0
    run_id: str = ""


def new_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"
