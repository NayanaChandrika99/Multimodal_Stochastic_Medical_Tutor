# ABOUTME: Runs the ACE reflector step to tag playbook bullets as helpful/harmful/neutral.
# ABOUTME: Emits reflection text and bullet_tags for playbook counter updates.

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from medical_tutor.ace.llm import call_llm as default_call_llm
from medical_tutor.ace.playbook import extract_json_from_text

REFLECTOR_PROMPT = """You are an expert analyst and educator. Your job is to diagnose why a model's answer or tool use failed.

**Instructions:**
- Use the environment feedback and tool traces to diagnose the failure.
- Treat tool failures as incorrect even when the answer seems correct.
- Identify root causes and what should be done differently.
- Tag each bullet from the playbook snippet as helpful, harmful, or neutral.
- Only tag bullet IDs that appear in the "Playbook Bullets Used" list.
- If no bullets are listed under "Playbook Bullets Used", return bullet_tags as an empty list.

Tool names (mention when relevant): zoom, enhance, segment, retrieve, web_browser
Failure tokens (use when relevant): parse_fail, unknown_tool, max_tools_exceeded

Your output must be a JSON object with these fields:
- error_identification
- root_cause_analysis
- correct_approach
- key_insights (list of strings)
- failure_modes (list of strings)
- bullet_tags (list of {{"id": "...", "tag": "helpful|harmful|neutral"}})

Question:
{question}

Model's Predicted Answer:
{predicted_answer}

Ground Truth Answer:
{ground_truth}

Environment Feedback:
{environment_feedback}

Tool Metrics:
{tool_metrics}

Tool Trace:
{tool_trace}

Playbook Bullets Used:
{bullets_used}

Answer in JSON only:
"""


class AceReflector:
    _BULLET_ID_PATTERN = re.compile(r"\[([a-z]{2,5}-\d{5})\]")
    _ALLOWED_TAGS = {"helpful", "harmful", "neutral"}

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int = 4096,
        call_llm: Callable[..., tuple[str, dict[str, Any]]] | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._call_llm = call_llm or default_call_llm

    def reflect(
        self,
        *,
        question: str,
        predicted_answer: str,
        ground_truth: str | None,
        environment_feedback: str,
        tool_metrics: dict[str, Any] | None,
        tool_trace: list[dict[str, Any]] | None,
        bullets_used: str,
        use_json_mode: bool = True,
        call_id: str = "reflect",
        log_dir: str | None = None,
    ) -> tuple[str, list[dict[str, str]], dict[str, Any]]:
        prompt_args = {
            "question": question,
            "predicted_answer": predicted_answer,
            "ground_truth": ground_truth or "",
            "environment_feedback": environment_feedback,
            "tool_metrics": json.dumps(tool_metrics or {}, indent=2, ensure_ascii=False),
            "tool_trace": json.dumps(tool_trace or [], indent=2, ensure_ascii=False),
            "bullets_used": bullets_used,
        }
        prompt = REFLECTOR_PROMPT.format(**prompt_args)
        response, call_info = self._call_llm(
            None,
            "openai",
            self.model,
            prompt,
            role="reflector",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
        )
        payload = extract_json_from_text(response) or {}
        if not isinstance(payload, dict):
            payload = {}
        allowed_ids = self._extract_allowed_bullet_ids(bullets_used)
        bullet_tags = self._extract_bullet_tags(payload, allowed_ids=allowed_ids)
        return response, bullet_tags, call_info

    @classmethod
    def _extract_allowed_bullet_ids(cls, bullets_used: str) -> set[str]:
        if not bullets_used.strip():
            return set()
        return set(cls._BULLET_ID_PATTERN.findall(bullets_used))

    @classmethod
    def _extract_bullet_tags(
        cls, payload: dict[str, Any], *, allowed_ids: set[str]
    ) -> list[dict[str, str]]:
        raw_tags = payload.get("bullet_tags", [])
        if not isinstance(raw_tags, list):
            return []
        if not allowed_ids:
            return []
        tags: list[dict[str, str]] = []
        for item in raw_tags:
            if not isinstance(item, dict):
                continue
            bullet_id = str(item.get("id") or "").strip()
            tag_value = str(item.get("tag") or "").strip().lower()
            if not bullet_id or tag_value not in cls._ALLOWED_TAGS:
                continue
            if bullet_id not in allowed_ids:
                continue
            tags.append({"id": bullet_id, "tag": tag_value})
        return tags
