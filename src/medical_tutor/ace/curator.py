# ABOUTME: Runs the ACE curator step to propose playbook operations from reflections.
# ABOUTME: Applies validated operations to update the playbook conservatively.

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from medical_tutor.ace.llm import call_llm as default_call_llm
from medical_tutor.ace.playbook import (
    apply_curator_operations,
    auto_prune_harmful_bullets,
    extract_json_from_text,
)

CURATOR_PROMPT = """You are a master curator of knowledge. Your job is to update an existing playbook based on a reflection.

CRITICAL: Respond with valid JSON only (no markdown, no code fences).

Tool names that must be mentioned in new bullets: zoom, enhance, segment, retrieve, web_browser
Failure tokens (use when relevant): parse_fail, unknown_tool, max_tools_exceeded

Training Context:
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

Current Playbook Stats (JSON):
{playbook_stats}

Recent Reflection:
{recent_reflection}

Current Playbook:
{current_playbook}

Question Context:
{question_context}

Your Task:
Output ONLY a valid JSON object with these exact fields:
- reasoning: string
- operations: list of operations
  - type: ADD | UPDATE | DELETE
  - For ADD: section, content
  - For UPDATE: bullet_id, content, reason (optional)
  - For DELETE: bullet_id, reason

Allowed ADD sections (normalized snake_case):
common_mistakes, problem_solving_heuristics, context_indicators, others
"""


_ALLOWED_ADD_SECTIONS = {
    "common_mistakes",
    "problem_solving_heuristics",
    "context_indicators",
    "others",
}


class AceCurator:
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

    def curate(
        self,
        *,
        current_playbook: str,
        recent_reflection: str,
        question_context: str,
        current_step: int,
        total_samples: int,
        token_budget: int,
        playbook_stats: dict[str, Any],
        use_json_mode: bool = True,
        call_id: str = "curate",
        log_dir: str | None = None,
        next_global_id: int,
    ) -> tuple[str, int, list[dict[str, Any]], dict[str, Any]]:
        prompt = CURATOR_PROMPT.format(
            current_step=current_step,
            total_samples=total_samples,
            token_budget=token_budget,
            playbook_stats=json.dumps(playbook_stats, indent=2, ensure_ascii=False),
            recent_reflection=recent_reflection,
            current_playbook=current_playbook,
            question_context=question_context,
        )
        response, call_info = self._call_llm(
            None,
            "openai",
            self.model,
            prompt,
            role="curator",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
        )

        operations: list[dict[str, Any]] = []
        payload = extract_json_from_text(response, "operations")
        if isinstance(payload, dict):
            operations = self._validate_operations(payload.get("operations"))

        updated_playbook, next_global_id = apply_curator_operations(
            current_playbook, operations, next_global_id
        )
        updated_playbook = auto_prune_harmful_bullets(updated_playbook)
        return updated_playbook, next_global_id, operations, call_info

    @classmethod
    def _validate_operations(cls, raw_ops: object) -> list[dict[str, Any]]:
        if not isinstance(raw_ops, list):
            return []
        validated: list[dict[str, Any]] = []
        for op in raw_ops:
            if not isinstance(op, dict):
                continue
            op_type = str(op.get("type") or "").strip().upper()
            if op_type == "ADD":
                section = str(op.get("section") or "").strip().lower()
                section = section.replace(" ", "_").replace("&", "and")
                content = str(op.get("content") or "").strip()
                if not content:
                    continue
                if section not in _ALLOWED_ADD_SECTIONS:
                    section = "others"
                validated.append({"type": "ADD", "section": section, "content": content})
            elif op_type == "UPDATE":
                bullet_id = str(op.get("bullet_id") or "").strip()
                content = str(op.get("content") or "").strip()
                if bullet_id and content:
                    validated.append({"type": "UPDATE", "bullet_id": bullet_id, "content": content})
            elif op_type == "DELETE":
                bullet_id = str(op.get("bullet_id") or "").strip()
                reason = str(op.get("reason") or "").strip()
                if bullet_id and reason:
                    validated.append({"type": "DELETE", "bullet_id": bullet_id, "reason": reason})
        return validated
