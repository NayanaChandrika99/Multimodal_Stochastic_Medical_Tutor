# ABOUTME: Provides a small, resilient LLM call helper for ACE workflows.
# ABOUTME: Records per-call metadata and returns errors in-band instead of raising.

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _write_llm_log(log_dir: str, call_info: dict[str, Any]) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    role = str(call_info.get("role") or "llm")
    call_id = str(call_info.get("call_id") or "call")
    filename = f"{role}_{call_id}_{timestamp}.json"
    call_info = dict(call_info)
    call_info["timestamp"] = timestamp
    call_info["datetime"] = datetime.now(timezone.utc).isoformat()
    (Path(log_dir) / filename).write_text(
        json.dumps(call_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def call_llm(
    _client: object | None,
    _api_provider: str,
    model: str,
    prompt: str,
    *,
    role: str,
    call_id: str,
    max_tokens: int = 4096,
    log_dir: str | None = None,
    use_json_mode: bool = True,
) -> tuple[str, dict[str, Any]]:
    del _client, _api_provider

    call_info: dict[str, Any] = {
        "role": role,
        "call_id": call_id,
        "model": model,
        "prompt": prompt,
        "prompt_length": len(prompt),
        "use_json_mode": use_json_mode,
    }
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"} if use_json_mode else None,
        )
        content = response.choices[0].message.content or ""
        call_info["response"] = content
        call_info["response_length"] = len(content)
        if getattr(response, "usage", None) is not None:
            call_info["prompt_num_tokens"] = getattr(response.usage, "prompt_tokens", None)
            call_info["response_num_tokens"] = getattr(response.usage, "completion_tokens", None)
        if log_dir:
            _write_llm_log(log_dir, call_info)
        return content, call_info
    except Exception as exc:  # pragma: no cover - network/credential errors
        call_info["error"] = str(exc)
        call_info["response"] = ""
        call_info["response_length"] = 0
        if log_dir:
            _write_llm_log(log_dir, call_info)
        return "", call_info
