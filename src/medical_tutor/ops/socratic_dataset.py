# ABOUTME: Builds a small Socratic tutoring dataset from MedXpert patient scripts.
# ABOUTME: Writes normalized case indices and action-only SFT examples as JSONL.

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

DATASET_VERSION = "socratic_tutor_v1"
PROFILES = ("novice", "medium", "expert")
TURNS_PER_CASE = 2


def build_case_index(*, text_path: Path, mm_path: Path) -> list[dict[str, Any]]:
    cases = []
    cases.extend(_load_cases(path=text_path, modality="text"))
    cases.extend(_load_cases(path=mm_path, modality="mm"))
    return sorted(cases, key=lambda item: item["case_id"])


def build_action_examples(
    cases: list[dict[str, Any]],
    *,
    seed: int,
    use_controller: bool = False,
    image_root: Path | None = None,
    playbook_text: str = "",
) -> list[dict[str, Any]]:
    examples, _ = _build_action_examples_with_mode(
        cases,
        seed=seed,
        use_controller=use_controller,
        image_root=image_root,
        playbook_text=playbook_text,
        log_events=None,
    )
    return examples


def _build_action_examples_with_mode(
    cases: list[dict[str, Any]],
    *,
    seed: int,
    use_controller: bool,
    image_root: Path | None,
    playbook_text: str,
    log_events: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], str]:
    examples: list[dict[str, Any]] = []
    labeling_mode = "heuristic"
    if use_controller:
        if os.environ.get("OPENAI_API_KEY"):
            labeling_mode = "controller"
        else:
            labeling_mode = "heuristic_fallback"
            if log_events is not None:
                log_events.append(
                    {
                        "event": "controller_unavailable",
                        "reason": "OPENAI_API_KEY not set",
                    }
                )
    for case in cases:
        for profile in PROFILES:
            history: list[dict[str, str]] = []
            tool_used = False
            for turn_index in range(TURNS_PER_CASE):
                rng = random.Random(_seed_from_parts(seed, case["case_id"], profile, turn_index))
                reply, assessment = _simulate_student_reply(case, profile, rng)
                history_with_reply = [*history, {"role": "student", "content": reply}]

                action = None
                if labeling_mode == "controller":
                    action, error = _label_action_with_controller(
                        case=case,
                        profile=profile,
                        history=history_with_reply,
                        assessment=assessment,
                        image_root=image_root,
                        playbook_text=playbook_text,
                        attempt_count=turn_index + 1,
                    )
                    if action is None:
                        labeling_mode = "heuristic_fallback"
                        if log_events is not None:
                            log_events.append(
                                {
                                    "event": "controller_error",
                                    "case_id": case.get("case_id"),
                                    "profile": profile,
                                    "turn_index": turn_index,
                                    "error": error or "unknown",
                                }
                            )

                if action is None:
                    action = _select_target_action(
                        case=case,
                        assessment=assessment,
                        turn_index=turn_index,
                        tool_used=tool_used,
                    )

                examples.append(
                    {
                        "dataset_version": DATASET_VERSION,
                        "case_id": case["case_id"],
                        "student_profile": profile,
                        "kc_tags": {
                            "body_system": case.get("body_system"),
                            "medical_task": case.get("medical_task"),
                            "question_type": case.get("question_type"),
                        },
                        "image_ids": list(case.get("images") or []),
                        "turn_index": turn_index,
                        "history": history_with_reply,
                        "tool_summaries": [],
                        "student_reply_assessment": assessment,
                        "target_action": action,
                    }
                )

                history = [
                    *history_with_reply,
                    {"role": "tutor", "content": _action_to_text(action)},
                ]
                if action["type"] == "REQUEST_TOOL":
                    tool_used = True
    return examples, labeling_mode


def write_dataset(
    *,
    text_path: Path,
    mm_path: Path,
    output_dir: Path,
    seed: int,
    use_controller: bool = False,
    image_root: Path | None = None,
    playbook_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_case_index(text_path=text_path, mm_path=mm_path)
    playbook_text = playbook_path.read_text(encoding="utf-8") if playbook_path else ""
    log_events: list[dict[str, Any]] = []
    examples, labeling_mode = _build_action_examples_with_mode(
        cases,
        seed=seed,
        use_controller=use_controller,
        image_root=image_root,
        playbook_text=playbook_text,
        log_events=log_events,
    )

    cases_path = output_dir / "cases.jsonl"
    actions_path = output_dir / "action_sft.jsonl"
    _write_jsonl(cases_path, cases)
    _write_jsonl(actions_path, examples)

    manifest = {
        "dataset_version": DATASET_VERSION,
        "source_files": [str(text_path), str(mm_path)],
        "num_cases_text": sum(1 for case in cases if case["modality"] == "text"),
        "num_cases_mm": sum(1 for case in cases if case["modality"] == "mm"),
        "num_examples": len(examples),
        "seed": seed,
        "action_labeling": labeling_mode,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if log_events:
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(logs_dir / "labeling.jsonl", log_events)

    return {
        "dataset_version": DATASET_VERSION,
        "cases_path": str(cases_path),
        "actions_path": str(actions_path),
        "manifest_path": str(manifest_path),
        "num_examples": len(examples),
        "labeling_mode": labeling_mode,
    }


def _load_cases(*, path: Path, modality: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    for item in payload:
        case_id = str(item.get("id") or "").strip()
        if not case_id:
            continue
        options = item.get("options") or {}
        if not isinstance(options, dict):
            options = {}
        images = item.get("images") or []
        if not isinstance(images, list):
            images = []
        model_response = item.get("model_response") or []
        if not isinstance(model_response, list):
            model_response = []

        steps = []
        for step in model_response:
            if not isinstance(step, dict):
                continue
            steps.append(
                {
                    "key_question": str(step.get("key_question") or "").strip(),
                    "step_summary": str(step.get("step_summary") or "").strip(),
                    "associated_image_id": step.get("associated_image_id"),
                }
            )

        cases.append(
            {
                "case_id": case_id,
                "modality": modality,
                "question": str(item.get("question") or "").strip(),
                "options": {str(k): str(v) for k, v in options.items()},
                "label": str(item.get("label") or "").strip(),
                "body_system": str(item.get("body_system") or "") or None,
                "medical_task": str(item.get("medical_task") or "") or None,
                "question_type": str(item.get("question_type") or "") or None,
                "images": [str(name) for name in images],
                "socratic_steps": steps,
            }
        )
    return cases


def _simulate_student_reply(
    case: dict[str, Any],
    profile: str,
    rng: random.Random,
) -> tuple[str, dict[str, Any]]:
    grade = _sample_grade(profile, rng)
    label = str(case.get("label") or "").strip().upper()
    options = sorted((case.get("options") or {}).keys())
    if not options:
        options = ["A", "B", "C", "D", "E"]

    if grade == "correct" and label in options:
        choice = label
    else:
        choice = next((opt for opt in options if opt != label), options[0])

    reply = f"Answer: {choice}"
    misconception = None
    if grade == "wrong":
        misconception = "Missed key finding."
    elif grade == "partial":
        misconception = "Partially correct reasoning."

    assessment = {"grade": grade, "misconception": misconception}
    return reply, assessment


def _sample_grade(profile: str, rng: random.Random) -> str:
    roll = rng.random()
    if profile == "novice":
        return "wrong" if roll < 0.7 else "partial" if roll < 0.9 else "correct"
    if profile == "medium":
        return "wrong" if roll < 0.4 else "partial" if roll < 0.8 else "correct"
    return "wrong" if roll < 0.15 else "partial" if roll < 0.5 else "correct"


def _select_target_action(
    *,
    case: dict[str, Any],
    assessment: dict[str, Any],
    turn_index: int,
    tool_used: bool,
) -> dict[str, Any]:
    grade = assessment.get("grade")
    steps = case.get("socratic_steps") or []
    step = steps[turn_index] if turn_index < len(steps) else {}

    if case.get("modality") == "mm" and not tool_used and case.get("images"):
        return {
            "type": "REQUEST_TOOL",
            "arguments": {
                "name": "zoom",
                "arguments": {"bbox_2d": [0.2, 0.2, 0.8, 0.8], "padding": 0.1},
            },
        }

    if grade == "wrong":
        if turn_index == 0:
            content = step.get("key_question") or "What key finding stands out to you?"
            return {"type": "ASK_PROBE", "arguments": {"content": content}}
        content = step.get("step_summary") or "Focus on the most discriminating finding."
        return {"type": "HINT", "arguments": {"content": content, "level": min(3, turn_index + 1)}}

    if grade == "partial":
        content = step.get("step_summary") or "You're close; refine the key detail."
        return {"type": "HINT", "arguments": {"content": content, "level": 1}}

    if turn_index == 0:
        content = step.get("key_question") or "Why is this the best choice?"
        return {"type": "QUIZ", "arguments": {"content": content}}

    return {"type": "REVEAL_ANSWER", "arguments": {}}


def _action_to_text(action: dict[str, Any]) -> str:
    action_type = action.get("type")
    arguments = action.get("arguments") or {}
    if isinstance(arguments, dict):
        content = arguments.get("content")
        if isinstance(content, str) and content.strip():
            return content
    if action_type == "REQUEST_TOOL":
        tool_name = arguments.get("name") if isinstance(arguments, dict) else None
        return f"REQUEST_TOOL: {tool_name or 'tool'}"
    return f"{action_type}"


def _label_action_with_controller(
    *,
    case: dict[str, Any],
    profile: str,
    history: list[dict[str, str]],
    assessment: dict[str, Any],
    image_root: Path | None,
    playbook_text: str,
    attempt_count: int,
) -> tuple[dict[str, Any] | None, str | None]:
    if not os.environ.get("OPENAI_API_KEY"):
        return None, "OPENAI_API_KEY not set"
    try:
        from medical_tutor.config import load_config
        from medical_tutor.contracts import AgentState
        from medical_tutor.runtime.tutor_policy import run_tutor_policy

        messages = _history_to_messages(history)
        case_prompt = _format_case_prompt(case)
        image_path = _resolve_image_path(case, image_root=image_root)

        state = AgentState(
            messages=messages,
            image=image_path,
            run_id="dataset_label",
            student_profile=profile,
            student_attempt_count=attempt_count,
            case_prompt=case_prompt,
            student_reply_text=_latest_student_reply(history),
            student_reply_grade=assessment.get("grade"),
            student_reply_misconception=assessment.get("misconception"),
        )
        action, _ = run_tutor_policy(state, load_config(), playbook_text, tool_registry=None)
        return action.model_dump(), None
    except Exception as exc:
        return None, str(exc)


def _seed_from_parts(*parts: Any) -> int:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _history_to_messages(history: list[dict[str, str]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for entry in history:
        role = entry.get("role")
        content = entry.get("content", "")
        if role == "tutor":
            messages.append({"role": "assistant", "content": content})
        else:
            messages.append({"role": "user", "content": content})
    return messages


def _format_case_prompt(case: dict[str, Any]) -> str:
    question = str(case.get("question") or "").strip()
    options = case.get("options") or {}
    lines = [question, "", "Answer Choices:"]
    for key in sorted(options.keys()):
        lines.append(f"({key}) {options[key]}")
    return "\n".join(lines).strip()


def _latest_student_reply(history: list[dict[str, str]]) -> str | None:
    for entry in reversed(history):
        if entry.get("role") == "student":
            content = entry.get("content")
            return content if isinstance(content, str) else None
    return None


def _resolve_image_path(case: dict[str, Any], *, image_root: Path | None) -> str | None:
    images = case.get("images") or []
    if not images:
        return None
    image_name = str(images[0])
    if image_root is None:
        return image_name
    return str(image_root / image_name)
