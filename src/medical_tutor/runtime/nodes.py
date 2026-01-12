# ABOUTME: Implements LangGraph nodes that transform runtime state for each step.
# ABOUTME: Handles both the base tool/answer routing flow and the Socratic tutor action routing flow.
"""LangGraph node implementations for Medical_Tutor runtime."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any, cast

from medical_tutor.config import Config
from medical_tutor.contracts import AgentState, ErrorRecord, ToolCall, ToolResult
from medical_tutor.runtime.answer import answer_question
from medical_tutor.runtime.orchestrator import run_orchestrator
from medical_tutor.runtime.tutor_policy import run_tutor_policy
from medical_tutor.tutoring import (
    TutorAction,
    apply_assessment_gate,
    apply_reveal_gate,
    student_requested_image_description,
    student_requested_reveal,
    update_tutor_state_from_assessment,
)


def _dump_state(state: AgentState) -> dict[str, Any]:
    return cast(dict[str, Any], state.model_dump())


def _latest_user_question(state: AgentState) -> str:
    for message in reversed(state.messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [
                    c.get("text")
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                ]
                return " ".join([t for t in texts if t])
    return ""


def _pending_tool_call(state: AgentState) -> ToolCall | None:
    if len(state.tool_results) >= len(state.tool_calls):
        return None
    return state.tool_calls[len(state.tool_results)]


def pre_retrieve(
    state: dict[str, Any],
    config: Config,
    retriever: Any | None,
    tracer: Any | None,
) -> dict[str, Any]:
    state_obj = AgentState.model_validate(state)
    state_before = _dump_state(state_obj)
    question = _latest_user_question(state_obj)
    image_path = state_obj.image

    hits = []
    summary = ""
    if retriever is not None and hasattr(retriever, "pre_retrieve"):
        try:
            hits, summary = retriever.pre_retrieve(question=question, image_path=image_path)
        except Exception as exc:
            state_obj.errors.append(ErrorRecord(code="retrieve_error", message=str(exc)))

    state_obj.retrieval_hits = hits
    state_obj.retrieval_summary = summary
    if tracer is not None and hasattr(tracer, "log_retrieval"):
        tracer.log_retrieval([hit.model_dump() for hit in hits])

    if tracer is not None:
        tracer.log_event(
            state_before=state_before,
            action={"node": "pre_retrieve", "question": question},
            observation={"retrieval_hits": len(hits)},
            state_after=_dump_state(state_obj),
        )

    return _dump_state(state_obj)


def orchestrate(
    state: dict[str, Any],
    config: Config,
    tool_registry: Any | None,
    playbook: str,
    tracer: Any | None,
) -> dict[str, Any]:
    state_obj = AgentState.model_validate(state)
    state_before = _dump_state(state_obj)
    decision, metadata = run_orchestrator(state_obj, config, playbook, tool_registry)
    can_use_tools = tool_registry is not None and hasattr(tool_registry, "call")
    state_obj.next_action = None

    if metadata.get("error"):
        state_obj.errors.append(ErrorRecord(code="orchestrator_error", message=metadata["error"]))

    if decision.get("type") == "tool" and can_use_tools:
        tool_call = ToolCall(
            id=str(uuid.uuid4()),
            name=decision.get("name", ""),
            arguments=decision.get("arguments", {}),
        )
        state_obj.tool_calls.append(tool_call)
        state_obj.next_action = "tool"
    else:
        state_obj.next_action = "answer"

    if tracer is not None:
        tracer.log_event(
            state_before=state_before,
            action={"node": "orchestrate", "decision": decision},
            observation={"metadata": metadata},
            state_after=_dump_state(state_obj),
        )

    return _dump_state(state_obj)


def decide(
    state: dict[str, Any],
    config: Config,
    tool_registry: Any | None,
    playbook: str,
    tracer: Any | None,
) -> dict[str, Any]:
    state_obj = AgentState.model_validate(state)
    if state_obj.student_profile:
        return tutor_decide(state, config, tool_registry, playbook, tracer)
    return orchestrate(state, config, tool_registry, playbook, tracer)


def tutor_decide(
    state: dict[str, Any],
    config: Config,
    tool_registry: Any | None,
    playbook: str,
    tracer: Any | None,
    *,
    policy: Callable[[AgentState], TutorAction] | None = None,
) -> dict[str, Any]:
    state_obj = AgentState.model_validate(state)
    state_before = _dump_state(state_obj)
    state_obj.next_action = None

    can_use_tools = tool_registry is not None and hasattr(tool_registry, "call")
    metadata: dict[str, Any] = {}

    update_tutor_state_from_assessment(state_obj)

    action: TutorAction | None = None
    if student_requested_image_description(state_obj.student_reply_text):
        summary = None
        for result in reversed(state_obj.tool_results):
            if result.tool_name != "image_findings":
                continue
            if not result.ok:
                continue
            output = result.output or {}
            if isinstance(output, dict):
                summary = output.get("summary")
            if isinstance(summary, str) and summary.strip():
                break
        if summary:
            action = TutorAction(
                type="MICROLESSON",
                arguments={
                    "content": (
                        f"Image findings (objective): {summary}\n\n"
                        "Based on those findings, which answer choice seems most likely and why?"
                    )
                },
            )
        else:
            pending = _pending_tool_call(state_obj)
            pending_is_image_findings = pending is not None and pending.name == "image_findings"
            if state_obj.image and not pending_is_image_findings and not state_obj.tool_results:
                action = TutorAction(
                    type="REQUEST_TOOL",
                    arguments={
                        "name": "image_findings",
                        "arguments": {"bbox_2d": [0.2, 0.2, 0.8, 0.8], "padding": 0.1},
                    },
                )

    if action is None:
        if policy is None:
            action, metadata = run_tutor_policy(state_obj, config, playbook, tool_registry)
        else:
            action = policy(state_obj)

    action = apply_reveal_gate(state_obj, action)
    action = apply_assessment_gate(state_obj, action)
    if state_obj.student_attempt_count > 0 and student_requested_reveal(
        state_obj.student_reply_text
    ):
        action = TutorAction(type="REVEAL_ANSWER", arguments={})
    if action.type == "REQUEST_TOOL" and state_obj.tool_results:
        last_tool = None
        last_summary = None
        for result in reversed(state_obj.tool_results):
            if not result.ok:
                continue
            last_tool = result.tool_name
            output = result.output or {}
            if isinstance(output, dict):
                last_summary = output.get("summary")
            break

        prefix = ""
        if isinstance(last_tool, str) and last_tool.strip():
            prefix = f"I already ran one tool ({last_tool}) this turn."
            if isinstance(last_summary, str) and last_summary.strip():
                prefix += f" Summary: {last_summary}"
            prefix += "\n\n"

        action = TutorAction(
            type="ASK_PROBE", arguments={"content": f"{prefix}What stands out to you now?"}
        )
    state_obj.tutor_action = action.model_dump()

    if action.type == "REQUEST_TOOL":
        pending = _pending_tool_call(state_obj)
        if pending is not None:
            state_obj.next_action = "tool"
            if tracer is not None:
                tracer.log_event(
                    state_before=state_before,
                    action={"node": "tutor_decide", "tutor_action": state_obj.tutor_action},
                    observation={"metadata": {**metadata, "pending_tool": pending.name}},
                    state_after=_dump_state(state_obj),
                )
            return _dump_state(state_obj)
        name = action.arguments.get("name")
        arguments = action.arguments.get("arguments")
        if can_use_tools and isinstance(name, str) and isinstance(arguments, dict):
            state_obj.tool_calls.append(
                ToolCall(id=str(uuid.uuid4()), name=name, arguments=cast(dict[str, Any], arguments))
            )
            state_obj.next_action = "tool"
        else:
            state_obj.errors.append(
                ErrorRecord(code="tutor_tool_unavailable", message="Tool registry not configured")
            )
            state_obj.messages.append(
                {
                    "role": "assistant",
                    "content": "Tools are not available right now. Let's reason it out.",
                }
            )
            state_obj.next_action = "respond"
    elif action.type == "REVEAL_ANSWER":
        state_obj.next_action = "answer"
    else:
        content = action.arguments.get("content")
        if not isinstance(content, str) or not content.strip():
            content = {
                "ASK_PROBE": "What's your reasoning so far?",
                "HINT": "Try focusing on the key finding that most strongly narrows the choices.",
                "MICROLESSON": "Let's quickly review the underlying concept before choosing.",
                "QUIZ": "Quick check: what single finding would you expect in this condition?",
                "SAFETY_REFUSE": "I can't help with that request.",
            }.get(action.type, "Let's continue step by step.")
        state_obj.messages.append({"role": "assistant", "content": content})
        state_obj.next_action = "respond"

    if tracer is not None:
        tracer.log_event(
            state_before=state_before,
            action={"node": "tutor_decide", "tutor_action": state_obj.tutor_action},
            observation={"metadata": metadata},
            state_after=_dump_state(state_obj),
        )

    return _dump_state(state_obj)


def tool_exec(
    state: dict[str, Any],
    tool_registry: Any | None,
    tracer: Any | None,
) -> dict[str, Any]:
    state_obj = AgentState.model_validate(state)
    state_before = _dump_state(state_obj)
    pending = _pending_tool_call(state_obj)
    state_obj.next_action = None

    if pending is None:
        state_obj.errors.append(ErrorRecord(code="tool_exec", message="No pending tool call"))
        if tracer is not None:
            tracer.log_event(
                state_before=state_before,
                action={"node": "tool_exec", "tool": None},
                observation={"ok": False, "error": "No pending tool call"},
                state_after=_dump_state(state_obj),
            )
        return _dump_state(state_obj)

    if tool_registry is None or not hasattr(tool_registry, "call"):
        result = ToolResult(tool_name=pending.name, ok=False, error="Tool registry not configured")
    else:
        try:
            result = tool_registry.call(pending.name, pending.arguments, state_obj)
        except Exception as exc:
            result = ToolResult(tool_name=pending.name, ok=False, error=str(exc))

    state_obj.tool_results.append(result)
    if result.artifact_refs:
        state_obj.artifacts.extend(result.artifact_refs)

    if tracer is not None:
        tracer.log_event(
            state_before=state_before,
            action={"node": "tool_exec", "tool": pending.name},
            observation={"ok": result.ok, "error": result.error},
            state_after=_dump_state(state_obj),
        )

    return _dump_state(state_obj)


def answer(
    state: dict[str, Any],
    config: Config,
    tracer: Any | None,
) -> dict[str, Any]:
    state_obj = AgentState.model_validate(state)
    state_before = _dump_state(state_obj)
    answer_text, metadata = answer_question(state_obj, config)
    state_obj.next_action = "respond"

    if metadata.get("error"):
        state_obj.errors.append(ErrorRecord(code="answer_error", message=metadata["error"]))

    if (
        "insufficient evidence" in answer_text.lower()
        and not metadata.get("error")
        and state_obj.retry_count < 1
    ):
        state_obj.retry_count += 1
        state_obj.next_action = "retry"
        if tracer is not None:
            tracer.log_event(
                state_before=state_before,
                action={"node": "answer", "retry": True},
                observation={"metadata": metadata},
                state_after=_dump_state(state_obj),
            )
        return _dump_state(state_obj)

    state_obj.messages.append({"role": "assistant", "content": answer_text})

    if tracer is not None:
        tracer.log_event(
            state_before=state_before,
            action={"node": "answer", "retry": False},
            observation={"metadata": metadata},
            state_after=_dump_state(state_obj),
        )

    return _dump_state(state_obj)


def respond(
    state: dict[str, Any],
    tracer: Any | None,
) -> dict[str, Any]:
    state_obj = AgentState.model_validate(state)
    state_before = _dump_state(state_obj)
    state_obj.next_action = None
    if tracer is not None:
        tracer.log_event(
            state_before=state_before,
            action={"node": "respond"},
            observation={},
            state_after=_dump_state(state_obj),
        )
    return _dump_state(state_obj)
