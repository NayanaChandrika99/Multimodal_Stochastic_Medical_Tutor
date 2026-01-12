# ABOUTME: Provides a minimal Gradio UI for the Medical_Tutor runtime.
# ABOUTME: Formats answers, tools, retrieval, and trace outputs for display.

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from medical_tutor.config import load_config
from medical_tutor.contracts import (
    AgentState,
    ArtifactRef,
    RetrievalItem,
    ToolCall,
    ToolResult,
    new_run_id,
)
from medical_tutor.ops.replay import ReplayRunner
from medical_tutor.runtime.runner import GraphRunner

_TUTOR_RUNNERS: dict[str, GraphRunner] = {}


def _coerce_message_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and item.get("text"):
                parts.append(str(item.get("text")))
        return " ".join([part.strip() for part in parts if part.strip()])
    return str(content)


def format_chat_history(messages: list[dict[str, object]]) -> list[dict[str, str]]:
    rendered: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = _coerce_message_content_to_text(message.get("content"))
        rendered.append({"role": str(role), "content": text})
    return rendered


def format_tool_cards(
    tool_calls: list[ToolCall],
    tool_results: list[ToolResult],
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for call, result in zip(tool_calls, tool_results, strict=False):
        summary = None
        output = result.output or {}
        if isinstance(output, dict):
            summary = output.get("summary")
        cards.append(
            {
                "name": call.name,
                "arguments": call.arguments,
                "ok": result.ok,
                "summary": summary,
                "error": result.error,
            }
        )
    return cards


def format_tool_images(artifacts: list[ArtifactRef]) -> list[tuple[str, str]]:
    rendered: list[tuple[str, str]] = []
    for artifact in artifacts:
        if artifact.kind not in {"image", "mask"}:
            continue
        rendered.append((artifact.path, artifact.summary))
    return rendered


def format_retrieval_hits(hits: list[RetrievalItem]) -> list[dict[str, Any]]:
    return [hit.model_dump() for hit in hits]


def load_trace_events(trace_path: str | Path) -> list[dict[str, Any]]:
    path = Path(trace_path)
    if not path.exists():
        return []
    return ReplayRunner(path).run()


def load_retrieval_events(retrieval_path: str | Path) -> list[dict[str, Any]]:
    path = Path(retrieval_path)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _extract_answer(state: AgentState) -> str:
    for message in reversed(state.messages):
        if message.get("role") == "assistant":
            return str(message.get("content") or "").strip()
    return ""


def _build_run_info(state: AgentState) -> str:
    config = load_config()
    output_dir = Path(config.output_dir) / state.run_id
    return f"run_id={state.run_id}\noutput_dir={output_dir}"


def run_single_turn(
    question: str, image_path: str | None
) -> tuple[
    str,
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    runner = GraphRunner(config=load_config())
    state = runner.run(question=question, image_path=image_path)

    config = load_config()
    run_dir = Path(config.output_dir) / state.run_id
    trace_events = load_trace_events(run_dir / "trace.jsonl")
    retrieval_events = load_retrieval_events(run_dir / "retrieval_artifacts.jsonl")

    answer = _extract_answer(state)
    tool_cards = format_tool_cards(state.tool_calls, state.tool_results)
    retrieval_hits = format_retrieval_hits(state.retrieval_hits)
    errors = [err.model_dump() for err in state.errors]
    run_info = _build_run_info(state)
    retrieval_summary = state.retrieval_summary or "(no retrieval summary)"

    return (
        answer,
        retrieval_summary,
        tool_cards,
        retrieval_hits,
        retrieval_events,
        trace_events,
        errors,
        run_info,
    )


def _load_text_file(path: str | None) -> str:
    if not path:
        return ""
    resolved = Path(path)
    if not resolved.exists():
        return ""
    return resolved.read_text(encoding="utf-8")


def _build_session_runner(*, playbook_text: str) -> GraphRunner:
    return GraphRunner(playbook_loader=lambda: playbook_text)


def _format_errors(state: AgentState) -> list[dict[str, Any]]:
    return [err.model_dump() for err in state.errors]


def tutor_start_session(
    profile: str,
    case_id: str,
    case_file: str,
    image_root: str,
    playbook_path: str | None,
) -> tuple[
    list[dict[str, str]],
    dict[str, Any],
    str,
    str | None,
    list[tuple[str, str]],
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if not os.environ.get("OPENAI_API_KEY"):
        return (
            [],
            {"error": "OPENAI_API_KEY is not set; cannot start tutor session."},
            "",
            None,
            [],
            "",
            [],
            [],
            "",
            [],
            [],
            [{"code": "OPENAI_API_KEY", "message": "OPENAI_API_KEY is not set."}],
        )

    from medical_tutor.medxpert import build_case_inputs, resolve_medxpert_case

    run_id = new_run_id("tutor_ui")
    playbook_text = _load_text_file(playbook_path) if playbook_path else ""
    runner = _build_session_runner(playbook_text=playbook_text)
    _TUTOR_RUNNERS[run_id] = runner

    case = resolve_medxpert_case(case_file=Path(case_file), case_id=case_id)
    case_prompt, resolved_image = build_case_inputs(case, image_root=Path(image_root))

    result = runner.run(
        question=case_prompt,
        image_path=resolved_image,
        conversation_history=[],
        run_id=run_id,
        student_profile=profile,
        student_attempt_count=0,
        case_prompt=case_prompt,
        case_label=case.label,
        tutor_hint_level=1,
        tutor_consecutive_wrong=0,
        tutor_last_grade=None,
        tutor_last_misconception=None,
    )

    session = {
        "run_id": run_id,
        "student_profile": profile,
        "case_id": case.case_id,
        "case_file": str(case_file),
        "image_root": str(image_root),
        "image_path": resolved_image,
        "case_prompt": case_prompt,
        "case_label": case.label,
        "messages": list(result.messages),
        "student_attempt_count": 0,
        "tutor_hint_level": result.tutor_hint_level,
        "tutor_consecutive_wrong": result.tutor_consecutive_wrong,
        "tutor_last_grade": result.tutor_last_grade,
        "tutor_last_misconception": result.tutor_last_misconception,
        "playbook_path": playbook_path,
    }

    config = load_config()
    run_dir = Path(config.output_dir) / run_id
    trace_events = load_trace_events(run_dir / "trace.jsonl")
    retrieval_events = load_retrieval_events(run_dir / "retrieval_artifacts.jsonl")

    return (
        format_chat_history(result.messages),
        session,
        case_prompt,
        resolved_image,
        format_tool_images(result.artifacts),
        _build_run_info(result),
        format_tool_cards(result.tool_calls, result.tool_results),
        format_retrieval_hits(result.retrieval_hits),
        result.retrieval_summary or "(no retrieval summary)",
        retrieval_events,
        trace_events,
        _format_errors(result),
    )


def tutor_send_reply(
    student_reply: str,
    session: dict[str, Any],
) -> tuple[
    str,
    list[dict[str, str]],
    dict[str, Any],
    list[tuple[str, str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    run_id = str(session.get("run_id") or "")
    if not run_id:
        return (
            "",
            [],
            {**session, "error": "Tutor session is not started yet."},
            [],
            [],
            [],
            "",
            [],
            [],
            [{"code": "session", "message": "Tutor session is not started yet."}],
        )

    runner = _TUTOR_RUNNERS.get(run_id)
    if runner is None:
        playbook_text = _load_text_file(session.get("playbook_path"))
        runner = _build_session_runner(playbook_text=playbook_text)
        _TUTOR_RUNNERS[run_id] = runner

    messages = list(session.get("messages") or [])
    profile = str(session.get("student_profile") or "")
    image_path = session.get("image_path")
    case_prompt = session.get("case_prompt")
    correct_label = session.get("case_label") or ""
    attempt_count = int(session.get("student_attempt_count") or 0) + 1

    assessment_grade = None
    assessment_misconception = None
    if correct_label:
        from medical_tutor.tutoring import assess_student_mcq_reply, extract_mcq_choice

        if extract_mcq_choice(student_reply):
            assessment = assess_student_mcq_reply(
                correct_label=str(correct_label),
                student_reply=student_reply,
            )
            assessment_grade = assessment.grade
            assessment_misconception = assessment.misconception

    result = runner.run(
        question=student_reply,
        image_path=str(image_path) if image_path else None,
        conversation_history=messages,
        run_id=run_id,
        student_profile=profile,
        student_attempt_count=attempt_count,
        case_prompt=str(case_prompt) if case_prompt else None,
        case_label=str(correct_label) if correct_label else None,
        student_reply_text=student_reply,
        student_reply_grade=assessment_grade,
        student_reply_misconception=assessment_misconception,
        tutor_hint_level=int(session.get("tutor_hint_level") or 1),
        tutor_consecutive_wrong=int(session.get("tutor_consecutive_wrong") or 0),
        tutor_last_grade=session.get("tutor_last_grade"),
        tutor_last_misconception=session.get("tutor_last_misconception"),
    )

    session_updated = dict(session)
    session_updated["messages"] = list(result.messages)
    session_updated["student_attempt_count"] = attempt_count
    session_updated["tutor_hint_level"] = result.tutor_hint_level
    session_updated["tutor_consecutive_wrong"] = result.tutor_consecutive_wrong
    session_updated["tutor_last_grade"] = result.tutor_last_grade
    session_updated["tutor_last_misconception"] = result.tutor_last_misconception

    config = load_config()
    run_dir = Path(config.output_dir) / run_id
    trace_events = load_trace_events(run_dir / "trace.jsonl")
    retrieval_events = load_retrieval_events(run_dir / "retrieval_artifacts.jsonl")

    return (
        "",
        format_chat_history(result.messages),
        session_updated,
        format_tool_images(result.artifacts),
        format_tool_cards(result.tool_calls, result.tool_results),
        format_retrieval_hits(result.retrieval_hits),
        result.retrieval_summary or "(no retrieval summary)",
        retrieval_events,
        trace_events,
        _format_errors(result),
    )


def tutor_reset(session: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    run_id = session.get("run_id")
    if run_id and run_id in _TUTOR_RUNNERS:
        _TUTOR_RUNNERS.pop(str(run_id), None)
    return [], {}


def _setup_logging() -> None:
    config = load_config()
    log_dir = Path(config.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ui.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def build_app() -> Any:
    import gradio as gr

    _setup_logging()

    with gr.Blocks() as app:
        gr.Markdown("# Medical Tutor")
        with gr.Tabs():
            with gr.Tab("Tutor"):
                gr.Markdown("Multi-turn Socratic tutoring for MedXpert cases (Stage A controller).")

                tutor_session = gr.State(value={})

                with gr.Row():
                    with gr.Column(scale=3):
                        profile = gr.Dropdown(
                            label="Student Profile",
                            choices=["novice", "medium", "expert"],
                            value="novice",
                        )
                        case_id = gr.Textbox(label="Case ID", value="MM-2000")
                        case_file = gr.Textbox(
                            label="Case File",
                            value="references/MedTutor-R1/code/Patient_simulate/MedXpert_patient_script_MM_dev.json",
                        )
                        image_root = gr.Textbox(
                            label="Image Root", value="cache/medxpert_images_mm_dev"
                        )
                        playbook_path = gr.Textbox(
                            label="Playbook Path (optional)",
                            value="Medical_Tutor/playbooks/phase1_tool_use.txt",
                        )
                        start_btn = gr.Button("Start Tutor Session", variant="primary")
                        reset_btn = gr.Button("Reset Session", variant="secondary")

                        run_info = gr.Markdown(label="Run Info")
                        case_prompt = gr.Markdown(label="Case Prompt (question + options)")
                        case_image = gr.Image(label="Case Image", type="filepath")

                    with gr.Column(scale=5):
                        chatbot = gr.Chatbot(label="Tutor Conversation")
                        student_reply = gr.Textbox(
                            label="Student Reply",
                            lines=1,
                            placeholder="Type an answer choice (e.g., A) or your reasoning...",
                        )
                        send_btn = gr.Button("Send Reply", variant="primary")

                        retrieval_summary = gr.Markdown(label="Retrieval Summary")
                        tool_images = gr.Gallery(label="Tool Images", columns=2, type="filepath")
                        tool_cards = gr.JSON(label="Tool Cards")
                        retrieval_hits = gr.JSON(label="Retrieval Hits")
                        retrieval_events = gr.JSON(label="Retrieval Events")
                        trace_events = gr.JSON(label="Trace Events")
                        errors = gr.JSON(label="Errors")

                start_btn.click(
                    fn=tutor_start_session,
                    inputs=[profile, case_id, case_file, image_root, playbook_path],
                    outputs=[
                        chatbot,
                        tutor_session,
                        case_prompt,
                        case_image,
                        tool_images,
                        run_info,
                        tool_cards,
                        retrieval_hits,
                        retrieval_summary,
                        retrieval_events,
                        trace_events,
                        errors,
                    ],
                )

                send_btn.click(
                    fn=tutor_send_reply,
                    inputs=[student_reply, tutor_session],
                    outputs=[
                        student_reply,
                        chatbot,
                        tutor_session,
                        tool_images,
                        tool_cards,
                        retrieval_hits,
                        retrieval_summary,
                        retrieval_events,
                        trace_events,
                        errors,
                    ],
                )

                reset_btn.click(
                    fn=tutor_reset, inputs=[tutor_session], outputs=[chatbot, tutor_session]
                )

            with gr.Tab("Debug Run"):
                gr.Markdown(
                    "One-shot debug run: upload an image (optional), enter a question, and review tools and retrieval."
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        image = gr.Image(label="Image", type="filepath")
                        question = gr.Textbox(
                            label="Question",
                            lines=3,
                            placeholder="Ask a medical question about the image...",
                        )
                        run_btn = gr.Button("Run", variant="primary")
                        answer = gr.Markdown(label="Answer")
                        debug_retrieval_summary = gr.Markdown(label="Retrieval Summary")

                    with gr.Column(scale=4):
                        debug_run_info = gr.Markdown(label="Run Info")
                        debug_tool_cards = gr.JSON(label="Tool Cards")
                        debug_retrieval_hits = gr.JSON(label="Retrieval Hits")
                        debug_retrieval_events = gr.JSON(label="Retrieval Events")
                        debug_trace_events = gr.JSON(label="Trace Events")
                        debug_errors = gr.JSON(label="Errors")

                run_btn.click(
                    fn=run_single_turn,
                    inputs=[question, image],
                    outputs=[
                        answer,
                        debug_retrieval_summary,
                        debug_tool_cards,
                        debug_retrieval_hits,
                        debug_retrieval_events,
                        debug_trace_events,
                        debug_errors,
                        debug_run_info,
                    ],
                )

    return app


def launch() -> None:
    app = build_app()
    import gradio as gr

    app.launch(theme=gr.themes.Soft())
