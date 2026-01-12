# ABOUTME: Runs the LangGraph runtime with configured tools, retrieval, tracing, and artifact storage.
# ABOUTME: Provides a simple entrypoint for both one-shot runs and multi-turn tutoring sessions.
"""Runtime runner for Medical_Tutor LangGraph workflow."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from medical_tutor.artifacts import ArtifactStore
from medical_tutor.config import Config, load_config
from medical_tutor.contracts import AgentState, ErrorRecord, new_run_id
from medical_tutor.ops.run_metadata import write_run_config
from medical_tutor.ops.tracing import TraceLogger
from medical_tutor.retrieval import NullRetriever, Retriever
from medical_tutor.runtime.graph import build_graph
from medical_tutor.tools.registry import ToolRegistry, build_default_registry


class NullTracer:
    def log_event(self, **_: Any) -> None:
        return None

    def log_retrieval(self, *_: Any, **__: Any) -> None:
        return None


class TracerProxy:
    def __init__(self, tracer: Any | None = None) -> None:
        self._tracer = tracer or NullTracer()

    def set_logger(self, tracer: Any | None) -> None:
        self._tracer = tracer or NullTracer()

    def log_event(self, **payload: Any) -> None:
        self._tracer.log_event(**payload)

    def log_retrieval(self, hits: list[Any]) -> None:
        if hasattr(self._tracer, "log_retrieval"):
            self._tracer.log_retrieval(hits)


@contextmanager
def _langsmith_env(config: Config, api_key: str) -> Any:
    updates = {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": api_key,
        "LANGSMITH_PROJECT": config.langsmith_project,
        "LANGSMITH_ENDPOINT": config.langsmith_endpoint,
    }
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        if value:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


class GraphRunner:
    def __init__(
        self,
        config: Config | None = None,
        retriever: Any | None = None,
        tool_registry: ToolRegistry | None = None,
        playbook_loader: Any | None = None,
        tracer: Any | None = None,
    ) -> None:
        self.config = config or load_config()
        self.retriever = retriever or self._build_retriever()
        self.tool_registry = tool_registry or build_default_registry()
        self._external_playbook_loader = playbook_loader is not None
        self._playbook_text = ""
        self.playbook_loader = playbook_loader or (lambda: self._playbook_text)
        self._external_tracer = tracer is not None
        self.tracer = TracerProxy(tracer)
        self.graph = build_graph(
            config=self.config,
            retriever=self.retriever,
            tool_registry=self.tool_registry,
            playbook_loader=self.playbook_loader,
            tracer=self.tracer,
        )

    def run(
        self,
        question: str,
        image_path: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        *,
        run_id: str | None = None,
        student_profile: str | None = None,
        student_attempt_count: int | None = None,
        case_prompt: str | None = None,
        case_label: str | None = None,
        student_reply_text: str | None = None,
        student_reply_grade: str | None = None,
        student_reply_misconception: str | None = None,
        tutor_hint_level: int | None = None,
        tutor_consecutive_wrong: int | None = None,
        tutor_last_grade: str | None = None,
        tutor_last_misconception: str | None = None,
    ) -> AgentState:
        playbook_error: str | None = None
        if not self._external_playbook_loader:
            self._playbook_text = ""
            if self.config.playbook_path:
                try:
                    self._playbook_text = Path(self.config.playbook_path).read_text(
                        encoding="utf-8"
                    )
                except Exception as exc:
                    playbook_error = (
                        f"Failed to load playbook from {self.config.playbook_path}: {exc}"
                    )

        messages = list(conversation_history or [])
        if image_path and not messages:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question},
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": question})

        messages = self._apply_history_window(messages)

        state = AgentState(
            messages=messages,
            image=image_path,
            run_id=run_id or new_run_id("run"),
            student_profile=student_profile,
            student_attempt_count=max(0, int(student_attempt_count or 0)),
            case_prompt=case_prompt,
            case_label=case_label,
            student_reply_text=student_reply_text,
            student_reply_grade=student_reply_grade,
            student_reply_misconception=student_reply_misconception,
            tutor_hint_level=int(tutor_hint_level) if tutor_hint_level is not None else 1,
            tutor_consecutive_wrong=int(tutor_consecutive_wrong or 0),
            tutor_last_grade=tutor_last_grade,
            tutor_last_misconception=tutor_last_misconception,
        )
        if playbook_error:
            state.errors.append(ErrorRecord(code="playbook", message=playbook_error))
        if hasattr(self.retriever, "clear_cache"):
            self.retriever.clear_cache()
        artifact_store = ArtifactStore(self.config.output_dir, state.run_id)
        if self.tool_registry is not None:
            self.tool_registry.set_context(artifact_store=artifact_store, retriever=self.retriever)
        if not self._external_tracer:
            self.tracer.set_logger(TraceLogger(self.config.output_dir, state.run_id))
        write_run_config(output_dir=self.config.output_dir, run_id=state.run_id, config=self.config)
        payload = state.model_dump()
        run_config = {"configurable": {"thread_id": state.run_id}}

        langsmith_warning: str | None = None
        result: Any | None = None

        if getattr(self.config, "langsmith_enabled", False):
            langsmith_key = self.config.langsmith_api_key or os.environ.get("LANGSMITH_API_KEY")
            if not langsmith_key:
                langsmith_warning = "LangSmith enabled but LANGSMITH_API_KEY is not set; continuing without LangSmith."
            else:
                try:
                    from langsmith import traceable  # pyright: ignore[reportMissingImports]
                except Exception as exc:
                    langsmith_warning = f"LangSmith enabled but could not import langsmith: {exc}"
                    langsmith_key = None

            if langsmith_key:
                try:
                    with _langsmith_env(self.config, langsmith_key):

                        @traceable(name="medical_tutor.run", run_type="chain")
                        def _invoke() -> Any:
                            return self.graph.invoke(payload, config=run_config)

                        result = _invoke()
                except Exception as exc:
                    langsmith_warning = f"LangSmith tracing failed: {exc}"
                    result = None

        if result is None:
            result = self.graph.invoke(payload, config=run_config)

        state_out = cast(AgentState, AgentState.model_validate(result))
        if langsmith_warning:
            state_out.errors.append(ErrorRecord(code="langsmith", message=langsmith_warning))
        return state_out

    def _apply_history_window(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        max_history = max(1, int(self.config.max_history_messages))
        if len(messages) <= max_history:
            return messages
        trimmed = len(messages) - max_history
        kept = messages[-max_history:]
        notice = {
            "role": "system",
            "content": f"[TRIMMED HISTORY] {trimmed} earlier messages removed.",
        }
        return [notice] + kept

    def _build_retriever(self) -> Any:
        try:
            return Retriever(
                bm25_path=self.config.bm25_path,
                text_index_path=self.config.text_index_path,
                image_index_path=self.config.image_index_path,
            )
        except Exception as exc:  # pragma: no cover - missing deps or indices
            return NullRetriever(str(exc))
