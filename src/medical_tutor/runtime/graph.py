# ABOUTME: Builds the LangGraph state machine for the Medical_Tutor runtime.
# ABOUTME: Routes between tutor decisions, tool execution, and answer generation, persisting a traceable state.
"""LangGraph graph construction for Medical_Tutor runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.memory import MemorySaver  # pyright: ignore[reportMissingImports]
from langgraph.graph import END, StateGraph  # pyright: ignore[reportMissingImports]

from medical_tutor.config import Config
from medical_tutor.runtime import nodes


def build_graph(
    config: Config,
    retriever: Any | None,
    tool_registry: Any | None,
    playbook_loader: Callable[[], str],
    tracer: Any | None,
) -> Any:
    graph = StateGraph(dict)

    graph.add_node(
        "pre_retrieve",
        lambda state: nodes.pre_retrieve(state, config, retriever, tracer),
    )
    graph.add_node(
        "decide",
        lambda state: nodes.decide(state, config, tool_registry, playbook_loader(), tracer),
    )
    graph.add_node(
        "tool_exec",
        lambda state: nodes.tool_exec(state, tool_registry, tracer),
    )
    graph.add_node(
        "answer",
        lambda state: nodes.answer(state, config, tracer),
    )
    graph.add_node(
        "respond",
        lambda state: nodes.respond(state, tracer),
    )

    graph.set_entry_point("pre_retrieve")
    graph.add_edge("pre_retrieve", "decide")

    def route_orchestrator(state: dict[str, Any]) -> str:
        return state.get("next_action") or "answer"

    def route_answer(state: dict[str, Any]) -> str:
        action = state.get("next_action")
        if action == "retry":
            return "decide"
        return "respond"

    graph.add_conditional_edges(
        "decide",
        route_orchestrator,
        {
            "tool": "tool_exec",
            "answer": "answer",
            "respond": "respond",
        },
    )

    graph.add_edge("tool_exec", "decide")
    graph.add_conditional_edges("answer", route_answer, {"decide": "decide", "respond": "respond"})
    graph.add_edge("respond", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
