"""Prompt templates for Medical_Tutor (ported from legacy system)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

TOOL_POLICY = (
    "Tool policy:\n"
    "- Use at most one tool per turn, only if needed.\n"
    "- If visual evidence is unclear, prefer zoom/segment/enhance.\n"
    "- If external medical knowledge is needed, retrieve first.\n"
    "- Call web_browser only if retrieval is insufficient or missing key facts.\n"
    "- web_browser accepts either a query (for search) or a url (for direct visit), not both.\n"
    "- Use web_browser query to discover sources; use url when a specific page is known.\n"
    "- Use retrieve when you want corpus-grounded info; use web_browser for up-to-date or external sources.\n"
    "- Only call tools listed below.\n"
    "- zoom bbox values <= 1.0 are normalized.\n"
    "- retrieve modality must be 'text' or 'image'.\n\n"
)

ORCHESTRATOR_INSTRUCTIONS = (
    "You are the orchestrator for a medical visual chat system.\n"
    "Decide the next action. Output JSON only with one of:\n"
    '{"type":"tool","name":"tool_name","arguments":{...}}\n'
    "or\n"
    '{"type":"answer","answer":"(optional short rationale)"}\n'
    "No extra keys or text.\n\n"
)

ANSWER_PROMPT_BASE = (
    "You are a medical imaging expert assistant. Use the image and any provided context.\n"
    "Answer in plain text (no tool tags). If evidence is insufficient, say exactly:\n"
    '"Insufficient evidence to answer confidently."'
)

PLAYBOOK_INSTRUCTIONS = (
    "Playbook:\n"
    "- Apply relevant strategies.\n"
    "- Avoid listed mistakes.\n"
    "- Do not mention bullet IDs.\n\n"
)


def format_playbook(playbook: str | None) -> str:
    if not playbook or not playbook.strip():
        return "(no playbook loaded)"
    return f"{PLAYBOOK_INSTRUCTIONS}{playbook.strip()}\n"


def format_tool_specs(tool_specs: Iterable[Mapping[str, object]]) -> str:
    lines = []
    for tool in tool_specs:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        args = tool.get("arguments", {})
        lines.append(f"- {name}: {desc}; args: {args}")
    return "\n".join(lines) if lines else "(no tools configured)"


def wrap_gemma_chat(prompt: str) -> str:
    return f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n<start_of_turn>model\n"
