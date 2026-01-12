# ABOUTME: Defines tutoring-specific action schemas and gating rules for Socratic tutoring.
# ABOUTME: Provides student reply assessment helpers used to update session tutoring state.

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from medical_tutor.contracts import AgentState

TutorActionType = Literal[
    "ASK_PROBE",
    "HINT",
    "MICROLESSON",
    "QUIZ",
    "REQUEST_TOOL",
    "REVEAL_ANSWER",
    "SAFETY_REFUSE",
]


class TutorAction(BaseModel):
    type: TutorActionType
    arguments: dict = Field(default_factory=dict)


StudentGrade = Literal["wrong", "partial", "correct"]


class StudentReplyAssessment(BaseModel):
    grade: StudentGrade
    misconception: str | None = None


def apply_reveal_gate(state: AgentState, action: TutorAction) -> TutorAction:
    if action.type != "REVEAL_ANSWER":
        return action
    if int(state.student_attempt_count) <= 0:
        return TutorAction(type="ASK_PROBE", arguments={})
    return action


_MCQ_BARE_CHOICE_RE = re.compile(r"^\s*([A-E])\s*[.)]?\s*$", re.IGNORECASE)
_MCQ_PAREN_CHOICE_RE = re.compile(r"\(([A-E])\)", re.IGNORECASE)
_MCQ_EXPLICIT_ANSWER_RE = re.compile(r"\banswer\s*[:\-]\s*([A-E])\b", re.IGNORECASE)
_MCQ_EXPLICIT_ANSWER_IS_RE = re.compile(r"\banswer\s+is\s*\(?([A-E])\)?\b", re.IGNORECASE)
_MCQ_EXPLICIT_PICK_RE = re.compile(r"\b(?:pick|choose|select)\s*\(?([A-E])\)?\b", re.IGNORECASE)


def extract_mcq_choice(student_reply: str) -> str | None:
    reply = (student_reply or "").strip()
    if not reply:
        return None

    bare_match = _MCQ_BARE_CHOICE_RE.match(reply)
    if bare_match:
        return str(bare_match.group(1)).upper()

    paren_matches = _MCQ_PAREN_CHOICE_RE.findall(reply)
    if paren_matches:
        if len(paren_matches) == 1:
            return str(paren_matches[-1]).upper()
        lowered = reply.lower()
        selection_signals = ("pick", "choose", "select", "answer is", "answer:")
        if any(signal in lowered for signal in selection_signals):
            return str(paren_matches[-1]).upper()
        return None

    for pattern in (_MCQ_EXPLICIT_ANSWER_RE, _MCQ_EXPLICIT_ANSWER_IS_RE, _MCQ_EXPLICIT_PICK_RE):
        matches = pattern.findall(reply)
        if matches:
            return str(matches[-1]).upper()

    return None


def assess_student_mcq_reply(*, correct_label: str, student_reply: str) -> StudentReplyAssessment:
    correct = (correct_label or "").strip().upper()
    if not correct or len(correct) != 1:
        return StudentReplyAssessment(grade="wrong", misconception="Missing correct_label.")

    chosen = extract_mcq_choice(student_reply)
    if not chosen:
        return StudentReplyAssessment(grade="wrong", misconception="No answer choice found.")

    if chosen == correct:
        return StudentReplyAssessment(grade="correct")
    return StudentReplyAssessment(grade="wrong")


def student_requested_reveal(student_reply_text: str | None) -> bool:
    text = (student_reply_text or "").strip().lower()
    if not text:
        return False

    if "do not reveal" in text or "don't reveal" in text or "dont reveal" in text:
        return False
    if "don't tell me the answer" in text or "do not tell me the answer" in text:
        return False

    if "reveal" in text and ("answer" in text or "solution" in text or "correct" in text):
        return True
    if "give me the answer" in text:
        return True
    if "what is the answer" in text or "what's the answer" in text:
        return True
    if "tell me the answer" in text:
        return True

    return False


def student_requested_image_description(student_reply_text: str | None) -> bool:
    text = (student_reply_text or "").strip().lower()
    if not text:
        return False

    if "do not describe" in text or "don't describe" in text or "dont describe" in text:
        return False

    keywords = (
        "what do you see",
        "what's in the image",
        "what is in the image",
        "what is shown",
        "describe",
        "abnormal",
        "what looks wrong",
        "what is wrong",
        "findings",
        "interpret",
    )
    return any(keyword in text for keyword in keywords)


def update_tutor_state_from_assessment(state: AgentState) -> None:
    grade = state.student_reply_grade
    if not grade:
        return
    state.tutor_last_grade = grade
    if grade == "correct":
        state.tutor_consecutive_wrong = 0
        state.tutor_hint_level = 1
        state.tutor_last_misconception = None
        return
    if state.student_reply_misconception:
        state.tutor_last_misconception = state.student_reply_misconception
    if grade == "wrong":
        state.tutor_consecutive_wrong += 1
        state.tutor_hint_level = min(3, max(2, state.tutor_hint_level + 1))
        return
    if grade == "partial":
        state.tutor_consecutive_wrong = max(state.tutor_consecutive_wrong, 1)
        state.tutor_hint_level = min(3, max(2, state.tutor_hint_level))


def apply_assessment_gate(state: AgentState, action: TutorAction) -> TutorAction:
    grade = state.tutor_last_grade or state.student_reply_grade
    if action.type == "HINT":
        return _hint_action_from_state(state, action=action)
    if grade == "wrong" and action.type in {"REVEAL_ANSWER"}:
        return _hint_action_from_state(state, action=action)
    if grade == "partial" and action.type in {"REVEAL_ANSWER"}:
        return _hint_action_from_state(state, action=action)
    return action


def _hint_action_from_state(state: AgentState, *, action: TutorAction | None = None) -> TutorAction:
    level = min(3, max(1, int(state.tutor_hint_level)))
    content = None
    if action and isinstance(action.arguments, dict):
        content = action.arguments.get("content")
    if not isinstance(content, str) or not content.strip():
        content = state.tutor_last_misconception or "Focus on the most discriminating finding."
    return TutorAction(type="HINT", arguments={"content": content, "level": level})
