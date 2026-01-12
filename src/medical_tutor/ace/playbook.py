# ABOUTME: Parses and edits ACE playbooks with bullet IDs and helpful/harmful counters.
# ABOUTME: Applies curator operations (ADD/UPDATE/DELETE) and supports bullet ID inference.

from __future__ import annotations

import json
import re
from typing import Any

_PLAYBOOK_LINE_RE = re.compile(r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)")
_BULLET_ID_RE = re.compile(r"\[([a-z]{2,5}-\d{5})\]")


def parse_playbook_line(line: str) -> dict[str, Any] | None:
    match = _PLAYBOOK_LINE_RE.match(line.strip())
    if not match:
        return None
    return {
        "id": match.group(1),
        "helpful": int(match.group(2)),
        "harmful": int(match.group(3)),
        "content": match.group(4),
        "raw_line": line,
    }


def format_playbook_line(bullet_id: str, helpful: int, harmful: int, content: str) -> str:
    return f"[{bullet_id}] helpful={helpful} harmful={harmful} :: {content}"


def extract_bullet_ids(text: str | None) -> list[str]:
    if not text:
        return []
    matches = _BULLET_ID_RE.findall(str(text))
    seen: set[str] = set()
    unique: list[str] = []
    for match in matches:
        if match in seen:
            continue
        seen.add(match)
        unique.append(match)
    return unique


def get_next_global_id(playbook_text: str) -> int:
    max_id = 0
    for line in playbook_text.strip().splitlines():
        parsed = parse_playbook_line(line)
        if not parsed:
            continue
        match = re.search(r"-(\d+)$", str(parsed["id"]))
        if not match:
            continue
        max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def _normalize_section_name(section_name: str) -> str:
    return section_name.strip().lower().replace(" ", "_").replace("&", "and")


def get_section_slug(section_name: str) -> str:
    slugs = {
        "common_mistakes": "err",
        "problem_solving_heuristics": "ctx",
        "context_indicators": "ind",
        "formulas_and_calculations": "calc",
        "others": "oth",
        "general": "gen",
    }
    return slugs.get(_normalize_section_name(section_name), "oth")


def extract_playbook_bullets(playbook_text: str, bullet_ids: list[str]) -> str:
    if not bullet_ids:
        return "(No bullets used by generator)"
    wanted = set(bullet_ids)
    found: list[str] = []
    for line in playbook_text.strip().splitlines():
        parsed = parse_playbook_line(line)
        if not parsed:
            continue
        if str(parsed["id"]) in wanted:
            found.append(
                format_playbook_line(
                    str(parsed["id"]),
                    int(parsed["helpful"]),
                    int(parsed["harmful"]),
                    str(parsed["content"]),
                )
            )
    if not found:
        return "(Generator referenced bullet IDs but none were found in playbook)"
    return "\n".join(found)


def update_bullet_counts(playbook_text: str, bullet_tags: list[dict[str, str]]) -> str:
    tag_map: dict[str, str] = {}
    for tag in bullet_tags:
        if not isinstance(tag, dict):
            continue
        bullet_id = str(tag.get("id") or "").strip()
        tag_value = str(tag.get("tag") or "").strip().lower()
        if bullet_id and tag_value in {"helpful", "harmful", "neutral"}:
            tag_map[bullet_id] = tag_value

    if not tag_map:
        return playbook_text

    updated: list[str] = []
    for line in playbook_text.strip().splitlines():
        if line.strip().startswith("##") or not line.strip():
            updated.append(line)
            continue
        parsed = parse_playbook_line(line)
        if not parsed:
            updated.append(line)
            continue
        bullet_id = str(parsed["id"])
        if bullet_id not in tag_map:
            updated.append(line)
            continue
        helpful = int(parsed["helpful"])
        harmful = int(parsed["harmful"])
        tag_value = tag_map[bullet_id]
        if tag_value == "helpful":
            helpful += 1
        elif tag_value == "harmful":
            harmful += 1
        updated.append(format_playbook_line(bullet_id, helpful, harmful, str(parsed["content"])))
    return "\n".join(updated) + "\n"


def get_playbook_stats(playbook_text: str) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "total_bullets": 0,
        "high_performing": 0,
        "problematic": 0,
        "unused": 0,
        "by_section": {},
    }
    current_section = "general"
    for line in playbook_text.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("##"):
            current_section = stripped[2:].strip() or "general"
            continue
        parsed = parse_playbook_line(line)
        if not parsed:
            continue
        helpful = int(parsed["helpful"])
        harmful = int(parsed["harmful"])
        stats["total_bullets"] += 1
        if helpful > 5 and harmful < 2:
            stats["high_performing"] += 1
        elif harmful >= helpful and harmful > 0:
            stats["problematic"] += 1
        elif helpful + harmful == 0:
            stats["unused"] += 1

        by_section = stats["by_section"]
        if current_section not in by_section:
            by_section[current_section] = {"count": 0, "helpful": 0, "harmful": 0}
        by_section[current_section]["count"] += 1
        by_section[current_section]["helpful"] += helpful
        by_section[current_section]["harmful"] += harmful
    return stats


def extract_json_from_text(text: str, _json_key: str | None = None) -> Any:
    del _json_key
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    fenced = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    for candidate in fenced:
        try:
            return json.loads(candidate.strip())
        except Exception:
            continue

    def _find_json_objects(raw: str) -> list[str]:
        objects: list[str] = []
        i = 0
        while i < len(raw):
            if raw[i] != "{":
                i += 1
                continue
            start = i
            brace_count = 1
            i += 1
            while i < len(raw) and brace_count > 0:
                if raw[i] == "{":
                    brace_count += 1
                elif raw[i] == "}":
                    brace_count -= 1
                elif raw[i] == '"':
                    i += 1
                    while i < len(raw) and raw[i] != '"':
                        if raw[i] == "\\":
                            i += 1
                        i += 1
                i += 1
            if brace_count == 0:
                objects.append(raw[start:i])
        return objects

    for candidate in _find_json_objects(text):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def apply_curator_operations(
    playbook_text: str,
    operations: list[dict[str, Any]],
    next_id: int,
) -> tuple[str, int]:
    lines = playbook_text.strip().splitlines()

    sections: set[str] = set()
    bullet_map: dict[str, tuple[int, dict[str, Any]]] = {}
    current_section = "general"
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("##"):
            current_section = _normalize_section_name(stripped[2:].strip() or "general")
            sections.add(current_section)
            continue
        parsed = parse_playbook_line(line)
        if parsed:
            bullet_map[str(parsed["id"])] = (idx, parsed)

    bullets_to_add: list[tuple[str, str]] = []
    bullets_to_update: dict[str, str] = {}
    bullets_to_delete: set[str] = set()

    for op in operations:
        if not isinstance(op, dict):
            continue
        op_type = str(op.get("type") or "").strip().upper()
        if op_type == "ADD":
            section_raw = str(op.get("section") or "others")
            section = _normalize_section_name(section_raw or "others")
            if section not in sections and section != "general":
                section = "others"
            slug = get_section_slug(section)
            new_id = f"{slug}-{next_id:05d}"
            next_id += 1
            content = str(op.get("content") or "").strip()
            if not content:
                continue
            bullets_to_add.append((section, format_playbook_line(new_id, 0, 0, content)))
        elif op_type == "UPDATE":
            bullet_id = str(op.get("bullet_id") or "").strip()
            content = str(op.get("content") or "").strip()
            if bullet_id and content and bullet_id in bullet_map:
                bullets_to_update[bullet_id] = content
        elif op_type == "DELETE":
            bullet_id = str(op.get("bullet_id") or "").strip()
            if not bullet_id or bullet_id not in bullet_map:
                continue
            _, parsed = bullet_map[bullet_id]
            helpful = int(parsed["helpful"])
            harmful = int(parsed["harmful"])
            if harmful > helpful + 2:
                bullets_to_delete.add(bullet_id)

    rebuilt: list[str] = []
    for line in lines:
        parsed = parse_playbook_line(line)
        if parsed:
            bullet_id = str(parsed["id"])
            if bullet_id in bullets_to_delete:
                continue
            if bullet_id in bullets_to_update:
                rebuilt.append(
                    format_playbook_line(
                        bullet_id,
                        int(parsed["helpful"]),
                        int(parsed["harmful"]),
                        bullets_to_update[bullet_id],
                    )
                )
            else:
                rebuilt.append(line)
        else:
            rebuilt.append(line)

    final_lines: list[str] = []
    current_section_key: str | None = None
    remaining_adds = list(bullets_to_add)
    for line in rebuilt:
        stripped = line.strip()
        if stripped.startswith("##"):
            if current_section_key:
                section_adds = [b for s, b in remaining_adds if s == current_section_key]
                final_lines.extend(section_adds)
                remaining_adds = [(s, b) for s, b in remaining_adds if s != current_section_key]
            current_section_key = _normalize_section_name(stripped[2:].strip() or "general")
        final_lines.append(line)

    if current_section_key:
        section_adds = [b for s, b in remaining_adds if s == current_section_key]
        final_lines.extend(section_adds)
        remaining_adds = [(s, b) for s, b in remaining_adds if s != current_section_key]

    if remaining_adds:
        others_bullets = [b for _, b in remaining_adds]
        final_lines.extend(others_bullets)

    return "\n".join(final_lines) + "\n", next_id


def auto_prune_harmful_bullets(playbook_text: str) -> str:
    kept: list[str] = []
    for line in playbook_text.strip().splitlines():
        parsed = parse_playbook_line(line)
        if parsed:
            helpful = int(parsed["helpful"])
            harmful = int(parsed["harmful"])
            if harmful > helpful + 2:
                continue
        kept.append(line)
    return "\n".join(kept) + "\n"


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "then",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def _tokenize_for_overlap(text: str) -> set[str]:
    tokens: list[str] = []
    for raw in re.split(r"[^a-zA-Z0-9_]+", text.lower()):
        raw = raw.strip()
        if not raw or raw in _STOPWORDS or len(raw) <= 2:
            continue
        tokens.append(raw)
    return set(tokens)


def infer_bullet_ids_from_episode(
    *,
    playbook: str,
    episode_text: str,
    top_k: int = 8,
) -> list[str]:
    if top_k <= 0:
        return []
    if not playbook.strip() or not episode_text.strip():
        return []

    episode_tokens = _tokenize_for_overlap(episode_text)
    if not episode_tokens:
        return []

    scored: list[tuple[float, str]] = []
    for line in playbook.splitlines():
        parsed = parse_playbook_line(line)
        if not parsed:
            continue
        bullet_tokens = _tokenize_for_overlap(str(parsed.get("content", "")))
        if not bullet_tokens:
            continue
        overlap = len(episode_tokens & bullet_tokens)
        if overlap <= 0:
            continue
        score = overlap / (len(bullet_tokens) ** 0.5)
        scored.append((score, str(parsed.get("id", ""))))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    seen: set[str] = set()
    inferred: list[str] = []
    for _, bullet_id in scored:
        if not bullet_id or bullet_id in seen:
            continue
        seen.add(bullet_id)
        inferred.append(bullet_id)
        if len(inferred) >= top_k:
            break
    return inferred
