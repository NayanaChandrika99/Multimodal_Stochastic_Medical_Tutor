"""Configuration defaults and environment overrides for Medical_Tutor."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

ENV_PREFIX = "MEDTUTOR_"


class Config(BaseModel):
    orchestrator_model: str = "gpt-4o-mini"
    answer_model: str = "google/medgemma-4b-it"
    ace_model: str = "gpt-4o-mini"
    bm25_path: str = "data/retrieval/v3/bm25.pkl"
    text_index_path: str = "data/retrieval/v3/text_enhanced_new"
    image_index_path: str = "data/retrieval/v3/images_pmc_vqa1"
    output_dir: str = "Medical_Tutor/outputs"
    max_history_messages: int = 20
    playbook_path: str | None = None
    langsmith_enabled: bool = False
    langsmith_project: str = "medical-tutor"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str | None = None

    def as_lines(self) -> str:
        lines = [
            f"orchestrator_model={self.orchestrator_model}",
            f"answer_model={self.answer_model}",
            f"ace_model={self.ace_model}",
            f"bm25_path={self.bm25_path}",
            f"text_index_path={self.text_index_path}",
            f"image_index_path={self.image_index_path}",
            f"output_dir={self.output_dir}",
            f"max_history_messages={self.max_history_messages}",
            f"playbook_path={self.playbook_path or '(unset)'}",
            f"langsmith_enabled={self.langsmith_enabled}",
            f"langsmith_project={self.langsmith_project}",
            f"langsmith_endpoint={self.langsmith_endpoint}",
            f"langsmith_api_key={'(set)' if self.langsmith_api_key else '(unset)'}",
        ]
        return "\n".join(lines)


def _env_override(key: str, default: str) -> str:
    return os.getenv(f"{ENV_PREFIX}{key}", default)


def _env_override_bool(key: str, default: bool) -> bool:
    value = os.getenv(f"{ENV_PREFIX}{key}")
    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized not in {"0", "false", "no", "off", ""}


def _env_override_optional(key: str) -> str | None:
    value = os.getenv(f"{ENV_PREFIX}{key}")
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _resolve_default_path(path: str) -> str:
    candidate = Path(path)
    if candidate.exists():
        return path
    alternate = Path("med-visual-tutor") / path
    if alternate.exists():
        return str(alternate)
    return path


def load_config() -> Config:
    defaults = Config()
    overrides: dict[str, object] = {
        "orchestrator_model": _env_override("ORCHESTRATOR_MODEL", defaults.orchestrator_model),
        "answer_model": _env_override("ANSWER_MODEL", defaults.answer_model),
        "ace_model": _env_override("ACE_MODEL", defaults.ace_model),
        "bm25_path": _resolve_default_path(_env_override("BM25_PATH", defaults.bm25_path)),
        "text_index_path": _resolve_default_path(
            _env_override("TEXT_INDEX_PATH", defaults.text_index_path)
        ),
        "image_index_path": _resolve_default_path(
            _env_override("IMAGE_INDEX_PATH", defaults.image_index_path)
        ),
        "output_dir": _env_override("OUTPUT_DIR", defaults.output_dir),
        "max_history_messages": int(
            _env_override("MAX_HISTORY_MESSAGES", str(defaults.max_history_messages))
        ),
        "playbook_path": _env_override_optional("PLAYBOOK_PATH"),
        "langsmith_enabled": _env_override_bool("LANGSMITH_ENABLED", defaults.langsmith_enabled),
        "langsmith_project": _env_override("LANGSMITH_PROJECT", defaults.langsmith_project),
        "langsmith_endpoint": _env_override("LANGSMITH_ENDPOINT", defaults.langsmith_endpoint),
        "langsmith_api_key": _env_override_optional("LANGSMITH_API_KEY"),
    }
    return Config(**overrides)
