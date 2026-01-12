# ABOUTME: Writes per-run metadata snapshots for debugging and reproducibility.
# ABOUTME: Persists non-secret config fields (models/paths/playbook) to run_config.json.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medical_tutor.config import Config


def write_run_config(*, output_dir: str | Path, run_id: str, config: Config) -> Path:
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = run_dir / "run_config.json"

    payload: dict[str, Any] = {
        "run_id": run_id,
        "orchestrator_model": config.orchestrator_model,
        "answer_model": config.answer_model,
        "ace_model": config.ace_model,
        "bm25_path": config.bm25_path,
        "text_index_path": config.text_index_path,
        "image_index_path": config.image_index_path,
        "output_dir": str(config.output_dir),
        "max_history_messages": int(config.max_history_messages),
        "playbook_path": config.playbook_path,
        "langsmith_enabled": bool(config.langsmith_enabled),
        "langsmith_project": config.langsmith_project,
        "langsmith_endpoint": config.langsmith_endpoint,
    }

    run_config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_config_path
