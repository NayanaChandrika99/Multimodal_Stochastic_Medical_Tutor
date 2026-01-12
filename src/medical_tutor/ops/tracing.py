"""Trace logging utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TraceLogger:
    def __init__(self, output_dir: str | Path, run_id: str) -> None:
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.trace_path = self.output_dir / run_id / "trace.jsonl"
        self.retrieval_path = self.output_dir / run_id / "retrieval_artifacts.jsonl"
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, **payload: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **self._sanitize(payload),
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_retrieval(self, hits: list[Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hits": self._sanitize(hits),
        }
        with self.retrieval_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _sanitize(self, payload: Any) -> Any:
        try:
            json.dumps(payload)
            return payload
        except TypeError:
            if isinstance(payload, dict):
                return {k: self._sanitize(v) for k, v in payload.items()}
            if isinstance(payload, list):
                return [self._sanitize(v) for v in payload]
            return str(payload)
