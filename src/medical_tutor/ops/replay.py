"""Replay trace logs for debugging."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class ReplayRunner:
    def __init__(self, trace_path: str | Path) -> None:
        self.trace_path = Path(trace_path)

    def run(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for event in self._read_events():
            events.append(event)
        return events

    def _read_events(self) -> Iterable[dict[str, Any]]:
        with self.trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
