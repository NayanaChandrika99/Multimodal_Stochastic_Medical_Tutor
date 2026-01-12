"""Artifact storage for tool outputs."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from medical_tutor.contracts import ArtifactRef


class ArtifactStore:
    def __init__(self, base_dir: str | Path, run_id: str) -> None:
        self.base_dir = Path(base_dir)
        self.run_id = run_id
        self.artifact_dir = self.base_dir / run_id / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def save_image(self, image, *, summary: str, kind: str = "image") -> ArtifactRef:
        artifact_id = self._new_id(kind)
        path = self.artifact_dir / f"{artifact_id}.png"
        image.save(path)
        return ArtifactRef(
            id=artifact_id,
            path=str(path),
            kind=kind,
            summary=summary,
        )

    def save_text(self, text: str, *, summary: str, kind: str = "text") -> ArtifactRef:
        artifact_id = self._new_id(kind)
        path = self.artifact_dir / f"{artifact_id}.txt"
        path.write_text(text)
        return ArtifactRef(
            id=artifact_id,
            path=str(path),
            kind=kind,
            summary=summary,
        )

    def save_json(self, payload: Any, *, summary: str, kind: str = "json") -> ArtifactRef:
        artifact_id = self._new_id(kind)
        path = self.artifact_dir / f"{artifact_id}.json"
        path.write_text(json.dumps(payload, indent=2))
        return ArtifactRef(
            id=artifact_id,
            path=str(path),
            kind=kind,
            summary=summary,
        )

    def _new_id(self, prefix: str) -> str:
        suffix = uuid.uuid4().hex[:8]
        return f"{prefix}_{suffix}"
