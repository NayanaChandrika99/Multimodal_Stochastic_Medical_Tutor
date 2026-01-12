"""FAISS vector index for semantic retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import faiss
except ModuleNotFoundError:  # pragma: no cover
    faiss = None
import numpy as np


class VectorIndex:
    def __init__(
        self,
        index_path: Path | str,
        *,
        embedder: Any | None = None,
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self._index_dir = Path(index_path)
        self._metadata_path = self._index_dir / "metadata.json"
        self._index_path = self._index_dir / "index.faiss"
        self._documents: list[str] = []
        self._metadata: list[dict] = []
        self._embedder_name = embedder_name
        self._embedder = embedder
        self._faiss_index: Any | None = None
        self._load()

    @property
    def documents(self) -> list[str]:
        return list(self._documents)

    @property
    def metadata(self) -> list[dict]:
        return list(self._metadata)

    @property
    def embedder_name(self) -> str:
        return str(self._embedder_name)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, object]]:
        if faiss is None:
            raise ImportError("faiss is required to use VectorIndex.")
        if not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if self._faiss_index is None:
            raise ValueError("Vector index is not loaded.")
        if not self._documents:
            raise ValueError("Vector index has no documents.")

        query_embeddings = self._encode([query])
        scores, indices = self._faiss_index.search(
            query_embeddings, min(top_k, len(self._documents))
        )
        results: list[dict[str, object]] = []
        for idx, score in zip(indices[0], scores[0], strict=False):
            idx_value = int(idx)
            results.append(
                {
                    "text": self._documents[idx_value],
                    "score": float(score),
                    "source": f"doc_{idx_value}",
                    "metadata": self._metadata[idx_value],
                }
            )
        return results

    def search_by_embedding(
        self, embedding: np.ndarray, top_k: int = 10
    ) -> list[dict[str, object]]:
        if faiss is None:
            raise ImportError("faiss is required to use VectorIndex.")
        if self._faiss_index is None:
            raise ValueError("Vector index is not loaded.")
        if not self._documents:
            raise ValueError("Vector index has no documents.")

        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self._faiss_index.d:
            return []

        scores, indices = self._faiss_index.search(vector, min(top_k, len(self._documents)))
        results: list[dict[str, object]] = []
        for idx, score in zip(indices[0], scores[0], strict=False):
            idx_value = int(idx)
            results.append(
                {
                    "text": self._documents[idx_value],
                    "score": float(score),
                    "source": f"doc_{idx_value}",
                    "metadata": self._metadata[idx_value],
                }
            )
        return results

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._embedder is None:
            self._embedder = self._resolve_embedder(self._embedder, self._embedder_name)
        return self._encode_with(self._embedder, texts)

    @staticmethod
    def _encode_with(embedder: Any, texts: list[str]) -> np.ndarray:
        embeddings = embedder.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)

    @staticmethod
    def _resolve_embedder(embedder: Any | None, embedder_name: str):
        if embedder is not None:
            return embedder
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise ImportError(
                "sentence-transformers is required to encode queries for VectorIndex."
            ) from exc
        return SentenceTransformer(embedder_name)

    def _load(self) -> None:
        if faiss is None:
            return
        if not self._index_path.exists() or not self._metadata_path.exists():
            return
        metadata = json.loads(self._metadata_path.read_text())
        self._documents = metadata.get("documents", [])
        self._metadata = metadata.get("metadata", [])
        self._embedder_name = metadata.get("embedder_name", self._embedder_name)
        self._faiss_index = faiss.read_index(str(self._index_path))
