"""BM25 index for text retrieval."""

from __future__ import annotations

import pickle
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError:  # pragma: no cover
    BM25Okapi = None


class BM25Index:
    def __init__(self, corpus_path: Path | str):
        if BM25Okapi is None:
            raise ImportError("rank-bm25 is required to use BM25Index.")
        self.corpus_path = Path(corpus_path)
        self._documents, self._tokenized, self._metadata = self._load_corpus()
        self._bm25 = BM25Okapi(self._tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def _load_corpus(self) -> tuple[list[str], list[list[str]], list[dict]]:
        with self.corpus_path.open("rb") as handle:
            payload = pickle.load(handle)  # nosec B301
        documents = payload["documents"]
        tokenized = payload["tokenized"]
        metadata = payload.get("metadata")
        if metadata is None:
            metadata = [{} for _ in documents]
        if len(metadata) != len(documents):
            raise ValueError("BM25 metadata length does not match documents.")
        return documents, tokenized, metadata

    def search(self, query: str, top_k: int = 10) -> list[dict[str, object]]:
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results: list[dict[str, object]] = []
        for idx in ranked_indices:
            results.append(
                {
                    "text": self._documents[idx],
                    "score": float(scores[idx]),
                    "source": f"doc_{idx}",
                    "metadata": self._metadata[idx],
                }
            )
        return results
