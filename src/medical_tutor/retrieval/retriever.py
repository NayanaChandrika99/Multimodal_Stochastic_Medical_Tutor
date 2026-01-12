"""Retriever wrapper for Medical_Tutor."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PIL import Image

from medical_tutor.contracts import RetrievalItem
from medical_tutor.retrieval.multimodal import MultiModalRetriever


class Retriever:
    def __init__(
        self,
        *,
        bm25_path: str,
        text_index_path: str,
        image_index_path: str,
        top_k_text: int = 5,
        top_k_image: int = 3,
    ) -> None:
        self.top_k_text = top_k_text
        self.top_k_image = top_k_image
        self._cache: dict[tuple, tuple[list[RetrievalItem], str]] = {}
        self._retriever = MultiModalRetriever.from_paths(
            bm25_path=Path(bm25_path),
            text_index_path=Path(text_index_path),
            image_index_path=Path(image_index_path),
        )

    def clear_cache(self) -> None:
        self._cache.clear()

    def pre_retrieve(
        self, *, question: str, image_path: str | None
    ) -> tuple[list[RetrievalItem], str]:
        try:
            hits, summary, _ = self._run_retrieval(
                question=question,
                image_path=image_path,
                modality=None,
                include_cache=False,
            )
            return hits, summary
        except Exception as exc:
            return [], f"Retrieval unavailable: {exc}"

    def retrieve(
        self,
        *,
        query: str,
        modality: str | None = None,
        top_k: int | None = None,
        image_path: str | None = None,
    ) -> tuple[list[RetrievalItem], str, bool]:
        try:
            hits, summary, cache_hit = self._run_retrieval(
                question=query,
                image_path=image_path,
                modality=modality,
                top_k=top_k,
                include_cache=True,
            )
            return hits, summary, cache_hit
        except Exception as exc:
            return [], f"Retrieval unavailable: {exc}", False

    def _run_retrieval(
        self,
        *,
        question: str,
        image_path: str | None,
        modality: str | None,
        top_k: int | None = None,
        include_cache: bool = False,
    ) -> tuple[list[RetrievalItem], str, bool]:
        cache_hit = False
        cache_key = (question, modality, top_k, image_path)
        if include_cache and cache_key in self._cache:
            cached_hits, cached_summary = self._cache[cache_key]
            return list(cached_hits), cached_summary, True

        modalities = None
        if modality in {"text", "image"}:
            modalities = {modality}

        max_k = top_k or max(self.top_k_text, self.top_k_image)
        results = self._retriever.search(query=question, top_k=max_k, modalities=modalities)

        image_results: list[dict] = []
        if image_path:
            try:
                image = Image.open(image_path).convert("RGB")
                image_results = self._retriever.search_by_image(
                    image, top_k=self.top_k_image, modalities={"image"}
                )
            except Exception:
                image_results = []

        merged = results + image_results
        retrieval_items = self._format_items(merged)

        text_hits = [item for item in retrieval_items if item.modality == "text"][: self.top_k_text]
        image_hits = [item for item in retrieval_items if item.modality == "image"][
            : self.top_k_image
        ]

        if modality == "text":
            retrieval_items = text_hits
        elif modality == "image":
            retrieval_items = image_hits
        else:
            retrieval_items = text_hits + image_hits

        summary = self._build_summary(text_hits=text_hits, image_hits=image_hits)
        if include_cache:
            self._cache[cache_key] = (list(retrieval_items), summary)
        return retrieval_items, summary, cache_hit

    @staticmethod
    def _format_items(results: Iterable[dict]) -> list[RetrievalItem]:
        formatted: list[RetrievalItem] = []
        for result in results:
            metadata = result.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            doc_id = str(metadata.get("doc_id") or result.get("source") or "unknown")
            modality = str(result.get("modality") or metadata.get("modality") or "text")
            score = float(result.get("score", 0.0))
            provenance = str(
                metadata.get("source") or metadata.get("path") or result.get("source") or "unknown"
            )
            snippet = str(result.get("text") or metadata.get("caption") or "")
            if len(snippet) > 240:
                snippet = snippet[:237] + "..."
            uri = metadata.get("uri")
            formatted.append(
                RetrievalItem(
                    doc_id=doc_id,
                    modality=modality,
                    score=score,
                    provenance=provenance,
                    snippet=snippet,
                    uri=uri,
                )
            )
        return formatted

    @staticmethod
    def _build_summary(*, text_hits: list[RetrievalItem], image_hits: list[RetrievalItem]) -> str:
        def _summarize(items: list[RetrievalItem]) -> str:
            return "; ".join([f"{item.doc_id} ({item.score:.3f})" for item in items])

        text_summary = _summarize(text_hits) if text_hits else "none"
        image_summary = _summarize(image_hits) if image_hits else "none"
        return f"Text hits: {text_summary}. Image hits: {image_summary}."


class NullRetriever:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def clear_cache(self) -> None:
        return None

    def pre_retrieve(self, *, question: str, image_path: str | None):
        return [], f"Retrieval unavailable: {self.reason}"

    def retrieve(
        self,
        *,
        query: str,
        modality: str | None = None,
        top_k: int | None = None,
        image_path: str | None = None,
    ):
        return [], f"Retrieval unavailable: {self.reason}", False
