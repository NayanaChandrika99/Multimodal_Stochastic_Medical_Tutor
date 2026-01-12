"""Cross-modal retrieval across text and image indices."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from medical_tutor.retrieval.bm25 import BM25Index
from medical_tutor.retrieval.vector import VectorIndex

if TYPE_CHECKING:
    from medical_tutor.retrieval.biomed_text_embedder import BiomedTextEmbedder
    from medical_tutor.retrieval.roi_embedder import ROIEmbedder


class MultiModalRetriever:
    def __init__(
        self,
        text_index: VectorIndex,
        image_index: VectorIndex,
        bm25_index: BM25Index | None = None,
        *,
        text_embedder: BiomedTextEmbedder | None = None,
        image_embedder: ROIEmbedder | None = None,
        k: int = 60,
    ) -> None:
        self.text_index = text_index
        self.image_index = image_index
        self.bm25_index = bm25_index
        self._text_embedder = text_embedder
        self._image_embedder = image_embedder
        self.k = k

    def search(
        self,
        query: str | None = None,
        query_embedding: np.ndarray | None = None,
        top_k: int = 10,
        modalities: set[str] | None = None,
        roi_embedding: object | None = None,
        source_filter: set[str] | None = None,
        modality_filter: set[str] | None = None,
    ) -> list[dict]:
        if roi_embedding is not None:
            query_embedding = roi_embedding
        if modality_filter is not None:
            modalities = modality_filter

        if query is None and query_embedding is None:
            raise ValueError("Must provide either query or query_embedding")

        if query is not None and query_embedding is None:
            query_embedding = self._encode_text(query)

        search_text = modalities is None or "text" in modalities
        search_images = modalities is None or "image" in modalities

        fused: dict[str, dict] = {}

        if search_text:
            text_results = self.text_index.search_by_embedding(query_embedding, top_k=top_k)
            self._apply_rrf(fused, text_results, modality="text")
            if self.bm25_index is not None and query is not None:
                bm25_results = self.bm25_index.search(query, top_k=top_k)
                self._apply_rrf(fused, bm25_results, modality="text")

        if search_images:
            image_results = self.image_index.search_by_embedding(query_embedding, top_k=top_k)
            self._apply_rrf(fused, image_results, modality="image")

        ranked = sorted(fused.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)
        if source_filter:
            ranked = [
                item for item in ranked if item.get("metadata", {}).get("source") in source_filter
            ]
        return ranked[:top_k]

    def search_by_image(
        self,
        image,
        top_k: int = 10,
        modalities: set[str] | None = None,
    ) -> list[dict]:
        embedding = self._encode_image(image)
        return self.search(query_embedding=embedding, top_k=top_k, modalities=modalities)

    def _encode_text(self, text: str) -> np.ndarray:
        if self._text_embedder is None:
            from medical_tutor.retrieval.biomed_text_embedder import BiomedTextEmbedder

            embedder_name = getattr(self.text_index, "embedder_name", None)
            if not embedder_name:
                embedder_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            self._text_embedder = BiomedTextEmbedder(model_name=embedder_name)

        embeddings = self._text_embedder.encode([text], normalize_embeddings=True)
        return np.asarray(embeddings[0], dtype=np.float32)

    def _encode_image(self, image) -> np.ndarray:
        if self._image_embedder is None:
            from medical_tutor.retrieval.roi_embedder import ROIEmbedder

            self._image_embedder = ROIEmbedder()
        return self._image_embedder.embed(image)

    def _apply_rrf(
        self,
        fused: dict[str, dict],
        results: list[dict],
        modality: str,
    ) -> None:
        for rank, result in enumerate(results, start=1):
            source = str(result.get("source", f"{modality}_{rank}"))
            key = f"{modality}:{source}"

            if key not in fused:
                fused[key] = {
                    "text": result.get("text", ""),
                    "source": source,
                    "modality": modality,
                    "score": 0.0,
                    "metadata": result.get("metadata", {}),
                }

            fused[key]["score"] = float(fused[key]["score"]) + (1.0 / (self.k + rank))

    @classmethod
    def from_paths(
        cls,
        *,
        text_index_path: Path | str,
        image_index_path: Path | str,
        bm25_path: Path | str | None = None,
        k: int = 60,
    ) -> MultiModalRetriever:
        text_index = VectorIndex(text_index_path)
        image_index = VectorIndex(image_index_path)
        bm25_index = BM25Index(bm25_path) if bm25_path else None

        return cls(
            text_index=text_index,
            image_index=image_index,
            bm25_index=bm25_index,
            k=k,
        )
