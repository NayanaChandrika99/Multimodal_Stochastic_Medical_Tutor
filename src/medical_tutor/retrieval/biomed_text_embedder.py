"""BiomedCLIP text embedder for retrieval."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class BiomedTextEmbedder:
    def __init__(
        self,
        model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        *,
        device: str | None = None,
        context_length: int = 256,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.context_length = context_length
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def encode(
        self,
        texts: Iterable[str],
        *,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ):
        if isinstance(texts, str):
            texts = [texts]

        model, tokenizer = self._load()

        import torch

        device = self._resolve_device()
        text_list = list(texts)
        batch_size = max(1, int(self.batch_size))
        chunks: list[np.ndarray] = []

        with torch.inference_mode():
            for start in range(0, len(text_list), batch_size):
                batch = text_list[start : start + batch_size]
                tokens = tokenizer(batch, context_length=self.context_length).to(device)
                features = model.encode_text(tokens)
                if normalize_embeddings:
                    features = features / features.norm(dim=-1, keepdim=True)
                chunks.append(features.detach().cpu().numpy())

        embeddings = np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)
        if convert_to_numpy:
            return embeddings
        return torch.from_numpy(embeddings)

    def _load(self):
        if self._model is None or self._tokenizer is None:
            try:
                import open_clip
            except ImportError as exc:
                raise ImportError(
                    "open-clip-torch is required to load BiomedCLIP text embeddings."
                ) from exc

            model_id = self._open_clip_identifier()
            device = self._resolve_device()
            self._model, _ = open_clip.create_model_from_pretrained(
                model_id,
                device=device,
            )
            self._model.eval()
            self._tokenizer = open_clip.get_tokenizer(model_id)

        return self._model, self._tokenizer

    def _open_clip_identifier(self) -> str:
        if self.model_name.startswith("hf-hub:"):
            return self.model_name
        return f"hf-hub:{self.model_name}"

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
