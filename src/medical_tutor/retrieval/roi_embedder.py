"""Embed image regions using BiomedCLIP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class _Region:
    left: int
    top: int
    right: int
    bottom: int


class ROIEmbedder:
    _DEFAULT_MODEL_NAME = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        *,
        device: str | None = None,
        backend: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._backend = backend
        self._model: Any | None = None
        self._processor: Any | None = None
        self._uses_open_clip = False

    def embed(
        self,
        image: Image.Image,
        region: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        view = image.convert("RGB")
        if region is not None:
            bounds = self._clamp_region(view, region)
            view = view.crop((bounds.left, bounds.top, bounds.right, bounds.bottom))

        if self._backend is not None:
            embedding = self._backend.embed(view)
        else:
            embedding = self._embed_with_biomedclip(view)

        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError("ROI embedder produced an empty embedding.")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("ROI embedder produced a zero-norm embedding.")
        return vector / norm

    def _embed_with_biomedclip(self, image: Image.Image) -> np.ndarray:
        model, processor = self._load_model()

        import torch

        device = self._resolve_device()
        if self._uses_open_clip:
            image_tensor = processor(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                features = model.encode_image(image_tensor)
        else:
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                if hasattr(model, "get_image_features"):
                    features = model.get_image_features(**inputs)
                else:
                    outputs = model(**inputs)
                    if hasattr(outputs, "image_embeds"):
                        features = outputs.image_embeds
                    elif hasattr(outputs, "pooler_output"):
                        features = outputs.pooler_output
                    else:
                        features = outputs.last_hidden_state.mean(dim=1)

        return features.squeeze(0).detach().cpu().numpy()

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            if self._should_use_open_clip():
                self._load_open_clip_model()
            else:
                self._load_transformers_model()

        return self._model, self._processor

    def _load_open_clip_model(self) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("open-clip-torch is required to load BiomedCLIP embeddings.") from exc

        model_id = self._open_clip_identifier()
        device = self._resolve_device()
        self._model, self._processor = open_clip.create_model_from_pretrained(
            model_id,
            device=device,
        )
        if self._model is None or self._processor is None:
            raise RuntimeError("Failed to initialize BiomedCLIP open_clip model.")
        self._model.eval()
        self._uses_open_clip = True

    def _load_transformers_model(self) -> None:
        from transformers import AutoModel, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self._model is None or self._processor is None:
            raise RuntimeError("Failed to initialize BiomedCLIP transformers model.")
        self._model = self._model.to(self._resolve_device())
        self._uses_open_clip = False

    def _open_clip_identifier(self) -> str:
        if self.model_name.startswith("hf-hub:"):
            return self.model_name
        return f"hf-hub:{self.model_name}"

    def _should_use_open_clip(self) -> bool:
        return "BiomedCLIP" in self.model_name or self.model_name.startswith("hf-hub:")

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _clamp_region(image: Image.Image, region: tuple[int, int, int, int]) -> _Region:
        left, top, right, bottom = region
        left = max(0, int(left))
        top = max(0, int(top))
        right = min(image.width, int(right))
        bottom = min(image.height, int(bottom))

        if right <= left or bottom <= top:
            raise ValueError("Region bounds must define a non-empty area.")

        return _Region(left=left, top=top, right=right, bottom=bottom)
