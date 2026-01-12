"""MedSAM2-based segmentation tool (optional dependency)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _resolve_medsam2_path() -> Path:
    env_path = os.getenv("MEDSAM2_PATH")
    if env_path:
        return Path(env_path).expanduser()
    medrax2_root = os.getenv("MEDRAX2_PATH")
    if medrax2_root:
        return Path(medrax2_root).expanduser() / "MedSAM2"
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "MedRAX2" / "MedSAM2"


def _ensure_medsam2_on_path() -> Path:
    medsam2_path = _resolve_medsam2_path()
    medsam2_parent = medsam2_path.parent
    if medsam2_path.exists() and str(medsam2_path) not in sys.path:
        sys.path.append(str(medsam2_path))
    if medsam2_parent.exists() and str(medsam2_parent) not in sys.path:
        sys.path.append(str(medsam2_parent))
    return medsam2_path


class MedSAM2:
    def __init__(
        self,
        checkpoint_path: str,
        *,
        model_file: str = "MedSAM2_latest.pt",
        model_cfg: str = "sam2.1_hiera_t512.yaml",
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.model_file = model_file
        self.model_cfg = model_cfg
        self.device = device or "cuda"
        self.cache_dir = Path(cache_dir) if cache_dir else Path("models")
        self._predictor = self._load_predictor()

    def _load_predictor(self) -> Any:
        medsam2_path = _ensure_medsam2_on_path()
        if not medsam2_path.exists():
            msg = (
                f"MedSAM2 path not found at {medsam2_path}. "
                "Set MEDSAM2_PATH or MEDRAX2_PATH to override."
            )
            raise FileNotFoundError(msg)

        try:
            from hydra import initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
        except ModuleNotFoundError as exc:
            raise ImportError("hydra-core is required for MedSAM2.") from exc

        from huggingface_hub import hf_hub_download
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        config_dir = medsam2_path / "sam2" / "configs"
        initialize_config_dir(config_dir=str(config_dir), version_base="1.2")

        model_path = hf_hub_download(
            repo_id=self.checkpoint_path,
            filename=self.model_file,
            local_dir=str(self.cache_dir),
            local_dir_use_symlinks=False,
        )
        config_name = self.model_cfg.replace(".yaml", "")
        sam2_model = build_sam2(config_name, model_path, device=self.device)
        return SAM2ImagePredictor(sam2_model)

    def _prepare_image(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.array(image, dtype=np.uint8)

    def segment_point(self, image: Image.Image, point: tuple[int, int]) -> np.ndarray:
        if not (0 <= point[0] < image.width and 0 <= point[1] < image.height):
            msg = f"Point {point} is outside image bounds ({image.width}x{image.height})."
            raise ValueError(msg)
        image_np = self._prepare_image(image)
        self._predictor.set_image(image_np)

        input_point = np.array([[point[0], point[1]]])
        input_label = np.array([1], dtype=np.int32)
        masks, scores, _ = self._predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        best_index = int(np.argmax(scores)) if len(scores) else 0
        mask = masks[best_index]
        return (mask > 0).astype(np.uint8)


class SegmentTool:
    _overlay_color = (255, 0, 0)
    _overlay_alpha = 0.35

    def __init__(
        self,
        checkpoint_path: str | None = None,
        *,
        model_file: str | None = None,
        model_cfg: str | None = None,
        device: str | None = None,
        cache_dir: str | None = None,
        medsam2: MedSAM2 | None = None,
    ) -> None:
        self.checkpoint_path = (
            checkpoint_path or os.getenv("MEDSAM2_CHECKPOINT") or "wanglab/MedSAM2"
        )
        self.model_file = model_file or os.getenv("MEDSAM2_MODEL_FILE") or "MedSAM2_latest.pt"
        self.model_cfg = model_cfg or os.getenv("MEDSAM2_MODEL_CFG") or "sam2.1_hiera_t512.yaml"
        self.device = device or os.getenv("MEDSAM2_DEVICE")
        self.cache_dir = cache_dir or os.getenv("MEDSAM2_CACHE_DIR")
        self._medsam2 = medsam2

    def run(
        self, *, image: Image.Image, point: tuple[int, int] | None = None
    ) -> tuple[Image.Image, str]:
        if point is None:
            point = (image.width // 2, image.height // 2)
        mask = self._get_medsam2().segment_point(image, point)
        overlay = self._overlay_mask(image, mask)
        summary = f"Segment region around point={point}."
        return overlay, summary

    def _get_medsam2(self) -> MedSAM2:
        if self._medsam2 is None:
            self._medsam2 = MedSAM2(
                checkpoint_path=self.checkpoint_path,
                model_file=self.model_file,
                model_cfg=self.model_cfg,
                device=self.device,
                cache_dir=self.cache_dir,
            )
        return self._medsam2

    def _overlay_mask(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        base = image.convert("RGB")
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_uint8, mode="L")
        overlay = Image.new("RGB", base.size, self._overlay_color)
        tinted = Image.blend(base, overlay, self._overlay_alpha)
        return Image.composite(tinted, base, mask_image)
