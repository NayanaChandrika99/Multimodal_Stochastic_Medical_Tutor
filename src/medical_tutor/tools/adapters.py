# ABOUTME: Implements tool handlers (zoom, enhance, retrieval) used by the LangGraph runtime.
# ABOUTME: Converts tool arguments + state into ToolResult objects and saved artifacts.
"""Tool adapters for Medical_Tutor."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from PIL import Image, ImageEnhance

from medical_tutor.contracts import ToolResult


def _coerce_float(value: object, *, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a number.")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"{name} must be a number.")


def _load_image(state: Any) -> Image.Image:
    if not getattr(state, "image", None):
        raise ValueError("No image provided.")
    return Image.open(state.image).convert("RGB")


def _compute_bounds(
    image: Image.Image,
    *,
    bbox_2d: Iterable[float],
    padding: float = 0.0,
) -> tuple[int, int, int, int]:
    coords = list(bbox_2d)
    if len(coords) != 4:
        raise ValueError("bbox_2d must contain exactly four values.")

    x1, y1, x2, y2 = (float(value) for value in coords)
    img_w, img_h = image.size

    if max(x1, y1, x2, y2) <= 1.0:
        norm_x1 = x1 - padding
        norm_y1 = y1 - padding
        norm_x2 = x2 + padding
        norm_y2 = y2 + padding
    else:
        norm_x1 = x1 / img_w - padding
        norm_y1 = y1 / img_h - padding
        norm_x2 = x2 / img_w + padding
        norm_y2 = y2 / img_h + padding

    norm_x1 = min(max(0.0, norm_x1), 1.0)
    norm_y1 = min(max(0.0, norm_y1), 1.0)
    norm_x2 = min(max(0.0, norm_x2), 1.0)
    norm_y2 = min(max(0.0, norm_y2), 1.0)

    left = int(norm_x1 * img_w)
    upper = int(norm_y1 * img_h)
    right = int(norm_x2 * img_w)
    lower = int(norm_y2 * img_h)

    if right <= left or lower <= upper:
        raise ValueError("Zoom region is empty after clamping.")

    return (left, upper, right, lower)


def zoom_adapter(*, arguments: dict, state: Any, registry: Any) -> ToolResult:
    try:
        image = _load_image(state)
        bbox_2d = arguments.get("bbox_2d")
        padding = _coerce_float(arguments.get("padding", 0.0), name="padding")
        if not isinstance(bbox_2d, Iterable):
            raise ValueError("bbox_2d must contain exactly four values.")
        left, upper, right, lower = _compute_bounds(image, bbox_2d=bbox_2d, padding=padding)
        cropped = image.crop((left, upper, right, lower))
        summary = f"Zoom into bbox_2d={bbox_2d} with padding={padding}."
        output = {
            "summary": summary,
            "bounds": [left, upper, right, lower],
            "image": {"width": cropped.width, "height": cropped.height, "mode": cropped.mode},
        }
        artifact_refs = []
        if registry.artifact_store is not None:
            artifact_refs.append(registry.artifact_store.save_image(cropped, summary=summary))
        return ToolResult(tool_name="zoom", ok=True, output=output, artifact_refs=artifact_refs)
    except Exception as exc:
        return ToolResult(tool_name="zoom", ok=False, error=str(exc))


def enhance_adapter(*, arguments: dict, state: Any, registry: Any) -> ToolResult:
    try:
        image = _load_image(state)
        factor = _coerce_float(arguments.get("factor", 1.2), name="factor")
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(factor)
        summary = f"Enhance contrast with factor={factor}."
        output = {
            "summary": summary,
            "image": {"width": enhanced.width, "height": enhanced.height, "mode": enhanced.mode},
        }
        artifact_refs = []
        if registry.artifact_store is not None:
            artifact_refs.append(registry.artifact_store.save_image(enhanced, summary=summary))
        return ToolResult(tool_name="enhance", ok=True, output=output, artifact_refs=artifact_refs)
    except Exception as exc:
        return ToolResult(tool_name="enhance", ok=False, error=str(exc))


def segment_adapter(*, arguments: dict, state: Any, registry: Any) -> ToolResult:
    try:
        from .segment import SegmentTool

        image = _load_image(state)
        point = arguments.get("point")
        bbox = arguments.get("bbox_2d")
        if point is None and bbox is not None and isinstance(bbox, Iterable):
            coords = list(bbox)
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        tool = SegmentTool()
        mask_image, summary = tool.run(image=image, point=point)
        output = {
            "summary": summary,
            "image": {
                "width": mask_image.width,
                "height": mask_image.height,
                "mode": mask_image.mode,
            },
        }
        artifact_refs = []
        if registry.artifact_store is not None:
            artifact_refs.append(
                registry.artifact_store.save_image(mask_image, summary=summary, kind="mask")
            )
        return ToolResult(tool_name="segment", ok=True, output=output, artifact_refs=artifact_refs)
    except Exception as exc:
        return ToolResult(tool_name="segment", ok=False, error=str(exc))


def ocr_adapter(*, arguments: dict, state: Any, registry: Any) -> ToolResult:
    try:
        import pytesseract

        image = _load_image(state)
        language = arguments.get("language")
        text = (
            pytesseract.image_to_string(image, lang=language)
            if language
            else pytesseract.image_to_string(image)
        )
        summary = "Extract text with OCR."
        output = {"summary": summary, "text": text.strip()}
        artifact_refs = []
        if registry.artifact_store is not None:
            artifact_refs.append(registry.artifact_store.save_text(text, summary=summary))
        return ToolResult(tool_name="ocr", ok=True, output=output, artifact_refs=artifact_refs)
    except Exception as exc:
        return ToolResult(tool_name="ocr", ok=False, error=str(exc))


def retrieve_adapter(*, arguments: dict, state: Any, registry: Any) -> ToolResult:
    retriever = getattr(registry, "retriever", None)
    if retriever is None:
        return ToolResult(tool_name="retrieve", ok=False, error="Retriever not configured")

    try:
        query = str(arguments.get("query", "")).strip()
        modality = arguments.get("modality")
        top_k = arguments.get("top_k")
        hits, summary, cache_hit = retriever.retrieve(
            query=query,
            modality=modality,
            top_k=top_k,
            image_path=state.image,
        )
        existing = {hit.doc_id for hit in getattr(state, "retrieval_hits", [])}
        if existing:
            hits = [hit for hit in hits if hit.doc_id not in existing]
        output = {
            "summary": summary,
            "cache_hit": cache_hit,
            "hits": [hit.model_dump() for hit in hits],
        }
        return ToolResult(tool_name="retrieve", ok=True, output=output, artifact_refs=[])
    except Exception as exc:
        return ToolResult(tool_name="retrieve", ok=False, error=str(exc))


def web_browser_adapter(*, arguments: dict, state: Any, registry: Any) -> ToolResult:
    try:
        from .web_browser import WebBrowserTool

        url = str(arguments.get("url", "")).strip()
        query = str(arguments.get("query", "")).strip()
        max_chars = arguments.get("max_content_length") or arguments.get("max_chars")
        max_links = arguments.get("max_links")
        tool = WebBrowserTool()
        output = tool.run(
            query=query,
            url=url,
            max_content_length=int(max_chars) if max_chars is not None else 5000,
            max_links=int(max_links) if max_links is not None else 5,
        )
        artifact_refs = []
        if registry.artifact_store is not None:
            text = output.get("content") or output.get("text", "")
            artifact_refs.append(
                registry.artifact_store.save_text(text, summary=output.get("summary", ""))
            )
        return ToolResult(
            tool_name="web_browser", ok=True, output=output, artifact_refs=artifact_refs
        )
    except Exception as exc:
        return ToolResult(tool_name="web_browser", ok=False, error=str(exc))


def image_findings_adapter(*, arguments: dict, state: Any, registry: Any) -> ToolResult:
    try:
        image = _load_image(state)
        bbox_2d = arguments.get("bbox_2d")
        padding = _coerce_float(arguments.get("padding", 0.0), name="padding")
        bounds = None
        if bbox_2d is not None:
            if not isinstance(bbox_2d, Iterable):
                raise ValueError("bbox_2d must contain exactly four values.")
            left, upper, right, lower = _compute_bounds(image, bbox_2d=bbox_2d, padding=padding)
            bounds = [left, upper, right, lower]
            image = image.crop((left, upper, right, lower))

        prompt = arguments.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = (
                "Describe the objective findings visible in this medical image. "
                "Focus on what is seen. Do not provide a diagnosis or select an answer choice."
            )

        max_new_tokens = arguments.get("max_new_tokens", 256)
        try:
            max_new_tokens = int(max_new_tokens)
        except Exception:
            max_new_tokens = 256

        import torch

        from medical_tutor.config import load_config
        from medical_tutor.runtime import answer as answer_runtime

        device, dtype_name = answer_runtime._detect_device()
        config = load_config()
        processor, model = answer_runtime._load_answer_model(
            config.answer_model, device, dtype_name
        )

        inputs = answer_runtime._encode_inputs(processor=processor, image=image, prompt=prompt)
        if hasattr(inputs, "to"):
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            target_device = getattr(model, "device", device)
            target_dtype = dtype_map.get(dtype_name)
            if target_dtype is None or dtype_name == "float32":
                inputs = inputs.to(target_device)
            else:
                inputs = inputs.to(target_device, dtype=target_dtype)

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=max_new_tokens)

        decoded = answer_runtime._decode_output(
            processor=processor, generation=generation, prepared_inputs=inputs
        )
        summary = decoded[0].strip() if decoded else ""
        output = {
            "summary": summary,
            "device": device,
            "dtype": dtype_name,
            "bounds": bounds,
        }
        artifact_refs = []
        if registry.artifact_store is not None:
            artifact_refs.append(
                registry.artifact_store.save_text(summary, summary="Image findings summary.")
            )
        return ToolResult(
            tool_name="image_findings", ok=True, output=output, artifact_refs=artifact_refs
        )
    except Exception as exc:
        return ToolResult(tool_name="image_findings", ok=False, error=str(exc))
