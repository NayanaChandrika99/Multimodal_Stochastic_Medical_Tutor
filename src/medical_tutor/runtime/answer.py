# ABOUTME: Generates final answers using a vision-language answer model over images and context.
# ABOUTME: Loads MedGemma via Transformers and runs inference on the best available device.
"""Answer model integration for Medical_Tutor."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, cast

from medical_tutor import prompts as prompt_templates
from medical_tutor.config import Config
from medical_tutor.contracts import AgentState


def select_device(*, cuda_available: bool, mps_available: bool) -> str:
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def select_dtype_name(*, device: str, cuda_bf16_supported: bool) -> str:
    if device == "cuda":
        return "bfloat16" if cuda_bf16_supported else "float16"
    if device == "mps":
        return "float16"
    return "float32"


def _detect_device() -> tuple[str, str]:
    try:
        import torch
    except Exception:
        return "cpu", "float32"

    cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    mps_available = False
    backends = getattr(torch, "backends", None)
    if backends is not None:
        mps_backend = getattr(backends, "mps", None)
        if mps_backend is not None and hasattr(mps_backend, "is_available"):
            try:
                mps_available = bool(mps_backend.is_available())
            except Exception:
                mps_available = False

    device = select_device(cuda_available=cuda_available, mps_available=mps_available)
    cuda_bf16_supported = False
    if device == "cuda" and hasattr(torch.cuda, "is_bf16_supported"):
        try:
            cuda_bf16_supported = bool(torch.cuda.is_bf16_supported())
        except Exception:
            cuda_bf16_supported = False

    dtype_name = select_dtype_name(device=device, cuda_bf16_supported=cuda_bf16_supported)
    return device, dtype_name


def _apply_transformer_workarounds() -> None:
    try:
        from torch import _dynamo as torch_dynamo

        torch_dynamo.config.suppress_errors = True
    except Exception:
        pass

    try:
        from transformers import modeling_utils

        if not getattr(modeling_utils, "ALL_PARALLEL_STYLES", None):
            modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]
    except Exception:
        pass

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.gemma3.modeling_gemma3 import (
            Gemma3Attention,
            Gemma3FlashAttention2,
            Gemma3SdpaAttention,
        )

        attention_mapping = {
            "eager": Gemma3Attention,
            "sdpa": Gemma3SdpaAttention,
            "flash_attention_2": Gemma3FlashAttention2,
        }
        for key, value in attention_mapping.items():
            if key not in ALL_ATTENTION_FUNCTIONS._global_mapping:
                ALL_ATTENTION_FUNCTIONS._global_mapping[key] = value
    except Exception:
        pass

    os.environ.setdefault("TRANSFORMERS_ATTENTION_TYPE", "eager")


@lru_cache(maxsize=2)
def _load_answer_model(model_name: str, device: str, dtype_name: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype_name, torch.float32)

    _apply_transformer_workarounds()

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        attn_implementation=os.environ.get("TRANSFORMERS_ATTENTION_TYPE", "eager"),
        trust_remote_code=True,
    )
    if device == "mps":
        model = model.to("mps")
    model.eval()
    return processor, model


def build_answer_prompt_text(state: AgentState) -> str:
    question = ""
    if state.case_prompt and state.case_prompt.strip():
        question = state.case_prompt.strip()
    for message in reversed(state.messages):
        if question:
            break
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                question = content
            elif isinstance(content, list):
                texts = [
                    c.get("text")
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                ]
                question = " ".join([t for t in texts if t])
            break

    prompt_parts = [prompt_templates.ANSWER_PROMPT_BASE, f"Question: {question}"]

    if state.retrieval_summary:
        prompt_parts.append("Retrieved context summary:")
        prompt_parts.append(state.retrieval_summary)

    if state.tool_results:
        prompt_parts.append("Tool results summary:")
        for result in state.tool_results:
            output = result.output or {}
            summary = None
            if isinstance(output, dict):
                summary = output.get("summary")
                if summary is None:
                    summary = {
                        key: value for key, value in output.items() if key not in {"hits", "image"}
                    }
            if summary is None:
                summary = output
            prompt_parts.append(f"- {result.tool_name}: {summary}")

    return "\n".join(prompt_parts)


_ANSWER_OPTION_RE = re.compile(r"^\s*\(([A-E])\)\s*(.+?)\s*$", re.MULTILINE)


def _extract_answer_options(prompt_text: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for label, text in _ANSWER_OPTION_RE.findall(prompt_text or ""):
        label_norm = str(label).strip().upper()
        if label_norm:
            options[label_norm] = str(text).strip()
    return options


def _build_reveal_answer(state: AgentState) -> tuple[str, dict[str, Any]] | None:
    if not state.case_label:
        return None
    tutor_action = state.tutor_action or {}
    if tutor_action.get("type") != "REVEAL_ANSWER":
        return None

    label = str(state.case_label).strip().upper()
    if not label or len(label) != 1:
        return None

    options = _extract_answer_options(state.case_prompt or "")
    option_text = options.get(label)
    if option_text:
        return f"Correct answer: ({label}) {option_text}.", {"source": "case_label"}
    return f"Correct answer: ({label}).", {"source": "case_label"}


def _encode_inputs(*, processor: Any, image: Any, prompt: str) -> Any:
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a medical imaging expert assistant.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            },
        ]
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    return processor(
        images=image,
        text=prompt_templates.wrap_gemma_chat(prompt),
        return_tensors="pt",
    )


def _decode_output(*, processor: Any, generation: Any, prepared_inputs: Any) -> list[str]:
    input_ids = None
    try:
        input_ids = prepared_inputs["input_ids"]
    except Exception:
        input_ids = None

    if input_ids is not None:
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generation, strict=False)
        ]
        return cast(
            list[str],
            processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True),
        )

    return cast(list[str], processor.batch_decode(generation, skip_special_tokens=True))


def answer_with_medgemma(
    state: AgentState,
    config: Config,
) -> tuple[str, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    prompt = build_answer_prompt_text(state)
    metadata["prompt"] = prompt

    if not state.image:
        metadata["error"] = "No image provided for medgemma answer"
        return "Insufficient evidence to answer confidently.", metadata

    try:
        import torch
        from PIL import Image

        device, dtype_name = _detect_device()
        metadata["device"] = device
        metadata["dtype"] = dtype_name

        processor, model = _load_answer_model(config.answer_model, device, dtype_name)
        image = Image.open(state.image).convert("RGB")

        inputs = _encode_inputs(processor=processor, image=image, prompt=prompt)
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
            outputs = model.generate(**inputs, max_new_tokens=256)
        decoded = _decode_output(processor=processor, generation=outputs, prepared_inputs=inputs)
        answer = decoded[0] if decoded else ""
        return answer.strip(), metadata
    except Exception as exc:  # pragma: no cover - heavy model load
        metadata["error"] = str(exc)
        return "Insufficient evidence to answer confidently.", metadata


def answer_question(state: AgentState, config: Config) -> tuple[str, dict[str, Any]]:
    reveal = _build_reveal_answer(state)
    if reveal is not None:
        return reveal
    return answer_with_medgemma(state, config)
