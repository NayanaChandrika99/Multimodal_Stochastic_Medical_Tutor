# ABOUTME: Evaluates action-only controller outputs for JSON validity and action rates.
# ABOUTME: Provides lightweight parsing and metric utilities for TutorAction predictions.

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from medical_tutor.ops.train_action_controller import format_action_prompt, load_action_dataset


def parse_action_output(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{[\s\S]*\})", text)
        if not match:
            return None
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def compute_metrics(
    predictions: list[dict[str, Any] | None],
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(examples)
    valid = sum(1 for pred in predictions if isinstance(pred, dict))
    json_validity = (valid / total) if total else 0.0

    reveal_turn0 = 0
    tool_turn0_mm = 0
    total_turn0 = 0
    total_mm_turn0 = 0
    action_counts_by_profile: dict[str, dict[str, int]] = {}
    for pred, example in zip(predictions, examples, strict=False):
        if example.get("turn_index") != 0:
            pass
        else:
            total_turn0 += 1
            if example.get("image_ids"):
                total_mm_turn0 += 1
            if isinstance(pred, dict):
                if pred.get("type") == "REVEAL_ANSWER":
                    reveal_turn0 += 1
                if pred.get("type") == "REQUEST_TOOL" and example.get("image_ids"):
                    tool_turn0_mm += 1

        profile = example.get("student_profile") or "unknown"
        if profile not in action_counts_by_profile:
            action_counts_by_profile[profile] = {}
        if not isinstance(pred, dict):
            continue
        action_type = pred.get("type")
        if not action_type:
            continue
        action_counts_by_profile[profile][action_type] = (
            action_counts_by_profile[profile].get(action_type, 0) + 1
        )

    action_distribution_by_profile: dict[str, dict[str, float]] = {}
    for profile, counts in action_counts_by_profile.items():
        total_actions = sum(counts.values())
        if total_actions == 0:
            action_distribution_by_profile[profile] = {}
        else:
            action_distribution_by_profile[profile] = {
                action_type: count / total_actions for action_type, count in counts.items()
            }

    return {
        "json_validity": json_validity,
        "reveal_turn0_rate": (reveal_turn0 / total_turn0) if total_turn0 else 0.0,
        "request_tool_turn0_rate_mm": (tool_turn0_mm / total_mm_turn0) if total_mm_turn0 else 0.0,
        "action_distribution_by_profile": action_distribution_by_profile,
    }


def evaluate_action_controller(
    *,
    dataset_path: Path,
    cases_path: Path | None,
    base_model: str,
    image_root: Path | None,
    max_examples: int | None,
    max_new_tokens: int,
    dry_run: bool,
) -> dict[str, Any]:
    examples = load_action_dataset(dataset_path, cases_path=cases_path)
    if max_examples is not None:
        examples = examples[: int(max_examples)]

    if dry_run:
        metrics = compute_metrics([None for _ in examples], examples)
        return {"metrics": metrics, "num_examples": len(examples)}

    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.eval()

    predictions: list[dict[str, Any] | None] = []
    for example in examples:
        prompt = format_action_prompt(example)
        image_path = None
        image_ids = example.get("image_ids") or []
        if image_ids:
            image_path = image_ids[0]
        image = None
        if image_path:
            resolved = (image_root / image_path) if image_root else Path(image_path)
            if resolved.exists():
                image = Image.open(resolved).convert("RGB")
        if image is not None:
            inputs = processor(text=prompt, images=image, return_tensors="pt")
        else:
            inputs = processor(text=prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        predictions.append(parse_action_output(decoded[0] if decoded else ""))

    metrics = compute_metrics(predictions, examples)
    return {"metrics": metrics, "num_examples": len(examples)}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate action-only tutor controller")
    parser.add_argument("--dataset", required=True, help="Path to action_sft.jsonl")
    parser.add_argument("--cases", default=None, help="Optional cases.jsonl path")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = evaluate_action_controller(
        dataset_path=Path(args.dataset),
        cases_path=Path(args.cases) if args.cases else None,
        base_model=args.base_model,
        image_root=Path(args.image_root) if args.image_root else None,
        max_examples=args.max_examples,
        max_new_tokens=args.max_new_tokens,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
