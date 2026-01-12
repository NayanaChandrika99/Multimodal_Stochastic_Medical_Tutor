# ABOUTME: Trains an action-only controller to emit TutorAction JSON.
# ABOUTME: Provides prompt formatting and dataset utilities for SFT training.

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

DATASET_VERSION = "socratic_tutor_v1"


@dataclass(frozen=True)
class TrainingExample:
    prompt: str
    target_json: str
    image_path: str | None
    raw: dict[str, Any]


def load_action_dataset(path: Path, *, cases_path: Path | None = None) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                examples.append(payload)
    resolved_cases_path = cases_path
    if resolved_cases_path is None:
        candidate = path.parent / "cases.jsonl"
        if candidate.exists():
            resolved_cases_path = candidate
    if resolved_cases_path is None:
        return examples

    cases_by_id = load_case_index(resolved_cases_path)
    return _hydrate_examples(examples, cases_by_id)


def load_case_index(path: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            case_id = payload.get("case_id")
            if case_id:
                cases[str(case_id)] = payload
    return cases


def _hydrate_examples(
    examples: list[dict[str, Any]],
    cases_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    hydrated: list[dict[str, Any]] = []
    for example in examples:
        case_id = str(example.get("case_id") or "")
        case = cases_by_id.get(case_id)
        if case is None:
            hydrated.append(example)
            continue
        merged = dict(example)
        if not merged.get("question"):
            merged["question"] = case.get("question", "")
        if not merged.get("options"):
            merged["options"] = case.get("options") or {}
        if not merged.get("image_ids"):
            images = case.get("images") or []
            merged["image_ids"] = list(images)
        hydrated.append(merged)
    return hydrated


def format_action_prompt(example: dict[str, Any]) -> str:
    profile = example.get("student_profile", "(unset)")
    kc_tags = example.get("kc_tags") or {}
    question = example.get("question", "")
    options = example.get("options") or {}
    history = example.get("history") or []
    tool_summaries = example.get("tool_summaries") or []
    assessment = example.get("student_reply_assessment") or {}

    lines = [
        "You are a Socratic tutor controller. Output JSON only.",
        "",
        f"Student profile: {profile}",
        (
            "KC tags: "
            f"body_system={kc_tags.get('body_system')}; "
            f"medical_task={kc_tags.get('medical_task')}; "
            f"question_type={kc_tags.get('question_type')}"
        ),
        "",
        f"Question: {question}",
        "",
        "Answer Choices:",
    ]
    for key in sorted(options.keys()):
        lines.append(f"({key}) {options[key]}")
    lines.append("")

    if history:
        lines.append("History:")
        for entry in history:
            role = entry.get("role")
            content = entry.get("content")
            lines.append(f"- {role}: {content}")
        lines.append("")

    if tool_summaries:
        lines.append("Tool summaries:")
        for summary in tool_summaries:
            lines.append(f"- {summary}")
        lines.append("")

    if assessment:
        grade = assessment.get("grade")
        misconception = assessment.get("misconception")
        if grade:
            lines.append(f"Student assessment: grade={grade}")
        if misconception:
            lines.append(f"Misconception: {misconception}")
        lines.append("")

    lines.append("Output schema:")
    lines.append(
        '{ "type": "ASK_PROBE|HINT|MICROLESSON|QUIZ|REQUEST_TOOL|REVEAL_ANSWER|SAFETY_REFUSE", '
        '"arguments": { ... } }'
    )
    return "\n".join(lines).strip()


def build_training_examples(examples: list[dict[str, Any]]) -> list[TrainingExample]:
    training: list[TrainingExample] = []
    for example in examples:
        prompt = format_action_prompt(example)
        target_json = json.dumps(example.get("target_action") or {}, ensure_ascii=False)
        image_ids = example.get("image_ids") or []
        image_path = str(image_ids[0]) if image_ids else None
        training.append(
            TrainingExample(
                prompt=prompt,
                target_json=target_json,
                image_path=image_path,
                raw=example,
            )
        )
    return training


def split_examples(
    examples: list[dict[str, Any]],
    *,
    seed: int,
    val_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    split = int(len(indices) * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    train = [examples[idx] for idx in train_idx]
    val = [examples[idx] for idx in val_idx]
    return train, val


def train_action_controller(
    *,
    dataset_path: Path,
    cases_path: Path | None,
    output_dir: Path,
    base_model: str,
    seed: int,
    val_ratio: float,
    image_root: Path | None,
    max_train_examples: int | None,
    dry_run: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    batch_size: int,
    grad_accum: int,
    epochs: int,
    learning_rate: float,
) -> dict[str, Any]:
    raw_examples = load_action_dataset(dataset_path, cases_path=cases_path)
    train_raw, val_raw = split_examples(raw_examples, seed=seed, val_ratio=val_ratio)
    if max_train_examples is not None:
        train_raw = train_raw[: int(max_train_examples)]

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_path": str(dataset_path),
        "cases_path": str(cases_path) if cases_path else None,
        "base_model": base_model,
        "seed": seed,
        "val_ratio": val_ratio,
        "num_train_examples": len(train_raw),
        "num_val_examples": len(val_raw),
        "dry_run": dry_run,
    }
    manifest_path = output_dir / "train_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if dry_run:
        return {"output_dir": str(output_dir), "manifest_path": str(manifest_path)}

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments

    processor = AutoProcessor.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    train_examples = build_training_examples(train_raw)
    val_examples = build_training_examples(val_raw)

    def resolve_image(path: str | None) -> Image.Image | None:
        if not path:
            return None
        if image_root is None:
            return Image.open(path).convert("RGB")
        resolved = image_root / path
        if not resolved.exists():
            return None
        return Image.open(resolved).convert("RGB")

    def collate(batch: list[TrainingExample]) -> dict[str, torch.Tensor]:
        texts = [example.prompt + "\n\n" + example.target_json for example in batch]
        images = [resolve_image(example.image_path) for example in batch]
        if any(images):
            filled_images = [
                image if image is not None else Image.new("RGB", (224, 224), color="black")
                for image in images
            ]
            inputs = cast(
                dict[str, torch.Tensor],
                processor(text=texts, images=filled_images, return_tensors="pt", padding=True),
            )
        else:
            inputs = cast(
                dict[str, torch.Tensor],
                processor(text=texts, return_tensors="pt", padding=True),
            )
        labels = inputs["input_ids"].clone()
        inputs["labels"] = labels
        return inputs

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if val_examples else "no",
        seed=seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        eval_dataset=val_examples if val_examples else None,
        data_collator=collate,
        tokenizer=processor,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    return {"output_dir": str(output_dir), "manifest_path": str(manifest_path)}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train action-only tutor controller")
    parser.add_argument("--dataset", required=True, help="Path to action_sft.jsonl")
    parser.add_argument("--cases", default=None, help="Optional cases.jsonl path")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,v_proj",
        help="Comma-separated target modules for LoRA",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = train_action_controller(
        dataset_path=Path(args.dataset),
        cases_path=Path(args.cases) if args.cases else None,
        output_dir=Path(args.output_dir),
        base_model=args.base_model,
        seed=args.seed,
        val_ratio=args.val_ratio,
        image_root=Path(args.image_root) if args.image_root else None,
        max_train_examples=args.max_train_examples,
        dry_run=args.dry_run,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
