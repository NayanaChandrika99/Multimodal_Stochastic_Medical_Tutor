# ABOUTME: Loads MedXpert patient-script cases for tutoring demos and dataset building.
# ABOUTME: Normalizes case fields and formats MCQ stems with answer choices.

from __future__ import annotations

import io
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MedXpertCase:
    case_id: str
    question: str
    options: dict[str, str]
    label: str
    body_system: str | None
    medical_task: str | None
    question_type: str | None
    images: list[str]
    question_steps: list[dict]

    def format_question(self) -> str:
        lines = [self.question.strip(), "", "Answer Choices:"]
        for key in sorted(self.options.keys()):
            value = self.options[key]
            lines.append(f"({key}) {value}")
        return "\n".join(lines).strip()


def load_medxpert_cases(path: Path) -> dict[str, MedXpertCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: dict[str, MedXpertCase] = {}
    for item in payload:
        case_id = str(item.get("id") or "").strip()
        if not case_id:
            continue
        options = item.get("options") or {}
        if not isinstance(options, dict):
            options = {}
        images = item.get("images") or []
        if not isinstance(images, list):
            images = []
        question_steps = item.get("question_steps") or []
        if not isinstance(question_steps, list):
            question_steps = []

        cases[case_id] = MedXpertCase(
            case_id=case_id,
            question=str(item.get("question") or "").strip(),
            options={str(k): str(v) for k, v in options.items()},
            label=str(item.get("label") or "").strip(),
            body_system=str(item.get("body_system") or "") or None,
            medical_task=str(item.get("medical_task") or "") or None,
            question_type=str(item.get("question_type") or "") or None,
            images=[str(name) for name in images],
            question_steps=list(question_steps),
        )
    return cases


def resolve_medxpert_case(*, case_file: Path, case_id: str) -> MedXpertCase:
    cases = load_medxpert_cases(case_file)
    case = cases.get(case_id)
    if case is None:
        raise KeyError(f"Case '{case_id}' not found in {case_file}.")
    return case


def resolve_case_image_path(case: MedXpertCase, *, image_root: Path | None) -> str | None:
    if not case.images:
        return None
    image_name = case.images[0]
    if image_root is None:
        return image_name
    return str(image_root / image_name)


def build_case_inputs(
    case: MedXpertCase,
    *,
    image_root: Path | None,
) -> tuple[str, str | None]:
    question_text = case.format_question()
    image_path = resolve_case_image_path(case, image_root=image_root)
    return question_text, image_path


def resolve_image_id(record: dict, *, image_field: str | None) -> str | None:
    for key in ("image_id", "image_name", "image_filename", "image_file", "img_id", "id"):
        value = record.get(key)
        if value:
            return str(value)
    if image_field:
        image_value = record.get(image_field)
        if isinstance(image_value, dict):
            path = image_value.get("path")
            if path:
                return Path(str(path)).name
    return None


def save_image_payload(payload: object, output_path: Path) -> Path:
    if hasattr(payload, "save"):
        payload.save(output_path)
        return output_path
    if isinstance(payload, dict):
        if payload.get("path"):
            source = Path(str(payload["path"]))
            shutil.copyfile(source, output_path)
            return output_path
        if payload.get("bytes"):
            from PIL import Image

            image = Image.open(io.BytesIO(payload["bytes"]))
            image.save(output_path)
            return output_path
    if isinstance(payload, bytes | bytearray):
        from PIL import Image

        image = Image.open(io.BytesIO(payload))
        image.save(output_path)
        return output_path
    raise ValueError("Unsupported image payload type.")


def download_medxpert_images(
    *,
    dataset_name: str,
    split: str,
    output_dir: Path,
    image_ids: Iterable[str] | None = None,
    max_images: int | None = None,
) -> dict[str, int]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ImportError(
            "datasets is required for downloading MedXpert images. Install it first."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    target_ids = set(str(image_id) for image_id in (image_ids or []))

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    image_field = None
    if getattr(dataset, "features", None):
        for name, feature in dataset.features.items():
            if feature.__class__.__name__ == "Image":
                image_field = name
                break
    if image_field is None:
        if "image" in dataset.column_names:
            image_field = "image"
    if image_field is None:
        raise ValueError("No image column found in dataset.")

    downloaded = 0
    skipped = 0
    for record in dataset:
        image_id = resolve_image_id(record, image_field=image_field)
        if target_ids and image_id not in target_ids:
            skipped += 1
            continue
        payload = record.get(image_field)
        if payload is None:
            skipped += 1
            continue
        if image_id:
            filename = image_id
        else:
            filename = f"medxpert_{downloaded:06d}.jpg"
        output_path = output_dir / filename
        save_image_payload(payload, output_path)
        downloaded += 1
        if max_images is not None and downloaded >= int(max_images):
            break

    return {"downloaded": downloaded, "skipped": skipped}
