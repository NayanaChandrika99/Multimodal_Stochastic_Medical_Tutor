# ABOUTME: Runs lightweight evaluation loops over datasets using the Medical_Tutor runtime.
# ABOUTME: Writes per-sample JSONL outputs and a summary JSON with aggregate metrics.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medical_tutor.config import Config, load_config
from medical_tutor.contracts import AgentState, new_run_id
from medical_tutor.runtime.runner import GraphRunner


def _extract_answer(state: AgentState) -> str:
    for message in reversed(state.messages):
        if message.get("role") == "assistant":
            return str(message.get("content") or "").strip()
    return ""


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _score_simple(predicted: str, ground_truth: str) -> float:
    pred = _normalize(predicted)
    gt = _normalize(ground_truth)
    if not gt:
        return 0.0
    if pred == gt:
        return 1.0
    return 1.0 if gt in pred else 0.0


class EvalRunner:
    def __init__(self, *, config: Config | None = None) -> None:
        self.config = config or load_config()

    def run(
        self,
        *,
        dataset_path: str,
        max_examples: int | None = None,
    ) -> dict[str, str]:
        run_id = new_run_id("eval_run")
        run_dir = Path(self.config.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(str(dataset_file))

        runner = GraphRunner(config=self.config)
        per_sample_path = run_dir / "per_sample.jsonl"

        scores: list[float] = []
        total = 0

        with (
            dataset_file.open("r", encoding="utf-8") as handle_in,
            per_sample_path.open("w", encoding="utf-8") as handle_out,
        ):
            for line in handle_in:
                if max_examples is not None and total >= int(max_examples):
                    break
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue

                sample_id = str(payload.get("id") or payload.get("sample_id") or "")
                question = str(payload.get("question") or "")
                image_path_val = payload.get("image_path")
                image_path = str(image_path_val) if image_path_val else None
                ground_truth = str(payload.get("answer") or payload.get("ground_truth") or "")

                state = runner.run(question=question, image_path=image_path)
                predicted = _extract_answer(state)
                score = _score_simple(predicted, ground_truth) if ground_truth else 0.0

                record: dict[str, Any] = {
                    "sample_id": sample_id,
                    "question": question,
                    "image_path": image_path,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "score": score,
                    "run_id": state.run_id,
                }
                handle_out.write(json.dumps(record, ensure_ascii=False) + "\n")

                scores.append(score)
                total += 1

        accuracy = float(sum(scores) / total) if total else 0.0
        summary = {
            "run_id": run_id,
            "dataset_path": str(dataset_file),
            "n_examples": total,
            "accuracy": accuracy,
            "per_sample_jsonl": str(per_sample_path),
        }
        summary_path = run_dir / "eval_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return {
            "run_id": run_id,
            "output_dir": str(run_dir),
            "eval_summary": str(summary_path),
        }
