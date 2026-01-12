# ABOUTME: Runs ACE-style workflows by repeatedly executing the Medical_Tutor runtime.
# ABOUTME: Writes per-sample JSONL artifacts and persists the active playbook text.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medical_tutor.ace.curator import AceCurator
from medical_tutor.ace.playbook import (
    extract_bullet_ids,
    extract_playbook_bullets,
    get_next_global_id,
    get_playbook_stats,
    infer_bullet_ids_from_episode,
    update_bullet_counts,
)
from medical_tutor.ace.reflector import AceReflector
from medical_tutor.config import Config, load_config
from medical_tutor.contracts import AgentState, new_run_id
from medical_tutor.runtime.runner import GraphRunner


def _extract_answer(state: AgentState) -> str:
    for message in reversed(state.messages):
        if message.get("role") == "assistant":
            return str(message.get("content") or "").strip()
    return ""


class AceRunner:
    def __init__(self, *, config: Config | None = None, call_llm: Any | None = None) -> None:
        self.config = config or load_config()
        self._call_llm = call_llm

    def run(
        self,
        *,
        samples: list[dict[str, Any]],
        playbook_path: str | None,
    ) -> dict[str, str]:
        run_id = new_run_id("ace_run")
        run_dir = Path(self.config.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        playbook_text = ""
        if playbook_path:
            playbook_text = Path(playbook_path).read_text(encoding="utf-8")

        playbook_file = run_dir / "playbook.txt"
        playbook_file.write_text(playbook_text, encoding="utf-8")

        final_playbook_path = run_dir / "final_playbook.txt"
        reflector_trace_path = run_dir / "reflector_trace.jsonl"
        reflector_trace_path.touch(exist_ok=True)
        curator_trace_path = run_dir / "curator_trace.jsonl"
        curator_trace_path.touch(exist_ok=True)

        next_id = get_next_global_id(playbook_text)

        run_config_path = run_dir / "run_config.json"
        run_config_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "token_budget": 80000,
                    "curator_frequency": 1,
                    "use_ground_truth": True,
                    "use_json_mode": True,
                    "initial_playbook_path": playbook_path,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        per_sample_path = run_dir / "per_sample.jsonl"
        graph_runner = GraphRunner(config=self.config, playbook_loader=lambda: playbook_text)
        reflector = AceReflector(model=self.config.ace_model, call_llm=self._call_llm)
        curator = AceCurator(model=self.config.ace_model, call_llm=self._call_llm)

        with per_sample_path.open("a", encoding="utf-8") as handle:
            for step, sample in enumerate(samples, start=1):
                sample_id = str(sample.get("id") or sample.get("sample_id") or "")
                question = str(sample.get("question") or "")
                image_path_val = sample.get("image_path")
                image_path = str(image_path_val) if image_path_val else None
                ground_truth_val = sample.get("answer") or sample.get("target")
                ground_truth = str(ground_truth_val).strip() if ground_truth_val else None

                state = graph_runner.run(question=question, image_path=image_path)
                answer = _extract_answer(state)

                tool_trace = [
                    {"call": call.model_dump(), "result": result.model_dump()}
                    for call, result in zip(state.tool_calls, state.tool_results, strict=False)
                ]
                tool_failure = any(
                    err.code
                    in {"retrieve_error", "orchestrator_error", "tool_exec", "answer_error"}
                    for err in state.errors
                ) or any(not r.ok for r in state.tool_results)

                answer_correct: bool | None
                if ground_truth is None:
                    answer_correct = None
                else:
                    answer_correct = answer.strip().lower() == ground_truth.strip().lower()

                effective_correct = (
                    bool(answer_correct) and not tool_failure
                    if answer_correct is not None
                    else False
                )

                episode_parts = [
                    f"Question: {question}",
                    f"Answer: {answer}",
                    f"Retrieval: {state.retrieval_summary}",
                    json.dumps(tool_trace, ensure_ascii=False),
                    json.dumps([err.model_dump() for err in state.errors], ensure_ascii=False),
                ]
                episode_text = "\n".join(part for part in episode_parts if part)

                bullet_ids_explicit = extract_bullet_ids(episode_text)
                bullet_ids_inferred = infer_bullet_ids_from_episode(
                    playbook=playbook_text, episode_text=episode_text, top_k=8
                )
                bullet_ids: list[str] = []
                seen: set[str] = set()
                for bullet_id in [*bullet_ids_explicit, *bullet_ids_inferred]:
                    if bullet_id in seen:
                        continue
                    seen.add(bullet_id)
                    bullet_ids.append(bullet_id)

                bullets_used = extract_playbook_bullets(playbook_text, bullet_ids)
                environment_feedback = (
                    f"answer_correct={answer_correct}, tool_failure={tool_failure}, "
                    f"errors={[err.code for err in state.errors]}"
                )

                reflection = ""
                bullet_tags: list[dict[str, str]] = []
                operations: list[dict[str, Any]] = []
                if not effective_correct:
                    reflection, bullet_tags, reflector_call_info = reflector.reflect(
                        question=question,
                        predicted_answer=answer,
                        ground_truth=ground_truth,
                        environment_feedback=environment_feedback,
                        tool_metrics={
                            "tool_calls": len(state.tool_calls),
                            "tool_failure": tool_failure,
                        },
                        tool_trace=tool_trace,
                        bullets_used=bullets_used,
                        use_json_mode=True,
                        call_id=f"train_{step}_reflect",
                        log_dir=str(run_dir / "detailed_llm_logs"),
                    )
                    if bullet_tags:
                        playbook_text = update_bullet_counts(playbook_text, bullet_tags)

                    reflector_record = {
                        "step": step,
                        "call_id": f"train_{step}_reflect",
                        "bullet_tags": bullet_tags,
                        "reflector_error": reflector_call_info.get("error")
                        if isinstance(reflector_call_info, dict)
                        else None,
                    }
                    with reflector_trace_path.open("a", encoding="utf-8") as trace_handle:
                        trace_handle.write(json.dumps(reflector_record, ensure_ascii=False) + "\n")

                    playbook_stats = get_playbook_stats(playbook_text)
                    playbook_text, next_id, operations, curator_call_info = curator.curate(
                        current_playbook=playbook_text,
                        recent_reflection=reflection,
                        question_context=f"Question: {question}",
                        current_step=step,
                        total_samples=len(samples),
                        token_budget=80000,
                        playbook_stats=playbook_stats,
                        use_json_mode=True,
                        call_id=f"train_{step}_curate",
                        log_dir=str(run_dir / "detailed_llm_logs"),
                        next_global_id=next_id,
                    )

                    trace_record = {
                        "step": step,
                        "call_id": f"train_{step}_curate",
                        "operation_counts": {
                            str(op.get("type") or "UNKNOWN"): operations.count(op)
                            for op in operations
                        },
                        "curator_error": curator_call_info.get("error")
                        if isinstance(curator_call_info, dict)
                        else None,
                    }
                    with curator_trace_path.open("a", encoding="utf-8") as trace_handle:
                        trace_handle.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

                record = {
                    "step": step,
                    "sample_id": sample_id,
                    "question": question,
                    "image_path": image_path,
                    "run_id": state.run_id,
                    "answer": answer,
                    "ground_truth": ground_truth,
                    "answer_correct": answer_correct,
                    "tool_failure": tool_failure,
                    "effective_correct": effective_correct,
                    "bullet_ids": bullet_ids,
                    "bullet_tags": bullet_tags,
                    "operations": operations,
                    "errors": [err.model_dump() for err in state.errors],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        final_playbook_path.write_text(playbook_text, encoding="utf-8")

        return {
            "run_id": run_id,
            "output_dir": str(run_dir),
            "per_sample_jsonl": str(per_sample_path),
            "playbook_path": str(playbook_file),
            "final_playbook_path": str(final_playbook_path),
            "reflector_trace_jsonl": str(reflector_trace_path),
            "curator_trace_jsonl": str(curator_trace_path),
        }
