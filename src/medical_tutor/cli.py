# ABOUTME: Command-line entrypoints for the standalone Medical_Tutor package.
# ABOUTME: Supports config inspection, one-shot runs, and trace replay.
"""Command-line interface for Medical_Tutor."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import load_config
from .contracts import new_run_id
from .ops.replay import ReplayRunner
from .ops.tracing import TraceLogger


def _find_workspace_root() -> Path | None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        references_dir = parent / "references"
        if references_dir.exists():
            return parent
    return None


def _resolve_reference_path(*, relative_path: str) -> Path | None:
    workspace_root = _find_workspace_root()
    if workspace_root is None:
        return None
    candidate = workspace_root / relative_path
    return candidate if candidate.exists() else None


def cmd_config() -> int:
    config = load_config()
    print(config.as_lines())
    return 0


def cmd_run(*, question: str, image: str | None) -> int:
    from medical_tutor.runtime.runner import GraphRunner

    runner = GraphRunner()
    result = runner.run(question=question, image_path=image)

    output_dir = Path(load_config().output_dir) / result.run_id
    print(f"run_id={result.run_id}")
    print(f"output_dir={output_dir}")
    if result.messages:
        for message in reversed(result.messages):
            if message.get("role") == "assistant":
                print(f"answer={message.get('content', '')}")
                break
    return 0


def cmd_tutor(
    *,
    profile: str,
    question: str | None,
    image: str | None,
    student_reply: str | None,
    max_turns: int,
    case_id: str | None,
    case_file: str | None,
    image_root: str | None,
) -> int:
    from medical_tutor.runtime.runner import GraphRunner

    if not os.environ.get("OPENAI_API_KEY"):
        print("error=OPENAI_API_KEY is not set; cannot run tutor sessions", file=sys.stderr)
        return 2

    config = load_config()
    run_id = new_run_id("tutor")
    trace_logger = TraceLogger(config.output_dir, run_id)
    trace_logger.log_event(action={"cli": "tutor_start", "profile": profile, "case_id": case_id})

    case_prompt = ""
    case_label: str | None = None
    case_metadata: dict[str, object] | None = None

    if case_id:
        from medical_tutor.medxpert import build_case_inputs, resolve_medxpert_case

        default_case_file = _resolve_reference_path(
            relative_path="references/MedTutor-R1/code/Patient_simulate/MedXpert_patient_script_MM_dev.json"
        )
        if case_file:
            case_path = Path(case_file)
        else:
            if default_case_file is None:
                print(
                    "error=--case-file is required when MedTutor-R1 references are unavailable",
                    file=sys.stderr,
                )
                return 2
            case_path = default_case_file
        try:
            case = resolve_medxpert_case(case_file=case_path, case_id=case_id)
        except (FileNotFoundError, KeyError) as exc:
            print(f"error={exc}", file=sys.stderr)
            return 2
        if case.images and image is None and image_root is None:
            print("error=--image-root is required for multimodal MedXpert cases", file=sys.stderr)
            return 2
        image_root_path = Path(image_root) if image_root else None
        case_prompt, case_image = build_case_inputs(case, image_root=image_root_path)
        if image is None:
            image = case_image
        case_label = case.label or None
        case_metadata = {
            "case_id": case.case_id,
            "label": case_label,
            "body_system": case.body_system,
            "medical_task": case.medical_task,
            "question_type": case.question_type,
            "images": case.images,
        }
    else:
        if not question:
            print("error=--question or --case-id is required", file=sys.stderr)
            return 2
        case_prompt = question

    output_dir = Path(config.output_dir) / run_id
    print(f"run_id={run_id}")
    print(f"output_dir={output_dir}")

    if case_metadata is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        case_path = output_dir / "tutor_case.json"
        case_path.write_text(json.dumps(case_metadata, indent=2), encoding="utf-8")

    if max_turns <= 0:
        return 0

    runner = GraphRunner()
    conversation: list[dict[str, object]] = []
    attempt_count = 0
    pending_student_reply = student_reply if (student_reply and student_reply.strip()) else None
    user_input = case_prompt
    last_student_reply: str | None = None
    last_assessment = None
    tutor_hint_level = 1
    tutor_consecutive_wrong = 0
    tutor_last_grade: str | None = None
    tutor_last_misconception: str | None = None

    try:
        for _turn_idx in range(max_turns):
            result = runner.run(
                question=user_input,
                image_path=image,
                conversation_history=conversation,
                run_id=run_id,
                student_profile=profile,
                student_attempt_count=attempt_count,
                case_prompt=case_prompt,
                case_label=case_label,
                student_reply_text=last_student_reply,
                student_reply_grade=getattr(last_assessment, "grade", None),
                student_reply_misconception=getattr(last_assessment, "misconception", None),
                tutor_hint_level=tutor_hint_level,
                tutor_consecutive_wrong=tutor_consecutive_wrong,
                tutor_last_grade=tutor_last_grade,
                tutor_last_misconception=tutor_last_misconception,
            )
            conversation = list(result.messages)
            tutor_hint_level = result.tutor_hint_level
            tutor_consecutive_wrong = result.tutor_consecutive_wrong
            tutor_last_grade = result.tutor_last_grade
            tutor_last_misconception = result.tutor_last_misconception

            tutor_text = ""
            for message in reversed(result.messages):
                if message.get("role") == "assistant":
                    tutor_text = str(message.get("content") or "")
                    break
            if tutor_text:
                print(f"Tutor: {tutor_text}")

            tutor_action = result.tutor_action or {}
            if tutor_action.get("type") == "REVEAL_ANSWER":
                break

            if pending_student_reply is not None:
                reply = pending_student_reply
                pending_student_reply = None
            else:
                reply = input("Student> ").strip()
            attempt_count += 1
            if case_label:
                from medical_tutor.tutoring import assess_student_mcq_reply, extract_mcq_choice

                if extract_mcq_choice(reply):
                    last_assessment = assess_student_mcq_reply(
                        correct_label=case_label, student_reply=reply
                    )
                else:
                    last_assessment = None
            else:
                last_assessment = None
            last_student_reply = reply
            user_input = reply
    except EOFError:
        return 0
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        return 2
    return 0


def cmd_ui() -> int:
    from medical_tutor.ui.app import launch

    launch()
    return 0


def cmd_ace(*, dataset: str, playbook: str | None) -> int:
    from medical_tutor.ace.runner import AceRunner

    dataset_path = Path(dataset)
    samples = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(__import__("json").loads(line))

    runner = AceRunner()
    result = runner.run(samples=samples, playbook_path=playbook)
    print(f"ace_run_id={result['run_id']}")
    print(f"output_dir={result['output_dir']}")
    return 0


def cmd_eval(*, dataset: str, max_examples: int | None) -> int:
    from medical_tutor.ops.eval import EvalRunner

    runner = EvalRunner()
    result = runner.run(dataset_path=dataset, max_examples=max_examples)
    print(f"eval_run_id={result['run_id']}")
    print(f"output_dir={result['output_dir']}")
    print(f"eval_summary={result['eval_summary']}")
    return 0


def cmd_download_medxpert_images(
    *,
    dataset: str,
    split: str,
    case_file: str | None,
    output_dir: str,
    max_images: int | None,
) -> int:
    from medical_tutor.medxpert import download_medxpert_images, load_medxpert_cases

    image_ids = None
    if case_file:
        cases = load_medxpert_cases(Path(case_file))
        image_ids = {image_id for case in cases.values() for image_id in case.images if image_id}

    result = download_medxpert_images(
        dataset_name=dataset,
        split=split,
        output_dir=Path(output_dir),
        image_ids=image_ids,
        max_images=max_images,
    )
    print(f"downloaded={result['downloaded']}")
    print(f"skipped={result['skipped']}")
    print(f"output_dir={output_dir}")
    return 0


def cmd_build_socratic_dataset(
    *,
    text_cases: str | None,
    mm_cases: str | None,
    output_dir: str | None,
    seed: int,
    use_controller: bool,
    image_root: str | None,
    playbook_path: str | None,
) -> int:
    from medical_tutor.ops.socratic_dataset import DATASET_VERSION, write_dataset

    default_text = _resolve_reference_path(
        relative_path="references/MedTutor-R1/code/Patient_simulate/MedXpert_patient_script_Text_for_test.json"
    )
    default_mm = _resolve_reference_path(
        relative_path="references/MedTutor-R1/code/Patient_simulate/MedXpert_patient_script_MM_for_test.json"
    )
    if text_cases:
        text_path = Path(text_cases)
    else:
        if default_text is None:
            print(
                "error=--text-cases is required when MedTutor-R1 references are unavailable",
                file=sys.stderr,
            )
            return 2
        text_path = default_text

    if mm_cases:
        mm_path = Path(mm_cases)
    else:
        if default_mm is None:
            print(
                "error=--mm-cases is required when MedTutor-R1 references are unavailable",
                file=sys.stderr,
            )
            return 2
        mm_path = default_mm

    config = load_config()
    default_output = Path(config.output_dir) / "datasets" / DATASET_VERSION
    output_path = Path(output_dir) if output_dir else default_output

    result = write_dataset(
        text_path=text_path,
        mm_path=mm_path,
        output_dir=output_path,
        seed=seed,
        use_controller=use_controller,
        image_root=Path(image_root) if image_root else None,
        playbook_path=Path(playbook_path) if playbook_path else None,
    )
    print(f"dataset_version={result['dataset_version']}")
    print(f"cases_path={result['cases_path']}")
    print(f"actions_path={result['actions_path']}")
    print(f"manifest_path={result['manifest_path']}")
    print(f"labeling_mode={result['labeling_mode']}")
    return 0


def cmd_replay(trace_path: str) -> int:
    runner = ReplayRunner(trace_path)
    events = runner.run()
    print(f"Loaded {len(events)} events from {trace_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medical-tutor")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("config", help="Print configuration summary")
    run_parser = subparsers.add_parser("run", help="Run a one-shot image+question")
    run_parser.add_argument("--question", required=True, help="User question")
    run_parser.add_argument("--image", default=None, help="Path to image file (optional)")
    tutor_parser = subparsers.add_parser("tutor", help="Run a multi-turn Socratic tutor session")
    tutor_parser.add_argument(
        "--profile",
        required=True,
        choices=["novice", "medium", "expert"],
        help="Student profile that controls tutoring behavior",
    )
    tutor_parser.add_argument("--question", default=None, help="Case question (MCQ stem)")
    tutor_parser.add_argument("--case-id", default=None, help="MedXpert case id (e.g. MM-2000)")
    tutor_parser.add_argument(
        "--case-file",
        default=None,
        help="Path to MedXpert case JSON (defaults to MM_dev file)",
    )
    tutor_parser.add_argument(
        "--image-root",
        default=None,
        help="Directory that contains MedXpert images (required for MM cases)",
    )
    tutor_parser.add_argument("--image", default=None, help="Path to image file (optional)")
    tutor_parser.add_argument(
        "--student-reply",
        default=None,
        help="Optional first student reply. If omitted, the command reads replies from stdin.",
    )
    tutor_parser.add_argument(
        "--max-turns",
        default=6,
        type=int,
        help="Maximum tutor turns before exiting (0 to only create a trace directory).",
    )
    subparsers.add_parser("ui", help="Launch the UI")
    ace_parser = subparsers.add_parser("ace", help="Run ACE workflow")
    ace_parser.add_argument("--dataset", required=True, help="Path to a JSONL dataset")
    ace_parser.add_argument("--playbook", default=None, help="Optional playbook text file")
    eval_parser = subparsers.add_parser("eval", help="Run evaluation workflow")
    eval_parser.add_argument("--dataset", required=True, help="Path to a JSONL dataset")
    eval_parser.add_argument("--max-examples", default=None, type=int, help="Optional max examples")
    download_parser = subparsers.add_parser(
        "download-medxpert-images", help="Download MedXpertQA images from Hugging Face"
    )
    download_parser.add_argument(
        "--dataset", default="TsinghuaC3I/MedXpertQA", help="Hugging Face dataset name"
    )
    download_parser.add_argument("--split", default="train", help="Dataset split to stream")
    download_parser.add_argument(
        "--case-file",
        default=None,
        help="Optional MedXpert case JSON to limit which images are downloaded",
    )
    download_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write downloaded image files",
    )
    download_parser.add_argument(
        "--max-images",
        default=None,
        type=int,
        help="Optional limit on the number of images to download",
    )
    build_dataset_parser = subparsers.add_parser(
        "build-socratic-dataset", help="Build Socratic tutor SFT datasets"
    )
    build_dataset_parser.add_argument(
        "--text-cases",
        default=None,
        help="Path to MedXpert text case JSON (defaults to MedTutor-R1 test file)",
    )
    build_dataset_parser.add_argument(
        "--mm-cases",
        default=None,
        help="Path to MedXpert MM case JSON (defaults to MedTutor-R1 test file)",
    )
    build_dataset_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the dataset (defaults to outputs/datasets/socratic_tutor_v1)",
    )
    build_dataset_parser.add_argument(
        "--seed",
        default=1234,
        type=int,
        help="Seed for deterministic dataset generation",
    )
    build_dataset_parser.add_argument(
        "--use-controller",
        action="store_true",
        help="Label actions with the tutor controller when OPENAI_API_KEY is set",
    )
    build_dataset_parser.add_argument(
        "--image-root",
        default=None,
        help="Optional image root for controller labeling on MM cases",
    )
    build_dataset_parser.add_argument(
        "--playbook-path",
        default=None,
        help="Optional playbook file for controller labeling",
    )
    replay_parser = subparsers.add_parser("replay", help="Replay a trace JSONL file")
    replay_parser.add_argument("trace_path", help="Path to trace.jsonl")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "config":
        return cmd_config()
    if args.command == "run":
        return cmd_run(question=args.question, image=args.image)
    if args.command == "tutor":
        return cmd_tutor(
            profile=args.profile,
            question=args.question,
            image=args.image,
            student_reply=args.student_reply,
            max_turns=args.max_turns,
            case_id=args.case_id,
            case_file=args.case_file,
            image_root=args.image_root,
        )
    if args.command == "ui":
        return cmd_ui()
    if args.command == "ace":
        return cmd_ace(dataset=args.dataset, playbook=args.playbook)
    if args.command == "eval":
        return cmd_eval(dataset=args.dataset, max_examples=args.max_examples)
    if args.command == "download-medxpert-images":
        return cmd_download_medxpert_images(
            dataset=args.dataset,
            split=args.split,
            case_file=args.case_file,
            output_dir=args.output_dir,
            max_images=args.max_images,
        )
    if args.command == "build-socratic-dataset":
        return cmd_build_socratic_dataset(
            text_cases=args.text_cases,
            mm_cases=args.mm_cases,
            output_dir=args.output_dir,
            seed=args.seed,
            use_controller=args.use_controller,
            image_root=args.image_root,
            playbook_path=args.playbook_path,
        )
    if args.command == "replay":
        return cmd_replay(args.trace_path)

    parser.print_help()
    return 0


def main_ui() -> int:
    return main(["ui", *sys.argv[1:]])


def main_ace() -> int:
    return main(["ace", *sys.argv[1:]])


def main_eval() -> int:
    return main(["eval", *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
