# ABOUTME: Evaluates tool-policy rollouts using Tinker sampling clients (pre/post LoRA checkpoints)
# ABOUTME: Runs N episodes and summarizes tool/retrieval/web-shaping metrics for comparison

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.training.rollout_eval import EpisodeStats, accumulate_metrics, summarize_rollouts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalConfig:
    model_name: str
    renderer_name: str
    task_type: Literal["chat", "caption", "report"]
    episodes: int
    max_tokens: int
    temperature: float
    bm25_index_path: str
    vector_index_path: str
    use_synthetic_data: bool
    seed: int


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate tool-policy rollouts by running episodes with a base model and/or "
            "a trained LoRA checkpoint on Tinker. Reports metrics like retrieval_called, "
            "web_search_called, repeat_region, curiosity."
        )
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="Tinker base model name (must be supported).",
    )
    parser.add_argument(
        "--renderer",
        default="qwen3_vl_instruct",
        help="tinker-cookbook renderer name (e.g., qwen3_vl_instruct).",
    )
    parser.add_argument("--task", choices=["chat", "caption", "report"], default="chat")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument(
        "--bm25-index",
        default="data/retrieval/bm25_v2.sample.pkl",
        help="Path to BM25 index pickle.",
    )
    parser.add_argument(
        "--vector-index",
        default="data/retrieval/vector_biomedclip_v2.sample",
        help="Path to vector index directory.",
    )
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic samples.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--pre",
        default="base",
        help="Pre policy: 'base' or a tinker model_path (tinker://.../weights/...).",
    )
    parser.add_argument(
        "--post",
        default="",
        help="Post policy: optional tinker model_path (tinker://.../weights/...).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path. If empty, prints only.",
    )
    return parser


def _load_renderer(model_name: str, renderer_name: str) -> Any:
    from tinker_cookbook import renderers, tokenizer_utils
    from tinker_cookbook.image_processing_utils import get_image_processor

    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    image_processor = get_image_processor(model_name)
    return renderers.get_renderer(renderer_name, tokenizer, image_processor=image_processor)


def _build_training_registry() -> Any:
    from src.tools.crop import CropTool
    from src.tools.enhance import EnhanceTool
    from src.tools.point_crop import PointCropTool
    from src.tools.registry import ToolRegistry
    from src.tools.zoom import ZoomTool

    registry = ToolRegistry()
    registry.register("crop", CropTool())
    registry.register("zoom", ZoomTool())
    registry.register("enhance", EnhanceTool())
    registry.register("point_crop", PointCropTool())
    return registry


def _load_retriever(bm25_path: str, vector_path: str) -> Any:
    from src.retrieval.hybrid import HybridRetriever

    bm25 = Path(bm25_path)
    vector = Path(vector_path)
    if not bm25.exists():
        raise FileNotFoundError(f"BM25 index not found at {bm25}.")
    if not vector.exists():
        raise FileNotFoundError(f"Vector index not found at {vector}.")
    return HybridRetriever.from_paths(bm25_path=bm25, vector_path=vector)


async def _build_envs(cfg: EvalConfig) -> list[Any]:
    from src.agent.actions import ActionParser
    from src.training.dataset import MedMaxRLDataset, SyntheticMedicalDataset
    from src.training.rewards import MedicalRewardFunction, RewardConfig

    renderer = _load_renderer(cfg.model_name, cfg.renderer_name)
    tool_registry = _build_training_registry()
    retriever = _load_retriever(cfg.bm25_index_path, cfg.vector_index_path)
    action_parser = ActionParser()

    reward_cfg = RewardConfig(format_weight=0.0, content_weight=0.0, grounding_weight=0.0)
    reward_fn = MedicalRewardFunction(config=reward_cfg, use_biomedclip=False)

    dataset: Any
    if cfg.use_synthetic_data:
        dataset = SyntheticMedicalDataset(
            batch_size=1,
            group_size=1,
            renderer=renderer,
            tool_registry=tool_registry,
            retriever=retriever,
            action_parser=action_parser,
            reward_function=reward_fn,
            num_samples=cfg.episodes,
            task_type=cfg.task_type,
            seed=cfg.seed,
        )
    else:
        dataset = MedMaxRLDataset(
            batch_size=1,
            group_size=1,
            renderer=renderer,
            tool_registry=tool_registry,
            retriever=retriever,
            action_parser=action_parser,
            reward_function=reward_fn,
            split="train",
            task_type=cfg.task_type,
            seed=cfg.seed,
            limit=cfg.episodes,
        )

    envs: list[Any] = []
    for batch_idx in range(len(dataset)):
        builders = dataset.get_batch(batch_idx)
        if not builders:
            continue
        builder = builders[0]
        group_envs = await builder.make_envs()
        envs.extend(list(group_envs))
        if len(envs) >= cfg.episodes:
            break
    return envs[: cfg.episodes]


def _build_sampling_client(*, base_model: str, model_path: str | None) -> Any:
    import tinker

    service_client = tinker.ServiceClient()
    if model_path:
        return service_client.create_sampling_client(model_path=model_path)
    return service_client.create_sampling_client(base_model=base_model)


async def _run_single_episode(env: Any, token_completer: Any) -> EpisodeStats:
    ob, stop = await env.initial_observation()
    total_reward = 0.0
    metrics_sum: dict[str, float] = {}
    steps = 0

    while True:
        action = await token_completer(ob, stop)
        step_result = await env.step(action.tokens)
        steps += 1
        total_reward += float(step_result.reward)
        accumulate_metrics(step_result.metrics, into=metrics_sum)
        if step_result.episode_done:
            break
        ob = step_result.next_observation
        stop = step_result.next_stop_condition

    return EpisodeStats(total_reward=total_reward, steps=steps, metrics_sum=metrics_sum)


async def _evaluate_policy(cfg: EvalConfig, *, model_path: str | None) -> dict[str, object]:
    from tinker_cookbook.completers import TinkerTokenCompleter

    envs = await _build_envs(cfg)
    sampling_client = _build_sampling_client(base_model=cfg.model_name, model_path=model_path)
    token_completer = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )

    episodes: list[EpisodeStats] = []
    for env in envs:
        episodes.append(await _run_single_episode(env, token_completer))

    summary = summarize_rollouts(episodes)
    return {
        "model_name": cfg.model_name,
        "model_path": model_path,
        "episodes": summary.episodes,
        "mean_total_reward": summary.mean_total_reward,
        "mean_steps": summary.mean_steps,
        "mean_metrics_sum": summary.mean_metrics_sum,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = _build_arg_parser().parse_args()

    cfg = EvalConfig(
        model_name=str(args.model),
        renderer_name=str(args.renderer),
        task_type=args.task,
        episodes=int(args.episodes),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        bm25_index_path=str(args.bm25_index),
        vector_index_path=str(args.vector_index),
        use_synthetic_data=bool(args.synthetic),
        seed=int(args.seed),
    )

    pre_model_path = None if str(args.pre).strip().lower() == "base" else str(args.pre).strip()
    post_model_path = str(args.post).strip() or None

    results: dict[str, object] = {"pre": None, "post": None}
    results["pre"] = asyncio.run(_evaluate_policy(cfg, model_path=pre_model_path))
    if post_model_path is not None:
        results["post"] = asyncio.run(_evaluate_policy(cfg, model_path=post_model_path))

    payload = json.dumps(results, indent=2, sort_keys=True, default=str)
    print(payload)
    if str(args.output).strip():
        out_path = Path(str(args.output)).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        logger.info("Wrote results to %s", out_path)


if __name__ == "__main__":
    main()
