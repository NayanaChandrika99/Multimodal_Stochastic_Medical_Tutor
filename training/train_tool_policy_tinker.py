# ABOUTME: Runs small Tinker-cookbook RL experiments to learn tool-use policy with a supported VLM
# ABOUTME: Trains a Qwen3-VL LoRA policy that emits our <tool_call> actions for downstream execution

from __future__ import annotations

import argparse
import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)
DEFAULT_LOG_PATH = str(Path(tempfile.gettempdir()) / "med-visual-tutor" / "tool_policy_rl")


@dataclass(frozen=True)
class ToolPolicyConfig:
    model_name: str
    renderer_name: str
    task_type: Literal["chat", "caption", "report"]
    batch_size: int
    group_size: int
    num_batches: int
    medmax_pool_size: int
    learning_rate: float
    lora_rank: int
    max_tokens: int
    temperature: float
    log_path: str
    bm25_index_path: str
    vector_index_path: str
    use_synthetic_data: bool
    synthetic_samples: int
    seed: int


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small RL run on Tinker for tool-use policy learning using a Tinker-supported VLM "
            "(e.g., Qwen3-VL). This is intended for cheap validation runs, not long training."
        )
    )

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="Tinker model name (must be in Tinker model lineup).",
    )
    parser.add_argument(
        "--renderer",
        default="qwen3_vl_instruct",
        help="tinker-cookbook renderer name (e.g., qwen3_vl_instruct).",
    )
    parser.add_argument("--task", choices=["chat", "caption", "report"], default="chat")

    parser.add_argument("--batch-size", type=int, default=2, help="Problems per batch.")
    parser.add_argument("--group-size", type=int, default=2, help="Rollouts per problem.")
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="Number of batches to train on (controls total steps/cost).",
    )
    parser.add_argument(
        "--medmax-pool-size",
        type=int,
        default=2000,
        help="How many MedMax examples to load from disk/cache before selecting the training subset.",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="LoRA learning rate.")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Max tokens per sampled action."
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH)

    parser.add_argument(
        "--bm25-index",
        default="data/retrieval/bm25_v2.sample.pkl",
        help="Path to BM25 index pickle (required for retrieval/web gating).",
    )
    parser.add_argument(
        "--vector-index",
        default="data/retrieval/vector_biomedclip_v2.sample",
        help="Path to vector index directory (required for retrieval/web gating).",
    )

    parser.add_argument("--synthetic", action="store_true", help="Use synthetic samples.")
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=0,
        help="Override number of synthetic samples (0 => derived from batch-size * num-batches).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


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


def _load_renderer(model_name: str, renderer_name: str, *, max_tokens: int) -> Any:
    try:
        from tinker_cookbook import renderers, tokenizer_utils
        from tinker_cookbook.image_processing_utils import get_image_processor
    except ImportError as exc:
        msg = (
            "tinker-cookbook is required for this training entrypoint. "
            "Install it in the active environment on your cluster."
        )
        raise ImportError(msg) from exc

    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    image_processor = get_image_processor(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer, image_processor=image_processor)
    renderer.max_tokens = max_tokens  # used only by some completers; safe to set
    return renderer


def _build_reward_function() -> Any:
    from src.training.rewards import MedicalRewardFunction, RewardConfig

    config = RewardConfig(
        format_weight=0.0,
        content_weight=0.0,
        grounding_weight=0.0,
    )
    return MedicalRewardFunction(config=config, use_biomedclip=False)


def _build_dataset_builder(cfg: ToolPolicyConfig) -> Any:
    try:
        from tinker_cookbook.rl.types import RLDatasetBuilder
    except ImportError:
        RLDatasetBuilder = object

    import chz

    @chz.chz
    class DatasetBuilder(RLDatasetBuilder):
        async def __call__(self) -> tuple[Any, Any | None]:
            from src.agent.actions import ActionParser
            from src.training.dataset import MedMaxRLDataset, SyntheticMedicalDataset

            renderer = _load_renderer(cfg.model_name, cfg.renderer_name, max_tokens=cfg.max_tokens)
            tool_registry = _build_training_registry()
            retriever = _load_retriever(cfg.bm25_index_path, cfg.vector_index_path)
            action_parser = ActionParser()
            reward_fn = _build_reward_function()

            limit = cfg.batch_size * cfg.num_batches
            if cfg.use_synthetic_data:
                num_samples = cfg.synthetic_samples or limit
                dataset = SyntheticMedicalDataset(
                    batch_size=cfg.batch_size,
                    group_size=cfg.group_size,
                    renderer=renderer,
                    tool_registry=tool_registry,
                    retriever=retriever,
                    action_parser=action_parser,
                    reward_function=reward_fn,
                    num_samples=num_samples,
                    task_type=cfg.task_type,
                    seed=cfg.seed,
                )
                return dataset, None

            dataset = MedMaxRLDataset(
                batch_size=cfg.batch_size,
                group_size=cfg.group_size,
                renderer=renderer,
                tool_registry=tool_registry,
                retriever=retriever,
                action_parser=action_parser,
                reward_function=reward_fn,
                split="train",
                # Tool-policy learning should be robust across tasks. We load a mixed pool
                # (chat/caption/report) rather than filtering to a single task_name.
                task_type=None,
                seed=cfg.seed,
                limit=limit,
                max_loaded_samples=max(limit, cfg.medmax_pool_size),
            )
            return dataset, None

    return DatasetBuilder()


async def _train(cfg: ToolPolicyConfig) -> None:
    try:
        from tinker_cookbook.rl import train as rl_train
    except ImportError as exc:
        msg = (
            "tinker-cookbook is required for this training entrypoint. "
            "Install it in the active environment on your cluster."
        )
        raise ImportError(msg) from exc

    dataset_builder = _build_dataset_builder(cfg)
    train_config = rl_train.Config(
        model_name=cfg.model_name,
        dataset_builder=dataset_builder,
        learning_rate=cfg.learning_rate,
        lora_rank=cfg.lora_rank,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        log_path=cfg.log_path,
        eval_every=0,
        save_every=max(1, cfg.num_batches),
    )

    logger.info(
        "Starting tool-policy RL with model=%s renderer=%s", cfg.model_name, cfg.renderer_name
    )
    logger.info(
        "Batches=%s batch_size=%s group_size=%s", cfg.num_batches, cfg.batch_size, cfg.group_size
    )
    logger.info("Retrieval bm25=%s vector=%s", cfg.bm25_index_path, cfg.vector_index_path)
    logger.info("Log path: %s", cfg.log_path)
    await rl_train.main(train_config)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg = ToolPolicyConfig(
        model_name=str(args.model),
        renderer_name=str(args.renderer),
        task_type=args.task,
        batch_size=int(args.batch_size),
        group_size=int(args.group_size),
        num_batches=int(args.num_batches),
        medmax_pool_size=int(args.medmax_pool_size),
        learning_rate=float(args.lr),
        lora_rank=int(args.lora_rank),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        log_path=str(args.log_path),
        bm25_index_path=str(args.bm25_index),
        vector_index_path=str(args.vector_index),
        use_synthetic_data=bool(args.synthetic),
        synthetic_samples=int(args.synthetic_samples),
        seed=int(args.seed),
    )
    asyncio.run(_train(cfg))


if __name__ == "__main__":
    main()
