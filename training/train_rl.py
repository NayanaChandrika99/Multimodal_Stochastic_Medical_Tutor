# ABOUTME: Tinker RL training entry point for medical visual agent
# ABOUTME: Configures and launches RL training with MedMax data

from __future__ import annotations

import argparse
import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
DEFAULT_LOG_PATH = str(Path(tempfile.gettempdir()) / "medgemma-rl")


@dataclass
class TrainingConfig:
    """Configuration for RL training."""

    # Model
    model_name: str = "google/medgemma-4b-it"
    lora_rank: int = 32
    lora_alpha: int = 64

    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 16
    group_size: int = 4
    max_tokens: int = 512
    gradient_accumulation_steps: int = 4

    # Task configuration
    task_type: str = "chat"  # "chat", "caption", "report"

    # Data paths
    data_dir: str = "data"
    bm25_index_path: str = "data/retrieval/bm25_index"
    vector_index_path: str = "data/retrieval/vector_index"

    # Training control
    max_steps: int = 1000
    eval_every: int = 100
    save_every: int = 500
    warmup_steps: int = 50

    # Logging
    log_path: str = DEFAULT_LOG_PATH
    wandb_project: str = "medical-visual-agent"
    use_wandb: bool = False

    # Reward configuration
    use_biomedclip: bool = True
    format_weight: float = 0.1
    content_weight: float = 0.6
    grounding_weight: float = 0.3

    # Environment
    max_tool_calls: int = 5
    max_trajectory_tokens: int = 16 * 1024

    # Debug
    use_synthetic_data: bool = False
    synthetic_samples: int = 100
    seed: int = 42


def load_components(config: TrainingConfig) -> dict[str, Any]:
    """Load all training components.

    Args:
        config: Training configuration

    Returns:
        Dict with renderer, tool_registry, retriever, action_parser
    """
    from transformers import AutoTokenizer

    from src.agent.actions import ActionParser
    from src.retrieval.hybrid import HybridRetriever
    from src.tools.registry import get_default_registry
    from src.training.renderers.gemma2 import Gemma2VLRenderer

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    logger.info("Creating renderer...")
    renderer = Gemma2VLRenderer(tokenizer, max_tokens=config.max_tokens)

    logger.info("Building tool registry...")
    tool_registry = get_default_registry()

    logger.info("Loading retriever...")
    try:
        retriever = HybridRetriever.from_paths(
            bm25_path=config.bm25_index_path,
            vector_path=config.vector_index_path,
        )
    except Exception as e:
        logger.warning(f"Failed to load retriever: {e}. Using mock.")
        retriever = None

    logger.info("Creating action parser...")
    action_parser = ActionParser()

    return {
        "tokenizer": tokenizer,
        "renderer": renderer,
        "tool_registry": tool_registry,
        "retriever": retriever,
        "action_parser": action_parser,
    }


def build_dataset(
    config: TrainingConfig,
    components: dict[str, Any],
    *,
    reward_function: Any,
) -> Any:
    """Build the training dataset.

    Args:
        config: Training configuration
        components: Loaded components dict

    Returns:
        RLDataset instance
    """
    from src.training.dataset import MedMaxRLDataset, SyntheticMedicalDataset

    if config.use_synthetic_data:
        logger.info("Using synthetic dataset for testing...")
        return SyntheticMedicalDataset(
            batch_size=config.batch_size,
            group_size=config.group_size,
            renderer=components["renderer"],
            tool_registry=components["tool_registry"],
            retriever=components["retriever"],
            action_parser=components["action_parser"],
            reward_function=reward_function,
            num_samples=config.synthetic_samples,
            task_type=config.task_type,
            seed=config.seed,
        )
    else:
        logger.info("Loading MedMax dataset...")
        return MedMaxRLDataset(
            batch_size=config.batch_size,
            group_size=config.group_size,
            renderer=components["renderer"],
            tool_registry=components["tool_registry"],
            retriever=components["retriever"],
            action_parser=components["action_parser"],
            reward_function=reward_function,
            split="train",
            task_type=config.task_type,
            seed=config.seed,
        )


def build_reward_function(config: TrainingConfig) -> Any:
    """Build the reward function.

    Args:
        config: Training configuration

    Returns:
        MedicalRewardFunction instance
    """
    from src.training.rewards import MedicalRewardFunction, RewardConfig

    reward_config = RewardConfig(
        format_weight=config.format_weight,
        content_weight=config.content_weight,
        grounding_weight=config.grounding_weight,
    )

    return MedicalRewardFunction(
        config=reward_config,
        use_biomedclip=config.use_biomedclip,
    )


async def train(config: TrainingConfig) -> None:
    """Main training function.

    Args:
        config: Training configuration
    """
    logger.info("=" * 60)
    logger.info("Starting Medical Visual Agent RL Training")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Task: {config.task_type}")
    logger.info(f"Batch size: {config.batch_size} x {config.group_size} rollouts")
    logger.info(f"Max steps: {config.max_steps}")
    logger.info("=" * 60)

    # Load components
    components = load_components(config)

    # Build reward function
    reward_fn = build_reward_function(config)
    logger.info("Reward function ready")

    # Build dataset
    dataset = build_dataset(config, components, reward_function=reward_fn)
    logger.info(f"Dataset size: {len(dataset)} batches")

    # Check for Tinker
    try:
        from tinker import train as tinker_train

        TINKER_AVAILABLE = True
    except ImportError:
        TINKER_AVAILABLE = False
        logger.warning("Tinker not available - running in dry-run mode")

    if not TINKER_AVAILABLE:
        # Dry run without Tinker
        logger.info("Running dry-run training loop...")
        await _dry_run_training(config, dataset, components, reward_fn)
        return

    # Real Tinker training
    logger.info("Configuring Tinker training...")

    train_config = tinker_train.Config(
        model_name=config.model_name,
        log_path=config.log_path,
        dataset_builder=dataset,
        learning_rate=config.learning_rate,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        eval_every=config.eval_every,
        save_every=config.save_every,
    )

    logger.info("Starting Tinker training...")
    await tinker_train.main(train_config)


async def _dry_run_training(
    config: TrainingConfig,
    dataset: Any,
    components: dict[str, Any],
    reward_fn: Any,
) -> None:
    """Dry run training without Tinker for testing.

    Args:
        config: Training configuration
        dataset: RLDataset instance
        components: Loaded components
        reward_fn: Reward function
    """
    from PIL import Image

    logger.info("Dry-run mode: simulating training loop")

    num_batches = min(5, len(dataset))

    for batch_idx in range(num_batches):
        logger.info(f"\nBatch {batch_idx + 1}/{num_batches}")

        batch = dataset.get_batch(batch_idx)
        logger.info(f"  Got {len(batch)} environment group builders")

        for i, env_builder in enumerate(batch[:2]):  # Only test first 2
            envs = await env_builder()
            logger.info(f"  Group {i}: Created {len(envs)} environments")

            # Test initial observation
            env = envs[0]
            obs, stop = await env.initial_observation()
            logger.info("    Initial observation created")

            # Simulate a simple action
            mock_action = [1, 2, 3, 4, 5]  # Fake token sequence
            result = await env.step(mock_action)
            logger.info(f"    Step result: done={result.episode_done}, reward={result.reward:.3f}")

            # Test reward function
            test_output = "<answer>This is a test response about pneumonia findings.</answer>"
            test_image = Image.new("RGB", (224, 224), color="gray")
            reward, metrics = reward_fn.compute(
                output=test_output,
                image=test_image,
                question="What is shown?",
                task_type=config.task_type,
            )
            logger.info(f"    Reward test: {reward:.3f}, metrics: {metrics}")

    logger.info("\nDry-run complete! Training infrastructure validated.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train medical visual agent with Tinker RL")

    # Model args
    parser.add_argument("--model", default="google/medgemma-4b-it", help="Base model name")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")

    # Training args
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--group-size", type=int, default=4, help="Rollouts per problem")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max training steps")

    # Task args
    parser.add_argument(
        "--task", choices=["chat", "caption", "report"], default="chat", help="Task type"
    )

    # Data args
    parser.add_argument("--bm25-index", default="data/retrieval/bm25_index", help="BM25 index path")
    parser.add_argument(
        "--vector-index", default="data/retrieval/vector_index", help="Vector index path"
    )
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    parser.add_argument(
        "--synthetic-samples", type=int, default=100, help="Number of synthetic samples"
    )

    # Logging args
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH, help="Log directory")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")

    # Reward args
    parser.add_argument("--no-biomedclip", action="store_true", help="Disable BioMedCLIP scoring")

    args = parser.parse_args()

    # Build config
    config = TrainingConfig(
        model_name=args.model,
        lora_rank=args.lora_rank,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_steps=args.max_steps,
        task_type=args.task,
        bm25_index_path=args.bm25_index,
        vector_index_path=args.vector_index,
        use_synthetic_data=args.synthetic,
        synthetic_samples=args.synthetic_samples,
        log_path=args.log_path,
        use_wandb=args.wandb,
        use_biomedclip=not args.no_biomedclip,
    )

    # Run training
    asyncio.run(train(config))


if __name__ == "__main__":
    main()
