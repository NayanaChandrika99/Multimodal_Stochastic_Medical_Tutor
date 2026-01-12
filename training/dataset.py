# ABOUTME: Tinker RL dataset wrapper for MedMax medical visual QA data
# ABOUTME: Converts MedMax samples to Tinker RLDataset batches

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from PIL import Image

from src.training.tinker_env import (
    MedicalDatum,
    MedicalEnvGroupBuilder,
    Renderer,
    Retriever,
)

if TYPE_CHECKING:
    from src.agent.actions import ActionParser
    from src.tools.registry import ToolRegistry
    from src.training.rewards import MedicalRewardFunction


# Try to import tinker types
try:
    from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset

    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False

    # Fallback abstract classes
    class RLDataset:  # type: ignore[no-redef]
        """Fallback RLDataset base class."""

        def get_batch(self, index: int) -> Sequence[Any]:
            raise NotImplementedError

        def __len__(self) -> int:
            raise NotImplementedError

    EnvGroupBuilder = Any


@dataclass
class MedMaxSample:
    """Raw sample from MedMax dataset."""

    image_path: str
    question: str
    answer: str
    task_type: str  # "visual_chat", "image_captioning", "report_generation"
    image_id: str
    metadata: dict[str, Any] | None = None


def load_medmax_samples(
    split: Literal["train", "val", "test"] = "train",
    task_type: str | None = None,
    limit: int | None = None,
    *,
    allowed_credentials: Sequence[str] | None = None,
    image_root: str | None = None,
) -> list[MedMaxSample]:
    """Load samples from MedMax dataset using the existing MedMaxDataset loader.

    Args:
        split: Dataset split to load ("train" or "eval")
        task_type: Filter to specific task type (optional)
        limit: Maximum number of samples to load

    Returns:
        List of MedMaxSample objects
    """
    from src.data.medmax import MedMaxDataset

    # Map split names
    medmax_split = "train" if split == "train" else "eval"

    # Load using existing loader
    dataset = MedMaxDataset(
        use_huggingface=True,
        split=medmax_split,
        max_records=limit,
        image_root=image_root or os.getenv("MEDMAX_IMAGE_ROOT"),
        allowed_credentials=allowed_credentials,
    )

    # Get examples based on task filter
    task_map = {
        "visual_chat": "visual_chat",
        "vqa": "vqa",
        "image_captioning": "image_captioning",
        "report_generation": "report_generation",
        "chat": "visual_chat",
        "caption": "image_captioning",
        "report": "report_generation",
    }

    samples: list[MedMaxSample] = []

    if task_type:
        mapped_task = task_map.get(task_type, task_type)
        task_candidates = [mapped_task]
        if mapped_task == "visual_chat":
            task_candidates.append("vqa")

        examples = []
        remaining = limit
        for task in task_candidates:
            max_examples = None if remaining is None else max(0, int(remaining))
            if max_examples == 0:
                break
            task_examples = dataset.get_generative_task(task, max_examples=max_examples)
            examples.extend(task_examples)
            if remaining is not None:
                remaining -= len(task_examples)
    else:
        # Get all generative tasks
        examples = []
        for task in ["visual_chat", "vqa", "image_captioning", "report_generation"]:
            task_examples = dataset.get_generative_task(task)
            examples.extend(task_examples)
            if limit and len(examples) >= limit:
                examples = examples[:limit]
                break

    # Convert MedMaxExample to MedMaxSample
    for ex in examples:
        samples.append(
            MedMaxSample(
                image_path=str(ex.image_path) if ex.image_path else "",
                question=ex.prompt,
                answer=ex.answer,
                task_type=ex.task_name,
                image_id=str(ex.image_path) if ex.image_path else "",
                metadata=None,
            )
        )

    return samples


def sample_to_datum(
    sample: MedMaxSample,
    preloaded_image: Image.Image | None = None,
) -> MedicalDatum:
    """Convert MedMaxSample to MedicalDatum.

    Args:
        sample: Raw MedMax sample
        preloaded_image: Optional preloaded PIL Image (from MedMaxDataset)

    Returns:
        MedicalDatum for use in MedicalAgentEnv
    """
    # Use preloaded image or load from path
    image = preloaded_image
    if image is None:
        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception:
            # Create placeholder image for testing
            image = Image.new("RGB", (224, 224), color="gray")

    # Map task types to our format
    task_map = {
        "visual_chat": "chat",
        "image_captioning": "caption",
        "report_generation": "report",
    }
    task_type = task_map.get(sample.task_type, "chat")

    return MedicalDatum(
        image=image,
        question=sample.question,
        task_type=task_type,
        ground_truth=sample.answer,
        image_id=sample.image_id,
    )


def example_to_datum(example: Any) -> MedicalDatum:
    """Convert MedMaxExample directly to MedicalDatum.

    This is the preferred conversion path as it preserves the
    already-loaded PIL Image from the dataset.

    Args:
        example: MedMaxExample from MedMaxDataset

    Returns:
        MedicalDatum for use in MedicalAgentEnv
    """
    # Map task types
    task_map = {
        "visual_chat": "chat",
        "image_captioning": "caption",
        "report_generation": "report",
        "chat": "chat",
        "caption": "caption",
        "report": "report",
        "open": "chat",
        "closed": "chat",
    }

    task_type = task_map.get(example.question_type, "chat")
    if hasattr(example, "task_name") and example.task_name:
        task_type = task_map.get(example.task_name, task_type)

    # Get image - prefer already-loaded image
    image = getattr(example, "image", None)
    if image is None and hasattr(example, "image_path") and example.image_path:
        try:
            image = Image.open(example.image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color="gray")

    if image is None:
        raise ValueError(f"No image available for example: {example.prompt[:50]}...")

    return MedicalDatum(
        image=image,
        question=example.prompt,
        task_type=task_type,
        ground_truth=example.answer,
        image_id=str(example.image_path)
        if hasattr(example, "image_path") and example.image_path
        else None,
    )


class MedMaxRLDataset(RLDataset):
    """Tinker RLDataset wrapper for MedMax.

    Loads MedMax samples and provides batches of environment groups
    for RL training.
    """

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: Renderer,
        tool_registry: ToolRegistry,
        retriever: Retriever | None,
        action_parser: ActionParser,
        *,
        reward_function: MedicalRewardFunction | None = None,
        split: Literal["train", "val", "test"] = "train",
        task_type: str | None = None,
        seed: int = 42,
        limit: int | None = None,
        max_loaded_samples: int | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            batch_size: Number of problems per batch
            group_size: Number of parallel rollouts per problem
            renderer: Renderer for message rendering
            tool_registry: Registry of available tools
            retriever: Retriever for RAG (optional)
            action_parser: Parser for model outputs
            split: Dataset split to use
            task_type: Optional filter for specific task type
            seed: Random seed for shuffling
            limit: Maximum samples to load (for testing)
            max_loaded_samples: Maximum samples to load before selecting the training subset
        """
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.tool_registry = tool_registry
        self.retriever = retriever
        self.action_parser = action_parser
        self.reward_function = reward_function

        loaded_limit: int | None = None
        if max_loaded_samples is not None:
            loaded_limit = int(max_loaded_samples)
        elif limit is not None:
            loaded_limit = int(limit)

        # Load and convert samples
        raw_samples = load_medmax_samples(
            split=split,
            task_type=task_type,
            limit=loaded_limit,
            allowed_credentials=["no"],
            image_root=os.getenv("MEDMAX_IMAGE_ROOT"),
        )
        self.data = [sample_to_datum(s) for s in raw_samples]

        # Shuffle with seed
        rng = random.Random(seed)
        rng.shuffle(self.data)

        if limit is not None:
            selected_limit = int(limit)
            if selected_limit > 0:
                self.data = self.data[:selected_limit]

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of environment group builders.

        Args:
            index: Batch index

        Returns:
            List of EnvGroupBuilder instances
        """
        start = index * self.batch_size
        end = start + self.batch_size
        batch_data = self.data[start:end]

        return [
            MedicalEnvGroupBuilder(
                datum=datum,
                renderer=self.renderer,
                tool_registry=self.tool_registry,
                retriever=self.retriever,
                action_parser=self.action_parser,
                num_envs=self.group_size,
                reward_function=self.reward_function,
            )
            for datum in batch_data
        ]

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.data) // self.batch_size


class SyntheticMedicalDataset(RLDataset):
    """Synthetic dataset for testing without MedMax access.

    Generates placeholder samples for development and testing.
    """

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: Renderer,
        tool_registry: ToolRegistry,
        retriever: Retriever | None,
        action_parser: ActionParser,
        *,
        reward_function: MedicalRewardFunction | None = None,
        num_samples: int = 100,
        task_type: str = "chat",
        seed: int = 42,
    ) -> None:
        """Initialize synthetic dataset.

        Args:
            batch_size: Number of problems per batch
            group_size: Number of parallel rollouts per problem
            renderer: Renderer for message rendering
            tool_registry: Registry of available tools
            retriever: Retriever for RAG (optional)
            action_parser: Parser for model outputs
            num_samples: Number of synthetic samples to generate
            task_type: Task type for all samples
            seed: Random seed
        """
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.tool_registry = tool_registry
        self.retriever = retriever
        self.action_parser = action_parser
        self.reward_function = reward_function

        # Generate synthetic samples
        rng = random.Random(seed)
        self.data = self._generate_samples(num_samples, task_type, rng)

    def _generate_samples(self, n: int, task_type: str, rng: random.Random) -> list[MedicalDatum]:
        """Generate synthetic samples for testing."""
        questions_by_task = {
            "chat": [
                "What abnormality is visible in this chest X-ray?",
                "Is there evidence of pneumonia in this image?",
                "Describe the cardiac silhouette in this X-ray.",
                "What findings suggest consolidation?",
                "Are there any nodules visible?",
            ],
            "caption": [
                "Describe this medical image.",
                "Provide a detailed caption for this X-ray.",
                "What does this image show?",
            ],
            "report": [
                "Generate a radiology report for this chest X-ray.",
                "Create a structured report with findings and impression.",
                "Provide a comprehensive radiological analysis.",
            ],
        }

        answers_by_task = {
            "chat": [
                "There is consolidation in the right lower lobe consistent with pneumonia.",
                "The cardiac silhouette is within normal limits.",
                "No significant abnormalities are identified.",
            ],
            "caption": [
                "Chest X-ray showing normal cardiopulmonary structures.",
                "PA and lateral chest radiograph demonstrating clear lungs.",
            ],
            "report": [
                "FINDINGS: The lungs are clear. No pneumothorax or pleural effusion. IMPRESSION: Normal chest radiograph.",
            ],
        }

        questions = questions_by_task.get(task_type, questions_by_task["chat"])
        answers = answers_by_task.get(task_type, answers_by_task["chat"])

        samples = []
        for i in range(n):
            # Create grayscale placeholder image
            image = Image.new("L", (224, 224), color=rng.randint(50, 200))

            samples.append(
                MedicalDatum(
                    image=image.convert("RGB"),
                    question=rng.choice(questions),
                    task_type=task_type,
                    ground_truth=rng.choice(answers),
                    image_id=f"synthetic_{i:06d}",
                )
            )

        return samples

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of environment group builders."""
        start = index * self.batch_size
        end = start + self.batch_size
        batch_data = self.data[start:end]

        return [
            MedicalEnvGroupBuilder(
                datum=datum,
                renderer=self.renderer,
                tool_registry=self.tool_registry,
                retriever=self.retriever,
                action_parser=self.action_parser,
                num_envs=self.group_size,
                reward_function=self.reward_function,
            )
            for datum in batch_data
        ]

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.data) // self.batch_size
