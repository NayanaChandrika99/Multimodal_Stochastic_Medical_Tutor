# ABOUTME: Reward functions for Tinker RL training of medical visual agent
# ABOUTME: Includes BioMedCLIPScore, format checking, and grounding validation

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch
from PIL import Image

if TYPE_CHECKING:
    pass


class RewardFunction(Protocol):
    """Protocol for reward functions."""

    def compute(
        self,
        output: str,
        image: Image.Image,
        question: str,
        task_type: str,
        ground_truth: str | None,
        retrieved_passages: list[dict],
    ) -> tuple[float, dict[str, float]]:
        """Compute reward for model output.

        Returns:
            Tuple of (total_reward, metrics_dict)
        """
        ...


@dataclass
class RewardConfig:
    """Configuration for reward computation."""

    # Component weights
    format_weight: float = 0.1
    content_weight: float = 0.6
    grounding_weight: float = 0.3

    # Thresholds
    min_output_length: int = 10
    max_output_length: int = 1000
    min_citations_for_full_credit: int = 2

    # BioMedCLIP settings
    biomedclip_model_name: str = (
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"  # pragma: allowlist secret
    )
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BioMedCLIPScorer:
    """Score image-text alignment using BioMedCLIP.

    Uses Microsoft's BioMedCLIP model trained on PMC-OA for
    computing similarity between medical images and text.
    """

    def __init__(
        self,
        model_name: str = (
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"  # pragma: allowlist secret
        ),
        device: str | None = None,
    ) -> None:
        """Initialize BioMedCLIP scorer.

        Args:
            model_name: HuggingFace model ID for BioMedCLIP
            device: Device to run on (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Any | None = None
        self._processor: Any | None = None

    def _load_model(self) -> None:
        """Lazy load the model."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModel, AutoProcessor

            processor = AutoProcessor.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name).to(self.device)
            model.eval()
            self._processor = processor
            self._model = model
        except Exception as e:
            raise RuntimeError(f"Failed to load BioMedCLIP: {e}") from e

    @torch.no_grad()
    def score(self, image: Image.Image, text: str) -> float:
        """Compute CLIP similarity score between image and text.

        Args:
            image: PIL Image
            text: Text description/caption

        Returns:
            Similarity score in [0, 1] range
        """
        self._load_model()

        if self._processor is None or self._model is None:
            return 0.5  # Fallback if model fails to load

        try:
            # Prepare inputs
            inputs = self._processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            # Get embeddings
            outputs = self._model(**inputs)

            # Compute cosine similarity
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # Normalize
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # Cosine similarity
            similarity = (image_embeds @ text_embeds.T).squeeze()

            # Convert to [0, 1] range (CLIP scores are typically in [-1, 1])
            score = (similarity.item() + 1) / 2

            return float(score)

        except Exception:
            return 0.5  # Fallback on error


class FormatChecker:
    """Check if output follows expected format for task type."""

    # Expected tags by task type
    EXPECTED_TAGS = {
        "chat": ("<answer>", "</answer>"),
        "caption": ("<caption>", "</caption>"),
        "report": ("<report>", "</report>"),
    }

    def check(self, output: str, task_type: str) -> float:
        """Check format compliance.

        Args:
            output: Model output text
            task_type: Type of task (chat, caption, report)

        Returns:
            Score in [0, 1]: 1.0 for correct format, 0.5 for partial, 0.0 for missing
        """
        tags = self.EXPECTED_TAGS.get(task_type, self.EXPECTED_TAGS["chat"])
        open_tag, close_tag = tags

        if open_tag in output and close_tag in output:
            return 1.0
        elif open_tag in output or close_tag in output:
            return 0.5
        else:
            return 0.0

    def extract_content(self, output: str, task_type: str) -> str:
        """Extract content from within tags.

        Args:
            output: Model output text
            task_type: Type of task

        Returns:
            Content within tags, or full output if no tags found
        """
        tags = self.EXPECTED_TAGS.get(task_type, self.EXPECTED_TAGS["chat"])
        open_tag, close_tag = tags

        pattern = f"{re.escape(open_tag)}(.*?){re.escape(close_tag)}"
        match = re.search(pattern, output, re.DOTALL)

        if match:
            return match.group(1).strip()
        return output


class GroundingChecker:
    """Check if output is grounded in retrieved passages."""

    def __init__(self, min_citations: int = 2) -> None:
        """Initialize grounding checker.

        Args:
            min_citations: Minimum citations for full credit
        """
        self.min_citations = min_citations

    def check(self, output: str, retrieved_passages: list[dict]) -> float:
        """Check grounding in retrieved passages.

        Args:
            output: Model output text
            retrieved_passages: List of retrieved passage dicts

        Returns:
            Score in [0, 1] based on grounding quality
        """
        if not retrieved_passages:
            # No passages to ground - can't evaluate
            return 0.5

        scores = []

        # 1. Citation score - explicit [1], [2] references
        citation_score = self._check_citations(output)
        scores.append(citation_score)

        # 2. Content overlap - key terms from passages in output
        overlap_score = self._check_overlap(output, retrieved_passages)
        scores.append(overlap_score)

        return sum(scores) / len(scores)

    def _check_citations(self, output: str) -> float:
        """Check for explicit citation markers like [1], [2]."""
        citations = re.findall(r"\[(\d+)\]", output)
        unique_citations = len(set(citations))

        if unique_citations >= self.min_citations:
            return 1.0
        elif unique_citations > 0:
            return unique_citations / self.min_citations
        else:
            return 0.0

    def _check_overlap(self, output: str, passages: list[dict]) -> float:
        """Check for key term overlap with passages."""
        output_lower = output.lower()
        overlap_count = 0

        for passage in passages[:5]:  # Check top 5 passages
            content = passage.get("content", passage.get("text", ""))
            # Extract key terms (words > 5 chars)
            key_terms = [w.lower() for w in content.split() if len(w) > 5][:15]

            for term in key_terms:
                if term in output_lower:
                    overlap_count += 1

        # Normalize: expect ~10 overlapping terms for full credit
        return min(1.0, overlap_count / 10)


class MedicalRewardFunction:
    """Combined reward function for medical visual agent.

    Combines format checking, content quality (BioMedCLIP),
    and grounding validation into a weighted score.
    """

    def __init__(
        self,
        config: RewardConfig | None = None,
        use_biomedclip: bool = True,
    ) -> None:
        """Initialize reward function.

        Args:
            config: Reward configuration
            use_biomedclip: Whether to use BioMedCLIP for content scoring
        """
        self.config = config or RewardConfig()
        self.format_checker = FormatChecker()
        self.grounding_checker = GroundingChecker(
            min_citations=self.config.min_citations_for_full_credit
        )

        self._biomedclip: BioMedCLIPScorer | None = None
        if use_biomedclip:
            try:
                self._biomedclip = BioMedCLIPScorer(
                    model_name=self.config.biomedclip_model_name,
                    device=self.config.device,
                )
            except Exception:
                pass  # Fall back to heuristic scoring

    def compute(
        self,
        output: str,
        image: Image.Image,
        question: str,
        task_type: str,
        ground_truth: str | None = None,
        retrieved_passages: list[dict] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute total reward for model output.

        Args:
            output: Model's generated output
            image: Input image
            question: Original question
            task_type: Task type (chat, caption, report)
            ground_truth: Optional ground truth for reference
            retrieved_passages: Passages retrieved by the agent

        Returns:
            Tuple of (total_reward, metrics_dict)
        """
        metrics: dict[str, float] = {}
        passages = retrieved_passages or []

        # 1. Format score
        format_score = self.format_checker.check(output, task_type)
        metrics["format"] = format_score

        # 2. Content score (BioMedCLIP or heuristic)
        content = self.format_checker.extract_content(output, task_type)
        if self._biomedclip is not None and task_type in ("caption", "report"):
            content_score = self._biomedclip.score(image, content)
        else:
            content_score = self._heuristic_content_score(content, task_type)
        metrics["content"] = content_score

        # 3. Grounding score
        grounding_score = self.grounding_checker.check(output, passages)
        metrics["grounding"] = grounding_score

        # 4. Length penalty
        length_score = self._length_score(content)
        metrics["length"] = length_score

        # Weighted combination
        total = (
            self.config.format_weight * format_score
            + self.config.content_weight * content_score
            + self.config.grounding_weight * grounding_score
        )

        # Apply length penalty as multiplier
        total *= length_score

        metrics["total"] = total
        return total, metrics

    def _heuristic_content_score(self, content: str, task_type: str) -> float:
        """Simple heuristic for content quality when CLIP unavailable."""
        word_count = len(content.split())

        # Check for medical terminology
        medical_terms = [
            "finding",
            "impression",
            "normal",
            "abnormal",
            "indicate",
            "suggest",
            "consistent",
            "diagnosis",
            "patient",
            "tissue",
            "lesion",
            "region",
            "opacity",
            "infiltrate",
            "cardiomegaly",
            "effusion",
            "nodule",
        ]
        term_count = sum(1 for term in medical_terms if term in content.lower())

        # Combine signals
        length_score = min(1.0, word_count / 50) if word_count < 50 else 1.0
        term_score = min(1.0, term_count / 4)

        return 0.4 * length_score + 0.6 * term_score

    def _length_score(self, content: str) -> float:
        """Compute length penalty/bonus."""
        word_count = len(content.split())

        if word_count < self.config.min_output_length:
            # Too short - penalize
            return 0.5
        elif word_count > self.config.max_output_length:
            # Too long - slight penalty
            return 0.8
        else:
            return 1.0


# Convenience function for creating reward function
def create_reward_function(
    use_biomedclip: bool = True,
    **config_kwargs: Any,
) -> MedicalRewardFunction:
    """Create a configured MedicalRewardFunction.

    Args:
        use_biomedclip: Whether to use BioMedCLIP for content scoring
        **config_kwargs: Override RewardConfig parameters

    Returns:
        Configured MedicalRewardFunction
    """
    config = RewardConfig(**config_kwargs) if config_kwargs else None
    return MedicalRewardFunction(config=config, use_biomedclip=use_biomedclip)
