# ABOUTME: Tinker RL environment wrapper for medical visual agent
# ABOUTME: Wraps agent loop as async Env for RL training

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from PIL import Image

from src.agent.prompts import TASK_PROMPTS

# Tinker imports (with fallback for testing)
try:
    import tinker
    from tinker import ModelInput

    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False

    # Fallback ModelInput for testing
    @dataclass
    class ModelInput:  # type: ignore[no-redef]
        """Fallback for tinker.ModelInput when tinker not installed."""

        chunks: list[Any] = field(default_factory=list)
        length: int = 0

        @classmethod
        def empty(cls) -> ModelInput:
            return cls(chunks=[], length=0)


if TYPE_CHECKING:
    from src.agent.actions import Action, ActionParser
    from src.tools.registry import ToolRegistry
    from src.training.rewards import MedicalRewardFunction


class Renderer(Protocol):
    tokenizer: Any

    def build_generation_prompt(self, messages: Sequence[dict[str, Any]]) -> ModelInput: ...

    def decode(self, action: ModelInput) -> str: ...

    def get_stop_sequences(self) -> list[str]: ...


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[dict[str, Any]]: ...


@dataclass
class StepResult:
    """Result of taking a step in the environment."""

    reward: float
    episode_done: bool
    next_observation: ModelInput
    next_stop_condition: Any = None
    metrics: dict[str, float] = field(default_factory=dict)
    logs: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.logs and self.metrics:
            self.logs = dict(self.metrics)
        elif not self.metrics and self.logs:
            self.metrics = dict(self.logs)


@dataclass
class MedicalDatum:
    """Single sample from MedMax dataset."""

    image: Image.Image
    question: str
    task_type: str  # "chat", "caption", "report"
    ground_truth: str | None = None
    image_id: str | None = None


# Use prompts from src/agent/prompts.py which include proper <tool_call> syntax
SYSTEM_PROMPTS = TASK_PROMPTS


class MedicalAgentEnv:
    """Tinker RL environment for medical visual QA.

    Wraps our agent loop as an async environment for RL training.
    Each episode is a single question-answer interaction with tool use.
    """

    def __init__(
        self,
        datum: MedicalDatum,
        renderer: Renderer,
        tool_registry: ToolRegistry,
        retriever: Retriever | None,
        action_parser: ActionParser,
        *,
        max_tool_calls: int = 5,
        max_trajectory_tokens: int = 16 * 1024,
        format_reward_weight: float = 0.1,
        reward_function: MedicalRewardFunction | None = None,
    ) -> None:
        """Initialize the environment.

        Args:
            datum: The sample to process (image, question, task type)
            renderer: Renderer for message rendering
            tool_registry: Registry of available tools
            retriever: Retriever for RAG (optional)
            action_parser: Parser for model outputs
            max_tool_calls: Maximum tool calls before forcing termination
            max_trajectory_tokens: Maximum tokens before truncation
            format_reward_weight: Weight for format compliance in reward
            reward_function: External reward function (if None, uses internal logic)
        """
        self.datum = datum
        self.renderer = renderer
        self.tool_registry = tool_registry
        self.retriever = retriever
        self.action_parser = action_parser
        self.max_tool_calls = max_tool_calls
        self.max_trajectory_tokens = max_trajectory_tokens
        self.format_reward_weight = format_reward_weight
        self.reward_function = reward_function

        # Episode state
        self.past_messages: list[dict[str, Any]] = []
        self.tool_calls_made: int = 0
        self.retrieved_passages: list[dict[str, Any]] = []
        self.current_image: Image.Image = datum.image  # Track image after tool calls
        self._seen_tool_calls: set[str] = set()

        # Tool-use policy shaping configuration (small rewards/penalties).
        self._db_score_threshold = 0.02
        self._web_search_penalty = 0.02
        self._image_tool_penalty = 0.005
        self._repeat_tool_penalty = 0.01
        self._curiosity_weight = 0.03
        self._retrieval_weight = 0.02

    @property
    def stop_condition(self) -> Any:
        """Return stop condition for generation."""
        if not TINKER_AVAILABLE:
            return None
        # Tinker cookbook RL uses stop conditions as stop sequences (list[int] or list[str]).
        # We return stop sequences directly to stay compatible with that interface.
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[ModelInput, Any]:
        """Return initial prompt with image + question.

        Returns:
            Tuple of (observation, stop_condition)
        """
        system_prompt = SYSTEM_PROMPTS.get(self.datum.task_type, SYSTEM_PROMPTS["chat"])

        # Build initial messages
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.datum.image},
                    {"type": "text", "text": self.datum.question},
                ],
            },
        ]
        self.past_messages = messages.copy()

        observation = self.renderer.build_generation_prompt(messages)
        return observation, self.stop_condition

    async def step(self, action: ModelInput | list[int]) -> StepResult:
        """Process model action and return result.

        Args:
            action: Generated tokens from the model

        Returns:
            StepResult with reward, done flag, and next observation
        """
        # Decode action to text
        if isinstance(action, list):
            action_text = self.renderer.tokenizer.decode(action, skip_special_tokens=False)
        else:
            action_text = self.renderer.decode(action)

        # Parse the action
        parsed_action = self.action_parser.parse(action_text)

        # Add assistant message to history
        self.past_messages.append({"role": "assistant", "content": action_text})

        # Check for terminal actions
        if parsed_action is None:
            # Parse failure - small penalty
            return self._make_terminal_result(reward=-0.1, metrics={"parse_fail": 1.0})

        if parsed_action.type in ("answer", "caption", "report"):
            # Episode complete - compute final reward
            content = str(parsed_action.kwargs.get("text", parsed_action.kwargs.get("content", "")))
            reward, metrics = await self._compute_final_reward(content)
            return self._make_terminal_result(reward=reward, metrics=metrics)

        # Tool call
        if self.tool_calls_made >= self.max_tool_calls:
            return self._make_terminal_result(reward=-0.1, metrics={"max_tools_exceeded": 1.0})

        self.tool_calls_made += 1

        tool_metrics: dict[str, float] = {"tool_call": 1.0}
        step_reward = 0.0
        tool_result: str | list[dict[str, Any]]

        if parsed_action.type == "retrieve":
            tool_result, step_reward, retrieval_metrics = self._handle_retrieve(parsed_action)
            tool_metrics.update(retrieval_metrics)
        elif parsed_action.type == "web_search":
            tool_result, step_reward, web_metrics = self._handle_web_search(parsed_action)
            tool_metrics.update(web_metrics)
        else:
            tool_result, step_reward, visual_metrics = await self._handle_visual_tool(parsed_action)
            tool_metrics.update(visual_metrics)

        # Add tool result to history
        self.past_messages.append({"role": "user", "content": tool_result})

        # Build next observation
        next_obs = self.renderer.build_generation_prompt(self.past_messages)

        # Check trajectory length
        if hasattr(next_obs, "length") and next_obs.length > self.max_trajectory_tokens:
            return self._make_terminal_result(reward=-0.1, metrics={"trajectory_truncated": 1.0})

        return StepResult(
            reward=step_reward,
            episode_done=False,
            next_observation=next_obs,
            next_stop_condition=self.stop_condition,
            metrics=tool_metrics,
        )

    def _handle_retrieve(self, action: Action) -> tuple[str, float, dict[str, float]]:
        """Execute retrieval and return (observation_text, step_reward, metrics)."""
        if self.retriever is None:
            return (
                "Error: Retriever unavailable. Cannot search medical literature.",
                -self._retrieval_weight,
                {"retrieval_called": 1.0, "retrieval_empty": 1.0, "retrieval_quality": 0.0},
            )

        query = str(action.kwargs.get("query", "")).strip()
        if not query:
            return (
                "Error: Missing retrieval query.",
                -self._retrieval_weight,
                {"retrieval_called": 1.0, "retrieval_empty": 1.0, "retrieval_quality": 0.0},
            )

        top_k_value = action.kwargs.get("top_k", 5)
        try:
            top_k = int(str(top_k_value))
        except Exception:
            top_k = 5
        if top_k <= 0:
            top_k = 5

        results = self.retriever.search(
            query=query,
            top_k=top_k,
        )
        self.retrieved_passages.extend(results)

        if not results:
            return (
                "No relevant passages found.",
                -self._retrieval_weight,
                {"retrieval_called": 1.0, "retrieval_empty": 1.0, "retrieval_quality": 0.0},
            )

        quality = self._retrieval_quality(results)
        return (
            self._format_retrieval_results(results),
            self._retrieval_weight * quality,
            {"retrieval_called": 1.0, "retrieval_empty": 0.0, "retrieval_quality": quality},
        )

    def _handle_web_search(self, action: Action) -> tuple[str, float, dict[str, float]]:
        """Execute web search with DB-first guardrail."""
        query = str(action.kwargs.get("query", "")).strip()
        if not query:
            return (
                "Error: Missing web search query.",
                -self._web_search_penalty,
                {
                    "web_search_called": 1.0,
                    "web_search_penalty": self._web_search_penalty,
                    "db_confidence": 0.0,
                },
            )

        db_conf = self._database_confidence(query)
        if db_conf >= 0.5:
            guidance = (
                "Web search is discouraged when the local medical database likely contains relevant information. "
                "Use retrieve(query=...) first."
            )
            return (
                guidance,
                -self._web_search_penalty,
                {
                    "web_search_called": 1.0,
                    "web_search_penalty": self._web_search_penalty,
                    "db_confidence": db_conf,
                },
            )

        return (
            "Web search may be useful, but prefer retrieve(query=...) when possible.",
            -self._web_search_penalty * 0.25,
            {
                "web_search_called": 1.0,
                "web_search_penalty": self._web_search_penalty * 0.25,
                "db_confidence": db_conf,
            },
        )

    async def _handle_visual_tool(
        self, action: Action
    ) -> tuple[str | list[dict[str, Any]], float, dict[str, float]]:
        """Execute visual tools and return (observation, step_reward, metrics)."""
        tool_type = action.type
        metrics: dict[str, float] = {"visual_tool": 1.0}

        signature = self._tool_signature(tool_type, action.kwargs)
        repeat = signature in self._seen_tool_calls
        if repeat:
            metrics["repeat_region"] = 1.0
        else:
            metrics["repeat_region"] = 0.0
            self._seen_tool_calls.add(signature)

        before_image = self.current_image

        try:
            tool = self.tool_registry.get(tool_type)
        except KeyError:
            return (
                f"Error: Unknown tool '{tool_type}'",
                -self._image_tool_penalty,
                {"visual_tool": 0.0, "unknown_tool": 1.0},
            )

        if not hasattr(tool, "apply"):
            return (
                f"Error: Tool '{tool_type}' is not a visual tool.",
                -self._image_tool_penalty,
                {"visual_tool": 0.0, "non_visual_tool": 1.0},
            )

        try:
            result_image = tool.apply(self.current_image, **action.kwargs)
            self.current_image = result_image
            description = tool.describe(**action.kwargs)

            novelty = self._image_novelty(before_image, result_image)
            metrics["curiosity"] = novelty
            curiosity_reward = self._curiosity_weight * novelty
            penalty = self._image_tool_penalty
            if repeat:
                penalty += self._repeat_tool_penalty

            step_reward = curiosity_reward - penalty

            return (
                [
                    {"type": "image", "image": result_image},
                    {"type": "text", "text": f"Tool {tool_type} executed: {description}"},
                ],
                step_reward,
                metrics,
            )
        except Exception as exc:
            return (
                f"Error executing {tool_type}: {exc}",
                -self._image_tool_penalty,
                {"visual_tool": 0.0, "tool_error": 1.0},
            )

    def _tool_signature(self, tool_type: str, kwargs: dict[str, object]) -> str:
        payload = {"tool": tool_type, "kwargs": kwargs}
        try:
            return json.dumps(payload, sort_keys=True, default=str)
        except TypeError:
            return f"{tool_type}:{sorted((k, str(v)) for k, v in kwargs.items())}"

    def _database_confidence(self, query: str) -> float:
        if self.retriever is None:
            return 0.0
        try:
            results = self.retriever.search(query=query, top_k=3)
        except Exception:
            return 0.0
        if not results:
            return 0.0

        best_score = 0.0
        for result in results[:3]:
            try:
                score = float(result.get("score", 0.0))
            except Exception:
                score = 0.0
            best_score = max(best_score, score)

        if best_score < self._db_score_threshold:
            return 0.0
        return 1.0

    def _retrieval_quality(self, results: list[dict[str, Any]]) -> float:
        reference = f"{self.datum.question} {self.datum.ground_truth or ''}".lower()
        tokens = {t for t in reference.split() if len(t) > 4}
        if not tokens:
            return 0.0

        hits = 0
        checked = 0
        for result in results[:5]:
            content = str(result.get("content", result.get("text", ""))).lower()
            if not content:
                continue
            checked += 1
            for token in list(tokens)[:25]:
                if token in content:
                    hits += 1
        if checked == 0:
            return 0.0
        return min(1.0, hits / 10.0)

    @staticmethod
    def _image_novelty(before: Image.Image, after: Image.Image) -> float:
        before_small = before.convert("RGB").resize((32, 32))
        after_small = after.convert("RGB").resize((32, 32))
        before_pixels = list(before_small.getdata())
        after_pixels = list(after_small.getdata())
        diffs = 0.0
        for (r1, g1, b1), (r2, g2, b2) in zip(before_pixels, after_pixels, strict=False):
            diffs += abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
        max_diff = 32 * 32 * 3 * 255
        if max_diff <= 0:
            return 0.0
        return float(diffs / max_diff)

    def _format_retrieval_results(self, results: list[dict[str, Any]]) -> str:
        """Format retrieval results for the model."""
        if not results:
            return "No relevant passages found."

        passages = []
        for i, r in enumerate(results[:5], 1):
            source = r.get("source", "unknown")
            content = r.get("content", r.get("text", ""))
            passages.append(f"[{i}] ({source}) {content}")

        return "Retrieved passages:\n" + "\n\n".join(passages)

    async def _compute_final_reward(self, output: str) -> tuple[float, dict[str, float]]:
        """Compute reward for final output.

        Args:
            output: The model's final response

        Returns:
            Tuple of (reward, metrics dict)
        """
        # Use provided reward function if available
        if self.reward_function is not None:
            return cast(
                tuple[float, dict[str, float]],
                self.reward_function.compute(
                    output=output,
                    image=self.current_image,
                    question=self.datum.question,
                    task_type=self.datum.task_type,
                    ground_truth=self.datum.ground_truth,
                    retrieved_passages=self.retrieved_passages,
                ),
            )

        # Fallback to internal logic if no reward function provided
        metrics: dict[str, float] = {}

        # 1. Format compliance (did the model use the right tags?)
        format_score = self._check_format(output)
        metrics["format"] = format_score

        # 2. Grounding score (did the model cite retrieved passages?)
        grounding_score = self._check_grounding(output)
        metrics["grounding"] = grounding_score

        # 3. Content quality (placeholder - will be BioMedCLIPScore or LLM judge)
        # For now, use a simple heuristic based on output length and structure
        content_score = self._simple_content_score(output)
        metrics["content"] = content_score

        # Weighted combination
        total_reward = (
            self.format_reward_weight * format_score + 0.3 * grounding_score + 0.6 * content_score
        )

        return total_reward, metrics

    def _check_format(self, output: str) -> float:
        """Check if output follows expected format.

        Returns:
            1.0 if format is correct, 0.0 otherwise
        """
        task = self.datum.task_type

        if task == "chat" and "<answer>" in output and "</answer>" in output:
            return 1.0
        if task == "caption" and "<caption>" in output and "</caption>" in output:
            return 1.0
        if task == "report" and "<report>" in output and "</report>" in output:
            return 1.0

        # Partial credit for having some structure
        if any(tag in output for tag in ["<answer>", "<caption>", "<report>"]):
            return 0.5

        return 0.0

    def _check_grounding(self, output: str) -> float:
        """Check if output references retrieved passages.

        Returns:
            Score between 0 and 1 based on grounding
        """
        if not self.retrieved_passages:
            # No passages retrieved - can't be grounded
            return 0.0

        # Check for citation patterns like [1], [2], etc.
        import re

        citations = re.findall(r"\[(\d+)\]", output)
        if citations:
            return min(1.0, len(citations) / 3)  # Max credit for 3+ citations

        # Check for mentions of passage content
        grounded_terms = 0
        for passage in self.retrieved_passages[:3]:
            content = passage.get("content", passage.get("text", ""))
            # Check for key term overlap (simplified)
            key_terms = content.split()[:10]
            for term in key_terms:
                if len(term) > 5 and term.lower() in output.lower():
                    grounded_terms += 1

        return min(1.0, grounded_terms / 5)

    def _simple_content_score(self, output: str) -> float:
        """Simple content quality heuristic.

        This is a placeholder - should be replaced with BioMedCLIPScore
        or LLM-as-judge for real training.

        Returns:
            Score between 0 and 1
        """
        # Basic quality indicators
        word_count = len(output.split())

        if word_count < 10:
            return 0.2  # Too short
        if word_count > 500:
            return 0.5  # Too long (may be repetitive)

        # Check for medical terminology (very simple heuristic)
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
        ]
        term_count = sum(1 for term in medical_terms if term in output.lower())
        term_score = min(1.0, term_count / 4)

        # Combine length and term scores
        length_score = min(1.0, word_count / 100)
        return 0.5 * length_score + 0.5 * term_score

    def _make_terminal_result(self, reward: float, metrics: dict[str, float]) -> StepResult:
        """Create a terminal StepResult.

        Args:
            reward: Final reward
            metrics: Metrics dictionary

        Returns:
            StepResult with episode_done=True
        """
        if TINKER_AVAILABLE and hasattr(tinker.ModelInput, "empty"):
            empty_obs = tinker.ModelInput.empty()
        else:
            empty_obs = ModelInput.empty()

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=empty_obs,
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )


class MedicalEnvGroupBuilder:
    """Builder for creating groups of MedicalAgentEnv instances.

    Used by Tinker's training loop to create environment groups.
    """

    def __init__(
        self,
        datum: MedicalDatum,
        renderer: Renderer,
        tool_registry: ToolRegistry,
        retriever: Retriever | None,
        action_parser: ActionParser,
        num_envs: int = 4,
        reward_function: MedicalRewardFunction | None = None,
    ) -> None:
        self.datum = datum
        self.renderer = renderer
        self.tool_registry = tool_registry
        self.retriever = retriever
        self.action_parser = action_parser
        self.num_envs = num_envs
        self.reward_function = reward_function

    async def __call__(self) -> list[MedicalAgentEnv]:
        """Create environment group.

        Returns:
            List of MedicalAgentEnv instances (one per group member)
        """
        return await self.make_envs()

    async def make_envs(self) -> list[MedicalAgentEnv]:
        """Create environment group for tinker-cookbook RL EnvGroupBuilder interface."""
        return [
            MedicalAgentEnv(
                datum=self.datum,
                renderer=self.renderer,
                tool_registry=self.tool_registry,
                retriever=self.retriever,
                action_parser=self.action_parser,
                reward_function=self.reward_function,
            )
            for _ in range(self.num_envs)
        ]

    async def compute_group_rewards(
        self,
        trajectory_group: list[object],
        env_group: Sequence[MedicalAgentEnv],
    ) -> list[tuple[float, dict[str, float]]]:
        """Compute per-trajectory group rewards.

        Our environments return per-step rewards (including final terminal reward) directly
        from Env.step(). We keep the group reward at 0.0 by default.
        """
        return [(0.0, {}) for _ in env_group]

    def logging_tags(self) -> list[str]:
        """Return tags for logging/aggregation."""
        return ["medical", str(self.datum.task_type)]
