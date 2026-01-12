# ABOUTME: Aggregates per-episode rollout metrics for tool-policy evaluation
# ABOUTME: Summarizes counts and averages from per-step metrics dictionaries

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EpisodeStats:
    total_reward: float
    steps: int
    metrics_sum: dict[str, float] = field(default_factory=dict)


def _coerce_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return 0.0


def accumulate_metrics(metrics: Mapping[str, object], *, into: dict[str, float]) -> None:
    for key, value in metrics.items():
        into[key] = into.get(key, 0.0) + _coerce_float(value)


@dataclass(frozen=True)
class RolloutSummary:
    episodes: int
    mean_total_reward: float
    mean_steps: float
    mean_metrics_sum: dict[str, float]


def summarize_rollouts(episodes: list[EpisodeStats]) -> RolloutSummary:
    if not episodes:
        return RolloutSummary(
            episodes=0,
            mean_total_reward=0.0,
            mean_steps=0.0,
            mean_metrics_sum={},
        )

    total_reward = sum(ep.total_reward for ep in episodes)
    total_steps = sum(ep.steps for ep in episodes)
    metrics_totals: dict[str, float] = {}
    for ep in episodes:
        accumulate_metrics(ep.metrics_sum, into=metrics_totals)

    n = float(len(episodes))
    mean_metrics = {k: v / n for k, v in metrics_totals.items()}
    return RolloutSummary(
        episodes=len(episodes),
        mean_total_reward=total_reward / n,
        mean_steps=float(total_steps) / n,
        mean_metrics_sum=mean_metrics,
    )
