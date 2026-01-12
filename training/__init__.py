# ABOUTME: Training module for Tinker RL and related training utilities
# ABOUTME: Contains renderers, environments, and reward functions

from src.training.dataset import (
    MedMaxRLDataset,
    SyntheticMedicalDataset,
    example_to_datum,
    sample_to_datum,
)
from src.training.rewards import (
    BioMedCLIPScorer,
    FormatChecker,
    GroundingChecker,
    MedicalRewardFunction,
    RewardConfig,
    create_reward_function,
)
from src.training.tinker_env import (
    MedicalAgentEnv,
    MedicalDatum,
    MedicalEnvGroupBuilder,
    StepResult,
)

__all__ = [
    # Environment
    "MedicalAgentEnv",
    "MedicalDatum",
    "MedicalEnvGroupBuilder",
    "StepResult",
    # Dataset
    "MedMaxRLDataset",
    "SyntheticMedicalDataset",
    "example_to_datum",
    "sample_to_datum",
    # Rewards
    "MedicalRewardFunction",
    "RewardConfig",
    "BioMedCLIPScorer",
    "FormatChecker",
    "GroundingChecker",
    "create_reward_function",
]
