"""Reward computation strategies for FPV environment."""

from embodied.envs.fpv.rewards.base import RewardStrategy
from embodied.envs.fpv.rewards.navigation import NavigationReward

__all__ = ["RewardStrategy", "NavigationReward"]
