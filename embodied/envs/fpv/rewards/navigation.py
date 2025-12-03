from __future__ import annotations

from typing import Optional

import numpy as np

from embodied.envs.fpv.types import DroneState, RewardConfig
from embodied.envs.fpv.rewards.base import RewardStrategy, RewardInfo


class NavigationReward(RewardStrategy):
    """Navigation reward for waypoint-following tasks."""

    def __init__(self, config: Optional[RewardConfig] = None):
        super().__init__(config)

    def compute(
        self,
        state: DroneState,
        target_position: np.ndarray,
        step: int,
        max_steps: int,
        has_collided: bool,
    ) -> RewardInfo:
        cfg = self.config
        info = RewardInfo(total=0.0, is_terminal=False)

        # Collision
        if has_collided:
            info.collision_penalty = cfg.collision_penalty
            info.total = cfg.collision_penalty
            info.is_terminal = True
            return info

        distance = float(np.linalg.norm(target_position - state.position))
        info.distance_to_target = distance

        # Survival bonus (decaying)
        alive_decay = np.exp(-step / cfg.survival_decay)
        info.survival = cfg.survival_bonus * alive_decay

        # Progress reward
        raw_progress = self._prev_distance - distance  # meters toward target (+ = good)
        info.progress = np.clip(
            raw_progress * cfg.progress_scale,
            cfg.progress_clip_min,
            cfg.progress_clip_max,
        )
        self._prev_distance = distance

        # Rate penalty
        if cfg.rate_penalty_scale > 0:
            omega = state.angular_velocity
            roll_pitch_l1 = abs(omega[0]) + abs(omega[1])
            yaw_l1 = abs(omega[2]) * cfg.rate_yaw_weight
            omega_weighted = roll_pitch_l1 + yaw_l1

            # Log-based penalty
            base_rate = -cfg.rate_penalty_scale * np.log(1.0 + omega_weighted ** 2)

            # Progress-gating: reduce penalty by up to 50% when making good progress
            progress_gate = max(0.0, 1.0 / (1.0 + np.exp(-5.0 * raw_progress)) - 0.5) * 2.0
            info.rate_penalty = base_rate * (1.0 - 0.5 * progress_gate)

        # Jerk penalty
        if cfg.jerk_penalty_scale > 0 and self._prev_omega is not None:
            delta_omega = state.angular_velocity - self._prev_omega
            info.jerk_penalty = -cfg.jerk_penalty_scale * np.sum(delta_omega**2)
        self._prev_omega = state.angular_velocity.copy()

        # Time penalty
        time_factor = 1 + step / cfg.time_penalty_growth
        info.time_penalty = cfg.time_penalty * time_factor

        info.total = (
            info.survival +
            info.progress +
            info.rate_penalty +
            info.jerk_penalty +
            info.time_penalty
        )

        return info

    def on_waypoint_reached(
        self,
        waypoint_index: int,
        num_waypoints: int,
        offset: Optional[np.ndarray] = None,
    ) -> float:
        cfg = self.config
        base_bonus = cfg.gate_bonus

        if cfg.gate_bonus_decay and offset is not None:
            max_offset = max(abs(float(offset[1])), abs(float(offset[2])))
            decay = max(0.0, 1.0 - max_offset / cfg.gate_size)
            base_bonus = base_bonus * decay

        if cfg.gate_bonus_progressive and num_waypoints > 1:
            progress = waypoint_index / (num_waypoints - 1)
            multiplier = 1.0 + (cfg.gate_bonus_max_scale - 1.0) * (progress ** 2)
            return base_bonus * multiplier

        return base_bonus