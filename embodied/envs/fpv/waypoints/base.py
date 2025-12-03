from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from embodied.envs.fpv.types import WaypointConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from embodied.envs.fpv.backends.base import SimulatorBackend


class WaypointManager(ABC):

    def __init__(self, config: Optional[WaypointConfig] = None):
        self._config = config or WaypointConfig()
        self._base_waypoints: List[np.ndarray] = [
            np.asarray(p, dtype=np.float32) for p in self._config.positions
        ]
        self._waypoints: List[np.ndarray] = [w.copy() for w in self._base_waypoints]
        self._current_index: int = 0
        self._initialized: bool = False
        self._rng: Optional[np.random.Generator] = None

    @property
    def config(self) -> WaypointConfig:
        return self._config

    @property
    def waypoints(self) -> List[np.ndarray]:
        return [w.copy() for w in self._waypoints]

    @property
    def num_waypoints(self) -> int:
        return len(self._waypoints)

    @property
    def current_index(self) -> int:
        return self._current_index

    def set_waypoints(self, waypoints: List[np.ndarray]) -> None:
        self._base_waypoints = [np.asarray(w, dtype=np.float32) for w in waypoints]
        self._waypoints = [w.copy() for w in self._base_waypoints]
        self._current_index = 0
        self._initialized = False

    def reset(self) -> None:
        self._current_index = 0
        self._initialized = False

        if self._config.randomization.enabled and self._base_waypoints:
            self._waypoints = self._apply_randomization(self._base_waypoints)
            logger.debug(
                f"[WP] Randomized {len(self._waypoints)} waypoints "
                f"(xy_radius={self._config.randomization.xy_radius}m, "
                f"smoothing={self._config.randomization.smoothing_factor})"
            )
        else:
            self._waypoints = [w.copy() for w in self._base_waypoints]

    def _apply_randomization(
        self, base_positions: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Apply momentum-smoothed random perturbations to waypoints."""
        cfg = self._config.randomization
        self._rng = np.random.default_rng(cfg.seed)

        randomized = []
        momentum = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        for i, base_pos in enumerate(base_positions):
            # Uniform disk sampling for XY offset
            angle = self._rng.uniform(0, 2 * np.pi)
            radius = self._rng.uniform(0, cfg.xy_radius)
            raw_offset = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                self._rng.uniform(cfg.z_range[0], cfg.z_range[1])
            ], dtype=np.float32)

            # Blend with momentum for smooth paths
            smoothed_offset = (
                cfg.smoothing_factor * momentum +
                (1 - cfg.smoothing_factor) * raw_offset
            )
            momentum = smoothed_offset.copy()
            new_pos = base_pos + smoothed_offset

            # Altitude constraints (NED: more negative = higher)
            new_pos[2] = np.clip(new_pos[2], cfg.max_altitude, cfg.min_altitude)

            # Enforce minimum waypoint spacing
            if i > 0 and len(randomized) > 0:
                dist = float(np.linalg.norm(new_pos - randomized[-1]))
                if dist < cfg.min_waypoint_distance:
                    direction = new_pos - randomized[-1]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        direction = direction / norm
                        new_pos = randomized[-1] + direction * cfg.min_waypoint_distance

            randomized.append(new_pos.astype(np.float32))

        return randomized

    def get_current_target(self) -> Optional[np.ndarray]:
        if not self._waypoints:
            return None
        return self._waypoints[self._current_index].copy()

    def get_distance_to_target(self, position: np.ndarray) -> float:
        target = self.get_current_target()
        if target is None:
            return 0.0
        return float(np.linalg.norm(target - position))

    def check_reached(self, position: np.ndarray) -> bool:
        target = self.get_current_target()
        if target is None:
            return False
        distance = float(np.linalg.norm(target - position))
        reached = distance < self._config.reach_radius
        if distance < 5.0:
            logger.info(
                f"[WP] check_reached: target={target}, pos={position}, "
                f"dist={distance:.2f}m, radius={self._config.reach_radius}m, reached={reached}"
            )
        return reached

    def advance(self) -> int:
        if not self._waypoints:
            return 0

        n = len(self._waypoints)
        next_index = (self._current_index + 1) % n

        if next_index == 0 and self._config.loop_start_index is not None:
            next_index = self._config.loop_start_index

        self._current_index = next_index
        return next_index

    def check_and_advance(self, position: np.ndarray) -> Optional[int]:
        if self.check_reached(position):
            reached_index = self._current_index
            self.advance()
            logger.info(f"[WP] check_and_advance: Reached waypoint {reached_index}, advanced to {self._current_index}")
            return reached_index
        return None

    def get_visible_range(self) -> tuple[int, int]:
        n = len(self._waypoints)
        if n == 0:
            return (0, 0)

        start = self._current_index
        if self._config.visible_count == 0:
            end = n
        else:
            end = min(self._current_index + self._config.visible_count, n)

        return (start, end)

    # Visual Marker Methods

    @abstractmethod
    def initialize_visuals(self, backend: "SimulatorBackend") -> None:
        """Initialize visual markers at episode start."""
        pass

    @abstractmethod
    def update_visuals(self, backend: "SimulatorBackend", reached_index: int) -> None:
        """Update visuals after a waypoint is reached."""
        pass

    @abstractmethod
    def cleanup_visuals(self, backend: "SimulatorBackend") -> None:
        """Remove all visual markers."""
        pass
