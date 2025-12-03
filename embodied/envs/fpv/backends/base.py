"""
Abstract base class for simulator backends

All backends expose a unified NED coordinate system and control interface.
Coordinate transforms, timing, and simulator-specific details are handled internally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any

import numpy as np

from embodied.envs.fpv.types import (
    DroneState,
    ImageData,
    ControlCommand,
    MarkerConfig,
)


class SimulatorBackend(ABC):

    # =========================================================================
    # Connection Management
    # =========================================================================

    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass

    # =========================================================================
    # Simulation Control
    # =========================================================================

    @abstractmethod
    def reset(self) -> DroneState:
        pass

    @abstractmethod
    def step(self, command: ControlCommand, dt: float) -> Tuple[DroneState, bool]:
        pass

    @abstractmethod
    def get_state(self) -> DroneState:
        pass

    # =========================================================================
    # Observations
    # =========================================================================

    @abstractmethod
    def get_images(self, size: Tuple[int, int], include_depth: bool = True) -> ImageData:
        pass

    # =========================================================================
    # Visual Markers
    # =========================================================================

    @property
    @abstractmethod
    def supports_visual_markers(self) -> bool:
        pass

    @abstractmethod
    def spawn_marker(
        self,
        name: str,
        position: np.ndarray,
        marker_type: str,
        scale: float,
        asset: str,
        is_blueprint: bool = False,
    ) -> Optional[str]:
        """Spawn a visual marker in the scene.

        Args:
            name: Unique identifier for the marker
            position: Position in NED coordinates [x, y, z]
            marker_type: Type hint ("target", "waypoint", "spline")
            scale: Uniform scale factor
            asset: Asset/prefab name (backend-specific)
            is_blueprint: Whether asset is a Blueprint (Colosseum-specific)

        Returns:
            Actual spawned name, or None if spawning not supported/failed
        """
        pass

    @abstractmethod
    def destroy_marker(self, name: str) -> bool:
        pass

    @abstractmethod
    def destroy_all_markers(self, prefix: str = "") -> int:
        pass

    @abstractmethod
    def list_assets(self) -> List[str]:
        pass

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def coordinate_system(self) -> str:
        return "NED"

    def get_info(self) -> dict:
        return {
            "name": self.name,
            "connected": self.is_connected(),
            "supports_markers": self.supports_visual_markers,
            "coordinate_system": self.coordinate_system,
        }
