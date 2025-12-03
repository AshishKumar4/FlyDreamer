from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

from embodied.envs.fpv.types import WaypointConfig
from embodied.envs.fpv.waypoints.base import WaypointManager

if TYPE_CHECKING:
    from embodied.envs.fpv.backends.base import SimulatorBackend

logger = logging.getLogger(__name__)


# =============================================================================
# Spline Interpolation (Pure Functions)
# =============================================================================

def catmull_rom_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    num_points: int = 20,
) -> List[np.ndarray]:
    """Generate points along a Catmull-Rom spline segment from p1 to p2."""
    if num_points < 2:
        return [p1.copy()]

    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        t2 = t * t
        t3 = t2 * t

        # Catmull-Rom basis functions (uniform parameterization)
        point = 0.5 * (
            (2.0 * p1) +
            (-p0 + p2) * t +
            (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
            (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        points.append(point.astype(np.float32))

    return points


def compute_spline_path(
    waypoints: List[np.ndarray],
    samples_per_segment: int = 20,
) -> List[np.ndarray]:
    """Compute a smooth Catmull-Rom spline path through all waypoints."""
    n = len(waypoints)
    if n == 0:
        return []
    if n == 1:
        return [waypoints[0].copy()]
    if n == 2:
        # Linear interpolation for 2 points
        return _linear_interpolate(waypoints[0], waypoints[1], samples_per_segment)

    all_points: List[np.ndarray] = []

    for i in range(n - 1):
        # Handle edge cases with phantom/reflected control points
        p0 = waypoints[max(0, i - 1)]
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        p3 = waypoints[min(n - 1, i + 2)]

        segment_points = catmull_rom_segment(p0, p1, p2, p3, samples_per_segment)

        # Avoid duplicate points at segment boundaries
        if i > 0 and segment_points:
            segment_points = segment_points[1:]

        all_points.extend(segment_points)

    return all_points


def _linear_interpolate(
    p1: np.ndarray,
    p2: np.ndarray,
    num_points: int,
) -> List[np.ndarray]:
    """Simple linear interpolation between two points."""
    if num_points < 2:
        return [p1.copy()]
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        point = p1 * (1 - t) + p2 * t
        points.append(point.astype(np.float32))
    return points


# =============================================================================
# Visual Waypoint Manager
# =============================================================================

class VisualWaypointManager(WaypointManager):
    """Waypoint manager with visual markers and spline path."""

    _global_spawn_counter: int = 0
    _session_prefix: str = f"S{int(time.time()) % 100000}"

    def __init__(self, config: Optional[WaypointConfig] = None):
        super().__init__(config)
        self._balloon_index_to_name: Dict[int, str] = {}
        self._spline_names: Set[str] = set()
        self._visible_start: int = 0
        self._visible_end: int = 0

    @classmethod
    def _get_unique_name(cls, prefix: str = "M") -> str:
        cls._global_spawn_counter += 1
        return f"{prefix}{cls._session_prefix}_{cls._global_spawn_counter}"

    def initialize_visuals(self, backend: "SimulatorBackend") -> None:
        if not backend.supports_visual_markers:
            logger.debug("Backend doesn't support visual markers, skipping")
            self._initialized = True
            return

        if not self._waypoints:
            self._initialized = True
            return

        self.cleanup_visuals(backend)
        self._update_visible_range()
        self._spawn_visible_balloons(backend)

        if self._config.show_spline:
            self._spawn_spline(backend)

        self._initialized = True
        logger.debug(
            f"Initialized visuals: {len(self._balloon_index_to_name)} balloons, "
            f"{len(self._spline_names)} spline markers"
        )

    def update_visuals(self, backend: "SimulatorBackend", reached_index: int) -> None:
        logger.info(f"[WP] update_visuals called! reached_index={reached_index}, current_index={self._current_index}")

        if not backend.supports_visual_markers:
            logger.warning("[WP] Backend doesn't support visual markers!")
            return

        logger.info(f"[WP] About to destroy balloon at index {reached_index}")
        self._destroy_balloon(backend, reached_index)

        # Check if we looped
        looped = (
            self._config.loop_start_index is not None
            and self._current_index < reached_index
        )

        if looped:
            logger.info(
                f"[WP] Loop detected after reaching {reached_index}; "
                f"resetting visuals (new current_index={self._current_index})"
            )
            self._destroy_all_balloons(backend)
            self._destroy_all_spline_markers(backend)
            self._update_visible_range()
            self._spawn_visible_balloons(backend)
            if self._config.show_spline:
                self._spawn_spline(backend)
            return

        old_start = self._visible_start
        old_end = self._visible_end
        self._update_visible_range()

        # Spawn new balloons in visible window
        for i in range(max(old_end, self._visible_start), self._visible_end):
            is_target = (i == self._current_index)
            self._spawn_balloon(backend, i, is_target)

        self._update_target_balloon(backend)

        if self._config.show_spline:
            self._destroy_all_spline_markers(backend)
            self._spawn_spline(backend)

    def cleanup_visuals(self, backend: "SimulatorBackend") -> None:
        self._destroy_all_balloons(backend)
        self._destroy_all_spline_markers(backend)

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _update_visible_range(self) -> None:
        n = len(self._waypoints)
        if n == 0:
            self._visible_start = 0
            self._visible_end = 0
            return

        self._visible_start = self._current_index

        if self._config.visible_count == 0:
            self._visible_end = n
        else:
            self._visible_end = min(self._current_index + self._config.visible_count, n)

    def _spawn_visible_balloons(self, backend: "SimulatorBackend") -> None:
        for i in range(self._visible_start, self._visible_end):
            is_target = (i == self._current_index)
            self._spawn_balloon(backend, i, is_target)

    def _spawn_balloon(
        self,
        backend: "SimulatorBackend",
        index: int,
        is_target: bool,
    ) -> Optional[str]:
        if index < 0 or index >= len(self._waypoints):
            return None

        pos = self._waypoints[index]
        name = self._get_unique_name("WP")
        markers = self._config.markers

        if is_target:
            asset = markers.target_asset
            scale = markers.target_scale
            is_blueprint = markers.target_is_blueprint
            marker_type = "target"
        else:
            asset = markers.waypoint_asset
            scale = markers.waypoint_scale
            is_blueprint = markers.waypoint_is_blueprint
            marker_type = "waypoint"

        actual_name = backend.spawn_marker(
            name=name,
            position=pos,
            marker_type=marker_type,
            scale=scale,
            asset=asset,
            is_blueprint=is_blueprint,
        )

        if actual_name:
            # Store mapping from index to actual name (AirSim may rename objects)
            self._balloon_index_to_name[index] = actual_name
            logger.debug(f"[WP] Spawned balloon: index={index}, requested='{name}', actual='{actual_name}'")

        return actual_name

    def _destroy_balloon(self, backend: "SimulatorBackend", index: int) -> None:
        """Destroy balloon at specified index."""
        # Look up the actual spawned name from our mapping
        actual_name = self._balloon_index_to_name.get(index)
        if actual_name is None:
            logger.warning(f"[WP] _destroy_balloon: No balloon tracked for index {index}")
            return

        logger.info(f"[WP] _destroy_balloon: Destroying index={index}, actual_name='{actual_name}'")
        result = backend.destroy_marker(actual_name)
        logger.info(f"[WP] _destroy_balloon: destroy_marker('{actual_name}') returned {result}")

        # Remove from tracking
        del self._balloon_index_to_name[index]
        logger.info(f"[WP] _destroy_balloon: Remaining tracked balloons: {self._balloon_index_to_name}")

    def _destroy_all_balloons(self, backend: "SimulatorBackend") -> None:
        """Destroy all spawned balloons."""
        for index, actual_name in list(self._balloon_index_to_name.items()):
            backend.destroy_marker(actual_name)
        self._balloon_index_to_name.clear()

    def _update_target_balloon(self, backend: "SimulatorBackend") -> None:
        """Respawn current waypoint as target marker."""
        idx = self._current_index
        if idx < 0 or idx >= len(self._waypoints):
            return

        # Respawn as target
        self._destroy_balloon(backend, idx)
        self._spawn_balloon(backend, idx, is_target=True)

    def _spawn_spline(self, backend: "SimulatorBackend") -> None:
        """Spawn spline markers for the visible waypoint segment."""
        if not self._config.show_spline:
            return

        # Get visible waypoints
        visible_waypoints = self._waypoints[self._visible_start:self._visible_end]
        if len(visible_waypoints) < 2:
            return

        # Compute spline points
        spline_points = compute_spline_path(
            visible_waypoints,
            self._config.spline_density,
        )

        markers = self._config.markers

        # Spawn markers along spline with globally unique names
        for i, point in enumerate(spline_points):
            name = self._get_unique_name("SP")
            actual_name = backend.spawn_marker(
                name=name,
                position=point,
                marker_type="spline",
                scale=markers.spline_scale,
                asset=markers.spline_asset,
                is_blueprint=markers.spline_is_blueprint,
            )
            if actual_name:
                self._spline_names.add(actual_name)

    def _destroy_all_spline_markers(self, backend: "SimulatorBackend") -> None:
        """Destroy all spawned spline markers."""
        for name in list(self._spline_names):
            backend.destroy_marker(name)
        self._spline_names.clear()

    def _respawn_loop_balloons(self, backend: "SimulatorBackend") -> None:
        """Respawn balloons from loop_start_index for subsequent laps."""
        if self._config.loop_start_index is None:
            return

        loop_start = self._config.loop_start_index
        n = len(self._waypoints)

        for i in range(loop_start, n):
            # Check if balloon for this index is not already spawned
            if i not in self._balloon_index_to_name:
                is_target = (i == self._current_index)
                self._spawn_balloon(backend, i, is_target)
