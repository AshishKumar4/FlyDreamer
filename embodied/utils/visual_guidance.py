from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set

import numpy as np

try:
    import airsim
except ImportError:
    airsim = None  # Allow import for testing without airsim


logger = logging.getLogger(__name__)


# =============================================================================
# Spline Interpolation
# =============================================================================

def catmull_rom_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    num_points: int = 20,
    alpha: float = 0.5,
) -> List[np.ndarray]:
    """Generate points along a Catmull-Rom spline segment.

    The spline passes exactly through p1 and p2, using p0 and p3 as
    control points for tangent computation.

    Args:
        p0: Control point before segment start
        p1: Segment start point (spline passes through here)
        p2: Segment end point (spline passes through here)
        p3: Control point after segment end
        num_points: Number of interpolated points to generate
        alpha: Tension parameter (0.5 = centripetal, recommended for no cusps)

    Returns:
        List of num_points interpolated 3D points from p1 to p2
    """
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
    """Compute a smooth spline path through all waypoints.

    Uses Catmull-Rom interpolation which guarantees the spline passes
    exactly through each waypoint while maintaining C1 continuity.

    Args:
        waypoints: List of 3D waypoint positions (NED coordinates)
        samples_per_segment: Number of interpolated points per segment

    Returns:
        List of interpolated 3D points forming the complete spline path
    """
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


def compute_spline_segment(
    waypoints: List[np.ndarray],
    start_idx: int,
    end_idx: int,
    samples_per_segment: int = 20,
) -> List[np.ndarray]:
    """Compute spline path for a subset of waypoints (rolling window).

    Args:
        waypoints: Full list of waypoints
        start_idx: Starting waypoint index (inclusive)
        end_idx: Ending waypoint index (exclusive)
        samples_per_segment: Points per segment

    Returns:
        Interpolated spline points for the specified range
    """
    n = len(waypoints)
    if n == 0 or start_idx >= n or end_idx <= start_idx:
        return []

    # Clamp indices
    start_idx = max(0, start_idx)
    end_idx = min(n, end_idx)

    subset = waypoints[start_idx:end_idx]
    return compute_spline_path(subset, samples_per_segment)


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
# Configuration
# =============================================================================

@dataclass
class VisualGuidanceConfig:
    """Configuration for the visual guidance system.

    All parameters are easily adjustable via this dataclass.
    Use different configs for training (minimal) vs demo (full visualization).

    Attributes:
        enabled: Master switch for the visual system
        target_asset: Unreal asset name for current target marker
        waypoint_asset: Unreal asset name for future waypoint markers
        target_scale: Scale (x, y, z) in meters for target marker
        waypoint_scale: Scale for future waypoint markers
        visible_waypoints: Number of upcoming waypoints to show (0 = show all)
        show_spline: Whether to render spline path between waypoints
        spline_asset: Asset for spline markers
        spline_scale: Scale for spline markers (should be small)
        spline_samples_per_segment: Density of spline (20+ for smooth ribbon)
        pop_on_reach: Remove marker when waypoint is reached
        loop_start_index: Index to loop back to after completing course (None = no loop)
        reach_radius: Distance threshold for "reaching" a waypoint (meters)

    Required UE5 Assets (create in BlocksV2):
        1. M_RedEmissive - Red glowing material for target marker
        2. M_GrayMarker - Gray matte material for waypoint markers
        3. BP_RedMarker - Blueprint: Sphere mesh + M_RedEmissive
        4. BP_GrayMarker - Blueprint: Sphere mesh + M_GrayMarker

    Run `client.simListAssets()` to verify assets are available.
    """
    # Marker settings (requires custom UE5 blueprints)
    enabled: bool = True
    target_asset: str = "BP_RedMarker"  # Red glowing sphere for current target
    waypoint_asset: str = "BP_GrayMarker"  # Gray matte sphere for future waypoints
    target_scale: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    waypoint_scale: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    visible_waypoints: int = 5  # 0 = show all

    # Asset type flags (True if asset is a Blueprint, False if StaticMesh)
    target_is_blueprint: bool = True  # BP_RedMarker is a Blueprint
    waypoint_is_blueprint: bool = True  # BP_GrayMarker is a Blueprint

    # Spline settings
    show_spline: bool = True
    spline_asset: str = "Sphere"  # Use basic Sphere for spline (StaticMesh)
    spline_scale: Tuple[float, float, float] = (0.12, 0.12, 0.12)
    spline_samples_per_segment: int = 20
    spline_is_blueprint: bool = False  # Sphere is a StaticMesh, not Blueprint

    # Behavior
    pop_on_reach: bool = True
    loop_start_index: Optional[int] = None
    reach_radius: float = 1.0

    def validate(self) -> None:
        """Validate configuration values."""
        if self.visible_waypoints < 0:
            raise ValueError("visible_waypoints must be >= 0")
        if self.spline_samples_per_segment < 2:
            raise ValueError("spline_samples_per_segment must be >= 2")
        if self.reach_radius <= 0:
            raise ValueError("reach_radius must be > 0")
        for scale_name in ["target_scale", "waypoint_scale", "spline_scale"]:
            scale = getattr(self, scale_name)
            if len(scale) != 3 or any(s <= 0 for s in scale):
                raise ValueError(f"{scale_name} must be 3 positive values")


# =============================================================================
# Visual Guidance System
# =============================================================================

class VisualGuidanceSystem:
    """Manages visual waypoint markers and spline path visualization.

    This class is decoupled from the RL environment and can be used
    standalone for manual control testing or integrated with any
    Colosseum/AirSim-based environment.

    The system supports:
      - Balloon markers at waypoints (different appearance for target vs future)
      - Smooth spline path visualization between waypoints
      - Rolling window for performance (only show N upcoming waypoints)
      - Course looping (return to specified index after completion)

    Thread Safety: Not thread-safe. Use from a single thread only.
    """

    def __init__(
        self,
        client: "airsim.MultirotorClient",
        config: Optional[VisualGuidanceConfig] = None,
    ):
        """Initialize the visual guidance system.

        Args:
            client: Connected AirSim client instance
            config: Configuration options (uses defaults if None)
        """
        if airsim is None:
            raise ImportError("airsim package required for VisualGuidanceSystem")

        self._client = client
        self._config = config or VisualGuidanceConfig()
        self._config.validate()

        # Waypoint state
        self._waypoints: List[np.ndarray] = []
        self._current_index: int = 0

        # Spawned object tracking
        self._balloon_names: Set[str] = set()
        self._spline_names: Set[str] = set()

        # Rolling window state
        self._visible_start: int = 0
        self._visible_end: int = 0

        # Cached spline points (for lazy updates)
        self._cached_spline_points: List[np.ndarray] = []
        self._spline_dirty: bool = True

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @property
    def config(self) -> VisualGuidanceConfig:
        """Get the current configuration."""
        return self._config

    @property
    def current_index(self) -> int:
        """Get the current target waypoint index."""
        return self._current_index

    @property
    def waypoints(self) -> List[np.ndarray]:
        """Get the list of waypoints (read-only copy)."""
        return [w.copy() for w in self._waypoints]

    @property
    def num_waypoints(self) -> int:
        """Get the number of waypoints."""
        return len(self._waypoints)

    def set_waypoints(self, waypoints: List[np.ndarray]) -> None:
        """Set the course waypoints.

        Call this before initialize() or when changing the course.
        Automatically cleans up existing visuals.

        Args:
            waypoints: List of 3D positions in NED coordinates
        """
        self.cleanup()
        self._waypoints = [np.asarray(w, dtype=np.float32) for w in waypoints]
        self._current_index = 0
        self._spline_dirty = True
        logger.debug(f"Set {len(waypoints)} waypoints")

    def initialize(self) -> None:
        """Spawn initial visual markers.

        Call this at episode start after set_waypoints().
        Spawns balloons and spline based on configuration.
        """
        if not self._config.enabled or not self._waypoints:
            return

        self.cleanup()
        self._current_index = 0
        self._update_visible_range()
        self._spawn_visible_balloons()

        if self._config.show_spline:
            self._spawn_spline()

        logger.debug(f"Initialized visuals: {len(self._balloon_names)} balloons, "
                    f"{len(self._spline_names)} spline markers")

    def on_waypoint_reached(self, reached_index: int) -> int:
        """Handle waypoint reached event.

        Updates visual state: pops reached balloon, updates target,
        spawns new balloon at end of rolling window.

        Args:
            reached_index: Index of the waypoint that was reached

        Returns:
            New current waypoint index (handles looping)
        """
        if not self._waypoints:
            return 0

        n = len(self._waypoints)

        # Pop the reached balloon if configured
        if self._config.pop_on_reach and self._config.enabled:
            self._destroy_balloon(reached_index)

        # Compute next index with looping support
        next_index = (reached_index + 1) % n

        # Handle course completion with loop
        if next_index == 0 and self._config.loop_start_index is not None:
            next_index = self._config.loop_start_index
            # Respawn loop portion balloons
            if self._config.enabled:
                self._respawn_loop_balloons()

        self._current_index = next_index

        # Update rolling window and spawn/destroy as needed
        if self._config.enabled:
            self._update_rolling_window()

        logger.debug(f"Waypoint {reached_index} reached, new target: {next_index}")
        return next_index

    def check_waypoint_reached(self, position: np.ndarray) -> Optional[int]:
        """Check if current waypoint is reached and handle it.

        Convenience method that checks distance and calls on_waypoint_reached
        if within reach_radius.

        Args:
            position: Current drone position in NED coordinates

        Returns:
            New waypoint index if reached, None otherwise
        """
        if not self._waypoints:
            return None

        target = self._waypoints[self._current_index]
        dist = float(np.linalg.norm(target - position))

        if dist < self._config.reach_radius:
            return self.on_waypoint_reached(self._current_index)
        return None

    def get_target_position(self) -> Optional[np.ndarray]:
        """Get the current target waypoint position.

        Returns:
            Current target position or None if no waypoints
        """
        if not self._waypoints:
            return None
        return self._waypoints[self._current_index].copy()

    def get_distance_to_target(self, position: np.ndarray) -> float:
        """Get distance from position to current target.

        Args:
            position: Query position in NED coordinates

        Returns:
            Euclidean distance to target (0.0 if no waypoints)
        """
        target = self.get_target_position()
        if target is None:
            return 0.0
        return float(np.linalg.norm(target - position))

    def cleanup(self) -> None:
        """Remove all spawned visual objects.

        Safe to call multiple times. Call at episode end or when
        changing waypoints.
        """
        self._destroy_all_balloons()
        self._destroy_all_spline_markers()
        self._spline_dirty = True
        logger.debug("Cleaned up all visuals")

    def list_available_assets(self) -> List[str]:
        """Query available Unreal assets for spawning.

        Useful for discovering what assets are available in the
        current environment.

        Returns:
            List of asset names, empty list on error
        """
        try:
            return self._client.simListAssets()
        except Exception as e:
            logger.warning(f"Failed to list assets: {e}")
            return []

    # -------------------------------------------------------------------------
    # Rolling Window Management
    # -------------------------------------------------------------------------

    def _update_visible_range(self) -> None:
        """Update the visible waypoint range based on current index."""
        n = len(self._waypoints)
        if n == 0:
            self._visible_start = 0
            self._visible_end = 0
            return

        self._visible_start = self._current_index

        if self._config.visible_waypoints == 0:
            # Show all waypoints
            self._visible_end = n
        else:
            # Rolling window
            self._visible_end = min(
                self._current_index + self._config.visible_waypoints,
                n
            )

    def _update_rolling_window(self) -> None:
        """Update visuals when rolling window shifts."""
        old_start = self._visible_start
        old_end = self._visible_end

        self._update_visible_range()

        # If window hasn't changed, just update target appearance
        if old_start == self._visible_start and old_end == self._visible_end:
            self._update_target_balloon()
            return

        # Destroy balloons that fell out of window
        for i in range(old_start, min(old_end, self._visible_start)):
            self._destroy_balloon(i)

        # Spawn new balloons that entered window
        for i in range(max(old_end, self._visible_start), self._visible_end):
            is_target = (i == self._current_index)
            self._spawn_balloon(i, is_target)

        # Update target balloon appearance
        self._update_target_balloon()

        # Respawn spline for new window
        if self._config.show_spline:
            self._destroy_all_spline_markers()
            self._spawn_spline()

    def _update_target_balloon(self) -> None:
        """Update which balloon appears as the target.

        Destroys and respawns the current target balloon with target asset/scale.
        """
        if not self._config.enabled:
            return

        idx = self._current_index
        if idx < 0 or idx >= len(self._waypoints):
            return

        # Respawn as target
        self._destroy_balloon(idx)
        self._spawn_balloon(idx, is_target=True)

    # -------------------------------------------------------------------------
    # Balloon Spawning
    # -------------------------------------------------------------------------

    def _spawn_visible_balloons(self) -> None:
        """Spawn balloons for all visible waypoints."""
        for i in range(self._visible_start, self._visible_end):
            is_target = (i == self._current_index)
            self._spawn_balloon(i, is_target)

    def _spawn_balloon(self, index: int, is_target: bool) -> Optional[str]:
        """Spawn a single balloon at the specified waypoint.

        Args:
            index: Waypoint index
            is_target: If True, use target appearance; else waypoint appearance

        Returns:
            Spawned object name or None on failure
        """
        if index < 0 or index >= len(self._waypoints):
            return None

        pos = self._waypoints[index]
        name = f"Balloon_{index}"

        # Choose asset, scale, and blueprint flag based on target/waypoint
        if is_target:
            asset = self._config.target_asset
            scale = self._config.target_scale
            is_blueprint = self._config.target_is_blueprint
        else:
            asset = self._config.waypoint_asset
            scale = self._config.waypoint_scale
            is_blueprint = self._config.waypoint_is_blueprint

        return self._spawn_object(name, asset, pos, scale, is_blueprint=is_blueprint)

    def _destroy_balloon(self, index: int) -> None:
        """Destroy balloon at specified index."""
        name = f"Balloon_{index}"
        self._destroy_object(name)
        self._balloon_names.discard(name)

    def _destroy_all_balloons(self) -> None:
        """Destroy all spawned balloons."""
        for name in list(self._balloon_names):
            self._destroy_object(name)
        self._balloon_names.clear()

    def _respawn_loop_balloons(self) -> None:
        """Respawn balloons for the loop portion after course completion."""
        if self._config.loop_start_index is None:
            return

        loop_start = self._config.loop_start_index
        n = len(self._waypoints)

        for i in range(loop_start, n):
            name = f"Balloon_{i}"
            if name not in self._balloon_names:
                is_target = (i == self._current_index)
                self._spawn_balloon(i, is_target)

    # -------------------------------------------------------------------------
    # Spline Spawning
    # -------------------------------------------------------------------------

    def _spawn_spline(self) -> None:
        """Spawn spline markers for the visible waypoint segment."""
        if not self._config.show_spline:
            return

        # Compute spline points for visible range
        visible_waypoints = self._waypoints[self._visible_start:self._visible_end]
        if len(visible_waypoints) < 2:
            return

        spline_points = compute_spline_path(
            visible_waypoints,
            self._config.spline_samples_per_segment,
        )

        # Spawn markers along spline
        for i, point in enumerate(spline_points):
            name = f"Spline_{self._visible_start}_{i}"
            self._spawn_object(
                name,
                self._config.spline_asset,
                point,
                self._config.spline_scale,
                is_blueprint=self._config.spline_is_blueprint,
            )
            self._spline_names.add(name)

    def _destroy_all_spline_markers(self) -> None:
        """Destroy all spawned spline markers."""
        for name in list(self._spline_names):
            self._destroy_object(name)
        self._spline_names.clear()

    # -------------------------------------------------------------------------
    # Low-Level Object Management
    # -------------------------------------------------------------------------

    def _spawn_object(
        self,
        name: str,
        asset: str,
        position: np.ndarray,
        scale: Tuple[float, float, float],
        is_blueprint: bool = False,
    ) -> Optional[str]:
        try:
            pose = airsim.Pose(
                airsim.Vector3r(float(position[0]), float(position[1]), float(position[2])),
                airsim.to_quaternion(0, 0, 0),
            )
            scale_vec = airsim.Vector3r(float(scale[0]), float(scale[1]), float(scale[2]))

            actual_name = self._client.simSpawnObject(
                name, asset, pose, scale_vec,
                physics_enabled=False,
                is_blueprint=is_blueprint,
            )

            if actual_name:
                self._balloon_names.add(actual_name)
                return actual_name
            else:
                logger.warning(f"Failed to spawn {name}: returned empty")
                return None

        except Exception as e:
            logger.warning(f"Failed to spawn {name}: {e}")
            return None

    def _destroy_object(self, name: str) -> bool:
        try:
            self._client.simDestroyObject(name)
            return True
        except Exception as e:
            logger.debug(f"Failed to destroy {name}: {e}")
            return False
