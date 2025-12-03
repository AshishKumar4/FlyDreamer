"""Waypoint management for FPV environment."""

from embodied.envs.fpv.waypoints.base import WaypointManager
from embodied.envs.fpv.waypoints.visual import VisualWaypointManager

__all__ = ["WaypointManager", "VisualWaypointManager"]
