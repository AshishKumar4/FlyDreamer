"""
FPV drone navigation environment
"""

# Main environment
from embodied.envs.fpv.env import FPVEnv

# Factory functions
from embodied.envs.fpv.factory import (
    create_fpv_env,
    create_colosseum_env,
    register_backend,
    register_reward,
    register_waypoint_manager,
)

# Types
from embodied.envs.fpv.types import (
    DroneState,
    ImageData,
    ControlCommand,
    EnvConfig,
    ControlConfig,
    RewardConfig,
    WaypointConfig,
    MarkerConfig,
)

# Base classes (for custom implementations)
from embodied.envs.fpv.backends.base import SimulatorBackend
from embodied.envs.fpv.rewards.base import RewardStrategy, RewardInfo
from embodied.envs.fpv.waypoints.base import WaypointManager

__all__ = [
    # Main environment
    "FPVEnv",
    # Factory functions
    "create_fpv_env",
    "create_colosseum_env",
    "create_isaacsim_env",
    "register_backend",
    "register_reward",
    "register_waypoint_manager",
    # Types
    "DroneState",
    "ImageData",
    "ControlCommand",
    "EnvConfig",
    "ControlConfig",
    "RewardConfig",
    "WaypointConfig",
    "MarkerConfig",
    # Base classes
    "SimulatorBackend",
    "RewardStrategy",
    "RewardInfo",
    "WaypointManager",
]
