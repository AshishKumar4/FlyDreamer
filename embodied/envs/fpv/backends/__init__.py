"""
Simulator backends for FPV environments
"""

from embodied.envs.fpv.backends.base import SimulatorBackend

# Import backends with fallbacks for missing dependencies
__all__ = ["SimulatorBackend"]

try:
    from embodied.envs.fpv.backends.colosseum import ColosseumBackend, ColosseumConfig
    __all__.extend(["ColosseumBackend", "ColosseumConfig"])
except ImportError:
    pass