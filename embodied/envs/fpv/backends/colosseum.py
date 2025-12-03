"""Colosseum (AirSim fork) simulator backend."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from embodied.envs.fpv.backends.base import SimulatorBackend
from embodied.envs.fpv.types import DroneState, ImageData, ControlCommand

logger = logging.getLogger(__name__)

try:
    import airsim
    import msgpackrpc
    AIRSIM_AVAILABLE = True
except ImportError:
    raise ImportError(
        "airsim package required for ColosseumBackend. "
        "Install with: pip install airsim"
    )


@dataclass
class ColosseumConfig:
    """Configuration for Colosseum backend."""
    vehicle_name: str = ""
    max_retries: int = 3
    spawn_position: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    spawn_yaw: float = 0.0


class ColosseumBackend(SimulatorBackend):

    def __init__(self, config: Optional[ColosseumConfig] = None):
        self._config = config or ColosseumConfig()
        self._client: Optional[airsim.MultirotorClient] = None
        self._connected: bool = False

        self._spawned_markers: Set[str] = set()
        self._last_state: Optional[DroneState] = None
        self._last_raw_frame: Optional[np.ndarray] = None
        self._last_collision_time: int = 0

    @property
    def name(self) -> str:
        return "Colosseum (AirSim)"

    @property
    def supports_visual_markers(self) -> bool:
        return True

    # =========================================================================
    # Connection Management
    # =========================================================================

    def connect(self) -> None:
        if self._connected:
            return

        logger.info("Connecting to Colosseum simulator...")

        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        self._client.enableApiControl(True, vehicle_name=self._config.vehicle_name)
        self._client.armDisarm(True, vehicle_name=self._config.vehicle_name)
        self._client.simPause(True)

        self._connected = True

        try:
            self._cleanup_stale_markers()
        except Exception as e:
            logger.warning(f"Failed to cleanup stale FPV markers on connect: {e}")

        logger.info("Connected to Colosseum simulator")

    def disconnect(self) -> None:
        if not self._connected or self._client is None:
            return

        try:
            self.destroy_all_markers()
            self._client.armDisarm(False, vehicle_name=self._config.vehicle_name)
            self._client.enableApiControl(False, vehicle_name=self._config.vehicle_name)
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")

        self._client = None
        self._connected = False
        logger.info("Disconnected from Colosseum simulator")

    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    def _reconnect(self) -> None:
        logger.info("Reconnecting to Colosseum...")
        time.sleep(0.5)

        try:
            self._client = airsim.MultirotorClient()
            self._client.confirmConnection()
            self._client.enableApiControl(True, vehicle_name=self._config.vehicle_name)
            self._client.armDisarm(True, vehicle_name=self._config.vehicle_name)
            self._client.simPause(True)
            logger.info("Reconnected successfully")
        except Exception as e:
            logger.error(f"Reconnect failed: {e}")
            raise

    # =========================================================================
    # Simulation Control
    # =========================================================================

    def reset(self) -> DroneState:
        if not self._connected:
            self.connect()

        for attempt in range(self._config.max_retries):
            try:
                self._client.reset()
                self._client.simPause(True)
                self._client.enableApiControl(True, vehicle_name=self._config.vehicle_name)
                self._client.armDisarm(True, vehicle_name=self._config.vehicle_name)

                spawn_pos = self._config.spawn_position
                spawn_pose = airsim.Pose(
                    airsim.Vector3r(spawn_pos[0], spawn_pos[1], spawn_pos[2]),
                    airsim.to_quaternion(0.0, 0.0, self._config.spawn_yaw),
                )

                try:
                    self._client.simSetVehiclePose(
                        spawn_pose, ignore_collisions=False,
                        vehicle_name=self._config.vehicle_name
                    )
                except TypeError:
                    try:
                        self._client.simSetVehiclePose(
                            spawn_pose, ignore_collision=False,
                            vehicle_name=self._config.vehicle_name
                        )
                    except TypeError:
                        self._client.simSetVehiclePose(
                            spawn_pose, False, self._config.vehicle_name
                        )

                self._spawned_markers.clear()

                collision_info = self._client.simGetCollisionInfo(
                    vehicle_name=self._config.vehicle_name
                )
                self._last_collision_time = collision_info.time_stamp

                state = self.get_state()
                self._last_state = state
                return state

            except (msgpackrpc.error.TransportError, UnicodeDecodeError, TypeError) as e:
                logger.warning(f"RPC error in reset (attempt {attempt + 1}): {e}")
                if attempt < self._config.max_retries - 1:
                    self._reconnect()
                else:
                    raise ConnectionError(f"Failed to reset after {self._config.max_retries} attempts")

    def step(self, command: ControlCommand, dt: float) -> Tuple[DroneState, bool]:
        if not self._connected:
            raise RuntimeError("Not connected to simulator")

        has_collided = False

        for attempt in range(self._config.max_retries):
            try:
                self._client.moveByAngleRatesThrottleAsync(
                    float(command.roll),
                    float(command.pitch),
                    float(command.yaw),
                    float(command.throttle),
                    float(dt),
                    vehicle_name=self._config.vehicle_name,
                )
                self._client.simContinueForTime(dt)

                # Detect new collisions by comparing timestamps
                collision_info = self._client.simGetCollisionInfo(
                    vehicle_name=self._config.vehicle_name
                )
                has_collided = False
                if collision_info.has_collided and collision_info.time_stamp != self._last_collision_time:
                    has_collided = True
                    self._last_collision_time = collision_info.time_stamp
                    logger.warning(f"[Collision] New collision at time={collision_info.time_stamp}")
                break

            except (msgpackrpc.error.TransportError, UnicodeDecodeError, TypeError) as e:
                logger.warning(f"RPC error in step (attempt {attempt + 1}): {e}")
                if attempt < self._config.max_retries - 1:
                    self._reconnect()
                else:
                    raise ConnectionError(f"Step failed after {self._config.max_retries} attempts")

        state = self.get_state()
        self._last_state = state
        return state, has_collided

    def get_state(self) -> DroneState:
        if not self._connected:
            raise RuntimeError("Not connected to simulator")

        state_data = self._client.getMultirotorState(
            vehicle_name=self._config.vehicle_name
        )

        pos = state_data.kinematics_estimated.position
        vel = state_data.kinematics_estimated.linear_velocity
        ori = state_data.kinematics_estimated.orientation
        ang_vel = state_data.kinematics_estimated.angular_velocity

        return DroneState(
            position=np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32),
            velocity=np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float32),
            orientation=np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val], dtype=np.float32),
            angular_velocity=np.array([ang_vel.x_val, ang_vel.y_val, ang_vel.z_val], dtype=np.float32),
            timestamp=state_data.timestamp,
        )

    def get_images(self, size: Tuple[int, int], include_depth: bool = True) -> ImageData:
        if not self._connected:
            raise RuntimeError("Not connected to simulator")

        requests = [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
        if include_depth:
            requests.append(airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False))

        responses = self._client.simGetImages(
            requests, vehicle_name=self._config.vehicle_name
        )

        rgb = np.zeros(size + (3,), dtype=np.uint8)
        raw_rgb = None

        if responses and responses[0].width > 0:
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img = img1d.reshape(responses[0].height, responses[0].width, 3)
            img = img[..., ::-1]  # BGR -> RGB
            raw_rgb = img.copy()

            if img.shape[:2] != size:
                img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
            rgb = img

        self._last_raw_frame = raw_rgb

        depth = None
        if include_depth:
            depth = np.full(size + (1,), 255, dtype=np.uint8)

            if len(responses) > 1 and responses[1].width > 0:
                depth_img = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
                depth_img = depth_img.reshape(responses[1].height, responses[1].width, 3)
                depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)

                if depth_img.shape[:2] != size:
                    depth_img = cv2.resize(
                        depth_img, (size[1], size[0]), interpolation=cv2.INTER_NEAREST
                    )
                depth = depth_img[..., None].astype(np.uint8)

        return ImageData(rgb=rgb, depth=depth, raw_rgb=raw_rgb)

    @property
    def last_raw_frame(self) -> Optional[np.ndarray]:
        return self._last_raw_frame

    # =========================================================================
    # Visual Markers
    # =========================================================================

    def spawn_marker(
        self,
        name: str,
        position: np.ndarray,
        marker_type: str,
        scale: float,
        asset: str,
        is_blueprint: bool = False,
    ) -> Optional[str]:
        if not self._connected:
            return None

        try:
            pose = airsim.Pose(
                airsim.Vector3r(float(position[0]), float(position[1]), float(position[2])),
                airsim.to_quaternion(0, 0, 0),
            )
            scale_vec = airsim.Vector3r(float(scale), float(scale), float(scale))

            actual_name = self._client.simSpawnObject(
                name, asset, pose, scale_vec,
                physics_enabled=False,
                is_blueprint=is_blueprint,
            )

            if actual_name:
                if isinstance(actual_name, bytes):
                    actual_name = actual_name.decode('utf-8')
                self._spawned_markers.add(actual_name)
                return actual_name

            logger.warning(f"Failed to spawn marker {name}: returned empty")
            return None

        except Exception as e:
            logger.warning(f"Failed to spawn marker {name}: {e}")
            return None

    def destroy_marker(self, name: str) -> bool:
        if not self._connected:
            return False

        try:
            self._client.simDestroyObject(name)
            self._spawned_markers.discard(name)
            return True
        except Exception as e:
            logger.warning(f"Failed to destroy marker {name}: {e}")
            return False

    def destroy_all_markers(self, prefix: str = "") -> int:
        count = 0
        for name in list(self._spawned_markers):
            if not prefix or name.startswith(prefix):
                if self.destroy_marker(name):
                    count += 1
        return count

    def _cleanup_stale_markers(self) -> None:
        if not self._connected or self._client is None:
            return

        patterns = ["WPS.*", "SPS.*"]
        removed = 0

        for pattern in patterns:
            try:
                names = self._client.simListSceneObjects(pattern)
            except Exception as e:
                logger.warning(f"Failed to list scene objects for pattern '{pattern}': {e}")
                continue

            decoded = [n.decode("utf-8") if isinstance(n, bytes) else n for n in names]
            if decoded:
                logger.info(f"Found {len(decoded)} existing FPV markers for pattern '{pattern}'")

            for name in decoded:
                try:
                    self._client.simDestroyObject(name)
                    removed += 1
                except Exception as e:
                    logger.warning(f"Failed to destroy stale marker {name}: {e}")

        if removed:
            logger.info(f"Removed {removed} stale FPV markers from scene")

    def list_assets(self) -> List[str]:
        if not self._connected:
            return []

        try:
            assets = self._client.simListAssets()
            return [a.decode('utf-8') if isinstance(a, bytes) else a for a in assets]
        except Exception as e:
            logger.warning(f"Failed to list assets: {e}")
            return []

    # =========================================================================
    # Utility
    # =========================================================================

    def get_info(self) -> dict:
        info = super().get_info()
        info.update({
            "vehicle_name": self._config.vehicle_name,
            "spawned_markers": len(self._spawned_markers),
        })
        return info

    def check_collision(self) -> bool:
        if not self._connected:
            return False

        try:
            info = self._client.simGetCollisionInfo(vehicle_name=self._config.vehicle_name)
            return info.has_collided
        except Exception:
            return False
