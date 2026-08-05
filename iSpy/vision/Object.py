import numpy as np
import math
import time
import itertools
_id_counter = itertools.count(1)

class Object:
    def __init__(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        id: int | None = None,
        alive_time: float = 0.4,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        name: str = "unknown",
        confidence: float = 0.0,
        keypoints_3d: list | None = None,
        ray_origin=None,
        ray_direction=None,
        depth_source: str = "monocular",
        vis_type: str = "generic",
        vis_meta: dict | None = None,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.id = id if id is not None else next(_id_counter)
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.name = name
        self.confidence = confidence
        self.keypoints_3d = keypoints_3d
        self.ray_origin = ray_origin        # np.ndarray(3,) robot-frame, or None
        self.ray_direction = ray_direction  # np.ndarray(3,) unit vector, or None
        self.depth_source = depth_source

        # Drives which renderer the 3D viewer uses for this object. New
        # detector types just need a value here + a matching entry in
        # VIS_RENDERERS in viewer3d.html - no other file needs to change.
        #   "generic" -> cube (default)
        #   "points"  -> raw keypoint dots, no bones (auto-selected whenever
        #                keypoints_3d is set, regardless of this field)
        #   "planar"  -> flat rectangle + orientation gizmo (AprilTags, QR, barcodes)
        self.vis_type = vis_type
        self.vis_meta = vis_meta or {}

        self.start_time = time.perf_counter()
        self.alive = 0
        self.destroyed = False
        self.alive_time = alive_time

    def relative_to(
        self,
        robot_x: float,
        robot_y: float,
        robot_z: float = 0.0,
        robot_roll: float = 0.0,
        robot_pitch: float = 0.0,
        robot_yaw: float = 0.0,
    ):
        # Frame: +X right, +Y forward, +Z up. robot_yaw is positive when the
        # robot has turned RIGHT (boresight rotated from +Y toward +X).
        cos_y = math.cos(robot_yaw)
        sin_y = math.sin(robot_yaw)
        field_x = self.x * cos_y + self.y * sin_y
        field_y = self.y * cos_y - self.x * sin_y
        self.x = field_x + robot_x
        self.y = field_y + robot_y
        self.z = self.z + robot_z
        self.yaw = (self.yaw + robot_yaw) % (2 * math.pi)
        self.roll = (self.roll + robot_roll) % (2 * math.pi)
        self.pitch = (self.pitch + robot_pitch) % (2 * math.pi)

    def get_position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def get_rotation(self) -> tuple[float, float, float]:
        return (self.roll, self.pitch, self.yaw)

    def has_rotation(self) -> bool:
        return self.roll != 0.0 or self.pitch != 0.0 or self.yaw != 0.0

    def reset_time(self):
        self.start_time = time.perf_counter()

    def get_position_normally(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def get_id(self) -> int:
        return self.id

    def set_id(self, id: int):
        self.id = id

    def update(self):
        self.alive = time.perf_counter() - self.start_time
        if self.alive >= self.alive_time:
            self.destroyed = True

    def __str__(self) -> str:
        rot = (
            f"  Roll: {math.degrees(self.roll):.1f}°"
            f"  Pitch: {math.degrees(self.pitch):.1f}°"
            f"  Yaw: {math.degrees(self.yaw):.1f}°"
            if self.has_rotation()
            else ""
        )
        return (
            f"Distance: {math.hypot(self.x, self.y):.3f}"
            f"  X: {self.x:.3f}  Y: {self.y:.3f}  Z: {self.z:.3f}"
            f"{rot}  Alive: {self.alive:.2f}s"
        )