import numpy as np
import math
import time


class Object:
    def __init__(
        self,
        x: float,
        y: float,
        id: int = -1,
        alive_time: float = 0.4,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ):
        self.x = x
        self.y = y
        self.id = id
        # 6DOF rotation in radians, zero for detect-only models
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

        self.start_time = time.perf_counter()
        self.alive = 0
        self.destroyed = False
        self.alive_time = alive_time

    def relative_to(self, robot_x: float, robot_y: float, robot_yaw_rad: float):
        cos_y = math.cos(robot_yaw_rad)
        sin_y = math.sin(robot_yaw_rad)
        field_x = cos_y * self.x - sin_y * self.y
        field_y = sin_y * self.x + cos_y * self.y
        self.x = field_x + robot_x
        self.y = field_y + robot_y
        # Object's heading in field frame = camera-frame yaw + robot heading
        # Roll and pitch stay in camera frame (no meaningful field transform)
        self.yaw = (self.yaw + robot_yaw_rad) % (2 * math.pi)

    def get_position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def get_rotation(self) -> tuple[float, float, float]:
        return (self.roll, self.pitch, self.yaw)

    def has_rotation(self) -> bool:
        """True if a pose model provided non-zero orientation data."""
        return self.roll != 0.0 or self.pitch != 0.0 or self.yaw != 0.0

    def reset_time(self):
        self.start_time = time.perf_counter()

    def get_position_normally(self) -> tuple[float, float]:
        return (self.x, self.y)

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
            f"  X: {self.x:.3f}  Y: {self.y:.3f}"
            f"{rot}  Alive: {self.alive:.2f}s"
        )