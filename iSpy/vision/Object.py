import numpy as np
import math
import time


class Object:
    def __init__(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        id: int = -1,
        alive_time: float = 0.4,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.id = id
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

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
        cos_y = math.cos(robot_yaw)
        sin_y = math.sin(robot_yaw)
        field_x = cos_y * self.x - sin_y * self.y
        field_y = sin_y * self.x + cos_y * self.y
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