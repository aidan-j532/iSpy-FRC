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
        source_camera: str | None = None,
        source_cameras: list | None = None,
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
        self.ray_origin = ray_origin  # np.ndarray(3,) robot-frame, or None
        self.ray_direction = ray_direction  # np.ndarray(3,) unit vector, or None
        self.depth_source = depth_source
        cameras = []
        if source_cameras:
            cameras.extend(str(name) for name in source_cameras if str(name).strip())
        if source_camera and str(source_camera) not in cameras:
            cameras.insert(0, str(source_camera))
        self.source_cameras = list(dict.fromkeys(cameras))
        self.source_camera = str(source_camera) if source_camera else (
            self.source_cameras[0] if self.source_cameras else None
        )
        self.frame = "field"

        self.vis_type = vis_type
        self.vis_meta = vis_meta or {}

        self.start_time = time.perf_counter()
        self.alive = 0
        self.destroyed = False
        self.alive_time = alive_time

    def get_position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def get_rotation(self) -> tuple[float, float, float]:
        return (self.roll, self.pitch, self.yaw)

    def has_rotation(self) -> bool:
        return self.roll != 0.0 or self.pitch != 0.0 or self.yaw != 0.0

    def reset_time(self):
        self.start_time = time.perf_counter()
        self.alive = 0
        self.destroyed = False

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

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "confidence": float(self.confidence),
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "roll": float(self.roll),
            "pitch": float(self.pitch),
            "yaw": float(self.yaw),
            "depth_source": self.depth_source,
            "vis_type": self.vis_type,
            "vis_meta": dict(self.vis_meta),
            "source_camera": self.source_camera,
            "source_cameras": list(self.source_cameras),
            "keypoints_3d": (
                [[float(c) for c in kpt] for kpt in self.keypoints_3d]
                if self.keypoints_3d is not None
                else None
            ),
            "ray_origin": (
                [float(c) for c in self.ray_origin]
                if self.ray_origin is not None
                else None
            ),
            "ray_direction": (
                [float(c) for c in self.ray_direction]
                if self.ray_direction is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Object":
        obj = cls(
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            z=float(data.get("z", 0.0)),
            id=data.get("id"),
            roll=float(data.get("roll", 0.0)),
            pitch=float(data.get("pitch", 0.0)),
            yaw=float(data.get("yaw", 0.0)),
            name=str(data.get("name", "unknown")),
            confidence=float(data.get("confidence", 0.0)),
            keypoints_3d=data.get("keypoints_3d"),
            depth_source=str(data.get("depth_source", "monocular")),
            vis_type=str(data.get("vis_type", "generic")),
            vis_meta=dict(data.get("vis_meta") or {}),
            source_camera=data.get("source_camera"),
            source_cameras=data.get("source_cameras"),
        )
        origin = data.get("ray_origin")
        direction = data.get("ray_direction")
        if origin is not None:
            obj.ray_origin = np.asarray(origin, dtype=float)
        if direction is not None:
            obj.ray_direction = np.asarray(direction, dtype=float)
        return obj

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
