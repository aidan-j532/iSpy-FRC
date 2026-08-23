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

        # vis_type -> renderer mapping (see VIS_RENDERERS in viewer3d.html)
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
        # frame: +X right, +Y forward, +Z up; robot_yaw is +ve when the
        # robot turned RIGHT (boresight rotated from +Y toward +X)
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

    # ------------------------------------------------------------------
    # Universal output schema (see VisionPipeline.OUTPUT_SCHEMA)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to the universal pipeline-output schema.

        JSON-safe, pipeline-agnostic: every pipeline's Objects flatten to the
        same keys; ``vis_type`` selects how to interpret them and ``vis_meta``
        carries the pipeline-specific payload (tag ids, flow vectors, depth
        estimates, ...).
        """
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
            "keypoints_3d": (
                [[float(c) for c in kpt] for kpt in self.keypoints_3d]
                if self.keypoints_3d is not None else None
            ),
            "ray_origin": (
                [float(c) for c in self.ray_origin]
                if self.ray_origin is not None else None
            ),
            "ray_direction": (
                [float(c) for c in self.ray_direction]
                if self.ray_direction is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Object":
        """Rebuild an Object from :meth:`to_dict` output.

        Tolerates missing keys and legacy flat dicts; unknown keys are ignored.
        """
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