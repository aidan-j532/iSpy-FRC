
import math

class Transform2D:

    def __init__(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0):
        self._x = float(x)
        self._y = float(y)
        self._heading = float(heading)

    def X(self) -> float:
        return self._x

    def Y(self) -> float:
        return self._y

    def rotation(self):
        outer = self

        class _Rot:
            def radians(self_inner):
                return outer._heading

        return _Rot()

def _as_transform(pose):
    if isinstance(pose, Transform2D):
        return pose
    if isinstance(pose, dict):
        return Transform2D(
            float(pose.get("x", 0) or 0),
            float(pose.get("y", 0) or 0),
            float(pose.get("heading", 0) or 0),
        )
    try:
        return Transform2D(
            float(pose.X()),
            float(pose.Y()),
            float(pose.rotation().radians()),
        )
    except Exception:
        return Transform2D()

def transform_2d(
    obj,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
):
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    field_x = obj.x * cos_y + obj.y * sin_y
    field_y = obj.y * cos_y - obj.x * sin_y
    obj.x = field_x + tx
    obj.y = field_y + ty
    obj.z = obj.z + tz
    obj.yaw = (obj.yaw + yaw) % (2 * math.pi)
    obj.roll = (obj.roll + roll) % (2 * math.pi)
    obj.pitch = (obj.pitch + pitch) % (2 * math.pi)
    return obj

robot_to_field = transform_2d
relative_to = transform_2d
Pose2D = Transform2D
_as_pose2d = _as_transform

