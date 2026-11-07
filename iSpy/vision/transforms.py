"""Generic coordinate transforms shared by trackers and pipelines.

These used to live as ``Object.relative_to``, which baked a robot-frame
transform (and the FRC yaw convention) into the detection data model.  Moving
them here keeps ``Object`` a plain data container and makes the transform an
explicit operation that callers opt into.
"""

import math


class Transform2D:
    """Generic 2-D rigid transform (x, y, heading) used by core as an opaque

    reference transform. It intentionally exposes only the minimal accessors
    core needs (``X()``, ``Y()``, ``rotation().radians()``) and carries no
    robot/field semantics.
    """

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
    """Coerce a WPILib pose (or dict) into the generic :class:`Transform2D`."""
    if isinstance(pose, Transform2D):
        return pose
    if isinstance(pose, dict):
        return Transform2D(
            float(pose.get("x", 0) or 0),
            float(pose.get("y", 0) or 0),
            float(pose.get("heading", 0) or 0),
        )
    # WPILib (or compatible) pose
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
    """Apply a 2-D rigid transform to an object's coordinates.

    ``tx, ty, yaw`` etc. describe the transform to apply.  Mutates ``obj`` in
    place and returns it.  This is a generic transform; FRC-specific
    conventions (axis swaps, robot pose) are applied by the caller (FRC plugin).
    """
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


# Backwards-compatible aliases for the old names.
robot_to_field = transform_2d
relative_to = transform_2d
Pose2D = Transform2D
_as_pose2d = _as_transform