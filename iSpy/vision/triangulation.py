import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Ray:
    origin: np.ndarray     # (3,) robot-frame, in the config's configured unit
    direction: np.ndarray  # (3,) unit vector, robot-frame


def pixel_to_ray(
    pixel_x: float,
    pixel_y: float,
    img_w: int,
    img_h: int,
    focal_length_px: float,
    camera_x: float,
    camera_y: float,
    camera_z: float,
    yaw_deg: float,
    pitch_deg: float,
) -> Ray:
    """Camera-local pixel -> a real 3D ray in robot frame. No object-size
    assumption anywhere in this function - just intrinsics + extrinsics."""
    cx, cy = img_w / 2.0, img_h / 2.0
    f = max(focal_length_px, 1e-6)

    dx = (pixel_x - cx) / f
    dy = (pixel_y - cy) / f
    dz = 1.0
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / norm, dy / norm, dz / norm

    # pitch: rotate (down, forward) about the camera's local right axis
    pitch = math.radians(pitch_deg)
    cp, sp = math.cos(pitch), math.sin(pitch)
    dy2 = dy * cp - dz * sp
    dz2 = dy * sp + dz * cp
    dx2 = dx

    # yaw: same convention as ObjectDetectionCamera._camera_point_to_robot
    yaw = math.radians(yaw_deg)
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    x_rot = dz2 * cy_ + (-dx2) * sy_
    y_rot = dz2 * sy_ - (-dx2) * cy_
    z_rot = -dy2  # down-positive camera axis -> up-positive world axis

    direction = np.array([x_rot, y_rot, z_rot], dtype=np.float64)
    direction /= (np.linalg.norm(direction) or 1.0)

    origin = np.array([camera_x, camera_y, camera_z], dtype=np.float64)
    return Ray(origin=origin, direction=direction)


def ground_plane_intersection(ray: Ray, ground_z: float = 0.0) -> np.ndarray | None:
    """Where a ray crosses the ground plane. None if parallel to the ground
    or the intersection is behind the camera (not physical)."""
    dz = ray.direction[2]
    if abs(dz) < 1e-9:
        return None
    t = (ground_z - ray.origin[2]) / dz
    if t <= 0:
        return None
    return ray.origin + t * ray.direction


def closest_point_between_rays(
    ray_a: Ray, ray_b: Ray, max_residual: float = 0.5
) -> tuple[np.ndarray, float] | None:
    """Midpoint of the shortest segment connecting two skew rays. residual
    is the gap between the two closest points - large residual means the
    rays don't converge, i.e. these are probably two different objects,
    not the same one seen twice. max_residual is in the config's unit.
    """
    o1, d1 = ray_a.origin, ray_a.direction
    o2, d2 = ray_b.origin, ray_b.direction

    d1d2 = float(np.dot(d1, d2))
    denom = 1.0 - d1d2 * d1d2
    if abs(denom) < 1e-9:
        return None  # rays (nearly) parallel - can't triangulate

    w0 = o1 - o2
    a = float(np.dot(d1, w0))
    b = float(np.dot(d2, w0))
    t1 = (d1d2 * b - a) / denom
    t2 = (b - d1d2 * a) / denom

    if t1 <= 0 or t2 <= 0:
        return None  # intersection behind one of the cameras

    p1 = o1 + t1 * d1
    p2 = o2 + t2 * d2
    residual = float(np.linalg.norm(p1 - p2))
    if residual > max_residual:
        return None

    return (p1 + p2) / 2.0, residual