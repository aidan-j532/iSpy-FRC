"""Sparse monocular voxel occupancy core.

The pipeline turns a Depth Anything depth map into world-frame points and
integrates them into a bounded, sparse voxel map. Nothing here is model aware -
it is pure geometry plus a dict-backed voxel store so it stays cheap on the
Orange Pi class hardware iSpy targets.

Camera convention matches iSpy/vision/triangulation.py:
    camera point = (right, down, forward), all in output units
    robot frame  = (+x right, +y forward, +z up)
"""

import math
import time

import numpy as np


def relative_depth_to_distance(
    depth: np.ndarray,
    d_min: float,
    d_max: float,
    max_depth: float,
    depth_scale: float = 1.0,
) -> np.ndarray:
    """Map Depth Anything's relative depth to an approximate forward distance.

    Depth Anything V2 emits relative inverse depth (larger = closer). The far
    plane ``max_depth`` pins the scale; ``depth_scale`` is a user trim.
    """
    depth = np.asarray(depth, dtype=np.float64)
    span = float(d_max) - float(d_min)
    if span <= 1e-9:
        # flat depth map: there is no relative signal to scale, but collapsing
        # to distance 0 would put every point at the origin and the min-depth
        # filter would then discard the whole frame. Mid-range is a safer guess.
        closeness = np.full_like(depth, 0.5)
    else:
        closeness = (depth - float(d_min)) / span
    closeness = np.clip(closeness, 0.0, 1.0)
    distance = (1.0 - closeness) * float(max_depth) * float(depth_scale)
    return np.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0)


def focal_length_pixels(
    frame_w: int, calibration: dict | None, default_fov: float = 60.0
) -> float:
    """Resolve a horizontal focal length in pixels for the live frame width."""
    calib = calibration or {}
    if not isinstance(calib, dict):
        calib = {}
    frame_w = max(1, int(frame_w))

    fov = calib.get("fov") or 0
    try:
        fov = float(fov)
    except (TypeError, ValueError):
        fov = 0.0
    if fov > 0:
        return (frame_w / 2.0) / math.tan(math.radians(fov / 2.0))

    focal = calib.get("focal_length_pixels")
    if focal:
        try:
            focal = float(focal)
            if focal > 0:
                return focal
        except (TypeError, ValueError):
            pass

    camera_matrix = calib.get("camera_matrix")
    if camera_matrix and len(camera_matrix) >= 3:
        try:
            fx = float(camera_matrix[0][0])
            resolution = calib.get("resolution")
            if (
                isinstance(resolution, (list, tuple))
                and len(resolution) == 2
                and resolution[0]
            ):
                fx *= frame_w / float(resolution[0])
            if fx > 0:
                return fx
        except (TypeError, ValueError, IndexError):
            pass

    return (frame_w / 2.0) / math.tan(math.radians(default_fov / 2.0))


def depth_to_camera_points(
    depth: np.ndarray, focal_px: float, stride: int = 8
) -> np.ndarray:
    """Back-project a depth map into camera-frame (right, down, forward) points.

    ``depth`` is forward distance in output units. Pixels are sampled on a
    ``stride`` grid so the point count (and therefore voxel cost) is bounded.
    Non-finite / non-positive samples are dropped.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2 or depth.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    h, w = depth.shape
    stride = max(1, int(stride))
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    sampled = depth[np.ix_(ys, xs)]

    valid = np.isfinite(sampled) & (sampled > 0.0)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)

    uu, vv = np.meshgrid(xs, ys)
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    d = sampled[valid]

    cx, cy = w / 2.0, h / 2.0
    f = max(float(focal_px), 1e-6)
    x = (u - cx) / f * d
    y = (v - cy) / f * d
    return np.stack([x, y, d], axis=1)


def camera_points_to_robot(
    points: np.ndarray,
    camera_x: float,
    camera_y: float,
    camera_z: float,
    yaw_deg: float,
    pitch_deg: float,
) -> np.ndarray:
    """Vectorized twin of triangulation.camera_point_to_robot.

    Kept numerically identical so a scalar call and a batch call agree.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    pitch = math.radians(pitch_deg)
    cp, sp = math.cos(pitch), math.sin(pitch)
    down = pts[:, 1] * cp + pts[:, 2] * sp
    forward = -pts[:, 1] * sp + pts[:, 2] * cp

    yaw = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    x_rot = pts[:, 0] * cos_y + forward * sin_y
    y_rot = forward * cos_y - pts[:, 0] * sin_y
    z_rot = -down

    out = np.empty_like(pts)
    out[:, 0] = x_rot + camera_x
    out[:, 1] = y_rot + camera_y
    out[:, 2] = z_rot + camera_z
    return out


class SparseVoxelMap:
    """Dict-backed voxel occupancy grid with touch-based decay and a hard cap."""

    def __init__(
        self,
        voxel_size: float = 0.1,
        max_voxels: int = 20000,
        decay_seconds: float = 8.0,
    ):
        self.voxel_size = max(float(voxel_size), 1e-4)
        self.max_voxels = max(1, int(max_voxels))
        self.decay_seconds = float(decay_seconds)
        self._voxels: dict[tuple[int, int, int], list] = {}

    def integrate(self, points: np.ndarray, now: float | None = None) -> int:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return 0
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.size == 0:
            return 0

        indices = np.floor(pts / self.voxel_size).astype(np.int64)
        unique, counts = np.unique(indices, axis=0, return_counts=True)
        now = time.monotonic() if now is None else now
        for key, count in zip(map(tuple, unique.tolist()), counts.tolist()):
            entry = self._voxels.get(key)
            if entry is None:
                self._voxels[key] = [int(count), now]
            else:
                entry[0] += int(count)
                entry[1] = now
        self._enforce_capacity()
        return int(unique.shape[0])

    def decay(self, now: float | None = None) -> int:
        if self.decay_seconds <= 0:
            return 0
        now = time.monotonic() if now is None else now
        stale = [
            key
            for key, entry in self._voxels.items()
            if now - entry[1] > self.decay_seconds
        ]
        for key in stale:
            del self._voxels[key]
        return len(stale)

    def _enforce_capacity(self) -> None:
        if len(self._voxels) <= self.max_voxels:
            return
        ordered = sorted(
            self._voxels.items(), key=lambda kv: (kv[1][1], kv[1][0])
        )
        for key, _entry in ordered[: len(self._voxels) - self.max_voxels]:
            del self._voxels[key]

    def clear(self) -> None:
        self._voxels.clear()

    def count(self) -> int:
        return len(self._voxels)

    def bounds(self) -> tuple[list[float], list[float]] | None:
        if not self._voxels:
            return None
        keys = np.array(list(self._voxels.keys()), dtype=np.float64)
        low = (keys.min(axis=0) + 0.0) * self.voxel_size
        high = (keys.max(axis=0) + 1.0) * self.voxel_size
        return low.tolist(), high.tolist()

    def export(
        self, max_points: int | None = None, min_count: int = 1
    ) -> list[list[float]]:
        """Return ``[x, y, z, count]`` entries, most-observed first."""
        items = [
            (key, entry)
            for key, entry in self._voxels.items()
            if entry[0] >= int(min_count)
        ]
        if not items:
            return []
        items.sort(key=lambda kv: kv[1][0], reverse=True)
        if max_points is not None:
            items = items[: max(0, int(max_points))]
        half = self.voxel_size / 2.0
        return [
            [
                key[0] * self.voxel_size + half,
                key[1] * self.voxel_size + half,
                key[2] * self.voxel_size + half,
                int(entry[0]),
            ]
            for key, entry in items
        ]
