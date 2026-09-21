"""Sparse monocular voxel occupancy core.

The pipeline turns a Depth Anything depth map into world-frame points and
integrates them into a bounded voxel map. The occupancy grid itself is an
Open3D ``VoxelGrid``; two small maps keep the per-voxel hit count and
last-observed time that Open3D has nowhere to store (they drive decay,
eviction, and the viewer's Min Hits filter). Nothing here is model aware -
pure geometry plus the voxel store.

Camera convention matches iSpy/vision/triangulation.py:
    camera point = (right, down, forward), all in output units
    robot frame  = (+x right, +y forward, +z up)
"""

import math
import time

import numpy as np

_o3d = None


def _open3d():
    global _o3d
    if _o3d is None:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError(
                "open3d is required for the voxel world pipeline and is a base "
                "dependency of ispy-frc, but it is not importable here. Fresh "
                'installs get it automatically; fix this one with: pip install '
                '"open3d>=0.18.0"'
            ) from exc
        _o3d = o3d
    return _o3d


def depth_plane_range(
    depth: np.ndarray,
    d_min: float,
    d_max: float,
    max_depth: float,
    depth_scale: float = 1.0,
    near_depth: float = 0.3,
) -> tuple[float, float, float, float]:
    """Resolve ``(d_lo, d_hi, z_near, z_far)`` for the inverse-depth model.

    Depth Anything emits relative *inverse* depth (disparity-like: larger =
    closer), so the only scale-free thing to recover is the scene's depth
    *ratio*. The nearest robust pixel (98th percentile of disparity) anchors to
    ``near_depth`` and the far plane follows ``d_hi / d_lo``, capped by
    ``max_depth * depth_scale`` - otherwise every scene gets force-stretched
    into a ~10 m funnel.
    """
    depth = np.asarray(depth, dtype=np.float64)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        z_near = max(float(near_depth), 1e-3)
        return 0.0, 1.0, z_near, z_near

    lo = float(np.percentile(finite, 2.0))
    hi = float(np.percentile(finite, 98.0))
    if hi - lo <= 1e-9:
        # no usable spread in the middle 96%: fall back to the caller's range,
        # and only then to a flat plane at a stable mid distance.
        lo, hi = float(d_min), float(d_max)
        if hi - lo <= 1e-9:
            # Fully degenerate frame (uniform depth). Bounds must stay
            # consistent with the model's real value range instead of the
            # hardcoded [0, 1] - a uniform map around 40.0 would otherwise
            # clip every pixel into [0, 1] and flatten a legitimately varied
            # depth map to a single distance.
            f_lo = float(finite.min())
            f_hi = float(finite.max())
            if f_hi - f_lo > 1e-9:
                lo, hi = f_lo, f_hi
            else:
                lo = hi = f_lo
            z_near = max(float(near_depth), 1e-3)
            z_far = max(float(max_depth) * float(depth_scale), z_near * 1.0001)
            mid = math.sqrt(z_near * z_far)
            return lo, hi, mid, mid

    z_near = max(float(near_depth), 1e-3)
    z_cap = max(float(max_depth) * float(depth_scale), z_near * 1.0001)
    ratio = hi / max(lo, 1e-6)
    z_far = min(z_near * ratio, z_cap)
    z_far = max(z_far, z_near * 1.0001)
    return lo, hi, z_near, z_far


def inverse_depth_to_distance(
    value: float, d_lo: float, d_hi: float, z_near: float, z_far: float
) -> float:
    """Map one relative (inverse-depth) value to a forward distance."""
    d = min(max(float(value), float(d_lo)), float(d_hi))
    distance = float(z_near) * (float(d_hi) / max(d, 1e-9))
    return min(max(distance, float(z_near)), float(z_far))


def relative_depth_to_distance(
    depth: np.ndarray,
    d_min: float,
    d_max: float,
    max_depth: float,
    depth_scale: float = 1.0,
    near_depth: float = 0.3,
) -> np.ndarray:
    """Map Depth Anything's relative depth to an approximate forward distance.

    Distance is disparity inverted so the nearest robust pixel lands at
    ``near_depth`` and the far plane follows the scene's own ratio (capped by
    ``max_depth * depth_scale``). The 2%-98% percentile band stops outliers
    from squashing the scene. ``near_depth`` is the global scale anchor: raise
    it to enlarge the whole reconstruction.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.size == 0:
        return depth

    d_lo, d_hi, z_near, z_far = depth_plane_range(
        depth, d_min, d_max, max_depth, depth_scale, near_depth
    )
    if z_near == z_far:
        return np.nan_to_num(
            np.full_like(depth, z_near), nan=0.0, posinf=0.0, neginf=0.0
        )

    d = np.clip(depth, d_lo, d_hi)
    distance = z_near * (d_hi / np.maximum(d, 1e-9))
    np.clip(distance, z_near, z_far, out=distance)
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
    depth: np.ndarray,
    focal_px: float,
    focal_px_y: float | None = None,
    stride: int = 8,
    return_pixels: bool = False,
):
    """Back-project a depth map into camera-frame (right, down, forward) points.

    Pixels are sampled on a ``stride`` grid so the point count (and therefore
    voxel cost) is bounded; non-finite / non-positive samples are dropped. With
    ``return_pixels`` the matching depth-map pixel coordinates are returned too,
    so callers can color the points from the frame.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2 or depth.size == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        return (empty, np.empty((0, 2), dtype=np.int64)) if return_pixels else empty

    h, w = depth.shape
    stride = max(1, int(stride))
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    sampled = depth[np.ix_(ys, xs)]

    valid = np.isfinite(sampled) & (sampled > 0.0)
    if not valid.any():
        empty = np.empty((0, 3), dtype=np.float64)
        return (empty, np.empty((0, 2), dtype=np.int64)) if return_pixels else empty

    uu, vv = np.meshgrid(xs, ys)
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    d = sampled[valid]

    cx, cy = w / 2.0, h / 2.0
    f = max(float(focal_px), 1e-6)
    fy = max(float(focal_px_y), 1e-6) if focal_px_y is not None else f
    x = (u - cx) / f * d
    y = (v - cy) / fy * d
    points = np.stack([x, y, d], axis=1)
    if not return_pixels:
        return points
    pixels = np.stack([u, v], axis=1).astype(np.int64)
    return points, pixels


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
    """Open3D ``VoxelGrid`` occupancy map with hit counts, decay, and a cap.

    ``_grid`` owns the voxels and their colors; ``_hits`` / ``_seen`` are
    observation metadata Open3D cannot represent (a VoxelGrid member stores
    only a grid index and a color).
    """

    def __init__(
        self,
        voxel_size: float = 0.1,
        max_voxels: int = 20000,
        decay_seconds: float = 8.0,
    ):
        _open3d()
        self.voxel_size = max(float(voxel_size), 1e-4)
        self.max_voxels = max(1, int(max_voxels))
        self.decay_seconds = float(decay_seconds)
        self._grid = _open3d().geometry.VoxelGrid()
        self._hits: dict[tuple[int, int, int], int] = {}
        self._seen: dict[tuple[int, int, int], float] = {}

    @staticmethod
    def _key(grid_index) -> tuple[int, int, int]:
        return int(grid_index[0]), int(grid_index[1]), int(grid_index[2])

    def _add_voxel(self, key: tuple[int, int, int], color) -> None:
        o3d = _open3d()
        idx = np.asarray(key, dtype=np.int32)
        self._grid.add_voxel(o3d.geometry.Voxel(idx, np.asarray(color)))

    def _remove_voxel(self, key: tuple[int, int, int]) -> None:
        self._grid.remove_voxel(np.asarray(key, dtype=np.int32))
        self._hits.pop(key, None)
        self._seen.pop(key, None)

    def integrate(
        self,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        now: float | None = None,
    ) -> int:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return 0
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if pts.size == 0:
            return 0

        col = None
        if colors is not None:
            col = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
            if col.shape[0] == finite.shape[0]:
                col = col[finite]
            else:
                col = None
        if col is None:
            col = np.full((pts.shape[0], 3), 170.0)

        # Bucket the frame, then merge its voxels into the live grid. Per-voxel
        # hit counts come from numpy (Open3D averages colors but cannot report
        # how many points landed in a voxel); the grid itself is Open3D's.
        indices = np.floor(pts / self.voxel_size).astype(np.int64)
        unique, inverse, counts = np.unique(
            indices, axis=0, return_inverse=True, return_counts=True
        )
        # np.unique(axis=0) has historically returned `inverse` with an extra
        # trailing dimension on some NumPy 2.0.x builds - flatten it to 1-D so
        # np.add.at scatters into rgb_sum correctly on every version.
        inverse = np.asarray(inverse).reshape(-1)
        rgb_sum = np.zeros((unique.shape[0], 3), dtype=np.float64)
        np.add.at(rgb_sum, inverse, col)

        existing = {
            self._key(v.grid_index): v
            for v in self._grid.get_voxels()
        }
        now = time.monotonic() if now is None else now
        for i, key in enumerate(map(tuple, unique.tolist())):
            count = int(counts[i])
            mean = rgb_sum[i] / count
            old_hits = self._hits.get(key, 0)
            total = old_hits + count
            old_voxel = existing.get(key)
            if old_voxel is not None and old_hits:
                merged = (old_voxel.color * 255.0 * old_hits + mean * count) / total
            else:
                merged = mean
            self._hits[key] = total
            self._seen[key] = now
            self._add_voxel(key, merged / 255.0)
        self._enforce_capacity()
        return int(unique.shape[0])

    def decay(self, now: float | None = None) -> int:
        """Remove voxels not re-observed within ``decay_seconds``.

        Callers should run decay aligned to the observation cadence (e.g. on
        integration frames only), never on a faster wall-clock cadence:
        voxels are only re-observed when new depth is integrated, so pruning
        between integrations starves any map whose ``decay_seconds`` is
        smaller than the frame period, permanently (see VoxelWorldPipeline.
        run()).
        """
        if self.decay_seconds <= 0:
            return 0
        now = time.monotonic() if now is None else now
        stale = [
            key
            for key, last in self._seen.items()
            if now - last > self.decay_seconds
        ]
        for key in stale:
            self._remove_voxel(key)
        return len(stale)

    def _enforce_capacity(self) -> None:
        if len(self._hits) <= self.max_voxels:
            return
        ordered = sorted(
            self._hits.keys(), key=lambda key: (self._seen[key], self._hits[key])
        )
        for key in ordered[: len(self._hits) - self.max_voxels]:
            self._remove_voxel(key)

    def clear(self) -> None:
        self._grid = _open3d().geometry.VoxelGrid()
        self._hits.clear()
        self._seen.clear()

    def count(self) -> int:
        return len(self._hits)

    def bounds(self) -> tuple[list[float], list[float]] | None:
        if not self._hits:
            return None
        keys = np.array(list(self._hits.keys()), dtype=np.float64)
        low = (keys.min(axis=0) + 0.0) * self.voxel_size
        high = (keys.max(axis=0) + 1.0) * self.voxel_size
        return low.tolist(), high.tolist()

    def export(
        self, max_points: int | None = None, min_count: int = 1
    ) -> list[list[float]]:
        """Return ``[x, y, z, count, r, g, b]`` entries, most-observed first.

        Positions are voxel centers; ``r, g, b`` are 0-255 hit-weighted frame
        colors (mid-grey when color was never supplied) for the 3D viewer.
        """
        items = []
        for v in self._grid.get_voxels():
            key = self._key(v.grid_index)
            count = self._hits.get(key, 0)
            if count >= int(min_count):
                items.append((key, count, v.color))
        items.sort(key=lambda item: item[1], reverse=True)
        if max_points is not None:
            items = items[: max(0, int(max_points))]
        half = self.voxel_size / 2.0
        return [
            [
                key[0] * self.voxel_size + half,
                key[1] * self.voxel_size + half,
                key[2] * self.voxel_size + half,
                int(count),
                *[
                    int(max(0, min(255, round(float(channel) * 255.0))))
                    for channel in color
                ],
            ]
            for key, count, color in items
        ]
