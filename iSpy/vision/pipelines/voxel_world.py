"""Monocular voxel world pipeline.

Reuses the existing Depth Anything V2 model/inference (and its optimization /
hardware backends) from :class:`DepthAnythingPipeline`, then lifts the relative
depth map into world-frame points and accumulates a sparse voxel occupancy map.

This is *approximate* geometry: Depth Anything outputs relative depth, so
distances are only as good as the far-plane scaling and camera calibration.
"""

import logging

import numpy as np

from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig, unit_to_inches
from iSpy.vision.Object import Object
from iSpy.vision.pipelines.depth_anything import DepthAnythingPipeline
from iSpy.vision.voxel_map import (
    SparseVoxelMap,
    camera_points_to_robot,
    depth_to_camera_points,
    focal_length_pixels,
    relative_depth_to_distance,
)

_INCHES_TO_OUTPUT_UNIT = {
    "meter": 0.0254,
    "meters": 0.0254,
    "inch": 1.0,
    "inches": 1.0,
    "foot": 1 / 12,
    "feet": 1 / 12,
    "centimeter": 2.54,
    "centimeters": 2.54,
    "frc": 0.0254,
}


class VoxelWorldPipeline(DepthAnythingPipeline):
    plugin_name = "voxel_world"
    beta = True

    # Voxels need real intrinsics for the back-projection to line up with the
    # field, so offer both the ChArUco and known-object focal wizards. The
    # pipeline still runs (approximate) without them - see _resolve_geometry.
    calibration_sections = ["charuco", "focal"]

    @classmethod
    def show_calibration(cls) -> bool:
        return True

    @classmethod
    def needs_calibration_to_run(cls) -> bool:
        # monocular depth is approximate anyway, so never hard-gate on this
        return False

    @classmethod
    def config_schema(cls) -> dict:
        schema = super().config_schema()
        schema.update(
            {
                "voxel_size": {
                    "type": "number",
                    "label": "Voxel Size (m)",
                    "default": 0.1,
                    "step": 0.01,
                    "help": "Edge length of each occupancy cube. Smaller = more "
                    "detail but more memory and a larger viewer payload.",
                },
                "pixel_stride": {
                    "type": "number",
                    "label": "Pixel Sampling Stride",
                    "default": 8,
                    "step": 1,
                    "help": "Sample every Nth pixel of the depth map. Raising "
                    "this is the cheapest way to cut voxel cost.",
                },
                "max_voxels": {
                    "type": "number",
                    "label": "Max Voxels",
                    "default": 20000,
                    "step": 1000,
                    "help": "Hard cap on retained voxels; the least-recently "
                    "seen ones are evicted first.",
                },
                "export_max_voxels": {
                    "type": "number",
                    "label": "Viewer Voxel Limit",
                    "default": 3000,
                    "step": 500,
                    "help": "How many voxels are sent to the 3D viewer per "
                    "tick, strongest hits first.",
                },
                "min_voxel_count": {
                    "type": "number",
                    "label": "Min Hits",
                    "default": 1,
                    "step": 1,
                    "help": "Voxels observed fewer times than this are not "
                    "exported - raise it to suppress depth noise.",
                },
                "decay_seconds": {
                    "type": "number",
                    "label": "Forget After (s)",
                    "default": 8.0,
                    "step": 1.0,
                    "help": "Voxels not re-observed for this long are removed, "
                    "so the map follows the scene instead of growing forever.",
                },
                "depth_scale": {
                    "type": "number",
                    "label": "Depth Scale",
                    "default": 1.0,
                    "step": 0.1,
                    "help": "Trim on the relative-depth-to-distance mapping. "
                    "Tune against a known distance in the scene.",
                },
                "min_height": {
                    "type": "number",
                    "label": "Min Height (m)",
                    "default": -0.25,
                    "step": 0.1,
                    "help": "World points below this height are discarded to "
                    "reject ground-plane noise. A small negative default keeps "
                    "the floor so low-mounted cameras still build a map; set "
                    "0 or higher to clip ground points.",
                },
                "voxel_min_depth": {
                    "type": "number",
                    "label": "Min Depth (m)",
                    "default": 0.05,
                    "step": 0.05,
                    "help": "Points closer than this are ignored (they are "
                    "usually on the robot itself).",
                },
                "near_depth": {
                    "type": "number",
                    "label": "Near Plane (m)",
                    "default": 0.3,
                    "step": 0.1,
                    "help": "Forward distance assigned to the nearest pixel. "
                    "Depth Anything is a relative (inverse-depth) model, so this "
                    "pins the near end of the scale against Max Depth. Lower it "
                    "if the near parts of the scene look too far away.",
                },
            }
        )
        return schema

    def __init__(
        self,
        camera_config: iSpyCameraConfig,
        config: iSpyConfig,
        core_mask=None,
    ):
        super().__init__(camera_config, config, core_mask)
        self.logger = logging.getLogger(__name__)

        self.conversions = _INCHES_TO_OUTPUT_UNIT

        def setting(key, default):
            value = camera_config.get_pipeline_setting(key)
            return default if value is None else value

        def as_float(key, default):
            try:
                return float(setting(key, default))
            except (TypeError, ValueError):
                return float(default)

        def as_int(key, default):
            try:
                return int(setting(key, default))
            except (TypeError, ValueError):
                return int(default)

        self.voxel_size_m = max(as_float("voxel_size", 0.1), 1e-3)
        self.voxel_size = self.voxel_size_m * self._z_scale
        self.pixel_stride = max(1, as_int("pixel_stride", 8))
        self.max_voxels = max(1, as_int("max_voxels", 20000))
        self.export_max_voxels = max(
            0, as_int("export_max_voxels", 3000)
        )
        self.min_voxel_count = max(1, as_int("min_voxel_count", 1))
        self.decay_seconds = max(0.0, as_float("decay_seconds", 8.0))
        self.depth_scale = max(as_float("depth_scale", 1.0), 1e-6)
        # near_depth is consumed in meters by relative_depth_to_distance (the
        # returned distance is scaled into output units afterwards), so it must
        # NOT be pre-scaled by _z_scale.
        self.near_depth = max(as_float("near_depth", 0.3), 1e-3)
        self.min_height = as_float("min_height", -0.25) * self._z_scale
        self.voxel_min_depth = (
            max(as_float("voxel_min_depth", 0.05), 0.0) * self._z_scale
        )

        self.voxel_map = SparseVoxelMap(
            voxel_size=self.voxel_size,
            max_voxels=self.max_voxels,
            decay_seconds=self.decay_seconds,
        )
        self._resolve_geometry(camera_config)
        # last tick's diagnostics, surfaced to the 3D viewer so an empty map
        # can explain itself instead of silently rendering nothing
        self._debug: dict = {}

    def _resolve_geometry(self, camera_config: iSpyCameraConfig) -> None:
        unit = self.unit
        scale = self.conversions.get(unit, self.conversions["frc"])

        def inches(value):
            return unit_to_inches(value or 0, unit) * scale

        self.camera_x = inches(camera_config.get("x", 0))
        self.camera_y = inches(camera_config.get("y", 0))
        # 'height' is the canonical mount height (same as every other pipeline)
        self.camera_height = inches(camera_config.get("height", 0))
        self.camera_pitch = float(camera_config.get("pitch", 0.0) or 0.0)
        self.camera_yaw = float(camera_config.get("yaw", 0.0) or 0.0)
        self.calibration = camera_config.get("calibration", {}) or {}

    # ------------------------------------------------------------------

    def _integrate_depth(self, depth: np.ndarray, frame: np.ndarray) -> None:
        self._debug = getattr(self, "_debug", None) or {}
        debug = self._debug
        debug["reason"] = ""
        if depth is None or getattr(depth, "size", 0) == 0:
            debug["reason"] = "no depth map produced"
            return

        finite = depth[np.isfinite(depth)]
        if finite.size == 0:
            debug["reason"] = "depth map has no finite values"
            return
        self._dmin = float(finite.min())
        self._dmax = float(finite.max())

        distance_m = relative_depth_to_distance(
            depth,
            self._dmin,
            self._dmax,
            self.max_depth,
            self.depth_scale,
            self.near_depth,
        )
        distance = distance_m * self._z_scale
        dist_finite = distance[np.isfinite(distance)]

        # focal is resolved in frame-pixel space (the calibration matrix /
        # FOV belong to the live camera resolution). The depth map can come
        # back at a different resolution (e.g. the HF pipeline sends ~518px
        # tensors), so rescale the focal to the grid we actually back-project
        # from - otherwise every point lands on the wrong ray.
        focal = focal_length_pixels(
            frame.shape[1], self.calibration, default_fov=60.0
        )
        if depth.shape[1] != frame.shape[1]:
            focal = focal * depth.shape[1] / float(max(frame.shape[1], 1))
        cam_points = depth_to_camera_points(
            distance, focal, stride=self.pixel_stride
        )
        far_plane_m = float(self.max_depth) * float(self.depth_scale)
        debug.update(
            dmin=round(self._dmin, 4),
            dmax=round(self._dmax, 4),
            dist_min=round(float(dist_finite.min()), 4) if dist_finite.size else 0.0,
            dist_max=round(float(dist_finite.max()), 4) if dist_finite.size else 0.0,
            max_depth=self.max_depth,
            depth_scale=self.depth_scale,
            near_depth=round(float(self.near_depth) * self._z_scale, 3),
            far_plane=round(far_plane_m * self._z_scale, 3),
            focal=round(float(focal), 1),
            depth_w=int(depth.shape[1]),
            frame_w=int(frame.shape[1]),
            back_projected=int(cam_points.shape[0]),
        )
        if far_plane_m < 1.0:
            # A far plane under a meter makes every point land almost on the
            # camera, so the whole map collapses into a tiny blob. Almost
            # always a mis-set Max Depth / Depth Scale.
            debug["warning"] = (
                f"far plane is only {far_plane_m * self._z_scale:.2f} "
                f"{self._unit_label} - raise Max Depth / Depth Scale or the "
                f"world collapses"
            )
            self.logger.warning(
                "Camera '%s': voxel far plane is %.2f %s (max_depth=%s x "
                "depth_scale=%s) - raise it or the world collapses.",
                self.config.get("name", "?"),
                far_plane_m * self._z_scale,
                self._unit_label,
                self.max_depth,
                self.depth_scale,
            )
        if cam_points.size == 0:
            debug["reason"] = "depth -> camera back-projection produced no points"
            return

        cam_points = cam_points[cam_points[:, 2] >= self.voxel_min_depth]
        debug["past_min_depth"] = int(cam_points.shape[0])
        if cam_points.size == 0:
            debug["reason"] = (
                f"every point is closer than Min Depth "
                f"({self.voxel_min_depth:.3f} {self._unit_label})"
            )
            return

        world = camera_points_to_robot(
            cam_points,
            self.camera_x,
            self.camera_y,
            self.camera_height,
            self.camera_yaw,
            self.camera_pitch,
        )
        world = world[world[:, 2] >= self.min_height]
        debug["past_min_height"] = int(world.shape[0])
        if world.size == 0:
            debug["reason"] = (
                f"every point is below Min Height "
                f"({self.min_height:.3f} {self._unit_label})"
            )
            return

        debug["integrated"] = int(self.voxel_map.integrate(world))

    def _voxel_object(self) -> Object:
        voxels = self.voxel_map.export(
            max_points=self.export_max_voxels,
            min_count=self.min_voxel_count,
        )
        total = self.voxel_map.count()

        if voxels:
            points = np.asarray([v[:3] for v in voxels], dtype=np.float64)
            centroid = points.mean(axis=0)
            extent = (points.max(axis=0) - points.min(axis=0)).tolist()
        else:
            centroid = np.zeros(3, dtype=np.float64)
            extent = [0.0, 0.0, 0.0]
        # A map that only spans a couple of voxel cells in every axis means the
        # depth->distance mapping collapsed (nothing survives much farther than
        # the camera). Surface that so the 3D viewer does not just silently
        # fail to show a world.
        degenerate = bool(
            voxels and max(float(e) for e in extent) < self.voxel_size * 4
        )

        meta = {
            "kind": "voxel_map",
            "voxels": voxels,
            "voxel_size": self.voxel_size,
            "count": int(total),
            "exported": len(voxels),
            "unit": self.unit,
            "extent": extent,
            "degenerate": degenerate,
            "geometry": {
                "camera_height": round(float(getattr(self, "camera_height", 0.0)), 4),
                "camera_pitch": getattr(self, "camera_pitch", 0.0),
                "camera_yaw": getattr(self, "camera_yaw", 0.0),
                "min_height": getattr(self, "min_height", 0.0),
                "min_depth": getattr(self, "voxel_min_depth", 0.0),
                "pixel_stride": getattr(self, "pixel_stride", 0),
                "process_every": getattr(self, "_every", 0),
            },
            "debug": dict(getattr(self, "_debug", {})),
            "model": {
                "estimate_depth": getattr(self, "estimate_depth", True),
                "backend": "onnx" if getattr(self, "_session", None) is not None else "torch",
                "load_error": getattr(self, "_load_error", None),
            },
        }

        # Reuse one Object so its identity stays stable across ticks (the map
        # contents change, the detection does not). Always return it - even an
        # empty map is reported so the 3D viewer can show that the world is
        # building rather than silently rendering nothing.
        obj = getattr(self, "_voxel_object_ref", None)
        if obj is None:
            obj = Object(
                x=float(centroid[0]),
                y=float(centroid[1]),
                z=float(centroid[2]),
                name="voxel_world",
                confidence=float(min(1.0, len(voxels) / 200.0)),
                depth_source="depth_model",
                vis_type="voxels",
                vis_meta=meta,
            )
            self._voxel_object_ref = obj
        else:
            obj.x = float(centroid[0])
            obj.y = float(centroid[1])
            obj.z = float(centroid[2])
            obj.confidence = float(min(1.0, len(voxels) / 200.0))
            obj.vis_meta = meta
        return obj

    # ------------------------------------------------------------------

    def run(self):
        frame = self.get_frame()
        if frame is None:
            return [], None
        if not self._is_processable():
            return [], frame

        self._debug = getattr(self, "_debug", None) or {}
        self._frame_count = getattr(self, "_frame_count", 0) + 1
        every = max(1, self._every)
        last_depth = self._last_depth

        refresh = (
            last_depth is None
            or every <= 1
            or self._frame_count % every == 0
        )
        if refresh:
            try:
                depth = self._infer_depth(frame)
                self._debug["infer"] = "ok"
            except Exception as exc:
                self.logger.exception("Voxel world depth inference failed.")
                self._debug["infer"] = f"error: {exc}"
                depth = last_depth
            if depth is not None:
                self._last_depth = depth
                last_depth = depth
                try:
                    self._integrate_depth(depth, frame)
                except Exception as exc:
                    # never let a single bad frame blank the whole pipeline -
                    # a geometry edge case must not wipe out the voxel world
                    self.logger.exception("Voxel world map integration failed.")
                    self._debug["reason"] = f"integration error: {exc}"

        self.voxel_map.decay()

        obj = self._voxel_object()
        self._last_objects = [obj]

        if last_depth is None:
            return self._last_objects, frame
        return self._last_objects, self._annotate(frame, last_depth)

    def destroy(self):
        if getattr(self, "voxel_map", None) is not None:
            self.voxel_map.clear()
        self._voxel_object_ref = None
        super().destroy()
