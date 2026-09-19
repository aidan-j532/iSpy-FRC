"""Monocular voxel world pipeline.

Reuses the existing Depth Anything V2 model/inference (and its optimization /
hardware backends) from :class:`DepthAnythingPipeline`, then lifts the relative
depth map into world-frame points and accumulates a sparse voxel occupancy map.

This is *approximate* geometry: Depth Anything outputs relative depth, so
distances are only as good as the far-plane scaling and camera calibration.

Coordinate frames
-----------------
``vis_meta["voxels"]`` (and the Object centroid) are **robot-relative**: +X
right, +Y forward, +Z up measured from the robot origin, i.e. exactly the
convention ``triangulation.camera_point_to_robot`` (and
``voxel_map.camera_points_to_robot``) back-projects into - NOT field-absolute.

Trackers convert only the Object's ``x/y/z`` (and rotation) into field
coordinates via :meth:`Object.relative_to`; they never touch ``vis_meta``. The
voxel points therefore intentionally stay robot-relative, and the 3D viewer
(parents ``viewer3d.html``'s voxel group to the live NetworkHandler robot pose
overlay) applies the robot->field transform at render time so the map rides
along with the robot marker. Do not switch the points to field coordinates in
this pipeline without changing the viewer to match, and vice versa.
"""

import logging

import cv2
import numpy as np

from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig, unit_to_inches
from iSpy.vision.Object import Object
from iSpy.vision.pipelines.depth_anything import DepthAnythingPipeline
from iSpy.vision.voxel_map import (
    SparseVoxelMap,
    camera_points_to_robot,
    depth_plane_range,
    depth_to_camera_points,
    focal_length_pixels,
    inverse_depth_to_distance,
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
                "depth_method": {
                    "type": "select",
                    "label": "Depth Algorithm",
                    "default": "depth_anything",
                    "options": ["depth_anything", "shadows"],
                    "help": "How per-pixel depth is estimated from the single "
                    "camera.\n"
                    "depth_anything: Depth Anything V2 neural model - relative "
                    "depth map, most accurate, weights are downloaded "
                    "automatically.\n"
                    "shadows: model-free 'shadows from HSV' heuristic - the "
                    "HSV value channel is read as shading, so shadowed "
                    "(darker) pixels map far and lit surfaces map near. The "
                    "reconstruction follows the scene's own lighting, which "
                    "lets it read the room's shape without any model, "
                    "downloads, or calibration.",
                },
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
                    "help": "Multiplies the Max Depth cap. The far plane is the "
                    "smaller of the scene's own depth ratio and Max Depth x "
                    "Depth Scale, so it only bites on very deep scenes.",
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
                    "Depth Anything is a relative (inverse-depth) model, so the "
                    "scene's true size is unknown - this anchors the whole "
                    "reconstruction: raise it to enlarge the world, lower it to "
                    "shrink it. Max Depth caps how far it can grow.",
                },
                "auto_ground": {
                    "type": "toggle",
                    "label": "Auto Ground Level",
                    "default": True,
                    "help": "When the camera mount height is 0/unknown, drop the "
                    "scene so its densest plane (the floor) sits at z=0 instead "
                    "of floating below the grid. Turn off once you set the real "
                    "camera height.",
                },
                "auto_world_scale": {
                    "type": "toggle",
                    "label": "Auto Scale World",
                    "default": True,
                    "help": "Depth Anything (and shadows) only output relative "
                    "depth, so the world's absolute size is unknown. When on, "
                    "the scale is stretched so the scene fills the Max Depth "
                    "range - a room a few metres deep renders a few metres "
                    "tall instead of collapsing into a small blob in front of "
                    "the camera. Turn off to control scale manually with Near "
                    "Plane / Max Depth.",
                },
                "debug_viz": {
                    "type": "toggle",
                    "label": "Debug Visualization",
                    "default": False,
                    "help": "Diagnostic/test mode. Emits overlay geometry into "
                    "the 3D viewer - the raw camera-frame back-projection, the "
                    "robot-frame cloud that became voxels, the camera frustum, "
                    "the robot axes and the settled floor - plus a one-time "
                    "geometry log on the first integration. Geometry mistakes "
                    "(wrong mount pitch/yaw/height, collapsed world) become "
                    "visually obvious instead of silently mangling the map. "
                    "Leave off for normal use.",
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
        # Read before super().__init__() - the parent's model load /
        # optimization decisions (which run in its own __init__) must know
        # whether a model-backed algorithm is even selected.
        raw_method = camera_config.get_pipeline_setting("depth_method")
        self.depth_method = str(raw_method or "depth_anything").strip().lower()
        if self.depth_method not in ("depth_anything", "shadows"):
            self.depth_method = "depth_anything"

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
        self.auto_ground = bool(setting("auto_ground", True))
        self.auto_world_scale = bool(setting("auto_world_scale", True))
        self.debug_viz = bool(setting("debug_viz", False))
        self._logged_geometry = False
        # Resolved on the first integrated frame (see _integrate_depth) and then
        # held constant so the accumulated map does not smear as the estimate
        # jitters frame to frame.
        self._ground_offset: float | None = None

        self.voxel_map = SparseVoxelMap(
            voxel_size=self.voxel_size,
            max_voxels=self.max_voxels,
            decay_seconds=self.decay_seconds,
        )
        self._resolve_geometry(camera_config)
        # last tick's diagnostics, surfaced to the 3D viewer so an empty map
        # can explain itself instead of silently rendering nothing
        self._debug: dict = {}

    def _method(self) -> str:
        # __new__-built instances (tests) never run __init__ - default to the
        # original Depth Anything behaviour.
        return getattr(self, "depth_method", "depth_anything")

    # ------------------------------------------------------------------
    # algorithm selection - the inherited Depth Anything machinery (model
    # download, optimization, ready/prepare gating) only applies to the
    # 'depth_anything' method; 'shadows' is a pure heuristic.
    # ------------------------------------------------------------------

    def _optimization_requested(self) -> bool:
        if self._method() != "depth_anything":
            return False
        return super()._optimization_requested()

    def _load_model(self):
        if self._method() != "depth_anything":
            self.logger.info(
                "Depth method '%s' needs no model - not loading weights.",
                self._method(),
            )
            return
        super()._load_model()

    def _is_processable(self) -> bool:
        if getattr(self, "_optimizing", False):
            return False
        if self._method() != "depth_anything":
            return True
        return super()._is_processable()

    def is_ready(self) -> tuple[bool, str]:
        if self._method() != "depth_anything":
            self._set_status("ready")
            return True, "ready"
        return super().is_ready()

    def _infer_depth(self, frame: np.ndarray):
        # the shadows heuristic is computed geometry, not a model - the dispatch
        # stays here so run() (and its last-frame fallbacks) are method-blind.
        if self._method() == "shadows":
            return self._estimate_shadows(frame)
        return super()._infer_depth(frame)

    def _distance_from_depth(self, raw: float) -> float:
        # Keep the label in lock-step with the voxel world: both use the
        # (possibly auto-scaled) plane resolved at the last integration.
        plane = getattr(self, "_eff_depth_plane", None)
        if plane is None:
            d_lo, d_hi, z_near, z_far = depth_plane_range(
                np.asarray([raw], dtype=np.float64),
                getattr(self, "_dmin", 0.0),
                getattr(self, "_dmax", 1.0),
                self.max_depth,
                1.0,
                getattr(self, "near_depth", 0.3),
            )
        else:
            d_lo, d_hi, z_near, z_far = plane
        return inverse_depth_to_distance(
            float(raw), d_lo, d_hi, z_near, z_far
        ) * self._z_scale

    def _annotate(self, frame, depth) -> np.ndarray:
        return super()._annotate(frame, depth)

    def plot(self, frame):
        return super().plot(frame)

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
        # diagnostics describe this frame only - without this the warning list
        # grows once per frame and the viewer shows the same text forever
        debug.pop("warning", None)

        def warn(message: str) -> None:
            existing = debug.get("warning")
            debug["warning"] = f"{existing}; {message}" if existing else message
        if depth is None or getattr(depth, "size", 0) == 0:
            debug["reason"] = "no depth map produced"
            return

        finite = depth[np.isfinite(depth)]
        if finite.size == 0:
            debug["reason"] = "depth map has no finite values"
            return
        self._dmin = float(finite.min())
        self._dmax = float(finite.max())

        # Every method emits relative inverse depth (larger = nearer); the
        # plane resolver anchors the nearest robust pixel at Near Plane and the
        # far plane follows the scene's own ratio, capped by Max Depth. With
        # Auto Scale World (default) the near anchor is raised so the far plane
        # reaches that cap - otherwise a room a few metres deep collapses into
        # a sub-metre blob right in front of the camera.
        d_lo, d_hi, z_near, z_far = depth_plane_range(
            depth,
            self._dmin,
            self._dmax,
            self.max_depth,
            self.depth_scale,
            self.near_depth,
        )
        far_cap = max(float(self.max_depth) * float(self.depth_scale), z_near)
        eff_near = z_near
        if bool(getattr(self, "auto_world_scale", True)) and z_far < far_cap - 1e-9:
            ratio = d_hi / max(d_lo, 1e-6)
            if ratio > 1.0:
                eff_near = max(min(far_cap / ratio, far_cap), z_near)
                z_far = min(eff_near * ratio, far_cap)
                z_far = max(z_far, eff_near * 1.0001)
        # The same plane feeds the on-screen label so the depth text shows the
        # same distances the voxel world is built from.
        self._eff_depth_plane = (d_lo, d_hi, eff_near, z_far)
        self._eff_near_depth = eff_near

        distance_m = relative_depth_to_distance(
            depth,
            self._dmin,
            self._dmax,
            self.max_depth,
            self.depth_scale,
            eff_near,
        )
        self._integrate_distance_m(
            distance_m * self._z_scale, frame, method=self._method()
        )

    def _estimate_shadows(self, frame: np.ndarray) -> np.ndarray:
        """Model-free 'shadows from HSV' relative depth.

        Shading is the strongest monocular depth cue a colour camera gives us
        for free: shadowed regions are simply darker than the lit surfaces
        around them. So the HSV value channel (V) is read as an inverse-depth
        map directly - bright, lit pixels land near and dark, shadowed pixels
        land far. Saturation is folded in lightly so saturated colour keeps
        reading as a lit surface even when its raw brightness is mid-range.

        The output is a relative inverse-depth map in the same convention as
        Depth Anything (larger = nearer), so the rest of the pipeline (plane
        resolver, back-projection, colour sampling) is method-blind. A uniform
        frame (no shading at all) degenerates to the plain top-far/bottom-near
        row ramp so the FOV still builds a wedge instead of an empty map.
        """
        frame = np.asarray(frame)
        h, w = frame.shape[0], frame.shape[1]
        if frame.ndim == 2 or frame.shape[2] < 3:
            hsv_input = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            hsv_input = frame[:, :, :3]
        hsv = cv2.cvtColor(hsv_input, cv2.COLOR_BGR2HSV)
        value = hsv[..., 2].astype(np.float64)
        saturation = hsv[..., 1].astype(np.float64)

        # lit = bright and/or saturated, shadow = dim; combine into a single
        # shading score and stretch it across the [0.05, 1] inverse-depth band
        # (0.05 keeps the percentile ratio finite, matching the old flat ramp).
        brightness = value * (0.55 + 0.45 * (saturation / 255.0))
        bmin = float(brightness.min())
        bmax = float(brightness.max())
        if bmax - bmin <= 1e-9:
            # no shading signal - fall back to the row ramp
            return np.tile(
                np.linspace(1.0, 0.05, h, dtype=np.float64)[:, None], (1, w)
            )
        relative = 0.05 + 0.95 * (brightness - bmin) / (bmax - bmin)
        return np.where(np.isfinite(relative), relative, 0.5).astype(np.float64)

    def _integrate_distance_m(
        self,
        distance: np.ndarray,
        frame: np.ndarray,
        method: str = "depth_anything",
    ) -> None:
        """Back-project a metric forward-distance map (output units) into the world."""
        self._debug = getattr(self, "_debug", None) or {}
        debug = self._debug
        distance = np.asarray(distance, dtype=np.float64)
        dist_finite = distance[np.isfinite(distance)]

        def warn(message: str) -> None:
            existing = debug.get("warning")
            debug["warning"] = f"{existing}; {message}" if existing else message

        # focal is resolved in frame-pixel space (the calibration matrix /
        # FOV belong to the live camera resolution). The depth map can come
        # back at a different resolution (e.g. the HF pipeline sends ~518px
        # tensors), so rescale the focal to the grid we actually back-project
        # from - otherwise every point lands on the wrong ray.
        focal = focal_length_pixels(
            frame.shape[1], self.calibration, default_fov=60.0
        )
        if distance.shape[1] != frame.shape[1]:
            focal = focal * distance.shape[1] / float(max(frame.shape[1], 1))
        cam_points, cam_pixels = depth_to_camera_points(
            distance, focal, stride=self.pixel_stride, return_pixels=True
        )
        far_plane_m = float(self.max_depth) * float(self.depth_scale)
        debug.update(
            dmin=round(self._dmin, 4),
            dmax=round(self._dmax, 4),
            method=method,
            dist_min=round(float(dist_finite.min()), 4) if dist_finite.size else 0.0,
            dist_max=round(float(dist_finite.max()), 4) if dist_finite.size else 0.0,
            max_depth=self.max_depth,
            depth_scale=self.depth_scale,
            near_depth=round(
                float(getattr(self, "_eff_near_depth", self.near_depth))
                * self._z_scale,
                3,
            ),
            auto_scale=bool(getattr(self, "auto_world_scale", True)),
            far_plane=round(far_plane_m * self._z_scale, 3),
            focal=round(float(focal), 1),
            depth_w=int(distance.shape[1]),
            frame_w=int(frame.shape[1]),
            back_projected=int(cam_points.shape[0]),
        )
        if far_plane_m < 1.0:
            # A far plane under a meter makes every point land almost on the
            # camera, so the whole map collapses into a tiny blob. Almost
            # always a mis-set Max Depth / Depth Scale.
            warn(
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

        keep = cam_points[:, 2] >= self.voxel_min_depth
        cam_points = cam_points[keep]
        cam_pixels = cam_pixels[keep]
        debug["past_min_depth"] = int(cam_points.shape[0])
        if cam_points.size == 0:
            debug["reason"] = (
                f"every point is closer than Min Depth "
                f"({self.voxel_min_depth:.3f} {self._unit_label})"
            )
            return

        # Sample the live frame so every voxel carries the color of the pixels
        # that produced it - this is what turns the map from a height ramp into
        # a recognizable, real-color reconstruction. Frames are BGR (OpenCV),
        # the viewer wants RGB. Depth-map pixels are scaled to frame pixels
        # because the model may run at a different resolution than the camera.
        colors = self._sample_colors(frame, cam_pixels, distance.shape)

        world_all = camera_points_to_robot(
            cam_points,
            self.camera_x,
            self.camera_y,
            self.camera_height,
            self.camera_yaw,
            self.camera_pitch,
        )

        ground_offset = getattr(self, "_ground_offset", None)
        # Auto-ground only matters when the mount height is unknown (0); once a
        # real height is set the world is already anchored to the floor.
        auto_level = bool(self.auto_ground) and abs(float(self.camera_height)) < 1e-9
        if ground_offset is None and auto_level:
            # The world origin is the floor but the camera height is not known
            # (unset -> 0), so the whole scene floats. Use the densest z plane -
            # which for the downward/robot view is the floor - as z=0. Held
            # constant afterwards so the map does not smear.
            ground_offset = float(np.median(world_all[:, 2]))
            self._ground_offset = ground_offset
        ground_offset = float(ground_offset or 0.0)
        if ground_offset:
            world_all = world_all.copy()
            world_all[:, 2] -= ground_offset

        # While auto-levelling the origin is arbitrary, so an absolute Min
        # Height would slice the scene in half - skip the clip and let the user
        # set a real camera height to get a true floor filter back.
        effective_min_height = float("-inf") if auto_level else float(self.min_height)
        world_z_min = float(world_all[:, 2].min())
        world_z_max = float(world_all[:, 2].max())
        debug.update(
            camera_height=round(float(self.camera_height), 3),
            camera_pitch=round(float(self.camera_pitch), 1),
            camera_yaw=round(float(self.camera_yaw), 1),
            min_height=(None if auto_level else round(float(self.min_height), 3)),
            ground_offset=round(ground_offset, 3),
            auto_level=auto_level,
            world_z_min=round(world_z_min, 3),
            world_z_max=round(world_z_max, 3),
        )
        keep = world_all[:, 2] >= effective_min_height
        world = world_all[keep]
        world_colors = colors[keep] if colors is not None else None
        debug["past_min_height"] = int(world.shape[0])
        if getattr(self, "debug_viz", False):
            self._debug["viz"] = self._viz_payload(
                cam_points,
                world_all,
                colors,
                focal,
                distance.shape,
                ground_offset,
            )

        if not getattr(self, "_logged_geometry", False):
            self._logged_geometry = True
            self._log_geometry_once(debug, ground_offset)
        if abs(float(self.camera_height)) < 1e-9 and not auto_level:
            # The world origin is the floor, so a mount height of 0 puts the
            # whole scene below z=0 and the Min Height floor then eats it.
            warn(
                "camera Height is 0 - set the real mount height (or enable "
                "Auto Ground) so the floor sits at z=0"
            )
        if world.size == 0:
            # The configured floor is below the entire scene (almost always a
            # camera-height/pitch that does not match the world origin). Rather
            # than silently blanking the whole world, keep the geometry and say
            # so - the user can then fix Min Height / the camera mount.
            world = world_all
            world_colors = colors
            warn(
                f"Min Height ({self.min_height:.2f} {self._unit_label}) "
                f"removed every point (world z {world_z_min:.2f}..{world_z_max:.2f})"
                f" - floor clip ignored, lower Min Height"
            )

        if world.size == 0:
            debug["reason"] = "no world points survived filtering"
            return

        debug["integrated"] = int(
            self.voxel_map.integrate(world, colors=world_colors)
        )

    # ------------------------------------------------------------------
    # debug visualization / test mode
    # ------------------------------------------------------------------

    _DEBUG_MAX_POINTS = 2000

    def _viz_payload(
        self,
        cam_points: np.ndarray,
        world_all: np.ndarray,
        colors: np.ndarray | None,
        focal: float,
        depth_shape: tuple,
        ground_offset: float,
    ) -> dict:
        """Intermediate geometry for the 3D viewer's debug/test mode.

        Everything is emitted in the SAME frame the voxels use (robot-relative,
        post ground-offset), so the viewer can literally overlay it on the map:

        ``raw``      - the depth grid back-projected into the camera frame
                       (right, down, forward) BEFORE the mount transform. The
                       pyramid-along-image-flat shape of this cloud is the
                       first thing that must look like the real scene.
        ``world``    - the same points AFTER camera-to-robot + ground offset,
                       i.e. exactly what got voxelized. If this does not line
                       up with the ``raw`` pyramid once the mount is applied,
                       the pitch/yaw/height config is wrong.
        ``frustum``  - the camera view volume (near + far rectangle corners) in
                       the same world frame, so the user sees where the config
                       says the camera actually points.
        ``cam_origin`` / ``axes_origin`` - mount position markers.

        Bound to ``_DEBUG_MAX_POINTS`` per cloud so the websocket payload and
        the browser stay cheap even at stride 1.
        """
        step = max(1, int(cam_points.shape[0] // self._DEBUG_MAX_POINTS)) \
            if cam_points.shape[0] else 1
        idx = np.arange(0, cam_points.shape[0], step)[: self._DEBUG_MAX_POINTS]

        raw = cam_points[idx]
        settled = world_all[idx]
        raw_colors = None
        if colors is not None and colors.shape[0] == world_all.shape[0]:
            raw_colors = colors[idx]

        plane = getattr(self, "_eff_depth_plane", None)
        near = float(self._eff_near_depth) * self._z_scale
        far = float(plane[3]) * self._z_scale if plane else near

        frustum = self._frustum_corners_world(depth_shape, focal, near, far, ground_offset)
        cam_origin = self._robot_frame([[0.0, 0.0, 0.0]], ground_offset)
        return {
            "raw": raw.tolist(),
            "world": settled.tolist(),
            "colors": (
                raw_colors.astype(np.float64).tolist()
                if raw_colors is not None
                else None
            ),
            "frustum": frustum.tolist(),
            "near": round(float(near), 4),
            "far": round(float(far), 4),
            "cam_origin": [round(float(v), 4) for v in cam_origin[0]],
            "axes_origin": [
                round(float(self.camera_x), 4),
                round(float(self.camera_y), 4),
                round(-ground_offset, 4),
            ],
            "unit": self.unit,
        }

    def _robot_frame(self, pts, ground_offset: float) -> np.ndarray:
        out = camera_points_to_robot(
            pts,
            self.camera_x,
            self.camera_y,
            self.camera_height,
            self.camera_yaw,
            self.camera_pitch,
        )
        if ground_offset:
            out = out.copy()
            out[:, 2] -= ground_offset
        return out

    def _frustum_corners_world(
        self,
        depth_shape: tuple,
        focal: float,
        near: float,
        far: float,
        ground_offset: float,
    ) -> np.ndarray:
        """Near + far image-rectangle corners, same transform as cam points."""
        h, w = depth_shape[0], depth_shape[1]
        cx, cy = w / 2.0, h / 2.0
        f = max(float(focal), 1e-6)
        corners = []
        for (u, v), z in (
            ((0.0, 0.0), near), ((w, 0.0), near), ((w, h), near), ((0.0, h), near),
            ((0.0, 0.0), far), ((w, 0.0), far), ((w, h), far), ((0.0, h), far),
        ):
            corners.append(((u - cx) / f * z, (v - cy) / f * z, z))
        return self._robot_frame(np.asarray(corners, dtype=np.float64), ground_offset)

    def _log_geometry_once(self, debug: dict, ground_offset: float) -> None:
        """One structured line per instance so boot-time geometry is checkable
        instead of guessed at across hundreds of frames."""
        pitch = float(debug.get("camera_pitch", 0.0))
        cam_h = float(debug.get("camera_height", 0.0))
        hints = []
        if abs(abs(pitch) - 90.0) < 1.0:
            hints.append(
                "pitch ~90 deg = camera points straight down; the scene maps to "
                "a vertical smear under the robot - check the mount is really "
                "straight down"
            )
        if cam_h == 0.0 and not debug.get("auto_level", False):
            hints.append(
                "height 0 = camera at floor level; set the real mount height or "
                "enable Auto Ground"
            )
        ext = float(debug.get("world_z_max", 0.0)) - float(debug.get("world_z_min", 0.0))
        if ext > 2.0 and abs(abs(pitch) - 90.0) < 1.0:
            hints.append(
                "world z-extent %.1f m with a sideways/near-down pitch - the "
                "depth range collapses into height instead of distance" % ext
            )
        if plane := getattr(self, "_eff_depth_plane", None):
            plane_txt = "d=(%.3f..%.3f) z=(%.2f..%.2f)" % (plane[0], plane[1], plane[2], plane[3])
        else:
            plane_txt = "n/a"
        logger = getattr(self, "logger", None)
        if logger is None or getattr(logger, "disabled", False):
            return
        cam_name = "?"
        cfg = getattr(self, "config", None)
        if cfg is not None:
            cam_name = cfg.get("name", "?") if isinstance(cfg, dict) else "?"
        logger.info(
            "VoxelWorld '%s' geometry: method=%s unit=%s depth=%sx%s focal=%.1fpx "
            "plane=%s far=%.2f%s mount=(x=%.3f y=%.3f h=%.3f pitch=%.1f yaw=%.1f) "
            "ground_offset=%.3f world_z=%.3f..%.3f%s",
            cam_name,
            self._method(),
            self.unit,
            debug.get("depth_w", "?"),
            debug.get("frame_w", "?"),
            float(debug.get("focal", 0.0)),
            plane_txt,
            float(debug.get("far_plane", 0.0)),
            self._unit_label,
            self.camera_x,
            self.camera_y,
            cam_h,
            pitch,
            float(debug.get("camera_yaw", 0.0)),
            ground_offset,
            float(debug.get("world_z_min", 0.0)),
            float(debug.get("world_z_max", 0.0)),
            f"; {'; '.join(hints)}" if hints else "",
        )

    def _sample_colors(
        self, frame: np.ndarray, pixels: np.ndarray, depth_shape: tuple
    ) -> np.ndarray | None:
        """Look up the RGB (0-255) of each back-projected pixel in the frame."""
        if pixels.size == 0 or frame is None or frame.ndim < 2:
            return None
        fh, fw = frame.shape[0], frame.shape[1]
        dh, dw = depth_shape[0], depth_shape[1]
        u = np.clip(
            (pixels[:, 0] * fw / max(dw, 1)).astype(np.int64), 0, max(fw - 1, 0)
        )
        v = np.clip(
            (pixels[:, 1] * fh / max(dh, 1)).astype(np.int64), 0, max(fh - 1, 0)
        )
        sampled = frame[v, u]
        if sampled.ndim == 1:
            return None
        # gray -> broadcast, BGR -> RGB, BGRA -> RGB
        if sampled.shape[1] >= 3:
            return sampled[:, 2::-1][:, :3].astype(np.float64)
        return np.repeat(sampled[:, :1], 3, axis=1).astype(np.float64)

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
                "ground_offset": round(float(getattr(self, "_ground_offset", 0.0) or 0.0), 4),
                "auto_ground": bool(getattr(self, "auto_ground", False)),
                "pixel_stride": getattr(self, "pixel_stride", 0),
                "process_every": getattr(self, "_every", 0),
            },
            "debug": dict(getattr(self, "_debug", {})),
            "model": {
                "method": self._method(),
                "estimate_depth": getattr(self, "estimate_depth", True),
                "backend": (
                    "synthetic"
                    if self._method() == "shadows"
                    else (
                        "onnx"
                        if getattr(self, "_session", None) is not None
                        else "torch"
                    )
                ),
                "load_error": getattr(self, "_load_error", None),
            },
        }

        # Test mode: ship the intermediate geometry under its own key so the
        # scalar diagnostics row stays numbers-only. (Drop the arrays out of
        # the scalar dict too - they have no right to bloat the status text.)
        if getattr(self, "debug_viz", False):
            viz = dict(getattr(self, "_debug", {}).get("viz") or {})
            if viz:
                meta["debug_viz"] = viz
            meta["debug"] = {
                k: v for k, v in meta["debug"].items() if k != "viz"
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
        # This single Object is field-converted in place by trackers
        # (relative_to). The same field pose gets re-added every tick, so re-zero
        # the rotation the tracker folded in last frame - otherwise yaw/roll/pitch
        # accumulate a robot_yaw every tick. A voxel occupancy map carries no
        # attitude; the viewer derives its orientation from the robot pose.
        obj.roll = 0.0
        obj.pitch = 0.0
        obj.yaw = 0.0
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

            # Decay is aligned to the integration cadence, not the wall-clock
            # frame cadence. A voxel is only ever re-observed on a refresh
            # frame, so pruning on the intermediate frames starves any map
            # whose decay_seconds is smaller than the frame time (a 0.1 s
            # decay on a ~10 fps Pi removes every cell before the next
            # integration -> a permanently empty/blinking world).
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
