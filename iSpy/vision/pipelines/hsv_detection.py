import logging
import math
import threading
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from iSpy.config.iSpyConfig import iSpyCameraConfig, iSpyConfig, unit_to_inches
from iSpy.vision import calibration as cam_calibration
from iSpy.vision import triangulation
from iSpy.vision.Object import Object
from iSpy.vision.pipelines.base import VisionPipeline

# presets replace hue/sat/val at runtime. red wraps around the hue wheel
_PRESETS = {
    "orange": ((5, 25), (110, 255), (80, 255)),
    "yellow": ((25, 40), (110, 255), (80, 255)),
    "red": ((170, 10), (100, 255), (70, 255)),
    "green": ((40, 85), (90, 255), (60, 255)),
    "blue": ((85, 130), (90, 255), (60, 255)),
    # white is low saturation, high value
    "white": ((0, 179), (0, 60), (180, 255)),
}

_MASK_STYLES = ("color", "binary")


def _auto_gamma_lut(mean_v: float) -> np.ndarray:
    # pull the mean toward the middle so harsh light stops eating one end of V
    mean = max(float(mean_v), 1.0) / 255.0
    if 0.02 < mean < 0.98:
        gamma = math.log(0.5) / math.log(mean)
    else:
        gamma = 1.0
    gamma = min(max(gamma, 0.4), 2.5)
    values = np.arange(256, dtype=np.float64) / 255.0
    return np.clip((values**gamma) * 255.0, 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class HSVParams:
    preset: str
    hue_range: tuple
    sat_range: tuple
    val_range: tuple
    lighting_normalize: str
    clahe_clip: float
    blur_kernel: int
    morph_open_iters: int
    morph_close_iters: int
    morph_kernel_size: int
    morph_kernel_shape: str
    process_scale: float
    roi_top_ignore_pct: float
    min_area_px: float
    max_area_px: float
    min_circularity: float
    min_solidity: float
    min_aspect: float
    max_aspect: float
    shape_hint: str
    max_detections: int
    merge_distance_px: float
    object_name: str
    object_size_in: float
    z_mode: str
    plane_height_in: float
    view_mode: str
    mask_style: str
    draw_labels: bool


@dataclass(frozen=True)
class _HSVRuntime:
    params: HSVParams
    lower_a: np.ndarray
    upper_a: np.ndarray
    lower_b: np.ndarray | None
    upper_b: np.ndarray | None
    morph_kernel: np.ndarray | None
    merge_kernel: np.ndarray | None
    blur_size: tuple | None
    clahe: object | None


class HSVDetectionPipeline(VisionPipeline):
    plugin_name = "hsv_detection"
    display_name = "HSV Color"
    calibration_sections = ["charuco", "focal"]
    live_tunable = True

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "preset": {
                "type": "select",
                "label": "Color preset",
                "options": [
                    "custom",
                    "orange",
                    "yellow",
                    "red",
                    "green",
                    "blue",
                    "white",
                ],
                "default": "custom",
                "help": "Presets take over the HSV ranges at runtime. "
                "custom keeps the ranges below.",
            },
            "hue_range": {
                "type": "range",
                "label": "Hue range",
                "min": 0,
                "max": 179,
                "step": 1,
                "default": [5, 25],
                "wrap": True,
                "gradient": "hue",
                "help": "OpenCV hue 0-179. Drop the low end below the high "
                "end to pick reds across the wrap.",
            },
            "sat_range": {
                "type": "range",
                "label": "Saturation range",
                "min": 0,
                "max": 255,
                "step": 1,
                "default": [110, 255],
                "help": "Low saturation lets washed-out pixels in.",
            },
            "val_range": {
                "type": "range",
                "label": "Value range",
                "min": 0,
                "max": 255,
                "step": 1,
                "default": [80, 255],
                "help": "Brightness band. Raise the low end to drop shadows.",
            },
            "lighting_normalize": {
                "type": "select",
                "label": "Lighting normalize",
                "options": ["off", "clahe", "auto_gamma"],
                "default": "off",
                "help": "Flattens harsh lighting on the V channel before "
                "thresholding, so hue stays stable.",
            },
            "clahe_clip": {
                "type": "number",
                "label": "CLAHE clip",
                "default": 3.0,
                "step": 0.5,
                "help": "Contrast limit for clahe. Higher pushes local contrast "
                "harder and can amplify noise.",
            },
            "blur_kernel": {
                "type": "number",
                "label": "Blur kernel",
                "default": 0,
                "step": 2,
                "help": "Odd size, 0 turns blur off. Softens sensor noise "
                "before thresholding.",
            },
            "morph_open_iters": {
                "type": "number",
                "label": "Morph open",
                "default": 0,
                "step": 1,
                "help": "Erosion passes that kill specks.",
            },
            "morph_close_iters": {
                "type": "number",
                "label": "Morph close",
                "default": 1,
                "step": 1,
                "help": "Dilation passes that fill pinholes in a blob.",
            },
            "morph_kernel_size": {
                "type": "number",
                "label": "Morph kernel",
                "default": 5,
                "step": 2,
                "help": "Odd kernel size used by the open/close passes.",
            },
            "morph_kernel_shape": {
                "type": "select",
                "label": "Morph shape",
                "options": ["ellipse", "rect"],
                "default": "ellipse",
            },
            "process_scale": {
                "type": "number",
                "label": "Process scale",
                "default": 1.0,
                "step": 0.05,
                "help": "Threshold and trace contours on a smaller frame, then "
                "scale the boxes back. Lower is faster.",
            },
            "roi_top_ignore_pct": {
                "type": "number",
                "label": "Ignore top (%)",
                "default": 0.0,
                "step": 1,
                "help": "Skip the top band of the frame, e.g. to hide a "
                "colored ceiling or field element.",
            },
            "min_area_px": {
                "type": "number",
                "label": "Min area (px)",
                "default": 150,
                "step": 10,
                "help": "Blobs smaller than this many pixels are noise.",
            },
            "max_area_px": {
                "type": "number",
                "label": "Max area (px)",
                "default": 250000,
                "step": 1000,
                "help": "Stops one huge blob (a whole colored wall) taking over.",
            },
            "min_circularity": {
                "type": "number",
                "label": "Min circularity",
                "default": 0.0,
                "step": 0.05,
                "help": "1.0 is a perfect circle. Raise it to reject ragged blobs.",
            },
            "min_solidity": {
                "type": "number",
                "label": "Min solidity",
                "default": 0.0,
                "step": 0.05,
                "help": "Blob area over its convex hull. Low values flag shapes "
                "with bites taken out of them.",
            },
            "min_aspect": {
                "type": "number",
                "label": "Min aspect",
                "default": 1.0,
                "step": 0.1,
                "help": "Long side over short side, so 1.0 is square.",
            },
            "max_aspect": {
                "type": "number",
                "label": "Max aspect",
                "default": 6.0,
                "step": 0.5,
                "help": "Drop long thin blobs like a cable or a lane line.",
            },
            "shape_hint": {
                "type": "select",
                "label": "Shape hint",
                "options": ["any", "circle", "rect"],
                "default": "any",
                "help": "Picks which shape term feeds confidence and depth.",
            },
            "max_detections": {
                "type": "number",
                "label": "Max detections",
                "default": 10,
                "step": 1,
                "help": "Keep the N biggest blobs.",
            },
            "merge_distance_px": {
                "type": "number",
                "label": "Merge distance (px)",
                "default": 0,
                "step": 5,
                "help": "Merge blobs closer than this. 0 turns merging off.",
            },
            "object_name": {
                "type": "text",
                "label": "Object name",
                "default": "hsv_object",
            },
            "object_size_in": {
                "type": "number",
                "label": "Object size (in)",
                "default": 4.0,
                "step": 0.1,
                "help": "Real diameter/width in inches, used by size_based depth.",
            },
            "z_mode": {
                "type": "select",
                "label": "Depth estimation",
                "options": ["ground_plane", "size_based"],
                "default": "ground_plane",
                "help": "ground_plane casts a ray onto the object plane. "
                "size_based uses the known object size instead.",
            },
            "plane_height_in": {
                "type": "number",
                "label": "Plane height (in)",
                "default": 0.0,
                "step": 0.5,
                "help": "Height of the plane the object center sits on. 0 is "
                "the floor.",
            },
            "view_mode": {
                "type": "select",
                "label": "View",
                "options": ["outline", "mask"],
                "default": "outline",
                "help": "outline keeps the camera picture with contours drawn "
                "on top. mask shows only the thresholded pixels.",
            },
            "mask_style": {
                "type": "select",
                "label": "Mask style",
                "options": ["color", "binary"],
                "default": "color",
                "help": "color keeps the source pixels inside the mask, binary "
                "is white on black.",
            },
            "draw_labels": {
                "type": "toggle",
                "label": "Draw labels",
                "default": True,
            },
        }

    def __init__(
        self, camera_config: iSpyCameraConfig, config: iSpyConfig, core_mask=None
    ):
        self.logger = logging.getLogger(__name__)
        self.config = camera_config
        self._ispy_config = config
        self._cam_name = camera_config.get("name", "?")
        self.core_mask = core_mask
        self._settings_lock = threading.Lock()
        self._pending_persist: dict = {}

        try:
            self.subsystem = camera_config.get("subsystem", "field")
            self.camera_bot_relative_yaw = camera_config.get("yaw", 0.0)
            self.camera_pitch_angle = camera_config.get("pitch", 0.0)

            _pos_unit = config.get("unit", "frc")
            # height is the single mount-height field feeding triangulation
            self.camera_height = unit_to_inches(
                camera_config.get("height", 0.0), _pos_unit
            )
            self.camera_x = unit_to_inches(camera_config.get("x", 0.0), _pos_unit)
            self.camera_y = unit_to_inches(camera_config.get("y", 0.0), _pos_unit)

            calib = camera_config.get("calibration", {}) or {}
            self.fov = calib.get("fov", 0.0)
            self.focal_length_pixels = calib.get("focal_length_pixels", 0.0)
            self.grayscale = camera_config.get("grayscale", False)
        except KeyError as e:
            raise ValueError(f"Missing camera config key for HSV detection: {e}")

        self.unit = config.get("unit", "meter")
        self.conversions = {
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

        super().__init__(camera_config, (640, 480), bool(self.grayscale))

        self._raw_settings = dict(camera_config.pipeline_settings() or {})
        self._runtime = self._build_runtime(self._raw_settings)
        self._last_objects: list[Object] = []
        self._last_rendered = None
        self._set_status("ready")

    def needs_calibration_to_run(self) -> bool:
        return False

    # settings parsing - everything goes through here, so a garbage value from
    # the json editor lands on the schema default instead of killing the tick

    def _warn_setting(self, key: str, value, fallback) -> None:
        self.logger.warning(
            "Camera '%s': hsv setting %s=%r is invalid - using %r",
            self._cam_name,
            key,
            value,
            fallback,
        )

    def _parse_settings(self, raw: dict) -> HSVParams:
        schema = self.config_schema()

        def value_of(key):
            if isinstance(raw, dict) and raw.get(key) is not None:
                return raw[key]
            return schema[key]["default"]

        def num(key, lo, hi):
            default = schema[key]["default"]
            value = value_of(key)
            try:
                number = float(value)
            except (TypeError, ValueError):
                self._warn_setting(key, value, default)
                return float(default)
            if not math.isfinite(number):
                self._warn_setting(key, value, default)
                return float(default)
            clamped = min(max(number, lo), hi)
            if clamped != number:
                self._warn_setting(key, value, clamped)
            return clamped

        def whole(key, lo, hi):
            return int(round(num(key, lo, hi)))

        def odd(key, lo, hi, off=False):
            value = whole(key, lo, hi)
            if off and value <= 0:
                return 0
            if value <= 0:
                value = 1
            if value % 2 == 0:
                value += 1
            return value

        def pair(key):
            default = schema[key]["default"]
            bounds = (schema[key]["min"], schema[key]["max"])
            wrap = bool(schema[key].get("wrap"))
            value = value_of(key)
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                self._warn_setting(key, value, default)
                return (int(default[0]), int(default[1]))
            try:
                first = int(round(float(value[0])))
                second = int(round(float(value[1])))
            except (TypeError, ValueError):
                self._warn_setting(key, value, default)
                return (int(default[0]), int(default[1]))
            first = min(max(first, bounds[0]), bounds[1])
            second = min(max(second, bounds[0]), bounds[1])
            if not wrap and first > second:
                self._warn_setting(key, value, default)
                return (int(default[0]), int(default[1]))
            return (first, second)

        def choice(key):
            options = schema[key]["options"]
            value = str(value_of(key) or "").strip().lower()
            if value not in options:
                self._warn_setting(key, value, schema[key]["default"])
                return schema[key]["default"]
            return value

        preset = choice("preset")
        hue_range = pair("hue_range")
        sat_range = pair("sat_range")
        val_range = pair("val_range")
        if preset != "custom":
            hue_range, sat_range, val_range = _PRESETS[preset]

        return HSVParams(
            preset=preset,
            hue_range=hue_range,
            sat_range=sat_range,
            val_range=val_range,
            lighting_normalize=choice("lighting_normalize"),
            clahe_clip=num("clahe_clip", 0.1, 40.0),
            blur_kernel=odd("blur_kernel", 0, 31, off=True),
            morph_open_iters=whole("morph_open_iters", 0, 10),
            morph_close_iters=whole("morph_close_iters", 0, 10),
            morph_kernel_size=odd("morph_kernel_size", 1, 31),
            morph_kernel_shape=choice("morph_kernel_shape"),
            process_scale=num("process_scale", 0.25, 1.0),
            roi_top_ignore_pct=num("roi_top_ignore_pct", 0.0, 90.0),
            min_area_px=num("min_area_px", 0.0, 10000000.0),
            max_area_px=num("max_area_px", 0.0, 10000000.0),
            min_circularity=num("min_circularity", 0.0, 1.0),
            min_solidity=num("min_solidity", 0.0, 1.0),
            min_aspect=num("min_aspect", 1.0, 100.0),
            max_aspect=num("max_aspect", 1.0, 100.0),
            shape_hint=choice("shape_hint"),
            max_detections=whole("max_detections", 1, 100),
            merge_distance_px=num("merge_distance_px", 0.0, 500.0),
            object_name=str(value_of("object_name") or "hsv_object"),
            object_size_in=num("object_size_in", 0.01, 1000.0),
            z_mode=choice("z_mode"),
            plane_height_in=num("plane_height_in", -1000.0, 1000.0),
            view_mode=choice("view_mode"),
            mask_style=choice("mask_style"),
            draw_labels=bool(value_of("draw_labels")),
        )

    def _build_runtime(self, raw: dict) -> _HSVRuntime:
        params = self._parse_settings(raw)
        h_lo, h_hi = params.hue_range
        s_lo, s_hi = params.sat_range
        v_lo, v_hi = params.val_range

        lower_a = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        upper_a = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
        lower_b = upper_b = None

        # wrapped hue needs two inRange passes: [lo, 179] plus [0, hi]
        if h_lo > h_hi:
            upper_a = np.array([179, s_hi, v_hi], dtype=np.uint8)
            lower_b = np.array([0, s_lo, v_lo], dtype=np.uint8)
            upper_b = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)

        morph_kernel = None
        if (
            params.morph_kernel_size > 0
            and (params.morph_open_iters > 0 or params.morph_close_iters > 0)
        ):
            shape = (
                cv2.MORPH_ELLIPSE
                if params.morph_kernel_shape == "ellipse"
                else cv2.MORPH_RECT
            )
            size = max(1, params.morph_kernel_size)
            morph_kernel = cv2.getStructuringElement(shape, (size, size))

        merge_kernel = None
        if params.merge_distance_px > 0:
            # close with a kernel big enough to bridge the requested gap, then
            # the erode half of the close restores the blob outline
            distance = max(1, int(round(params.merge_distance_px * params.process_scale)))
            size = 2 * distance + 1
            merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

        blur_size = (params.blur_kernel, params.blur_kernel)
        if params.blur_kernel < 3:
            blur_size = None

        clahe = None
        if params.lighting_normalize == "clahe":
            clahe = cv2.createCLAHE(
                clipLimit=params.clahe_clip, tileGridSize=(8, 8)
            )

        return _HSVRuntime(
            params=params,
            lower_a=lower_a,
            upper_a=upper_a,
            lower_b=lower_b,
            upper_b=upper_b,
            morph_kernel=morph_kernel,
            merge_kernel=merge_kernel,
            blur_size=blur_size,
            clahe=clahe,
        )

    # live tuning -----------------------------------------------------------------

    def take_pending_settings(self) -> dict:
        with self._settings_lock:
            pending = dict(self._pending_persist)
            self._pending_persist.clear()
        return pending

    def current_settings(self) -> dict:
        # the parsed values, so the tuning panel shows what is actually running
        return asdict(self._runtime.params)

    def update_settings(self, partial: dict) -> dict:
        schema = self.config_schema()
        allowed = {
            key: value for key, value in (partial or {}).items() if key in schema
        }
        if not allowed:
            return {}
        with self._settings_lock:
            merged = dict(self._raw_settings)
            merged.update(allowed)
            runtime = self._build_runtime(merged)
            # one attribute swap, so run() never sees half of an update
            self._runtime = runtime
            self._raw_settings = merged
        return {key: getattr(runtime.params, key) for key in allowed}

    # action buttons ---------------------------------------------------------------

    def get_actions(self) -> list[dict]:
        try:
            params = self._runtime.params
        except Exception:
            return []
        return [
            {
                "id": "mask_view",
                "label": "Mask",
                "kind": "toggle",
                "value": params.view_mode == "mask",
                "title": "Show the threshold mask instead of the camera view",
            },
            {
                "id": "cycle_mask_style",
                "label": "Mask style",
                "kind": "button",
                "title": "Switch the mask between color and white on black",
            },
        ]

    def invoke_action(self, action_id: str, value=None) -> dict:
        if action_id == "mask_view":
            mode = "mask" if value else "outline"
            applied = self.update_settings({"view_mode": mode})
            if not applied:
                return {"ok": False, "error": "could not apply view_mode"}
            with self._settings_lock:
                self._pending_persist["view_mode"] = applied["view_mode"]
            return {"ok": True, "actions": self.get_actions()}

        if action_id == "cycle_mask_style":
            current = self._runtime.params.mask_style
            index = _MASK_STYLES.index(current) if current in _MASK_STYLES else -1
            nxt = _MASK_STYLES[(index + 1) % len(_MASK_STYLES)]
            applied = self.update_settings({"mask_style": nxt})
            if not applied:
                return {"ok": False, "error": "could not apply mask_style"}
            with self._settings_lock:
                self._pending_persist["mask_style"] = applied["mask_style"]
            return {"ok": True, "actions": self.get_actions()}

        return {"ok": False, "error": "unsupported"}

    # geometry ---------------------------------------------------------------------

    def _focal_length_px_fov(self, img_w: int) -> float:
        intr = cam_calibration.intrinsics_for_frame(
            self.config.get("calibration", {}), img_w, 1
        )
        if intr is not None:
            return float(intr[0][0, 0])
        if self.fov and self.fov > 0:
            return (img_w / 2.0) / math.tan(math.radians(self.fov / 2.0))
        configured = getattr(self, "focal_length_pixels", 0.0)
        if configured and configured > 1:
            return float(configured)
        # no calibration: assume a typical 60 deg horizontal fov so depth is
        # at least in the right ballpark
        return (img_w / 2.0) / math.tan(math.radians(60.0 / 2.0))

    def _pixel_ray(self, pixel_x: float, pixel_y: float, img_w: int, img_h: int):
        return triangulation.pixel_to_ray(
            pixel_x,
            pixel_y,
            img_w,
            img_h,
            self._focal_length_px_fov(img_w),
            self.camera_x,
            self.camera_y,
            self.camera_height,
            self.camera_bot_relative_yaw,
            self.camera_pitch_angle,
        )

    def _pixel_to_robot(
        self, pixel_x, pixel_y, distance_los, img_w, img_h, obj_height_z=0.0
    ) -> np.ndarray:
        f = max(self._focal_length_px_fov(img_w), 1e-6)
        horizontal_angle_rad = math.atan((pixel_x - img_w / 2.0) / f)

        # same triangle as object_detection: an elevated object shortens the
        # vertical leg between camera and assumed plane
        vertical_delta = max(self.camera_height - obj_height_z, 0.0)
        if vertical_delta > 0 and distance_los > vertical_delta:
            true_horiz = math.sqrt(distance_los**2 - vertical_delta**2)
        else:
            true_horiz = distance_los * math.cos(math.radians(self.camera_pitch_angle))

        left_right = true_horiz * math.sin(horizontal_angle_rad)
        forward = true_horiz * math.cos(horizontal_angle_rad)

        yaw_rad = math.radians(self.camera_bot_relative_yaw)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        x_rot = left_right * cos_y + forward * sin_y
        y_rot = forward * cos_y - left_right * sin_y

        scale = self.conversions.get(self.unit, self.conversions["meter"])
        return np.array(
            [(x_rot + self.camera_x) * scale, (y_rot + self.camera_y) * scale],
            dtype=np.float64,
        )

    def _size_based_point(self, cx, cy, width_px, img_w, img_h, params):
        f = self._focal_length_px_fov(img_w)
        if width_px <= 0 or f <= 0 or params.object_size_in <= 0:
            return None
        distance_los = (params.object_size_in * f) / float(width_px)
        base = self._pixel_to_robot(
            cx, cy, distance_los, img_w, img_h, params.plane_height_in
        )
        scale = self.conversions.get(self.unit, self.conversions["meter"])
        return np.array(
            [base[0], base[1], params.plane_height_in * scale], dtype=np.float64
        )

    # detection --------------------------------------------------------------------

    def _detect(self, frame, params, runtime):
        img_h, img_w = frame.shape[:2]
        scale = params.process_scale
        if scale < 1.0:
            work = cv2.resize(
                frame,
                (max(1, int(img_w * scale)), max(1, int(img_h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            work = frame

        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

        if runtime.clahe is not None:
            hsv[:, :, 2] = runtime.clahe.apply(hsv[:, :, 2])
        elif params.lighting_normalize == "auto_gamma":
            hsv[:, :, 2] = cv2.LUT(
                hsv[:, :, 2], _auto_gamma_lut(float(hsv[:, :, 2].mean()))
            )

        if runtime.blur_size is not None:
            # blur s/v only - blurring h smears colors that touch into a third
            # hue that neither blob actually has
            hsv[:, :, 1] = cv2.blur(hsv[:, :, 1], runtime.blur_size)
            hsv[:, :, 2] = cv2.blur(hsv[:, :, 2], runtime.blur_size)

        mask = cv2.inRange(hsv, runtime.lower_a, runtime.upper_a)
        if runtime.lower_b is not None:
            mask = cv2.bitwise_or(
                mask, cv2.inRange(hsv, runtime.lower_b, runtime.upper_b)
            )

        if params.roi_top_ignore_pct > 0:
            cut = int(mask.shape[0] * params.roi_top_ignore_pct / 100.0)
            if cut > 0:
                mask[:cut, :] = 0

        if runtime.morph_kernel is not None:
            if params.morph_open_iters > 0:
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_OPEN,
                    runtime.morph_kernel,
                    iterations=params.morph_open_iters,
                )
            if params.morph_close_iters > 0:
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_CLOSE,
                    runtime.morph_kernel,
                    iterations=params.morph_close_iters,
                )

        if runtime.merge_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, runtime.merge_kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        inv = 1.0 / scale if scale > 0 else 1.0
        area_scale = inv * inv
        found = []
        for contour in contours:
            area_work = cv2.contourArea(contour)
            area_px = area_work * area_scale
            if area_px < params.min_area_px or area_px > params.max_area_px:
                continue

            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw <= 0 or bh <= 0:
                continue
            rect = cv2.minAreaRect(contour)
            long_side = max(rect[1][0], rect[1][1])
            short_side = min(rect[1][0], rect[1][1])
            aspect = long_side / short_side if short_side > 1e-6 else 999.0
            if aspect < params.min_aspect or aspect > params.max_aspect:
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = (
                4.0 * math.pi * area_work / (perimeter * perimeter)
                if perimeter > 0
                else 0.0
            )
            if circularity < params.min_circularity:
                continue

            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area_work / hull_area if hull_area > 0 else 0.0
            if solidity < params.min_solidity:
                continue

            rect_area = rect[1][0] * rect[1][1]
            rectangularity = area_work / rect_area if rect_area > 0 else 0.0
            rectangularity = min(rectangularity, 1.0)

            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                cx = (moments["m10"] / moments["m00"]) * inv
                cy = (moments["m01"] / moments["m00"]) * inv
            else:
                cx = (bx + bw / 2.0) * inv
                cy = (by + bh / 2.0) * inv

            # confidence blends how much of the shape box the blob fills with
            # how well it matches the shape hint
            if params.shape_hint == "circle":
                shape_score = circularity
            elif params.shape_hint == "rect":
                shape_score = rectangularity
            else:
                shape_score = max(circularity, rectangularity)
            confidence = min(max(0.5 * rectangularity + 0.5 * shape_score, 0.0), 1.0)

            box = (
                int(round(bx * inv)),
                int(round(by * inv)),
                max(1, int(round(bw * inv))),
                max(1, int(round(bh * inv))),
            )
            contour_full = (contour.astype(np.float32) * inv).astype(np.int32)
            found.append(
                (
                    area_px,
                    contour_full,
                    float(cx),
                    float(cy),
                    box,
                    float(circularity),
                    float(confidence),
                )
            )

        found.sort(key=lambda item: item[0], reverse=True)
        return found[: params.max_detections], mask, scale

    def _make_object(self, detection, params, img_w, img_h):
        _, _contour, cx, cy, box, _circularity, confidence = detection
        # a sphere's center is on the plane, everything else sits with its
        # bottom edge on it
        pixel_y = cy if params.shape_hint == "circle" else box[1] + box[3]
        ray = self._pixel_ray(cx, pixel_y, img_w, img_h)
        on_plane = triangulation.ground_plane_intersection(
            ray, ground_z=params.plane_height_in
        )
        size_based = self._size_based_point(cx, cy, box[2], img_w, img_h, params)
        scale = self.conversions.get(self.unit, self.conversions["meter"])

        if params.z_mode == "size_based":
            if size_based is not None:
                point, depth_source = size_based, "size_based"
            elif on_plane is not None:
                point, depth_source = on_plane * scale, "ground_plane"
            else:
                return None
        else:
            if on_plane is not None:
                point, depth_source = on_plane * scale, "ground_plane"
            elif size_based is not None:
                point, depth_source = size_based, "size_based"
            else:
                return None

        return Object(
            float(point[0]),
            float(point[1]),
            z=float(params.plane_height_in * scale),
            name=params.object_name,
            confidence=confidence,
            ray_origin=ray.origin,
            ray_direction=ray.direction,
            depth_source=depth_source,
            vis_type="generic",
            vis_meta={
                "kind": "hsv",
                "area_px": round(detection[0], 1),
                "circularity": round(detection[5], 3),
                "bbox": list(box),
            },
        )

    # rendering --------------------------------------------------------------------

    def _render(self, frame, mask, detections, params):
        if params.view_mode == "mask":
            if params.mask_style == "binary":
                view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                color = (0, 0, 0)
            else:
                view = cv2.bitwise_and(frame, frame, mask=mask)
                color = (0, 255, 0)
        else:
            view = frame
            color = (0, 255, 0)

        for area_px, contour, _cx, _cy, box, _circularity, confidence in detections:
            cv2.drawContours(view, [contour], -1, color, 2)
            if params.draw_labels:
                label = f"{params.object_name} {confidence:.2f}"
                cv2.putText(
                    view,
                    label,
                    (box[0], max(14, box[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        return view

    def run(self):
        frame = self.get_frame()
        if frame is None:
            return [], None

        # one read so a live update can swap the whole runtime underneath us
        runtime = self._runtime
        params = runtime.params

        gated = self._gate_uncalibrated(frame)
        if gated is not None:
            return gated

        try:
            detections, mask, scale = self._detect(frame, params, runtime)
        except Exception:
            self.logger.exception("HSV detection failed")
            return [], frame

        img_h, img_w = frame.shape[:2]
        objects = []
        for detection in detections:
            obj = self._make_object(detection, params, img_w, img_h)
            if obj is not None:
                objects.append(obj)
        self._last_objects = objects

        if params.view_mode == "mask" and scale < 1.0:
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        view = self._render(frame, mask, detections, params)
        self._last_rendered = view
        return objects, view

    def get_data_for_subsystem(self, target: str):
        if getattr(self, "subsystem", "field") != target:
            return None
        return self._last_objects

    def plot(self, frame):
        # run() already returned the rendered view (solo mode passes it here),
        # so there is nothing left to draw
        return frame

    def destroy(self):
        super().destroy()
