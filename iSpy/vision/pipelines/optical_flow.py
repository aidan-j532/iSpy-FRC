import cv2
import logging
import math
import numpy as np
import time

from iSpy.vision.pipelines.base import VisionPipeline
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision.Object import Object


class OpticalFlowCamera(VisionPipeline):
    plugin_name = "optical_flow"
    beta = True

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "method": {
                "type": "select", "label": "Flow Method",
                "options": ["farneback", "lk"], "default": "farneback",
                "help": "farneback = dense flow (smoother, slower). lk = sparse "
                        "corner tracking (faster, noisier).",
            },
            "flow_scale": {
                "type": "number", "label": "Processing Scale", "default": 0.5, "step": 0.1,
                "help": "Downscale factor applied before computing flow. 0.5 = "
                        "half resolution, 4x faster.",
            },
            "win_size": {
                "type": "number", "label": "Window Size", "default": 15, "step": 2,
            },
            "lk_win_size": {
                "type": "number", "label": "LK Window Size", "default": 21, "step": 2,
            },
            "max_corners": {
                "type": "number", "label": "LK Max Corners", "default": 250, "step": 25,
            },
            "min_flow": {
                "type": "number", "label": "Min Flow (px)", "default": 0.2, "step": 0.05,
                "help": "Flows below this magnitude are ignored for the average.",
            },
            "smoothing": {
                "type": "number", "label": "Velocity Smoothing", "default": 0.3,
                "step": 0.05,
                "help": "EMA alpha (0..1) applied to the velocity estimate. "
                        "Lower = smoother but laggier.",
            },
            "ground_ratio": {
                "type": "number", "label": "Ground Region", "default": 0.6, "step": 0.05,
                "help": "Fraction of the frame (from the bottom) treated as ground.",
            },
            "nominal_range": {
                "type": "number", "label": "Nominal Range (in)", "default": 60, "step": 5,
                "help": "Used when the camera has no downward pitch: assumed "
                        "distance to the ground region in inches.",
            },
            "flow_saturation": {
                "type": "number", "label": "Flow Saturation (px)", "default": 24, "step": 4,
                "help": "Flow magnitude (px) that renders as full color intensity "
                        "in the visualization.",
            },
            "draw_debug": {
                "type": "toggle", "label": "Draw Debug Overlay", "default": True,
            },
        }

    def __init__(self, camera_config: iSpyCameraConfig, config: iSpyConfig, core_mask=None):
        self.logger = logging.getLogger(__name__)
        self.config = camera_config

        self.subsystem = camera_config.get("subsystem", "field")
        self.camera_pitch_angle = camera_config.get("pitch", 0.0)
        _pos_unit = config.get("unit", "frc")
        from iSpy.config.iSpyConfig import unit_to_inches
        # 'height' is the single mount-height field (matches other pipelines);
        # legacy 'z' is deprecated and ignored
        self.camera_height = unit_to_inches(camera_config.get("height", 0.0), _pos_unit)
        _legacy_z = camera_config.get("z")
        if _legacy_z not in (None, 0, 0.0):
            self.logger.warning(
                "Camera config key 'z' (%s) is deprecated and ignored - "
                "'height' is the single mount-height field.",
                _legacy_z,
            )

        calib = camera_config.get("calibration", {}) or {}
        self.fov = calib.get("fov", 0.0) or 0.0
        self.grayscale = camera_config.get("grayscale", False)

        self.unit = config.get("unit", "meter")
        self.conversions = {
            "meter": 0.0254, "meters": 0.0254,
            "inch": 1.0, "inches": 1.0,
            "foot": 1 / 12, "feet": 1 / 12,
            "centimeter": 2.54, "centimeters": 2.54,
            "frc": 0.0254,
        }

        super().__init__(camera_config, (640, 480), self.grayscale)

        self._prev_gray = None
        self._prev_points = None
        self._last_ts = time.perf_counter()
        self._fps_smooth = 0.0
        self._svx = 0.0
        self._svy = 0.0
        self._last_objects: list[Object] = []
        self._last_motion: dict = {}
        self._last_viz: dict | None = None
        self._set_status("ready")

    # ------------------------------------------------------------------

    def _setting(self, key, default):
        try:
            v = self.config.get_pipeline_setting(key)
        except Exception:
            v = None
        return default if v is None else v

    def _focal_length_px_fov(self, img_w: int) -> float:
        if self.fov and self.fov > 0:
            return (img_w / 2.0) / math.tan(math.radians(self.fov / 2.0))
        return (img_w / 2.0) / math.tan(math.radians(60.0 / 2.0))

    def _range_inches(self) -> float:
        pitch = math.radians(self.camera_pitch_angle)
        cam_h = abs(self.camera_height)
        if pitch > 1e-3 and cam_h > 1e-6:
            return max(cam_h / math.tan(pitch), 1.0)
        return max(float(self._setting("nominal_range", 60)), 1.0)

    def _unit_label(self) -> str:
        return {
            "meter": "m/s", "meters": "m/s", "frc": "m/s",
            "inch": "in/s", "inches": "in/s",
            "foot": "ft/s", "feet": "ft/s",
            "centimeter": "cm/s", "centimeters": "cm/s",
        }.get(self.unit, self.unit + "/s")

    def _flow_measure(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        scale = float(self._setting("flow_scale", 0.5))
        if 0 < scale < 1.0:
            small = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = gray

        prev = self._prev_gray
        self._prev_gray = small
        if prev is None or prev.shape != small.shape:
            return 0.0, 0.0, 0.0, None

        method = str(self._setting("method", "farneback"))
        min_flow = float(self._setting("min_flow", 0.2))
        sat_px = max(1.0, float(self._setting("flow_saturation", 24)))
        ground_ratio = float(np.clip(self._setting("ground_ratio", 0.6), 0.1, 1.0))
        start_row = int(small.shape[0] * (1.0 - ground_ratio))

        if method == "lk":
            win = int(self._setting("lk_win_size", 21))
            corners = int(self._setting("max_corners", 250))
            pts = cv2.goodFeaturesToTrack(prev, maxCorners=corners, qualityLevel=0.01,
                                          minDistance=8, blockSize=win)
            if pts is None:
                return 0.0, 0.0, 0.0, None
            cur, st, _ = cv2.calcOpticalFlowPyrLK(prev, small, pts, None,
                                                  winSize=(win, win), maxLevel=3)
            ok = st[:, 0] == 1
            if not ok.any():
                return 0.0, 0.0, 0.0, None
            tracked = pts[ok].reshape(-1, 2)
            disp = cur[ok].reshape(-1, 2) - tracked
            in_ground = tracked[:, 1] >= start_row
            if in_ground.any():
                dg = disp[in_ground]
                mags = np.hypot(dg[:, 0], dg[:, 1])
                valid = mags >= min_flow
                coverage = float(valid.mean())
                if valid.any():
                    dx = float(np.median(dg[valid, 0]))
                    dy = float(np.median(dg[valid, 1]))
                else:
                    dx = dy = 0.0
            else:
                coverage = 0.0
                dx = dy = 0.0
            arrows = []
            points = []
            for (px, py), (dxd, dyd) in zip(tracked[in_ground], disp[in_ground]):
                points.append((float(px) / scale, float(py) / scale))
                if math.hypot(dxd, dyd) >= min_flow:
                    arrows.append((float(px) / scale, float(py) / scale,
                                   float(dxd) / scale, float(dyd) / scale))
            viz = {"kind": "lk", "arrows": arrows, "points": points, "scale": scale}
            return dx / scale, dy / scale, coverage, viz

        # farneback (dense)
        win = max(3, int(self._setting("win_size", 15)))
        flow = cv2.calcOpticalFlowFarneback(
            prev, small, None, pyr_scale=0.5, levels=3, winsize=win,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        region = flow[start_row:, :, :]
        mags = np.hypot(region[..., 0], region[..., 1])
        mask = mags >= min_flow
        coverage = float(mask.mean()) if mask.size else 0.0
        if mask.any():
            dx = float(np.median(region[..., 0][mask]))
            dy = float(np.median(region[..., 1][mask]))
        else:
            dx = dy = 0.0

        arrows = []
        step = max(8, int(24 * scale))
        for yy in range(start_row, small.shape[0], step):
            for xx in range(0, small.shape[1], step):
                dv = flow[yy, xx]
                if math.hypot(dv[0], dv[1]) >= min_flow:
                    arrows.append((float(xx) / scale, float(yy) / scale,
                                   float(dv[0]) / scale, float(dv[1]) / scale))
        viz = {
            "kind": "farneback",
            "flow_small": flow, "mask_small": mask,
            "start_row_small": start_row, "scale": scale,
            "arrows": arrows,
        }
        return dx / scale, dy / scale, coverage, viz

    def is_ready(self) -> tuple[bool, str]:
        return True, self.get_status()

    def run(self):
        frame = self.get_frame()
        if frame is None:
            return [], None

        now = time.perf_counter()
        fps = 1.0 / max(now - self._last_ts, 1e-6)
        self._last_ts = now
        self._fps_smooth = self._fps_smooth * 0.9 + fps * 0.1
        fps = self._fps_smooth

        img_h, img_w = frame.shape[:2]
        f = self._focal_length_px_fov(img_w)
        scale = self.conversions.get(self.unit, self.conversions["meter"])

        dx_px, dy_px, coverage, viz = 0.0, 0.0, 0.0, None
        if f > 1:
            dx_px, dy_px, coverage, viz = self._flow_measure(frame)
        self._last_viz = viz

        rng = self._range_inches()
        k = rng / f if f > 1 else 0.0
        # robot +X is right, +Y is forward; ground features moving down the
        # image (positive dy) mean forward motion, leftward flow means rightward motion
        vx = -dx_px * k * fps * scale
        vy = dy_px * k * fps * scale
        mag_px = math.hypot(dx_px, dy_px)

        alpha = float(np.clip(self._setting("smoothing", 0.3), 0.05, 1.0))
        self._svx = alpha * vx + (1.0 - alpha) * self._svx
        self._svy = alpha * vy + (1.0 - alpha) * self._svy
        svx, svy = self._svx, self._svy

        speed = math.hypot(svx, svy)
        heading = math.degrees(math.atan2(svx, svy)) if speed > 1e-9 else 0.0
        conf = float(np.clip(coverage * min(1.0, 0.3 + 0.7 * min(1.0, mag_px / 8.0)),
                             0.0, 1.0))

        self._last_motion = {
            "kind": "velocity",
            "vx": svx, "vy": svy,
            "lateral": svx, "forward": svy,
            "speed": speed, "heading_deg": heading,
            "vx_px": dx_px, "vy_px": dy_px,
            "flow_magnitude_px": mag_px,
            "coverage": coverage,
            "range_inches": rng,
            "fps": fps,
            "confidence": conf,
        }

        obj = Object(
            x=svx, y=svy, z=0.0,
            name="flow",
            confidence=conf,
            depth_source="optical_flow",
            vis_type="generic",
            vis_meta=dict(self._last_motion),
        )
        self._last_objects = [obj] if mag_px > 0.05 else []
        return self._last_objects, frame

    # ------------------------------------------------------------------
    # visualization (called by the run loop after run())

    def plot(self, frame):
        if frame is None:
            return None
        try:
            out = frame.copy()
            if not bool(self._setting("draw_debug", True)):
                return out
            img_h, img_w = out.shape[:2]

            ground_ratio = float(np.clip(self._setting("ground_ratio", 0.6), 0.1, 1.0))
            start_row_full = int(img_h * (1.0 - ground_ratio))
            sat_px = max(1.0, float(self._setting("flow_saturation", 24)))

            viz = self._last_viz
            if viz is not None and viz.get("kind") == "farneback":
                flow_small = viz["flow_small"]
                mask_small = viz["mask_small"]
                flow = cv2.resize(flow_small, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                fx = flow[..., 0].astype(np.float32)
                fy = flow[..., 1].astype(np.float32)
                mags = np.hypot(fx, fy)
                hue = ((np.degrees(np.arctan2(fy, fx)) % 360.0) / 2.0).astype(np.uint8)
                val = np.clip(mags * (255.0 / sat_px), 0, 255).astype(np.uint8)
                hsv = np.dstack([hue, np.full_like(hue, 255), val])
                color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                mask_full = cv2.resize(
                    mask_small.astype(np.uint8) * 255, (img_w, img_h),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
                live = np.zeros((img_h, img_w), dtype=bool)
                live[start_row_full:, :] = True
                live &= mask_full
                if color is not None and live.any():
                    blend = cv2.addWeighted(out, 0.55, color, 0.45, 0)
                    out[live] = blend[live]
                cv2.line(out, (0, start_row_full), (img_w, start_row_full),
                         (0, 255, 255), 1, cv2.LINE_AA)

            motion = self._last_motion or {}
            self._draw_hud(out, motion)
            return out
        except Exception:
            return frame

    def _draw_hud(self, out, motion: dict):
        panel_w, line_h = 250, 20
        x0, y0 = 12, 12
        rows = 5
        ph = y0 + rows * line_h + 12
        sub = out[y0:y0 + ph, x0:x0 + panel_w]
        if sub.size:
            dark = np.full_like(sub, (12, 14, 18))
            cv2.addWeighted(sub, 0.62, dark, 0.38, 0, dst=sub)

        def text(line_idx, s, col, scale=0.5, bold=1):
            cv2.putText(out, s, (x0 + 10, y0 + line_idx * line_h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, col, bold, cv2.LINE_AA)

        unit = self._unit_label()
        text(0, f"OPTICAL FLOW   {self._last_viz.get('kind', '-').upper() if self._last_viz else '-'}",
             (88, 166, 255), 0.55, 2)
        speed = motion.get("speed", 0.0)
        text(1, f"SPEED   {speed:.2f} {unit}", (120, 255, 120), 0.6, 2)
        heading = float(motion.get("heading_deg", 0.0))
        text(2, f"HEADING {heading:+.0f} deg   FWD {motion.get('forward', 0.0):+.2f}",
             (230, 230, 230))
        cov = motion.get("coverage", 0.0)
        text(3, f"COVERAGE {cov * 100:.0f}%   {motion.get('fps', 0.0):.0f} FPS",
             (230, 230, 230))

        # confidence bar
        conf = float(np.clip(motion.get("confidence", 0.0), 0.0, 1.0))
        bw = int((panel_w - 24) * conf)
        cv2.rectangle(out, (x0 + 10, y0 + 4 * line_h + 4),
                      (x0 + panel_w - 14, y0 + 4 * line_h + 10), (60, 60, 70), -1)
        bar_col = (60, 170, 255) if conf < 0.4 else ((90, 230, 140) if conf < 0.75 else (80, 255, 120))
        cv2.rectangle(out, (x0 + 10, y0 + 4 * line_h + 4),
                      (x0 + 10 + bw, y0 + 4 * line_h + 10), bar_col, -1)
        text(4, f"CONF {conf * 100:.0f}%", (230, 230, 230))

    # ------------------------------------------------------------------

    def get_data_for_subsystem(self, target: str):
        if getattr(self, "subsystem", "field") != target:
            return None
        return self._last_objects

    def get_motion(self) -> dict:
        return dict(self._last_motion)

    def get_speed(self) -> dict:
        m = self._last_motion or {}
        return {
            "forward": m.get("forward", 0.0),
            "lateral": m.get("lateral", 0.0),
            "speed": m.get("speed", 0.0),
            "heading_deg": m.get("heading_deg", 0.0),
            "unit": self.unit,
            "confidence": m.get("confidence", 0.0),
        }

    def destroy(self):
        super().destroy()
