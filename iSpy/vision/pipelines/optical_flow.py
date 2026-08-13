"""Optical flow odometry.

[BETA] Estimates robot ground velocity from optical flow. Farneback dense
flow (or sparse Lucas-Kanade) is measured on the ground region of the
frame; pixel flow is converted to robot-frame speed using the pinhole
model and the camera's pitch/height (features moving *down* the image mean
the robot is moving *forward*). The emitted Object's x/y carry the
lateral/forward velocity in configured units per second - best treated as
a soft dead-reckoning signal between vision updates, not an odometry source.
"""

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
            "min_flow": {
                "type": "number", "label": "Min Flow (px)", "default": 0.2, "step": 0.05,
                "help": "Flows below this magnitude are ignored for the average.",
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
            "draw_debug": {
                "type": "toggle", "label": "Draw Debug Overlay", "default": True,
            },
        }

    def __init__(self, camera_config: iSpyCameraConfig, config: iSpyConfig, core_mask=None):
        self.logger = logging.getLogger(__name__)
        self.config = camera_config

        self.subsystem = camera_config.get("subsystem", "field")
        self.camera_pitch_angle = camera_config.get("pitch", 0.0)
        self.camera_z = camera_config.get("z", 0.0)

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
        self._last_objects: list[Object] = []
        self._last_motion: dict = {}
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
        """distance from the camera to the ground region along the optical axis"""
        pitch = math.radians(self.camera_pitch_angle)
        cam_h = abs(self.camera_z)
        if pitch > 1e-3 and cam_h > 1e-6:
            return max(cam_h / math.tan(pitch), 1.0)
        return max(float(self._setting("nominal_range", 60)), 1.0)

    def _flow_measure(self, frame):
        """returns (mean_dx_px, mean_dy_px) in full-resolution pixels/frame,
        plus a list of (x, y, dx, dy) arrows in downscaled coords for debug."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        scale = float(self._setting("flow_scale", 0.5))
        if 0 < scale < 1.0:
            small = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = gray

        prev = self._prev_gray
        self._prev_gray = small
        if prev is None or prev.shape != small.shape:
            return (0.0, 0.0), []

        method = str(self._setting("method", "farneback"))
        min_flow = float(self._setting("min_flow", 0.2))
        ground_ratio = float(np.clip(self._setting("ground_ratio", 0.6), 0.1, 1.0))
        start_row = int(small.shape[0] * (1.0 - ground_ratio))

        if method == "lk":
            win = int(self._setting("lk_win_size", 21))
            pts = cv2.goodFeaturesToTrack(prev, maxCorners=250, qualityLevel=0.01,
                                          minDistance=8, blockSize=win)
            if pts is None:
                return (0.0, 0.0), []
            cur, st, _ = cv2.calcOpticalFlowPyrLK(prev, small, pts, None,
                                                  winSize=(win, win), maxLevel=3)
            ok = st[:, 0] == 1
            if not ok.any():
                return (0.0, 0.0), []
            disp = cur[ok] - pts[ok]
            rows_ok = pts[ok][:, 1] >= start_row
            if rows_ok.any():
                mags = np.hypot(disp[rows_ok, 0], disp[rows_ok, 1])
                keep = mags >= min_flow
                if keep.any():
                    dx = float(disp[rows_ok][keep, 0].mean())
                    dy = float(disp[rows_ok][keep, 1].mean())
                else:
                    dx = dy = 0.0
            else:
                dx = dy = 0.0
            arrows = [(float(p[0]), float(p[1]), float(d[0]), float(d[1]))
                      for p, d in zip(pts[ok], disp) if p[1] >= start_row]
            return (dx / scale, dy / scale), arrows

        # farneback (dense)
        win = max(3, int(self._setting("win_size", 15)))
        flow = cv2.calcOpticalFlowFarneback(
            prev, small, None, pyr_scale=0.5, levels=3, winsize=win,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        region = flow[start_row:, :, :]
        mags = np.hypot(region[..., 0], region[..., 1])
        if region.size and (mags >= min_flow).any():
            keep = mags >= min_flow
            dx = float(region[..., 0][keep].mean())
            dy = float(region[..., 1][keep].mean())
        else:
            dx = dy = 0.0
        arrows = []
        step = max(8, int(24 * scale))
        for yy in range(start_row, small.shape[0], step):
            for xx in range(0, small.shape[1], step):
                dv = flow[yy, xx]
                arrows.append((float(xx), float(yy), float(dv[0]), float(dv[1])))
        return (dx / scale, dy / scale), arrows

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

        dx_px, dy_px, arrows = 0.0, 0.0, []
        if f > 1:
            _m, arrows = self._flow_measure(frame)
            dx_px, dy_px = _m

        rng = self._range_inches()
        k = rng / f if f > 1 else 0.0
        # robot +X is right, +Y is forward; ground features moving down the
        # image (positive dy) mean forward motion, leftward flow means rightward motion
        vx_in_per_fr = -dx_px * k
        vy_in_per_fr = dy_px * k
        vx = vx_in_per_fr * fps * scale
        vy = vy_in_per_fr * fps * scale
        mag_px = math.hypot(dx_px, dy_px)

        draw = bool(self._setting("draw_debug", True))
        if draw:
            flow_scale = float(self._setting("flow_scale", 0.5))
            for ax, ay, adx, ady in arrows:
                sx, sy = int(ax / flow_scale), int(ay / flow_scale)
                ex, ey = int(sx + adx / flow_scale), int(sy + ady / flow_scale)
                cv2.arrowedLine(frame, (sx, sy), (ex, ey), (0, 255, 0), 1, tipLength=0.3)
            unit = self.unit
            speed_txt = f"V {vy:+.2f} {unit}/s" if unit in ("meter", "meters", "frc") else f"V {vy:+.1f} {unit}/s"
            cv2.putText(frame, speed_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"flow {mag_px:.1f}px", (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 200, 255), 2)

        conf = min(1.0, mag_px * 0.2)
        self._last_motion = {
            "vx": vx, "vy": vy,
            "vx_px": dx_px, "vy_px": dy_px,
            "flow_magnitude_px": mag_px,
            "range_inches": rng,
            "fps": fps,
        }

        obj = Object(
            x=vx, y=vy, z=0.0,
            name="flow",
            confidence=conf,
            depth_source="optical_flow",
            vis_type="generic",
            vis_meta=dict(self._last_motion),
        )
        self._last_objects = [obj] if mag_px > 0 else []
        return self._last_objects, frame

    def get_data_for_subsystem(self, target: str):
        if getattr(self, "subsystem", "field") != target:
            return None
        return self._last_objects

    def get_motion(self) -> dict:
        """lateral/forward velocity (configured units/s) + raw pixel flow"""
        return dict(self._last_motion)

    def plot(self, frame):
        if frame is None:
            return None
        try:
            overlay = frame.copy()
            cv2.putText(overlay, "Optical Flow", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return overlay
        except Exception:
            return frame

    def destroy(self):
        super().destroy()
