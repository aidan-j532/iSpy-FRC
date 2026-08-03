import logging
from pathlib import Path

import cv2
import numpy as np

from iSpy.vision.Camera import Camera
from iSpy.plugins.bases import VisionBase
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision.Object import Object


class DepthAnythingCamera(Camera, VisionBase):
    """Depth estimation using Depth Anything V2.

    Runs the bundled `assets/_depth_anything.pt` (small variant) checkpoint
    through a self-contained DINOv2 + DPT graph (see `_depth_anything_model`)
    so no upstream package is required.  Falls back to a synthetic heatmap
    when the weights or torch are unavailable.
    """

    plugin_name = "depth_anything"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "model_size": {
                "type": "select",
                "label": "Model Size",
                "options": ["small", "base", "large"],
                "default": "small",
                "help": "Only 'small' is bundled with the repo.",
            },
            "estimate_depth": {
                "type": "boolean",
                "label": "Estimate Depth",
                "default": True,
            },
            "model_path": {
                "type": "text",
                "label": "Model Path (.pt)",
                "default": "assets/_depth_anything.pt",
            },
            "max_depth": {
                "type": "number",
                "label": "Max Depth (m)",
                "default": 10.0,
                "step": 1.0,
            },
            "process_every": {
                "type": "number",
                "label": "Infer Every N Frames",
                "default": 5,
                "step": 1,
            },
        }

    def __init__(self, camera_config: iSpyCameraConfig, config: iSpyConfig, core_mask=None):
        self.logger = logging.getLogger(__name__)
        self.config = camera_config
        self._model = None
        self._frame_count = 0
        self._every = 5
        self._last_depth: np.ndarray | None = None
        self._last_objects: list[Object] = []
        self._last_annotated: np.ndarray | None = None

        self.unit = config.get("unit", "meter")
        self.max_depth = float(camera_config.get("max_depth", 10.0))
        self.estimate_depth = bool(camera_config.get("estimate_depth", True))
        try:
            self._every = max(1, int(camera_config.get("process_every", 5)))
        except (TypeError, ValueError):
            self._every = 5

        super().__init__(camera_config, (640, 480), camera_config.get("grayscale", False))
        self._load_model()

    def _resolve_model_path(self) -> Path | None:
        configured = self.config.get("model_path")
        if configured:
            candidate = Path(str(configured))
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if candidate.exists():
                return candidate
        bundled = Path(__file__).resolve().parents[3] / "assets" / "_depth_anything.pt"
        return bundled if bundled.exists() else None

    def _load_model(self):
        if not self.estimate_depth:
            self.logger.info("Depth estimation disabled by config.")
            return
        path = self._resolve_model_path()
        if path is None:
            self.logger.error(
                "Depth Anything weights not found (set 'model_path' or place "
                "assets/_depth_anything.pt) - falling back to the synthetic heatmap."
            )
            return
        try:
            from iSpy.plugins.vision.BuiltIn import _depth_anything_model
            self._depth_model = _depth_anything_model
        except Exception as exc:
            self.logger.error("torch is required for depth estimation: %s", exc)
            return
        try:
            self._model = _depth_anything_model.DepthAnythingV2()
            self._model.load_checkpoint(str(path))
            self._model.eval()
            self.logger.info("Loaded Depth Anything V2 model from %s", path)
        except Exception:
            self.logger.exception("Failed to load Depth Anything model from %s", path)
            self._model = None

    # -- inference helpers --------------------------------------------------

    def _distance_from_depth(self, raw: float) -> float:
        # Depth Anything V2 emits inverse depth: larger = closer.  Scale it
        # into the configured output unit so "closer" reads as a smaller
        # distance in meters.
        norm = float(np.clip(raw, 0.0, 1e6))
        d_min = getattr(self, "_dmin", 0.0)
        d_max = getattr(self, "_dmax", 1.0)
        span = d_max - d_min
        if span <= 1e-9:
            return self.max_depth
        closeness = (norm - d_min) / span
        distance_m = self.max_depth * float(np.clip(1.0 - closeness, 0.0, 1.0))
        return distance_m

    def _objects_from_depth(self, depth: np.ndarray, frame: np.ndarray) -> list[Object]:
        h, w = depth.shape
        self._dmin = float(depth.min())
        self._dmax = float(depth.max())

        cx, cy = w // 2, h // 2
        center_d = self._distance_from_depth(float(depth[cy, cx]))

        objects = [Object(
            x=0.0,
            y=0.0,
            z=center_d,
            name="depth_center",
            confidence=0.8,
            vis_type="generic",
            vis_meta={
                "kind": "depth",
                "heatmap": True,
                "depth_estimate": round(center_d, 3),
                "max_depth": self.max_depth,
            },
        )]

        # Nearest point in the frame (highest inverse-depth) - useful as a
        # crude obstacle distance.
        flat_near = np.unravel_index(np.argmax(depth), depth.shape)
        near_y, near_x = int(flat_near[0]), int(flat_near[1])
        near_d = self._distance_from_depth(float(depth[near_y, near_x]))
        objects.append(Object(
            x=float((near_x - cx) / max(w, 1)),
            y=float((near_y - cy) / max(h, 1)),
            z=near_d,
            name="depth_nearest",
            confidence=0.9,
            vis_type="generic",
            vis_meta={
                "kind": "depth",
                "heatmap": True,
                "depth_estimate": round(near_d, 3),
                "nearest_px": [near_x, near_y],
                "max_depth": self.max_depth,
            },
        ))
        return objects

    def _annotate(self, frame: np.ndarray, depth: np.ndarray) -> np.ndarray:
        h, w = depth.shape
        normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.55, heatmap, 0.45, 0)

        cx, cy = w // 2, h // 2
        radius = max(6, min(w, h) // 30)
        cv2.circle(blended, (cx, cy), radius, (255, 255, 255), 2)
        center_d = self._distance_from_depth(float(depth[cy, cx]))
        label = f"Depth {center_d:.2f} {self.unit}"
        cv2.putText(blended, label, (cx - radius * 2, cy + radius + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return blended

    def _fallback_run(self, frame: np.ndarray):
        """Synthetic grayscale heatmap used when no model is available."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.55, heatmap, 0.45, 0)

        center_x, center_y = w // 2, h // 2
        radius = max(12, min(w, h) // 6)
        cv2.circle(blended, (center_x, center_y), radius, (255, 255, 255), 2)
        cv2.putText(blended, "Depth (no model)", (center_x - 90, center_y + radius + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        depth_estimate = 1.0 - (normalized[center_y, center_x] / 255.0)
        objects = [Object(
            x=0.0, y=0.0, z=float(depth_estimate),
            name="demo_depth", confidence=0.8,
            vis_type="generic",
            vis_meta={
                "kind": "depth",
                "heatmap": True,
                "depth_estimate": round(depth_estimate, 3),
                "radius": radius / max(w, h),
            },
        )]
        self._last_objects = objects
        return objects, blended

    def run(self):
        frame = self.get_frame()
        if frame is None:
            return [], None

        model = getattr(self, "_model", None)
        self._frame_count = getattr(self, "_frame_count", 0) + 1
        if model is None:
            return self._fallback_run(frame)

        every = max(1, getattr(self, "_every", 5))
        last_depth = getattr(self, "_last_depth", None)

        # Reuse the last depth map on frames where we skip inference so the
        # loop doesn't stall on every frame.
        if (last_depth is not None
                and every > 1
                and self._frame_count % every != 0):
            objects = self._objects_from_depth(last_depth, frame)
            self._last_objects = objects
            return objects, self._annotate(frame, last_depth)

        try:
            depth = self._depth_model.infer_depth(model, frame)
        except Exception:
            self.logger.exception("Depth inference failed")
            if last_depth is not None:
                depth = last_depth
            else:
                return self._fallback_run(frame)
        if depth is None:
            return [], frame

        self._last_depth = depth
        objects = self._objects_from_depth(depth, frame)
        self._last_objects = objects
        annotated = self._annotate(frame, depth)
        self._last_annotated = annotated
        return objects, annotated

    def get_data_for_subsystem(self, target: str):
        if getattr(self, "subsystem", "field") != target:
            return None
        return self._last_objects

    def plot(self, frame):
        if frame is None:
            return None
        try:
            overlay = frame.copy()
            h, w = overlay.shape[:2]
            cv2.putText(overlay, "Depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.rectangle(overlay, (8, 38), (w - 8, h - 8), (255, 255, 255), 1)
            return overlay
        except Exception:
            return frame

    def destroy(self):
        self._model = None
        self._depth_model = None
        if super().destroy is not None:
            super().destroy()
