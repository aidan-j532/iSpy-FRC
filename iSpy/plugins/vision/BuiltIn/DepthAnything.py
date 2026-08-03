import logging

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

from iSpy.vision.Camera import Camera
from iSpy.plugins.bases import VisionBase
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision.Object import Object


class DepthAnythingCamera(Camera, VisionBase):
    plugin_name = "depth_anything"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "model_size": {
                "type": "select",
                "label": "Model Size",
                "options": ["small"],
                "default": "small",
                "help": "Depth Anything V2 Small is downloaded automatically from Hugging Face.",
            },
            "estimate_depth": {
                "type": "boolean",
                "label": "Estimate Depth",
                "default": True,
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

    def __init__(
        self,
        camera_config: iSpyCameraConfig,
        config: iSpyConfig,
        core_mask=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.config = camera_config

        self._model = None
        self._frame_count = 0
        self._every = 5
        self._last_depth = None
        self._last_objects = []
        self._last_annotated = None

        self.unit = config.get("unit", "meter")
        self.max_depth = float(camera_config.get("max_depth", 10.0))
        self.estimate_depth = bool(camera_config.get("estimate_depth", True))

        try:
            self._every = max(1, int(camera_config.get("process_every", 5)))
        except (TypeError, ValueError):
            self._every = 5

        super().__init__(
            camera_config,
            (640, 480),
            camera_config.get("grayscale", False),
        )

        self._load_model()

    def _load_model(self):
        if not self.estimate_depth:
            self.logger.info("Depth estimation disabled by config.")
            return

        try:
            self.logger.info(
                "Loading Depth Anything V2 Small from Hugging Face..."
            )

            self._model = pipeline(
                "depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
            )

            self.logger.info("Loaded Depth Anything V2 Small.")

        except Exception:
            self.logger.exception(
                "Failed to load Depth Anything V2 from Hugging Face."
            )
            self._model = None

    def _infer_depth(self, frame: np.ndarray):
        image = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        result = self._model(image)

        # Transformers versions can expose the depth as either
        # predicted_depth or depth. Prefer the actual tensor output.
        depth = result.get("predicted_depth")

        if depth is not None:
            if hasattr(depth, "detach"):
                depth = depth.detach().cpu().numpy()

            depth = np.asarray(depth)

            # Remove batch/channel dimensions.
            while depth.ndim > 2:
                depth = depth[0]

            return depth.astype(np.float32)

        depth_image = result.get("depth")
        if depth_image is not None:
            depth = np.asarray(depth_image).astype(np.float32)
            return depth

        raise RuntimeError(
            "Depth Anything pipeline returned no depth output."
        )

    def _distance_from_depth(self, raw: float) -> float:
        norm = float(np.clip(raw, 0.0, 1e6))

        d_min = getattr(self, "_dmin", 0.0)
        d_max = getattr(self, "_dmax", 1.0)

        span = d_max - d_min
        if span <= 1e-9:
            return self.max_depth

        closeness = (norm - d_min) / span

        # Depth Anything provides relative depth, not real-world meters.
        distance_m = self.max_depth * float(
            np.clip(1.0 - closeness, 0.0, 1.0)
        )

        return distance_m

    def _objects_from_depth(
        self,
        depth: np.ndarray,
        frame: np.ndarray,
    ) -> list[Object]:
        h, w = depth.shape

        self._dmin = float(depth.min())
        self._dmax = float(depth.max())

        cx, cy = w // 2, h // 2

        center_d = self._distance_from_depth(
            float(depth[cy, cx])
        )

        objects = [
            Object(
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
            )
        ]

        # Nearest point = highest relative inverse-depth value.
        flat_near = np.unravel_index(
            np.argmax(depth),
            depth.shape,
        )

        near_y, near_x = int(flat_near[0]), int(flat_near[1])
        near_d = self._distance_from_depth(
            float(depth[near_y, near_x])
        )

        objects.append(
            Object(
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
            )
        )

        return objects

    def _annotate(
        self,
        frame: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        h, w = depth.shape

        normalized = cv2.normalize(
            depth,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

        heatmap = cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_JET,
        )

        # Resize if the model's output resolution differs from the frame.
        if heatmap.shape[:2] != frame.shape[:2]:
            heatmap = cv2.resize(
                heatmap,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        blended = cv2.addWeighted(
            frame,
            0.55,
            heatmap,
            0.45,
            0,
        )

        cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
        radius = max(
            6,
            min(frame.shape[1], frame.shape[0]) // 30,
        )

        cv2.circle(
            blended,
            (cx, cy),
            radius,
            (255, 255, 255),
            2,
        )

        # Map the center pixel to the depth-map coordinates.
        depth_x = min(
            depth.shape[1] - 1,
            int(cx * depth.shape[1] / frame.shape[1]),
        )
        depth_y = min(
            depth.shape[0] - 1,
            int(cy * depth.shape[0] / frame.shape[0]),
        )

        center_d = self._distance_from_depth(
            float(depth[depth_y, depth_x])
        )

        label = f"Depth {center_d:.2f} {self.unit}"

        cv2.putText(
            blended,
            label,
            (cx - radius * 2, cy + radius + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return blended

    def _fallback_run(self, frame: np.ndarray):
        """Synthetic heatmap used when the model cannot be loaded."""
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        normalized = cv2.normalize(
            gray,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

        heatmap = cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_JET,
        )

        blended = cv2.addWeighted(
            frame,
            0.55,
            heatmap,
            0.45,
            0,
        )

        center_x, center_y = w // 2, h // 2
        radius = max(12, min(w, h) // 6)

        cv2.circle(
            blended,
            (center_x, center_y),
            radius,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            blended,
            "Depth (no model)",
            (center_x - 90, center_y + radius + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        depth_estimate = 1.0 - (
            normalized[center_y, center_x] / 255.0
        )

        objects = [
            Object(
                x=0.0,
                y=0.0,
                z=float(depth_estimate),
                name="demo_depth",
                confidence=0.8,
                vis_type="generic",
                vis_meta={
                    "kind": "depth",
                    "heatmap": True,
                    "depth_estimate": round(depth_estimate, 3),
                    "radius": radius / max(w, h),
                },
            )
        ]

        self._last_objects = objects

        return objects, blended

    def run(self):
        frame = self.get_frame()

        if frame is None:
            return [], None

        model = getattr(self, "_model", None)

        self._frame_count += 1

        if model is None:
            return self._fallback_run(frame)

        every = max(1, self._every)
        last_depth = self._last_depth

        # Reuse the previous depth map between inference frames.
        if (
            last_depth is not None
            and every > 1
            and self._frame_count % every != 0
        ):
            objects = self._objects_from_depth(
                last_depth,
                frame,
            )

            self._last_objects = objects

            return (
                objects,
                self._annotate(frame, last_depth),
            )

        try:
            depth = self._infer_depth(frame)

        except Exception:
            self.logger.exception(
                "Depth Anything inference failed."
            )

            if last_depth is not None:
                depth = last_depth
            else:
                return self._fallback_run(frame)

        if depth is None:
            return [], frame

        self._last_depth = depth

        objects = self._objects_from_depth(
            depth,
            frame,
        )

        self._last_objects = objects

        annotated = self._annotate(
            frame,
            depth,
        )

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

            cv2.putText(
                overlay,
                "Depth",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.rectangle(
                overlay,
                (8, 38),
                (w - 8, h - 8),
                (255, 255, 255),
                1,
            )

            return overlay

        except Exception:
            return frame

    def destroy(self):
        self._model = None
        self._last_depth = None
        self._last_objects = []
        self._last_annotated = None

        if hasattr(super(), "destroy"):
            super().destroy()
