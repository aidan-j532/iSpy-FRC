import logging
import threading

import cv2
import numpy as np
from PIL import Image

from iSpy.vision.pipelines.base import BackgroundPreparedPipeline
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision.Object import Object

_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
_DEPTH_INPUT_SIZE = 518
_DEPTH_ARTIFACT_STEM = "depth_anything_v2_small"

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class DepthAnythingCamera(BackgroundPreparedPipeline):
    plugin_name = "depth_anything"

    @classmethod
    def needs_model_backend(cls) -> bool:
        return True

    def is_ready(self) -> tuple[bool, str]:
        ready, status = self._readiness()
        self._set_status(status)
        return ready, status

    def _readiness(self) -> tuple[bool, str]:
        if not self.estimate_depth:
            return True, "ready"
        if getattr(self, "_optimizing", False):
            return False, "optimizing (onnx build)"
        if self._preparing():
            status = "optimizing (onnx build)" if self.optimize else "downloading (model weights)"
            return False, status
        if self.optimize:
            if self._session is not None:
                return True, "ready"
            if getattr(self, "_optimize_fallback", False):
                return True, "error: optimized ONNX build failed - using unoptimized fallback"
            reason = getattr(self, "_load_error", None)
            if reason:
                return False, f"error: {reason}"
            return False, "error: optimized ONNX model not built"
        if self._model is not None:
            return True, "ready"
        reason = getattr(self, "_load_error", None)
        if reason:
            return False, f"error: {reason}"
        return False, "error: model weights not downloaded/loaded"

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
            "optimize": {
                "type": "boolean",
                "label": "Optimize",
                "default": True,
                "help": "Export the model once to an int8-quantized ONNX via iSpy's export framework and run it through onnxruntime for fast CPU inference.",
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
        self._session = None
        self._onnx = False
        self._load_error = None
        self._optimize_fallback = False
        self._optimizing = False
        self._frame_count = 0
        self._every = 5
        self._last_depth = None
        self._last_objects = []
        self._last_annotated = None

        self.unit = config.get("unit", "meter")
        self.max_depth = float(camera_config.get("max_depth", 10.0))
        self.estimate_depth = bool(camera_config.get("estimate_depth", True))

        # max_depth is configured in meters; scale z output into the
        # configured unit so every pipeline emits the same unit.
        self._z_scale = {
            "meter": 1.0, "meters": 1.0,
            "inch": 39.37007874, "inches": 39.37007874,
            "foot": 3.280839895, "feet": 3.280839895,
            "centimeter": 100.0, "centimeters": 100.0,
        }.get(self.unit, 1.0)

        raw_optimize = camera_config.get("optimize", True)
        if isinstance(raw_optimize, str):
            raw_optimize = raw_optimize.strip().lower() in ("1", "true", "yes", "on")
        self.optimize = bool(raw_optimize)

        try:
            self._every = max(1, int(camera_config.get("process_every", 5)))
        except (TypeError, ValueError):
            self._every = 5

        super().__init__(
            camera_config,
            (640, 480),
            camera_config.get("grayscale", False),
        )

        self.prepare()

    def _prepare(self):
        """Background preparation: download/export the depth model without
        blocking construction of the other cameras."""
        self._load_model()

    def get_optimization_options(self) -> dict:
        schema = self.config_schema()
        return {
            key: schema[key]
            for key in ("optimize", "model_size")
            if key in schema
        }

    def optimize(self, **kwargs) -> str:
        """Start a forced ONNX rebuild of the depth model as a background
        job (generic entry point over request_optimize())."""
        return self.request_optimize()

    def request_optimize(self) -> str:
        """Force-rebuild the optimized ONNX artifact (downloads the base
        weights again if needed) without blocking. No-op if a rebuild is
        already running. Never blocks."""
        if not self.optimize:
            return "optimization disabled for this camera (set 'Optimize' in camera settings)"
        if getattr(self, "_optimizing", False):
            return "optimizing (onnx build)"
        if self._preparing():
            return "initializing (model preparation in progress)"
        self._optimizing = True
        self._set_status("optimizing (onnx build)")
        thread = threading.Thread(
            target=self._optimize_worker,
            daemon=True,
            name="Optimize-DepthAnything",
        )
        thread.start()
        return "optimizing (onnx build)"

    def _optimize_worker(self):
        try:
            self._load_optimized(force=True)
        except Exception as exc:
            self._load_error = f"optimized ONNX rebuild failed: {exc}"
            self.logger.exception("Optimized ONNX depth model rebuild failed.")
        finally:
            self._optimizing = False
        if self._session is not None:
            self._load_error = None
            self._set_status("ready")
        else:
            self._set_status("error: optimized ONNX build failed")

    def _load_model(self):
        if not self.estimate_depth:
            self.logger.info("Depth estimation disabled by config.")
            return

        if self.optimize:
            try:
                self._load_optimized()
            except Exception as exc:
                self._load_error = f"optimized ONNX export/load failed: {exc}"
                self.logger.exception("Optimized ONNX depth model failed to load.")
            if self._session is not None:
                self._load_error = None
                return
            self.logger.warning(
                "Optimized ONNX depth model unavailable for camera '%s' - "
                "falling back to the transformers pipeline.",
                self.config.get("name", "?"),
            )

        self._optimize_fallback = self.optimize and self._session is None
        self._load_pipeline()

    def _load_optimized(self, force: bool = False):
        import torch.nn as nn

        from transformers import AutoModelForDepthEstimation

        from iSpy.vision.QuantizedModel import ensure_onnx_model

        class _DepthModule(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, pixel_values):
                return self.model(pixel_values=pixel_values).predicted_depth

        def build():
            self.logger.info(
                "Loading Depth Anything V2 Small weights from Hugging Face..."
            )
            model = AutoModelForDepthEstimation.from_pretrained(_DEPTH_MODEL_ID)
            model.eval()
            return _DepthModule(model)

        artifact, converted = ensure_onnx_model(
            build,
            _DEPTH_ARTIFACT_STEM,
            input_size=(_DEPTH_INPUT_SIZE, _DEPTH_INPUT_SIZE),
            quantize=True,
            force=force,
        )
        if not converted or not artifact:
            self._session = None
            self._load_error = "optimized ONNX conversion produced no artifact"
            return

        import onnxruntime as ort

        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self._session = ort.InferenceSession(artifact, providers=providers)
        self._onnx = True
        self._load_error = None
        self.logger.info("Loaded optimized Depth Anything ONNX from %s", artifact)

    def _load_pipeline(self):
        if not self.estimate_depth:
            self.logger.info("Depth estimation disabled by config.")
            return

        try:
            self.logger.info(
                "Loading Depth Anything V2 Small from Hugging Face..."
            )

            from transformers import pipeline
            self._model = pipeline(
                "depth-estimation",
                model=_DEPTH_MODEL_ID,
            )

            self._load_error = None
            self.logger.info("Loaded Depth Anything V2 Small.")

        except Exception as exc:
            self._load_error = f"failed to load Depth Anything V2 from Hugging Face: {exc}"
            self.logger.exception(
                "Failed to load Depth Anything V2 from Hugging Face."
            )
            self._model = None

    def _infer_depth(self, frame: np.ndarray):
        if self._onnx and self._session is not None:
            return self._infer_depth_onnx(frame)

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

    def _infer_depth_onnx(self, frame: np.ndarray) -> np.ndarray:
        size = _DEPTH_INPUT_SIZE
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = (
            cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
            .astype(np.float32)
            / 255.0
        )
        pixel_values = (
            (img.transpose(2, 0, 1) - _IMAGENET_MEAN) / _IMAGENET_STD
        ).astype(np.float32)[None]

        depth = self._session.run(None, {"pixel_values": pixel_values})[0][0]

        return cv2.resize(
            np.asarray(depth, dtype=np.float32),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
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

        return distance_m * self._z_scale

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
        session = getattr(self, "_session", None)

        self._frame_count = getattr(self, "_frame_count", 0) + 1

        if model is None and session is None:
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
        self._session = None
        self._onnx = False
        self._last_depth = None
        self._last_objects = []
        self._last_annotated = None

        if hasattr(super(), "destroy"):
            super().destroy()
