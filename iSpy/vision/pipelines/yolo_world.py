import hashlib
import logging
import threading
from pathlib import Path

import cv2
import numpy as np
import requests

from iSpy.vision.pipelines.base import BackgroundPreparedPipeline
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision.Object import Object

_WORLD_MODEL_DIR = Path(__file__).resolve().parents[3] / "YoloModels" / "pytorch"

_WORLD_MODEL_URLS = {
    "s": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt",
    "m": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m-worldv2.pt",
    "l": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8l-worldv2.pt",
}


class YoloWorldCamera(BackgroundPreparedPipeline):
    plugin_name = "yolo_world"

    @classmethod
    def needs_model_backend(cls) -> bool:
        return True

    def is_ready(self) -> tuple[bool, str]:
        # Pure status report - never triggers or blocks on optimization.
        # The optimize build is started at construction when needed.
        if not self._optimization_requested():
            if self._preparing():
                self._set_status("downloading (model weights)")
                return False, "downloading (model weights)"
            if self.model is not None:
                self._set_status("ready")
                return True, "ready"
            reason = getattr(self, "_load_error", None)
            status = f"error: {reason}" if reason else "error: model weights not downloaded/loaded"
            self._set_status(status)
            return False, status

        if self._optimizing:
            self._set_status("optimizing")
            return False, "optimizing"

        if self._optimized_active():
            self._set_status("ready")
            return True, "ready"

        reason = self._optimize_error or getattr(self, "_load_error", None)
        status = reason or "optimizing"
        self._set_status(status)
        return False, status

    def _set_load_error(self, reason: str):
        self._load_error = reason
        self.logger.error("YOLO World camera '%s': %s", self.config.get("name", "?"), reason)

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "prompt": {
                "type": "text",
                "label": "Prompt",
                "default": "A dog.",
            },
            "model_size": {
                "type": "select",
                "label": "Model Size",
                "options": ["s", "m", "l"],
                "default": "s",
                "help": "YOLO World v2 weights are downloaded automatically from Ultralytics on first use.",
            },
            "optimize": {
                "type": "toggle",
                "label": "Optimize/Convert",
                "default": False,
                "optimize_toggle": True,
                "help": "Build the best optimized backend artifact for this device "
                        "(rknn on Rockchip NPU, engine on NVIDIA, onnx elsewhere, "
                        "etc.) in the background. Falls back to the top-level "
                        "config 'optimize' when unset.",
            },
            "target_format": {
                "type": "select",
                "label": "Target Format",
                "options": ["auto", "onnx", "rknn", "tflite", "openvino", "engine", "coreml"],
                "default": "auto",
                "quantization": True,
                "help": "'auto' picks the best format for this device (rknn on Rockchip NPUs, tflite on Edge TPU, engine on NVIDIA, onnx elsewhere).",
            },
            "quantize": {
                "type": "toggle",
                "label": "Quantize model",
                "default": False,
                "quantization": True,
                "help": "Quantize the optimized artifact (int8). Only meaningful "
                        "with optimize or target_format set.",
            },
            "quantization_dataset": {
                "type": "browse",
                "label": "Quantization dataset",
                "default": "",
                "browse_root": "QuantizeDataset",
                "quantization": True,
                "gated_by": "quantize",
                "help": "Optional folder of calibration images used for "
                        "quantization. Leave empty to auto-download images "
                        "from the model's calibration keywords.",
            },
            "input_size": {
                "type": "number",
                "label": "Input Size",
                "default": 640,
                "quantization": True,
                "help": "Letterbox resolution used for the quantized model conversion and inference.",
            },
        }

    @staticmethod
    def _parse_classes(prompt: str) -> list[str]:
        """Split a comma-separated prompt into YOLO World class tokens,
        stripping leading articles and trailing punctuation."""
        classes = []
        for part in str(prompt).split(","):
            token = part.strip().rstrip(".!?").strip()
            if not token:
                continue
            lower = token.lower()
            for article in ("a ", "an ", "the "):
                if lower.startswith(article):
                    token = token[len(article):]
                    break
            token = token.strip()
            if token and token not in classes:
                classes.append(token)
        return classes or ["object"]

    def __init__(self, camera_config: iSpyCameraConfig, config: iSpyConfig, core_mask=None):
        self.logger = logging.getLogger(__name__)
        self.config = camera_config
        self._ispy_config = config
        self.core_mask = core_mask
        self.prompt = str(camera_config.get_pipeline_setting("prompt") or "A dog.")
        self.classes = self._parse_classes(self.prompt)
        self.model_size = str(camera_config.get_pipeline_setting("model_size") or "s").lower()

        raw_quantize = camera_config.get_pipeline_setting("quantize")
        if raw_quantize is None:
            raw_quantize = camera_config.get_pipeline_setting("quantized")  # legacy key
        if raw_quantize is None:
            raw_quantize = False
        if isinstance(raw_quantize, str):
            raw_quantize = raw_quantize.strip().lower() in ("1", "true", "yes", "on")
        self.quantize = bool(raw_quantize)

        self.target_format = str(camera_config.get_pipeline_setting("target_format") or "auto").lower()
        self._quantization_dataset = camera_config.get_pipeline_setting("quantization_dataset") or None
        self._model_input_size = int(camera_config.get_pipeline_setting("input_size") or 640)

        raw_optimize = camera_config.get_pipeline_setting("optimize")
        if raw_optimize is None:
            raw_optimize = camera_config.get_pipeline_setting("auto_opt")  # legacy key
        if raw_optimize is None:
            raw_optimize = config.get("optimize", config.get("auto_opt", False)) if config is not None else False
        if isinstance(raw_optimize, str):
            raw_optimize = raw_optimize.strip().lower() in ("1", "true", "yes", "on")
        self._auto_opt = bool(raw_optimize)
        self.model = None
        self._model_path = None
        self._quantized = False
        self._load_error = None
        self._class_names = {i: name for i, name in enumerate(self.classes)}
        self._optimizing = False
        self._optimize_error: str | None = None
        self._target_format: str | None = None
        super().__init__(camera_config, (640, 480), camera_config.get("grayscale", False))

        # If the config requests optimization and no matching artifact is
        # active yet, kick off the build on a simple background thread so the
        # app can keep running (is_ready() reports "optimizing" until the
        # artifact is active; run() passes frames through untouched).
        if self._optimization_requested() and not self._optimized_active():
            self.logger.info(
                "Camera '%s': optimization requested - building %s artifact",
                self.config.get("name", "?"), self._target_format_cached(),
            )
            threading.Thread(
                target=self._optimize_runner,
                daemon=True,
                name=f"Optimize-{self.config.get('name', 'yolo_world')}",
            ).start()
        else:
            self.prepare()

    def _prepare(self):
        """Background preparation: download/convert the YOLO World model
        without blocking construction of the other cameras."""
        self._load_model()

    def get_optimization_options(self) -> dict:
        schema = self.config_schema()
        return {
            key: schema[key]
            for key in ("optimize", "quantize", "target_format", "quantization_dataset", "input_size")
            if key in schema
        }

    def optimize(self, **kwargs) -> str:
        """Build the quantized backend artifact synchronously. Blocks until
        the build finishes; is_ready() reports (False, "optimizing") while it
        runs and (True, "ready") once it has produced a matching artifact."""
        if not self._optimization_requested():
            return "optimization disabled for this camera (set 'Quantize' in camera settings)"
        if self._optimizing:
            return "optimizing"

        if kwargs.pop("force", False):
            return self._optimize_forced()

        self._optimizing = True
        self._set_status("optimizing (backend build)")
        try:
            # Reuse a cached artifact when one matches the requested
            # target format; only a full rebuild re-exports and re-quantizes.
            self._load_quantized_model(force=False)
            if not self._optimized_active():
                self._load_quantized_model(force=True)
        except Exception as exc:
            self._optimize_error = f"quantized build failed: {exc}"
            self._set_load_error(self._optimize_error)
            self._optimizing = False
            return f"error: quantized build failed - {exc}"
        self._optimizing = False

        if self._optimized_active():
            self._load_error = None
            self._optimize_error = None
            self._set_status("ready")
            return "ready"
        status = self._optimize_error or "error: quantized build failed - no artifact produced"
        self._optimize_error = status
        self._set_status(status)
        return status

    def _optimize_forced(self) -> str:
        """Force a full rebuild: re-export and re-quantize even when a
        matching artifact is already cached (manual rebuild path)."""
        self._optimizing = True
        self._set_status("optimizing (backend build)")
        try:
            self._load_quantized_model(force=True)
        except Exception as exc:
            self._optimize_error = f"quantized build failed: {exc}"
            self._set_load_error(self._optimize_error)
            self._optimizing = False
            return f"error: quantized build failed - {exc}"
        self._optimizing = False

        if self._optimized_active():
            self._load_error = None
            self._optimize_error = None
            self._set_status("ready")
            return "ready"
        status = self._optimize_error or "error: quantized build failed - no artifact produced"
        self._optimize_error = status
        self._set_status(status)
        return status

    def _optimize_runner(self):
        """Run the synchronous optimize() off the main thread (started at
        construction when the config requests a build). is_ready() reports
        "optimizing" until this finishes."""
        status = self.optimize()
        if not self._optimized_active():
            self._optimize_error = status
        self._set_status(status)

    def _optimization_requested(self) -> bool:
        return bool(getattr(self, "quantize", False)) or bool(
            getattr(self, "_auto_opt", False)
        )

    def _resolve_target_format(self) -> str:
        explicit = str(getattr(self, "target_format", "") or "").strip().lower()
        if explicit and explicit != "auto":
            target = explicit
        else:
            # Same resolution ensure_quantized_model uses so the readiness
            # format check agrees with the artifact that actually gets built.
            from iSpy.config.AutoOpt import recommend_format
            target = recommend_format()
        supported = {"onnx", "rknn", "tflite", "openvino", "engine", "coreml", "tpu"}
        if target not in supported:
            self.logger.warning(
                "Recommended target format %r unsupported - using onnx", target,
            )
            return "onnx"
        return target

    def _target_format_cached(self) -> str:
        if self._target_format is None:
            self._target_format = self._resolve_target_format()
        return self._target_format

    @staticmethod
    def _path_format(path: str) -> str:
        """Format of a model path: 'onnx', 'openvino', ... or ''."""
        p = str(path).lower()
        if "openvino_model" in p or p.endswith(".xml"):
            return "openvino"
        for ext, fmt in (
            (".pt", "pytorch"),
            (".onnx", "onnx"),
            (".rknn", "rknn"),
            (".tflite", "tflite"),
            (".engine", "engine"),
            (".mlpackage", "coreml"),
        ):
            if p.endswith(ext):
                return fmt
        return ""

    def _optimized_active(self) -> bool:
        """True once the loaded model is the quantized backend artifact in
        the requested target format (a full-precision fallback or an artifact
        in another format does not count)."""
        if getattr(self, "model", None) is None or not getattr(self, "_quantized", False):
            return False
        if getattr(self.model, "model_type", "") == "tpu":
            return True
        path = str(getattr(self, "_model_path", "") or "")
        if not path:
            return False
        return self._path_format(path) == self._target_format_cached()

    def _is_processable(self) -> bool:
        """True when run() may actually run inference. When False, run()
        passes the raw camera feed through untouched."""
        if getattr(self, "_optimizing", False):
            return False
        if self.model is None:
            return False
        if self._optimization_requested():
            return self._optimized_active()
        return True

    def _ensure_world_model(self, size: str) -> str | None:
        """Download YOLO World weights into <project>/YoloModels/pytorch on first use."""
        url = _WORLD_MODEL_URLS.get(size, _WORLD_MODEL_URLS["s"])
        filename = url.rsplit("/", 1)[-1]
        target = _WORLD_MODEL_DIR / filename

        if target.exists() and target.stat().st_size >= 1024:
            return str(target)

        self.logger.info("Downloading YOLO World weights %s to %s ...", filename, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
            tmp.replace(target)
        except Exception as exc:  # pragma: no cover - network dependent
            self.logger.exception("Failed to download YOLO World weights: %s", exc)
            try:
                tmp.unlink()
            except OSError:
                pass
            return None

        if target.stat().st_size < 1024:
            self.logger.error("YOLO World weights appear truncated (%s bytes).", target.stat().st_size)
            return None
        return str(target)

    def _resolve_weights(self) -> str | None:
        """Bundled asset takes priority, then auto-downloaded weights."""
        # Bundled asset takes priority, then auto-downloaded weights.
        # assets lives at iSpy/assets; this file moved to iSpy/vision/pipelines/,
        # so parents[2] resolves to the iSpy package directory.
        asset_path = Path(__file__).resolve().parents[2] / "assets" / "yolo-world.pt"
        if asset_path.exists():
            self.logger.info("Using bundled YOLO World weights at %s", asset_path)
            return str(asset_path)
        return self._ensure_world_model(self.model_size)

    def _reparameterize_world(self, weights: str) -> str | None:
        """Bake the configured classes into the open-vocabulary YOLO World
        weights so the result is a standard fixed-vocabulary detector that the
        conversion framework can export and quantize."""
        try:
            from ultralytics import YOLOWorld, YOLO
        except Exception as exc:  # pragma: no cover - runtime dependency fallback
            self.logger.error("Ultralytics is required to reparameterize YOLO World: %s", exc)
            return None

        classes_key = hashlib.sha1("|".join(self.classes).encode("utf-8")).hexdigest()[:8]
        fixed = _WORLD_MODEL_DIR / "world" / f"{Path(weights).stem}-{classes_key}.pt"
        if fixed.exists() and fixed.stat().st_size >= 1024:
            return str(fixed)

        fixed.parent.mkdir(parents=True, exist_ok=True)
        try:
            model = YOLOWorld(weights, verbose=False)
            model.set_classes(self.classes)
            model.save(str(fixed))
            YOLO(str(fixed), task="detect", verbose=False)
            self.logger.info("Reparameterized YOLO World model -> %s (classes=%s)", fixed, self.classes)
            return str(fixed)
        except Exception as exc:  # pragma: no cover - runtime dependency fallback
            self.logger.exception("Failed to reparameterize YOLO World model: %s", exc)
            return None

    def _load_model(self):
        if self._optimization_requested():
            self._load_quantized_model()
            if self.model is not None:
                return
            self.logger.warning(
                "Quantized YOLO World model failed to load for camera '%s' - "
                "falling back to full-precision inference.",
                self.config.get("name", "?"),
            )
        self._load_full_precision()

    def _load_full_precision(self):
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - runtime dependency fallback
            self._set_load_error(
                f"ultralytics is required for YOLO World inference: {exc}"
            )
            self.model = None
            return

        weights = self._resolve_weights()
        if not weights:
            self._set_load_error(
                "failed to resolve YOLO World weights (bundled asset missing "
                "and download failed)"
            )
            self.model = None
            return

        try:
            # Bake the configured classes into the open-vocabulary weights and
            # run the result as a plain fixed-vocabulary detector. Running the
            # world head directly after set_classes() hits an ultralytics
            # channel mismatch (convs built for the checkpoint's nc=80, head
            # switched to nc=len(classes) at runtime) -> RuntimeError:
            # shape '[1, <nc+64>, -1]' is invalid for input of size <n>.
            fixed = self._reparameterize_world(weights)
            if not fixed:
                self._set_load_error("failed to reparameterize YOLO World model")
                self.model = None
                return

            self.model = YOLO(fixed, task="detect", verbose=False)
            self._model_path = fixed
            self._quantized = False
            self._load_error = None
            self.logger.info(
                "Loaded YOLO World model from %s (classes=%s)",
                fixed, self.classes,
            )
        except Exception as exc:  # pragma: no cover - runtime dependency fallback
            self._set_load_error(f"failed to load YOLO World model: {exc}")
            self.model = None

    def _load_quantized_model(self, force: bool = False):
        try:
            weights = self._resolve_weights()
            if not weights:
                self._set_load_error(
                    "failed to resolve YOLO World weights (bundled asset "
                    "missing and download failed)"
                )
                self.model = None
                return

            fixed = self._reparameterize_world(weights)
            if not fixed:
                self._set_load_error("failed to reparameterize YOLO World model")
                self.model = None
                return

            from iSpy.vision.QuantizedModel import ensure_quantized_model
            artifact, converted = ensure_quantized_model(
                fixed,
                self.target_format,
                (self._model_input_size, self._model_input_size),
                quantize=True,
                dataset_path=self._quantization_dataset,
                force=force,
            )
            if not converted:
                self._set_load_error(
                    f"quantized conversion of {Path(fixed).name} produced no artifact"
                )
                self.model = None
                return

            from iSpy.vision.genericYolo import GenericYolo
            from iSpy.vision.metadata import read_metadata
            self.model = GenericYolo(
                {
                    "file_path": artifact,
                    "input_size": [self._model_input_size, self._model_input_size],
                    "min_conf": 0.25,
                },
                self.core_mask,
                iSpy_config=self._ispy_config,
            )
            self._model_path = artifact
            self._quantized = True
            self._load_error = None

            meta = read_metadata(Path(artifact))
            if meta and isinstance(meta.get("names"), dict):
                self._class_names = {int(k): str(v) for k, v in meta["names"].items()}
            self.logger.info(
                "Loaded quantized YOLO World model from %s (classes=%s)",
                artifact, self.classes,
            )
        except Exception as exc:  # pragma: no cover - runtime dependency fallback
            self._set_load_error(f"failed to load quantized YOLO World model: {exc}")
            self.model = None

    def run(self):
        frame = self.get_frame()
        if frame is None:
            return [], None

        if not self._is_processable():
            return [], frame

        try:
            if self._quantized:
                results = self.model.predict(frame, orig_shape=frame.shape)
                objects: list[Object] = []
                for box in results.boxes:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy]
                    conf = float(box.conf)
                    cls_id = int(box.cls_id)
                    name = self._class_names.get(cls_id, str(cls_id))
                    objects.append(
                        Object(
                            x=float((x1 + x2) / 2.0),
                            y=float((y1 + y2) / 2.0),
                            z=0.0,
                            name=name,
                            confidence=conf,
                            vis_type="generic",
                            vis_meta={
                                "prompt": self.prompt,
                                "classes": self.classes,
                                "kind": "detection",
                                "quantized": True,
                            },
                        )
                    )
                annotated = results.plot(frame.copy())
                return objects, annotated

            results = self.model(frame, stream=False, conf=0.25, imgsz=640, verbose=False)
            objects: list[Object] = []
            annotated = frame.copy()

            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    names = getattr(result, "names", None)
                    if isinstance(names, dict):
                        name = names.get(cls_id, names.get(str(cls_id), str(cls_id)))
                    elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                        name = names[cls_id]
                    else:
                        name = str(cls_id)
                    objects.append(
                        Object(
                            x=float((x1 + x2) / 2.0),
                            y=float((y1 + y2) / 2.0),
                            z=0.0,
                            name=name,
                            confidence=conf,
                            vis_type="generic",
                            vis_meta={"prompt": self.prompt, "classes": self.classes, "kind": "detection"},
                        )
                    )

            if isinstance(results, list) and results:
                annotated = results[0].plot() if hasattr(results[0], "plot") else annotated
            elif hasattr(results, "plot"):
                annotated = results.plot()

            if annotated is None:
                annotated = frame
            return objects, annotated
        except Exception as exc:  # pragma: no cover - runtime dependency fallback
            self.logger.exception("YOLO World inference failed: %s", exc)
            return [], frame

    def plot(self, frame):
        if frame is None:
            return None
        try:
            overlay = frame.copy()
            cv2.putText(overlay, "YOLO World", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            return overlay
        except Exception:
            return frame

    def destroy(self):
        self.stopped = True
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
