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

_WORLD_MODEL_CACHE = Path.home() / ".cache" / "ispy"

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
        ready, status = self._readiness()
        self._set_status(status)
        return ready, status

    def _readiness(self) -> tuple[bool, str]:
        if getattr(self, "_optimizing", False):
            return False, "optimizing (backend build)"
        if self._preparing():
            return False, "downloading (model weights)"
        if self.model is not None:
            if self.quantize and not self._quantized:
                return True, "error: quantized build failed - using full precision fallback"
            return True, "ready"
        reason = getattr(self, "_load_error", None)
        if reason:
            return False, f"error: {reason}"
        return False, "error: model weights not downloaded/loaded"

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
            "quantize": {
                "type": "boolean",
                "label": "Quantize",
                "default": False,
                "help": "Convert the YOLO World model to a quantized backend artifact using iSpy's export framework.",
            },
            "quantization_dataset": {
                "type": "text",
                "label": "Quantization dataset",
                "default": "",
                "help": "Optional path to a folder of calibration images used "
                        "for quantization. Leave empty to auto-download images "
                        "from the model's calibration keywords.",
            },
            "target_format": {
                "type": "select",
                "label": "Target Format",
                "options": ["auto", "onnx", "rknn", "tflite", "openvino", "engine", "coreml"],
                "default": "auto",
                "help": "'auto' picks the best format for this device (rknn on Rockchip NPUs, tflite on Edge TPU, engine on NVIDIA, onnx elsewhere).",
            },
            "input_size": {
                "type": "number",
                "label": "Input Size",
                "default": 640,
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
        self.prompt = str(camera_config.get("prompt") or "A dog.")
        self.classes = self._parse_classes(self.prompt)
        self.model_size = str(camera_config.get("model_size") or "s").lower()

        raw_quantize = camera_config.get("quantize", False)
        if isinstance(raw_quantize, str):
            raw_quantize = raw_quantize.strip().lower() in ("1", "true", "yes", "on")
        self.quantize = bool(raw_quantize)

        self.target_format = str(camera_config.get("target_format") or "auto").lower()
        self._quantization_dataset = camera_config.get("quantization_dataset") or None
        self._model_input_size = int(camera_config.get("input_size") or 640)
        self.model = None
        self._model_path = None
        self._quantized = False
        self._load_error = None
        self._class_names = {i: name for i, name in enumerate(self.classes)}
        self._optimizing = False
        super().__init__(camera_config, (640, 480), camera_config.get("grayscale", False))

        self.prepare()

    def _prepare(self):
        """Background preparation: download/convert the YOLO World model
        without blocking construction of the other cameras."""
        self._load_model()

    def get_optimization_options(self) -> dict:
        schema = self.config_schema()
        return {
            key: schema[key]
            for key in ("quantize", "target_format", "quantization_dataset", "input_size")
            if key in schema
        }

    def optimize(self, **kwargs) -> str:
        """Start a forced backend build of this camera's YOLO World model
        as a background job (generic entry point over request_optimize())."""
        return self.request_optimize()

    def request_optimize(self) -> str:
        """Force-rebuild the quantized backend artifact (re-downloading or
        re-reparameterizing weights if needed) without blocking. No-op if a
        rebuild is already running. Never blocks."""
        if not self.quantize:
            return "optimization disabled for this camera (set 'Quantize' in camera settings)"
        if getattr(self, "_optimizing", False):
            return "optimizing (backend build)"
        if self._preparing():
            return "initializing (model preparation in progress)"
        self._optimizing = True
        self._set_status("optimizing (backend build)")
        thread = threading.Thread(
            target=self._optimize_worker,
            daemon=True,
            name="Optimize-YoloWorld",
        )
        thread.start()
        return "optimizing (backend build)"

    def _optimize_worker(self):
        try:
            self._load_quantized_model(force=True)
        except Exception as exc:
            self._set_load_error(f"quantized rebuild failed: {exc}")
        finally:
            self._optimizing = False
        if self.model is not None and self._quantized:
            self._load_error = None
            self._set_status("ready")
        else:
            self._set_status("error: quantized build failed - using fallback")

    def _ensure_world_model(self, size: str) -> str | None:
        """Download YOLO World weights into the iSpy cache on first use."""
        url = _WORLD_MODEL_URLS.get(size, _WORLD_MODEL_URLS["s"])
        filename = url.rsplit("/", 1)[-1]
        target = _WORLD_MODEL_CACHE / filename

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
        fixed = _WORLD_MODEL_CACHE / "world" / f"{Path(weights).stem}-{classes_key}.pt"
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
        if self.quantize:
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

        if self.model is None:
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
