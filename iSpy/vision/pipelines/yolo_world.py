import hashlib
import json
import logging
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
import requests

from iSpy.vision.pipelines.base import BackgroundPreparedPipeline
from iSpy.vision.pipelines.optimizable import OptimizableModelPipeline, SUPPORTED_TARGET_FORMATS
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision.Object import Object

_WORLD_MODEL_DIR = Path(__file__).resolve().parents[3] / "YoloModels" / "pytorch"

_WORLD_MODEL_URLS = {
    "s": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt",
    "m": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m-worldv2.pt",
    "l": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8l-worldv2.pt",
}


class YoloWorldPipeline(OptimizableModelPipeline, BackgroundPreparedPipeline):
    plugin_name = "yolo_world"
    # zero-shot detection needs no intrinsics calibration - disable the default tab
    calibration_sections = []

    _OPT_OPTIONS_EXTRA = ("input_size",)

    @classmethod
    def show_calibration(cls) -> bool:
        return False

    def is_ready(self) -> tuple[bool, str]:
        # pure status report - never triggers/blocks on optimization
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
        schema = {
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
        }
        schema.update(cls._optimization_schema(
            target_formats=("auto", "onnx", "rknn", "tflite", "openvino", "engine", "coreml"),
            input_size_default=640,
        ))
        return schema

    @staticmethod
    def _parse_classes(prompt: str) -> list[str]:
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

        # same name the mixin contract expects (_requested_format); the raw
        # setting ("auto" allowed) - resolution happens lazily via
        # _target_format_cached()
        self._requested_format = str(camera_config.get_pipeline_setting("target_format") or "auto").lower()
        self._quantization_dataset = camera_config.get_pipeline_setting("quantization_dataset") or None
        self._model_input_size = int(camera_config.get_pipeline_setting("input_size") or 640)

        raw_optimize = camera_config.get_pipeline_setting("optimize")
        if raw_optimize is None:
            raw_optimize = camera_config.get_pipeline_setting("auto_opt")  # legacy key
        if raw_optimize is None:
            raw_optimize = False
        self._auto_opt = self._normalize_auto_opt(raw_optimize)
        self.model = None
        self._model_path = None
        self._quantized = False
        self._load_error = None
        self._class_names = {i: name for i, name in enumerate(self.classes)}
        self._optimizing = False
        self._optimize_error: str | None = None
        self._target_format: str | None = None
        super().__init__(camera_config, (640, 480), camera_config.get("grayscale", False))

        # file_path drift guard - see OptimizableModelPipeline._resync_stale_model_file_path
        if self._optimization_requested():
            self._resync_stale_model_file_path(config)

        # optimization requested + no active artifact yet -> kick off the
        # build on a bg thread so the app keeps running
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
        self._load_model()

    def optimize(self, **kwargs) -> str:
        if not self._optimization_requested():
            return "optimization disabled for this camera (set 'Quantize' in camera settings)"
        if self._optimizing:
            return "optimizing"

        if kwargs.pop("force", False):
            return self._optimize_forced()

        self._optimizing = True
        self._set_status("optimizing (backend build)")
        try:
            # reuse a cached artifact matching the target format;
            # only a full rebuild re-exports and re-quantizes
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

    def _resolve_target_format(self) -> str:
        explicit = str(getattr(self, "_requested_format", "") or "").strip().lower()
        auto_opt_fmt = self._auto_opt_to_target_format()
        if explicit and explicit != "auto":
            target = explicit
        elif auto_opt_fmt:
            target = auto_opt_fmt
        else:
            # same resolution ensure_quantized_model uses so readiness
            # agrees with the artifact that actually gets built
            # (deliberately NOT ignore_dependencies - unlike the other
            # model-backed pipelines, a dependency-less guess here could
            # disagree with the artifact ensure_quantized_model builds)
            from iSpy.config.AutoOpt import recommend_format
            target = recommend_format()
        if target not in SUPPORTED_TARGET_FORMATS:
            self.logger.warning(
                "Recommended target format %r unsupported - using onnx", target,
            )
            return "onnx"
        return target

    def _optimized_active(self) -> bool:
        if getattr(self, "model", None) is None or not getattr(self, "_quantized", False):
            return False
        if getattr(self.model, "model_type", "") == "tpu":
            return True
        path = str(getattr(self, "_model_path", "") or "")
        if not path:
            return False
        return self._path_format(path) == self._target_format_cached()

    # ------------------------------------------------------------------
    # stale-artifact resync hooks (see OptimizableModelPipeline)
    #
    # yolo_world derives every model path from config at boot (model_size
    # -> weights -> class-hashed reparameterization -> artifact), so there
    # is no persisted file_path that can drift the way object_detection's
    # vision_model.file_path can. The hooks exist so the boot guard is
    # callable on every model-backed pipeline instead of raising
    # AttributeError if a future refactor adds persisted state.
    # ------------------------------------------------------------------

    @property
    def yolo_model_file(self):
        # mixin-facing alias: the artifact currently loaded/selected
        return getattr(self, "_model_path", None)

    @yolo_model_file.setter
    def yolo_model_file(self, value):
        self._model_path = value

    def _resolve_model_path(self, path: str) -> Path | None:
        if not path:
            return None
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[3] / p
        return p

    def _source_model_path(self) -> Path | None:
        # authoritative source = the reparameterized .pt for the CURRENT
        # model_size/class set. Existence-check only - never download or
        # build here; nothing on disk yet means nothing to sync against.
        asset = Path(__file__).resolve().parents[2] / "assets" / "yolo-world.pt"
        weights = asset if asset.exists() else None
        if weights is None:
            url = _WORLD_MODEL_URLS.get(self.model_size, _WORLD_MODEL_URLS["s"])
            downloaded = _WORLD_MODEL_DIR / url.rsplit("/", 1)[-1]
            if downloaded.exists() and downloaded.stat().st_size >= 1024:
                weights = downloaded
        if weights is None:
            return None
        classes_key = hashlib.sha1("|".join(self.classes).encode("utf-8")).hexdigest()[:8]
        fixed = _WORLD_MODEL_DIR / "world" / f"{weights.stem}-{classes_key}.pt"
        if fixed.exists() and fixed.stat().st_size >= 1024:
            return fixed
        return None

    def _persist_file_path(self, file_path: str, config: iSpyConfig | None):
        # no model path is stored in config - loads are re-derived from
        # model_size/classes on every boot, so there is nothing to write;
        # the mixin's in-memory correction via the yolo_model_file alias
        # already keeps the running session consistent.
        pass

    def _ensure_world_model(self, size: str) -> str | None:
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
        asset_path = Path(__file__).resolve().parents[2] / "assets" / "yolo-world.pt"
        if asset_path.exists():
            self.logger.info("Using bundled YOLO World weights at %s", asset_path)
            return str(asset_path)
        return self._ensure_world_model(self.model_size)

    def _reparameterize_world(self, weights: str) -> str | None:
        # NOTE: This is an optional *build-time* step (like the optimizer's
        # exporter). It bakes the text prompt's class vocabulary into a
        # fixed-vocab .pt. The produced .pt is a plain fixed-vocab detector
        # that the on-device loader (load_yolo_pt) consumes at runtime WITHOUT
        # any Ultralytics dependency; Ultralytics is only required in a build
        # environment to produce the bundled/fixed weights, never for iSpy
        # runtime inference. That AGPL build step is isolated in a subprocess
        # (_yoloworld_reparam_worker) so the network-serving process never
        # imports ultralytics.
        classes_key = hashlib.sha1("|".join(self.classes).encode("utf-8")).hexdigest()[:8]
        fixed = _WORLD_MODEL_DIR / "world" / f"{Path(weights).stem}-{classes_key}.pt"
        if fixed.exists() and fixed.stat().st_size >= 1024:
            return str(fixed)

        try:
            self._reparameterize_world_subprocess(weights, fixed)
        except Exception as exc:  # pragma: no cover - runtime dependency fallback
            self.logger.exception("Failed to reparameterize YOLO World model: %s", exc)
            return None

        if not (fixed.exists() and fixed.stat().st_size >= 1024):
            self.logger.error(
                "Reparameterized YOLO World model missing/empty at %s", fixed,
            )
            return None

        self.logger.info("Reparameterized YOLO World model -> %s (classes=%s)", fixed, self.classes)
        return str(fixed)

    def _reparameterize_world_subprocess(self, weights: str, output: Path) -> None:
        # Mirror the optimizer's _convert_model_subprocess: write a JSON args
        # file, run the isolated worker, read back <args>.result.json. All
        # ultralytics imports live inside the subprocess.
        outputs_dir = Path(__file__).resolve().parents[3] / "Outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        args = {
            "weights": weights,
            "classes": list(self.classes),
            "output_path": str(output),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=str(outputs_dir), encoding="utf-8"
        ) as f:
            args_path = f.name
            json.dump(args, f)

        result_path = args_path + ".result.json"
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "iSpy.boot._yoloworld_reparam_worker", args_path],
                cwd=str(Path(__file__).resolve().parents[3]),
                capture_output=True,
                text=True,
            )
            if proc.stderr:
                self.logger.debug("yoloworld reparam worker stderr: %s", proc.stderr.strip())
            if proc.returncode != 0 or not Path(result_path).exists():
                detail = ""
                try:
                    with open(result_path, encoding="utf-8") as rf:
                        detail = json.load(rf).get("error", "")
                except (OSError, ValueError):
                    detail = proc.stderr.strip()
                raise RuntimeError(
                    f"yolo_world reparameterization subprocess failed (exit {proc.returncode}): {detail}"
                )
            with open(result_path, encoding="utf-8") as f:
                result = json.load(f)
            if "error" in result:
                raise RuntimeError(f"yolo_world reparameterization failed: {result['error']}")
            if result.get("result") != str(output):
                raise RuntimeError(
                    f"yolo_world reparameterization produced unexpected path {result.get('result')}"
                )
        finally:
            for p in (args_path, result_path):
                try:
                    Path(p).unlink()
                except OSError:
                    pass

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
        weights = self._resolve_weights()
        if not weights:
            self._set_load_error(
                "failed to resolve YOLO World weights (bundled asset missing "
                "and download failed)"
            )
            self.model = None
            return

        try:
            # bake classes into the weights and load the result as a plain
            # fixed-vocab detector. the on-device loader (load_yolo_pt) never
            # touches Ultralytics - only the optional build-time
            # _reparameterize_world step does.
            fixed = self._reparameterize_world(weights)
            if not fixed:
                self._set_load_error("failed to reparameterize YOLO World model")
                self.model = None
                return

            from iSpy.vision.genericYolo import GenericYolo
            self.model = GenericYolo(
                {
                    "file_path": fixed,
                    "task": "detect",
                    "input_size": [self._model_input_size, self._model_input_size],
                    "min_conf": 0.25,
                },
                self.core_mask,
                iSpy_config=self._ispy_config,
            )
            self._model_path = fixed
            self._quantized = False
            self._load_error = None

            from iSpy.vision.metadata import read_metadata
            meta = read_metadata(Path(fixed))
            if meta and isinstance(meta.get("names"), dict):
                self._class_names = {int(k): str(v) for k, v in meta["names"].items()}
            else:
                self._class_names = {i: name for i, name in enumerate(self.classes)}
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
            # cached resolved format (not the raw "auto" setting) so the
            # artifact built here is always the one _optimized_active()
            # expects to find
            artifact, converted = ensure_quantized_model(
                fixed,
                self._target_format_cached(),
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
            # self.model is always a GenericYolo (both full-precision and
            # quantized paths) whose predict() returns Results with iterable
            # boxes + plot(); no Ultralytics objects are ever involved.
            results = self.model.predict(frame, orig_shape=frame.shape)
            objects: list[Object] = []
            annotated = frame.copy()
            if results is not None:
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
                                "quantized": self._quantized,
                            },
                        )
                    )
                if getattr(results, "boxes", None) is not None and len(results.boxes) > 0:
                    drawn = results.plot(frame.copy())
                    if drawn is not None:
                        annotated = drawn
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

