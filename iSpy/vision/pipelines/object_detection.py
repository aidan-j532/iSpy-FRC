from pathlib import Path
from iSpy.vision.Object import Object
import cv2
import math
import numpy as np
import time
import logging
import threading
import queue
import json
from iSpy.vision.pipelines.base import VisionPipeline
from iSpy.vision.genericYolo import Box, Results, GenericYolo, ModelFileError
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig, get_pipeline_settings, unit_to_inches
from iSpy.vision import triangulation
from iSpy.vision import calibration as cam_calibration

class ObjectDetectionCamera(VisionPipeline):
    plugin_name = "object_detection"

    def __init__(
        self,
        camera_config: iSpyCameraConfig,
        config: iSpyConfig,
        core_mask=None,
    ):
        self.logger = logging.getLogger(__name__)

        self.config = camera_config
        self._ispy_config = config
        self._cam_name = camera_config.get("name", "?")
        self._preproc_thread: threading.Thread | None = None

        try:
            self.known_calibration_distance = camera_config["calibration"]["distance"]
            self.ball_d_inches = camera_config["calibration"]["game_piece_size"]
            self.known_calibration_pixel_height = camera_config["calibration"]["size"]
            self.fov = camera_config["calibration"]["fov"]
            self.grayscale = camera_config.get("grayscale", False)
            self.subsystem = camera_config["subsystem"]

            self.camera_bot_relative_yaw = camera_config["yaw"]
            self.camera_pitch_angle = camera_config["pitch"]

            _pos_unit = config.get("unit", "frc")
            self.camera_height = unit_to_inches(camera_config["height"], _pos_unit)
            self.camera_x = unit_to_inches(camera_config["x"], _pos_unit)
            self.camera_y = unit_to_inches(camera_config["y"], _pos_unit)
            self.camera_z = unit_to_inches(camera_config.get("z", 0.0), _pos_unit)
        except KeyError as e:
            raise ValueError(f"Missing camera config key: {e}")

        # Model architecture fields come only from metadata sidecars, not config.
        # Config holds user-preference fields; vision_model is per-camera
        # pipeline settings (migrated from the old top-level key).
        from iSpy.vision.ModelInspector import fill_missing_config
        vm_cfg = camera_config.get_pipeline_setting("vision_model")
        if not isinstance(vm_cfg, dict) or not vm_cfg:
            raise RuntimeError(
                f"Camera '{self._cam_name}' uses pipeline 'object_detection' "
                "but has no vision_model block in its pipeline settings."
            )
        # schema fields set via the config UI live in pipeline settings; merge
        # them into the model config. Merge on a copy so the persisted
        # vision_model keeps only model identity - otherwise every setting
        # gets duplicated on the next save.
        vm_cfg = dict(vm_cfg)
        for _k in (
            "quantize", "min_conf", "target_format", "input_size",
            "quantization_dataset", "optimize",
        ):
            _v = camera_config.get_pipeline_setting(_k)
            if _v is None:
                _legacy = {"quantize": "quantized", "optimize": "auto_opt"}.get(_k)
                if _legacy is not None:
                    _v = camera_config.get_pipeline_setting(_legacy)
            if _v is not None:
                vm_cfg[_k] = _v
        vm_filled = fill_missing_config(dict(vm_cfg))
        self.margin = vm_filled.get("margin", vm_cfg.get("margin", 0))
        raw_min_conf = vm_filled.get("min_conf", vm_cfg.get("min_conf", 0.5))
        self.min_confidence = float(raw_min_conf) if raw_min_conf is not None else 0.5
        self.z_mode = vm_cfg.get("z_mode", "size_based")  # "size_based" | "ground_plane"
        self.yolo_model_file = vm_filled.get("file_path", vm_cfg.get("file_path", ""))
        self.input_size = tuple(vm_filled.get("input_size", (640, 640)))
        raw_quantize = vm_filled.get("quantize", vm_cfg.get("quantize"))
        if raw_quantize is None:
            raw_quantize = vm_filled.get("quantized", vm_cfg.get("quantized"))
        if raw_quantize is None:
            raw_quantize = False
        self.quantize = bool(raw_quantize)
        cam_auto_opt = vm_cfg.get("optimize")
        if cam_auto_opt is None:
            cam_auto_opt = vm_cfg.get("auto_opt")  # legacy key
        if cam_auto_opt is None:
            cam_auto_opt = (
                config.get("optimize", config.get("auto_opt", False))
                if config is not None
                else False
            )
        self._auto_opt = bool(cam_auto_opt)
        self._requested_format = str(vm_cfg.get("target_format") or "auto")
        self.quantization_dataset = vm_cfg.get("quantization_dataset") or None
        self._target_format: str | None = None
        self.frame_sync = config.get("frame_sync", False)
        if self.frame_sync:
            self.logger.warning("Frame sync is enabled. This may introduce latency in detection (you probaly don't want this).")
            
        self.core_mask = core_mask
        self.unit = config["unit"]
        self.debug_mode = config["debug_mode"]
        self.gui_available = False

        self.conversions = {
            "meter": 0.0254,
            "meters": 0.0254,
            "inch": 1.0,
            "inches": 1.0,
            "foot": 1 / 12,
            "feet": 1 / 12,
            "centimeter": 2.54,
            "centimeters": 2.54,
            # FRC/WPILib convention: meters out (robot code), calibration in inches
            "frc": 0.0254,
        }

        try:
            if self.known_calibration_pixel_height <= 0 or self.known_calibration_distance <= 0:
                self.logger.info("Calibration values must be positive, defaulting focal length to 1")
                self.focal_length_pixels = 1.0
            else:
                self.focal_length_pixels = (
                    self.known_calibration_pixel_height * self.known_calibration_distance
                ) / self.ball_d_inches
        except ZeroDivisionError:
            self.logger.warning(
                "Calibration game_piece_size is 0, defaulting focal length to 1"
            )
            self.focal_length_pixels = 1.0

        super().__init__(camera_config, self.input_size, self.grayscale)

        # file_path can drift from source_pt when the model was re-picked in
        # the UI: it may point at a stale artifact built for an older model,
        # which would silently keep that old model running. Trust the source -
        # load its already-built artifact when one exists, else the .pt itself
        # (the background optimizer swaps file_path to the fresh artifact once
        # its build lands).
        if self._optimization_requested():
            source = self._source_model_path()
            current = str(self.yolo_model_file or "")
            current_stem = Path(current).stem if current else None
            source_stem = source.stem if source is not None else None
            if source is not None and current and current_stem != source_stem:
                from iSpy.vision.optimizer import existing_artifact_for
                artifact = existing_artifact_for(source, self._target_format_cached())
                preferred = artifact or str(self._resolve_model_path(source) or source)
                if self._resolve_model_path(current) != self._resolve_model_path(preferred):
                    self.logger.warning(
                        "Camera '%s': vision_model.file_path (%s) doesn't match "
                        "source_pt (%s) - correcting to %s and persisting.",
                        self._cam_name, current, source, preferred,
                    )
                    vm_filled["file_path"] = preferred
                    self.yolo_model_file = preferred
                    self._persist_file_path(preferred, config)
        try:
            self.model = GenericYolo(vm_filled, self.core_mask, iSpy_config=config)
        except ModelFileError as e:
            self.logger.error(
                "Camera '%s': %s — this camera will run without detection until fixed.",
                camera_config.get("name", "?"), e,
            )
            self.model = None

        self._class_names: dict[int, str] = {0: "object"}
        try:
            from iSpy.vision.metadata import read_metadata
            meta = read_metadata(Path(self.yolo_model_file))
            if meta and isinstance(meta.get("names"), dict):
                self._class_names = {int(k): str(v) for k, v in meta["names"].items()}
        except Exception:
            pass

        self._preproc_q: queue.Queue = queue.Queue(maxsize=1)
        self._use_pipeline = self.model is not None and self.model.model_type in ("rknn", "onnx", "tflite")

        self._last_result: Results | None = None
        self._last_frame: np.ndarray | None = None
        self.last_time = time.perf_counter()
        self._pipeline_timeout = 0.1
        self._last_objects: list[Object] = []
        
        if self._use_pipeline:
            self._preproc_thread = threading.Thread(
                target=self._preprocess_worker,
                daemon=True,
                name=f"PreProc-{self.source}",
            )
            self._preproc_thread.start()

        # optimization requested + no active artifact yet -> kick off the
        # build on a bg thread so the app keeps running
        self._optimizing = False
        self._optimize_error: str | None = None
        if self._optimization_requested() and not self._optimized_active():
            self.logger.info(
                "Camera '%s': optimization requested - building %s artifact",
                self._cam_name, self._target_format_cached(),
            )
            threading.Thread(
                target=self._optimize_runner,
                daemon=True,
                name=f"Optimize-{self._cam_name}",
            ).start()

    def _persist_file_path(self, file_path: str, config: iSpyConfig | None):
        vm_out = self.config.get_pipeline_setting("vision_model")
        if isinstance(vm_out, dict):
            vm_out["file_path"] = file_path
        if config is None:
            return
        try:
            cams = config.config.get("camera_configs", {})
            for key, entry in cams.items():
                if not isinstance(entry, dict):
                    continue
                if entry.get("name") == self._cam_name or key == self._cam_name:
                    entry_vm = get_pipeline_settings(entry).get("vision_model")
                    if isinstance(entry_vm, dict):
                        entry_vm["file_path"] = file_path
            config.save(quiet=True)
        except Exception:
            self.logger.exception("Failed to persist corrected vision_model.file_path")

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "vision_model": {
                "type": "model",
                "label": "YOLO Model (.pt)",
                "default": "",
                "help": "Pick a .pt model from the Yolo Models library. The "
                        "chosen file is used for detection and as the source "
                        "for optimization builds.",
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
                "label": "Target format",
                "options": ["auto", "onnx"],
                "default": "auto",
                "quantization": True,
                "help": "'auto' picks the best backend for this device via "
                        "recommend_format(). Set an explicit format to override.",
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
            "min_conf": {
                "type": "number",
                "label": "Min confidence",
                "default": 0.5,
                "step": 0.05,
            },
        }

    @classmethod
    def recommended_format(cls) -> str:
        try:
            from iSpy.config.AutoOpt import recommend_format

            return recommend_format(ignore_dependencies=True)
        except Exception:
            logging.getLogger(__name__).warning(
                "AutoOpt.recommend_format did NOT work for your device, falling back to ONNX!"
            )
            return "onnx"

    @classmethod
    def needs_model_backend(cls) -> bool:
        return True

    @classmethod
    def uses_user_model(cls) -> bool:
        return True

    def _resolve_model_path(self, path: str) -> Path | None:
        if not path:
            return None
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[3] / p
        return p

    def _resolve_target_format(self) -> str:
        explicit = str(getattr(self, "_requested_format", "") or "").strip().lower()
        if explicit and explicit != "auto":
            target = explicit
        else:
            target = self.recommended_format()
            
        # Note right now the user only has access to ONNX for a reason!
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

    def _is_processable(self) -> bool:
        if getattr(self, "_optimizing", False):
            return False
        if self.model is None:
            return False
        if self._optimization_requested():
            return self._optimized_active()
        return True

    def _source_model_path(self) -> Path | None:
        vm = self._current_vm_config()
        source = vm.get("source_pt") or vm.get("file_path")
        p = self._resolve_model_path(str(source or ""))
        if p is not None and p.exists() and p.suffix.lower() == ".pt":
            return p
        return None

    def _optimized_artifact_path(self) -> Path | None:
        src = self._source_model_path()
        if src is None:
            return None
        from iSpy.vision.optimizer import _desired_output_path
        try:
            p = _desired_output_path(src, self._target_format_cached())
        except Exception:
            return None
        if p.exists():
            return p
        return None

    def _optimization_requested(self) -> bool:
        # quantize flag OR the old boot auto_opt behavior; legacy
        # 'quantized'/'auto_opt' keys still honored
        if bool(getattr(self, "quantize", False)):
            return True
        if getattr(self, "_auto_opt", False):
            return True
        vm = self._current_vm_config()
        return bool(vm.get("optimize") or vm.get("auto_opt")) or bool(
            vm.get("quantize") or vm.get("quantized")
        )

    @staticmethod
    def _path_format(path: str) -> str:
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
        if getattr(self, "model", None) is None:
            return False
        if getattr(self.model, "model_type", "") == "tpu":
            return True
        path = str(getattr(self, "yolo_model_file", "") or "")
        if not path or self._path_format(path) != self._target_format_cached():
            return False
        expected = self._optimized_artifact_path()
        if expected is not None:
            return self._resolve_model_path(path) == expected
        # artifact for this source not built yet (or built under a name that
        # no longer matches). A leftover artifact from an older model must be
        # rejected, not trusted, or an old model silently keeps running.
        src = self._source_model_path()
        if src is not None:
            from iSpy.vision.optimizer import _artifact_name
            return Path(path).name == _artifact_name(src, self._target_format_cached())
        return True

    def is_ready(self) -> tuple[bool, str]:
        # pure status report - never triggers/blocks on optimization
        if not self._optimization_requested():
            status = "ready" if self.model is not None else "error: no model configured/found"
            self._set_status(status)
            return self.model is not None, status

        if self._optimizing:
            self._set_status("optimizing")
            return False, "optimizing"

        if self._optimized_active():
            self._set_status("ready")
            return True, "ready"

        status = self._optimize_error or "optimizing"
        self._set_status(status)
        return False, status

    def get_optimization_options(self) -> dict:
        schema = self.config_schema()
        return {
            key: schema[key]
            for key in ("optimize", "target_format", "quantize", "quantization_dataset")
            if key in schema
        }

    def optimize(self, **kwargs) -> str:
        if self._optimizing:
            return "optimizing"

        artifact = self._optimized_artifact_path()
        if artifact is not None:
            try:
                self._activate_optimized_model(str(artifact))
                return "ready"
            except Exception:
                pass  # stale/broken artifact - rebuild below

        source_pt = self._source_model_path()
        if source_pt is None:
            return "error: optimize failed - source .pt not found"

        self._optimizing = True
        try:
            target = self._target_format_cached()
            if target == "tpu":
                # TPU runs the .pt directly via torch_xla, no conversion
                self.logger.info(
                    "Camera '%s': TPU backend - keeping .pt, no conversion.",
                    self._cam_name,
                )
                self._activate_optimized_model(str(source_pt), vm_extra={"device": "tpu"})
                return "ready"

            from iSpy.vision.optimizer import _convert_model_subprocess

            converted = _convert_model_subprocess(
                str(source_pt),
                target,
                list(self.input_size),
                quantize=bool(self.quantize),
                force=True,
                dataset_path=self.quantization_dataset,
            )
            converted_path = self._resolve_model_path(str(converted or ""))
            if (
                converted_path is None
                or not converted_path.exists()
                or converted_path.suffix.lower() == ".pt"
            ):
                raise RuntimeError(
                    f"{target} build produced no artifact (got '{converted}')"
                )
            self._activate_optimized_model(str(converted_path))
            return "ready"
        except Exception as exc:
            self.logger.exception(
                "Optimization failed for camera '%s': %s", self._cam_name, exc
            )
            return f"error: optimize failed - {exc}"
        finally:
            self._optimizing = False

    def _optimize_runner(self):
        status = self.optimize()
        if not self._optimized_active():
            self._optimize_error = status
        self._set_status(status)

    def _current_vm_config(self) -> dict:
        vm = self.config.get_pipeline_setting("vision_model")
        if isinstance(vm, dict):
            return json.loads(json.dumps(vm))
        return {
            "file_path": getattr(self, "yolo_model_file", ""),
            "input_size": list(getattr(self, "input_size", (640, 640))),
        }

    def _activate_optimized_model(self, artifact_path: str, vm_extra: dict | None = None):
        vm = self._current_vm_config()
        vm["file_path"] = artifact_path
        vm["quantize"] = True
        if vm_extra:
            vm.update(vm_extra)

        from iSpy.vision.ModelInspector import fill_missing_config
        new_model = GenericYolo(
            fill_missing_config(vm),
            self.core_mask,
            iSpy_config=self._ispy_config,
        )
        self.model = new_model
        self.yolo_model_file = artifact_path
        self.quantize = True
        self._use_pipeline = new_model.model_type in ("rknn", "onnx", "tflite")
        if self._use_pipeline and (
            self._preproc_thread is None or not self._preproc_thread.is_alive()
        ):
            self._preproc_thread = threading.Thread(
                target=self._preprocess_worker,
                daemon=True,
                name=f"PreProc-{self.source}",
            )
            self._preproc_thread.start()

        try:
            from iSpy.vision.metadata import read_metadata
            meta = read_metadata(Path(artifact_path))
            if meta and isinstance(meta.get("names"), dict):
                self._class_names = {int(k): str(v) for k, v in meta["names"].items()}
        except Exception:
            pass

        vm_out = self.config.get_pipeline_setting("vision_model")
        if isinstance(vm_out, dict):
            vm_out["file_path"] = artifact_path
            if vm_extra:
                vm_out.update(vm_extra)
        if self._ispy_config is not None:
            cams = self._ispy_config.config.get("camera_configs", {})
            for key, entry in cams.items():
                if not isinstance(entry, dict):
                    continue
                if entry.get("name") == self._cam_name or key == self._cam_name:
                    entry_vm = get_pipeline_settings(entry).get("vision_model")
                    if isinstance(entry_vm, dict):
                        entry_vm["file_path"] = artifact_path
                        if vm_extra:
                            entry_vm.update(vm_extra)
            try:
                self._ispy_config.save(quiet=True)
            except Exception:
                pass
        self.logger.info(
            "Camera '%s': optimized model active at %s", self._cam_name, artifact_path
        )

    def _letterbox(self, img: np.ndarray, target_size: tuple) -> tuple:
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        left = pad_w // 2
        padded = cv2.copyMakeBorder(
            resized,
            top,
            pad_h - top,
            left,
            pad_w - left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return padded, scale, left, top

    def _letterbox_into(
        self, img: np.ndarray, dst: np.ndarray, target_size: tuple
    ) -> None:
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        dst[:] = 114
        dst[top : top + new_h, left : left + new_w] = resized

    def _preprocess_worker(self):
        last_ts = None
        while not self.stopped:
            if self.is_image:
                frame = self.get_frame()
                ts = 0
                
                time.sleep(1 / 100) # Simulate a 100 fps camera
            else:
                with self.frame_lock:
                    frame = self.frame
                    ts = self.frame_timestamp
                if frame is None or ts == last_ts:
                    self._frame_event.wait(timeout=0.05)
                    self._frame_event.clear()
                    continue

            last_ts = ts
            orig_shape = frame.shape
            preprocessed = self.model._preprocess_frame(frame)
            self._preproc_q.put((preprocessed, frame, orig_shape))

    def _focal_length_px_fov(self, img_w: int) -> float:
        # FOV-derived intrinsic - doesnt rely on a game piece's known size.
        # A chessboard-calibrated camera matrix is the most accurate source.
        intr = cam_calibration.intrinsics_for_frame(
            self.config.get("calibration", {}), img_w, 1
        )
        if intr is not None:
            return float(intr[0][0, 0])
        if self.fov and self.fov > 0:
            return (img_w / 2.0) / math.tan(math.radians(self.fov / 2.0))
        return self.focal_length_pixels  # fallback when FOV isnt calibrated

    def _pixel_ray(self, pixel_x: float, pixel_y: float, img_w: int, img_h: int) -> triangulation.Ray:
        f = self._focal_length_px_fov(img_w)
        return triangulation.pixel_to_ray(
            pixel_x, pixel_y, img_w, img_h, f,
            self.camera_x, self.camera_y, self.camera_z,
            self.camera_bot_relative_yaw, self.camera_pitch_angle,
        )

    def _filter_box(self, box: Box, img_w: int, img_h: int) -> bool:
        x1, y1, x2, y2 = box.xyxy
        w_px = x2 - x1
        h_px = y2 - y1
        if (
            x1 < self.margin
            or y1 < self.margin
            or x2 > (img_w - self.margin)
            or y2 > (img_h - self.margin)
        ):
            return False
        if h_px == 0:
            return False
        aspect = w_px / h_px # Aspect is calculate but I won't use it because
        # I want it to continue detections partial objectcs/rectangles
        return True
        # return 0.8 <= aspect <= 1.2

    def _box_to_robot_point(
        self, box: Box, img_w: int, img_h: int
    ) -> np.ndarray | None:
        # unified depth model: cast the bottom-center ray and hit the ground
        # plane (objects assumed to sit on the ground). size-based heuristic
        # is only a fallback for when the ray runs parallel to the ground.
        x1, y1, x2, y2 = box.xyxy
        ray = self._pixel_ray((x1 + x2) / 2.0, y2, img_w, img_h)
        gp = triangulation.ground_plane_intersection(ray, ground_z=0.0)
        if gp is not None:
            scale = self.conversions.get(self.unit, self.conversions["meter"])
            return gp * scale
        return self._size_based_point(box, img_w, img_h)

    def _size_based_point(
        self, box: Box, img_w: int, img_h: int
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = box.xyxy
        avg_px = ((x2 - x1) + (y2 - y1)) / 2.0
        if avg_px <= 0:
            return None
        cx = (x1 + x2) / 2.0
        distance_los = (self.ball_d_inches * self.focal_length_pixels) / avg_px
        return self._pixel_to_robot_coordinates(
            cx, (y1 + y2) / 2.0, distance_los, img_w, img_h
        )

    def _pixel_to_robot_coordinates(
        self,
        pixel_x: float,
        pixel_y: float,
        distance_los: float,
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        pixel_offset_x = pixel_x - img_w / 2.0
        horizontal_angle_rad = math.atan(pixel_offset_x / self.focal_length_pixels)

        if self.camera_height > 0 and distance_los > self.camera_height:
            true_horiz = math.sqrt(distance_los**2 - self.camera_height**2)
        else:
            true_horiz = distance_los * math.cos(math.radians(self.camera_pitch_angle))

        left_right = true_horiz * math.sin(horizontal_angle_rad)
        forward = true_horiz * math.cos(horizontal_angle_rad)

        # +X right, +Y forward, +Z up; yaw 0 = facing +Y, positive yaw turns right.
        yaw_rad = math.radians(self.camera_bot_relative_yaw)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        x_rot = left_right * cos_y + forward * sin_y
        y_rot = forward * cos_y - left_right * sin_y

        scale = self.conversions.get(self.unit, self.conversions["meter"])
        return np.array(
            [(x_rot + self.camera_x) * scale, (y_rot + self.camera_y) * scale],
            dtype=np.float32,
        )

    def get_yolo_data(self) -> tuple[Results | None, np.ndarray | None]:
        if self.model is None:
            frame = self.get_frame()
            return None, frame

        if self._use_pipeline:
            try:
                preprocessed, orig_frame, orig_shape = self._preproc_q.get(
                    timeout=None if self.frame_sync else self._pipeline_timeout
                )
            except queue.Empty:
                return self._last_result, self._last_frame

            results = self.model.predict_preprocessed(preprocessed, orig_shape)
            annotated_frame = orig_frame.copy()
            self._last_result = results
            self._last_frame = annotated_frame
        else:
            frame = self.get_frame()
            if frame is None:
                self.logger.warning("No frame available.")
                return None, None
            clean_frame = frame.copy()  # keep clean copy before prediction
            results = self.model.predict(frame, orig_shape=frame.shape)
            annotated_frame = clean_frame  # use untouched frame
            self._last_result = results
            self._last_frame = annotated_frame

        if annotated_frame is not None:
            annotated_frame = results.plot(annotated_frame.copy())
            if self.debug_mode:
                new_time = time.perf_counter()
                fps = 1 / max(new_time - self.last_time, 1e-6)
                self.last_time = new_time
                cv2.putText(
                    annotated_frame,
                    f"FPS: {int(fps)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                if self.gui_available:
                    cv2.imshow("YOLO Detections", annotated_frame)
                    cv2.waitKey(1)
            self._last_frame = annotated_frame

        return results, annotated_frame

    def _camera_point_to_robot(self, pt: tuple[float, float, float]) -> np.ndarray:
        scale = self.conversions.get(self.unit, self.conversions["meter"])
        return triangulation.camera_point_to_robot(
            pt,
            self.camera_x,
            self.camera_y,
            self.camera_z,
            self.camera_bot_relative_yaw,
            self.camera_pitch_angle,
        ) * scale

    def _pnp_point_to_robot(self, pt: tuple[float, float, float]) -> np.ndarray:
        # solvePnP output is in the units of pnp.object_points (meters, ~1.8m
        # tall COCO skeleton), but the rest of the pipeline works in inches -
        # without this conversion things render ~39x too small
        meters_to_inches = 1.0 / self.conversions["meter"]
        return self._camera_point_to_robot(
            (
                pt[0] * meters_to_inches,
                pt[1] * meters_to_inches,
                pt[2] * meters_to_inches,
            )
        )

    def _pnp_to_robot_coordinates(
        self, tvec: tuple[float, float, float]
    ) -> np.ndarray:
        return self._pnp_point_to_robot(tvec)

    def _box_to_object(self, box: Box, img_w: int, img_h: int, keypoints_2d: np.ndarray | None = None) -> Object | None:
        bottom_x = (box.xyxy[0] + box.xyxy[2]) / 2.0
        bottom_y = box.xyxy[3]
        ray = self._pixel_ray(bottom_x, bottom_y, img_w, img_h)

        depth_source = "monocular"
        if box.translation is not None:
            pt = self._pnp_to_robot_coordinates(box.translation)
        else:
            pt = self._box_to_robot_point(box, img_w, img_h)
            if pt is None:
                return None
            # both depth models assume the object sits on the ground, so z
            # is the ground-plane intersection (0). z_mode kept for back-compat
            if self.z_mode == "ground_plane":
                depth_source = "ground_plane"
            pt = np.array([pt[0], pt[1], 0.0])

        roll, pitch, yaw = 0.0, 0.0, 0.0
        if box.rotation is not None:
            roll, pitch, yaw = box.rotation
        z = float(pt[2]) if len(pt) > 2 else 0.0
        class_name = self._class_names.get(box.cls_id, f"class_{box.cls_id}")

        kpts_3d_robot = None
        if box.keypoints_3d is not None:
            kpts_3d_robot = [self._pnp_point_to_robot(tuple(kpt)).tolist() for kpt in box.keypoints_3d]
        elif keypoints_2d is not None:
            x1, y1, x2, y2 = box.xyxy
            avg_px = ((x2 - x1) + (y2 - y1)) / 2.0
            if avg_px > 0:
                distance_los = (self.ball_d_inches * self.focal_length_pixels) / avg_px
                f = self._focal_length_px_fov(img_w)
                cx = img_w / 2.0
                cy = img_h / 2.0
                kpts_3d_robot = []
                for kp in keypoints_2d:
                    kp_x, kp_y = kp[0], kp[1]
                    fx_cam = (kp_x - cx) * distance_los / f
                    fy_cam = (kp_y - cy) * distance_los / f
                    fz_cam = distance_los
                    rpt = self._camera_point_to_robot((fx_cam, fy_cam, fz_cam))
                    kpts_3d_robot.append(rpt.tolist())

        return Object(
            float(pt[0]), float(pt[1]), z=z,
            roll=roll, pitch=pitch, yaw=yaw,
            name=class_name, confidence=box.conf,
            keypoints_3d=kpts_3d_robot,
            ray_origin=ray.origin, ray_direction=ray.direction,
            depth_source=depth_source,
        )

    def run(self):
        if self.is_image:
            frame = self.get_frame()
        else:
            with self.frame_lock:
                frame = self.frame
        if frame is None:
            return [], None
        if not self._is_processable():
            return [], frame

        data, annotated = self.get_yolo_data()
        if data is None or annotated is None:
            return [], frame

        img_h, img_w = frame.shape[:2]
        objects: list[Object] = []
        kp_iter = iter(data.keypoints) if data.keypoints else None
        for box in data.boxes:
            if not self._filter_box(box, img_w, img_h):
                if kp_iter:
                    next(kp_iter, None)
                continue
            kp = next(kp_iter) if kp_iter else None
            obj = self._box_to_object(box, img_w, img_h, keypoints_2d=kp)
            if obj is not None:
                objects.append(obj)
        self._last_objects = objects
        return objects, annotated
 
    def run_with_supplied_data(self, data: Results) -> list[Object]:
        if not self._is_processable():
            return []
        img_h, img_w = data.orig_shape[:2]
        objects: list[Object] = []
        kp_iter = iter(data.keypoints) if data.keypoints else None
        for box in data.boxes:
            if not self._filter_box(box, img_w, img_h):
                if kp_iter:
                    next(kp_iter, None)
                continue
            kp = next(kp_iter) if kp_iter else None
            obj = self._box_to_object(box, img_w, img_h, keypoints_2d=kp)
            if obj is not None:
                objects.append(obj)
        return objects
 

    def plot(self, frame):
        if frame is None:
            return None
        result = getattr(self, "_last_result", None)
        if result is None:
            return frame
        try:
            return result.plot(frame.copy())
        except Exception:
            return frame

    def get_data_for_subsystem(self, target: str):
        if self.subsystem != target:
            return None
        positions = self._last_objects
        if self.subsystem == "hopper":
            return len(positions) > 0
        return positions

    def get_subsystem(self) -> str:
        return self.subsystem

    def destroy(self):
        self.stopped = True
        if not self.is_image and hasattr(self, "cap") and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def release(self):
        self.destroy()