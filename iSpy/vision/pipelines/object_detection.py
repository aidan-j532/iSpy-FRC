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
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision import triangulation

# In-process registry of per-camera optimization jobs, keyed by camera name.
# Mirrors the state-in-dict pattern used by service_daemon.VisionSupervisor:
# only the status string and the worker thread live here.
_OPTIMIZE_JOBS: dict[str, dict] = {}
_OPTIMIZE_LOCK = threading.Lock()

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
            self.camera_height = camera_config["height"]
            self.camera_x = camera_config["x"]
            self.camera_y = camera_config["y"]
            self.camera_z = camera_config["z"]
        except KeyError as e:
            raise ValueError(f"Missing camera config key: {e}")

        # Model architecture fields come exclusively from metadata sidecars,
        # not from config.  Config holds only user-preference fields.  The
        # vision_model block is per-camera (migrated from the legacy top-level
        # key in a prior pass).
        from iSpy.vision.ModelInspector import fill_missing_config
        vm_cfg = camera_config.get("vision_model")
        if not isinstance(vm_cfg, dict) or not vm_cfg:
            raise RuntimeError(
                f"Camera '{self._cam_name}' uses pipeline 'object_detection' "
                "but has no per-camera vision_model block."
            )
        # Schema fields set via the config UI live at the camera level; merge
        # them into the model config so they drive loading, readiness gating
        # and optimization identically to the nested vision_model keys.
        for _k in (
            "quantized", "min_conf", "target_format", "input_size",
            "quantization_dataset", "auto_opt",
        ):
            if _k in camera_config.data:
                vm_cfg[_k] = camera_config.data[_k]
        camera_config.data["vision_model"] = vm_cfg
        vm_filled = fill_missing_config(dict(vm_cfg))
        self.margin = vm_filled.get("margin", vm_cfg.get("margin", 0))
        raw_min_conf = vm_filled.get("min_conf", vm_cfg.get("min_conf", 0.5))
        self.min_confidence = float(raw_min_conf) if raw_min_conf is not None else 0.5
        self.z_mode = vm_cfg.get("z_mode", "size_based")  # "size_based" | "ground_plane"
        self.yolo_model_file = vm_filled.get("file_path", vm_cfg.get("file_path", ""))
        self.input_size = tuple(vm_filled.get("input_size", (640, 640)))
        self.quantized = vm_filled.get("quantized", vm_cfg.get("quantized", False))
        cam_auto_opt = vm_cfg.get("auto_opt")
        if cam_auto_opt is None:
            cam_auto_opt = config.get("auto_opt", False) if config is not None else False
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

        # The pipeline kicks off its own readiness work: if the config
        # requests quantization/optimization and the optimized artifact
        # isn't loaded yet, start the background RKNN build right here so
        # is_ready() reports "optimizing (rknn build)" and boot only has to
        # wait for it to finish.
        if self._optimization_requested() and not self._optimized_active():
            self.logger.info(
                "Camera '%s': quantization requested - starting background "
                "RKNN build", self._cam_name,
            )
            self.request_optimize()

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
            "auto_opt": {
                "type": "toggle",
                "label": "Optimize",
                "default": False,
                "optimize_toggle": True,
                "help": "Build the best optimized backend artifact for this device "
                        "(rknn on Rockchip NPU, engine on NVIDIA, onnx elsewhere, "
                        "etc.) in the background. Falls back to the top-level "
                        "config 'auto_opt' when unset.",
            },
            "target_format": {
                "type": "select",
                "label": "Target format",
                "options": ["auto", "onnx", "rknn", "tflite", "openvino", "engine", "coreml"],
                "default": "auto",
                "quantization": True,
                "help": "'auto' picks the best backend for this device via "
                        "recommend_format(). Set an explicit format to override.",
            },
            "quantized": {
                "type": "toggle",
                "label": "Quantize model",
                "default": False,
                "quantization": True,
                "help": "Quantize the optimized artifact (int8). Only meaningful "
                        "with auto_opt or target_format set.",
            },
            "quantization_dataset": {
                "type": "browse",
                "label": "Quantization dataset",
                "default": "",
                "browse_root": "QuantizeDataset",
                "quantization": True,
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
        """Best backend for this device (what 'auto' resolves to)."""
        try:
            from iSpy.config.AutoOpt import recommend_format

            return recommend_format(ignore_dependencies=True)
        except Exception:
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
        """Effective conversion backend: explicit target_format if set (and not
        'auto'), otherwise the old boot recommend_format() detection."""
        explicit = str(getattr(self, "_requested_format", "") or "").strip().lower()
        if explicit and explicit != "auto":
            target = explicit
        else:
            target = self.recommended_format()
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

    def _optimizing_status(self) -> str:
        return f"optimizing ({self._target_format_cached()} build)"

    def _has_fallback(self) -> bool:
        """True if ANY loadable model artifact exists for this camera's
        configured vision_model (file_path or source_pt) - including a raw
        .pt that runs on CPU/ONNX. Purely filesystem-level; never converts."""
        vm = self.config.get("vision_model") or {}
        for key in ("file_path", "source_pt"):
            p = self._resolve_model_path(str(vm.get(key, "")))
            if p is not None and p.exists():
                return True
        return False

    def _model_file_is_pt(self) -> bool:
        return str(getattr(self, "yolo_model_file", "")).lower().endswith(".pt")

    def _current_job(self) -> dict | None:
        with _OPTIMIZE_LOCK:
            return _OPTIMIZE_JOBS.get(self._cam_name)

    def _optimization_requested(self) -> bool:
        # Quantize flag OR the old boot auto_opt behavior (camera-level
        # auto_opt, falling back to the top-level config auto_opt).
        if bool(getattr(self, "quantized", False)):
            return True
        if getattr(self, "_auto_opt", False):
            return True
        vm = self.config.get("vision_model") or {}
        return bool(vm.get("auto_opt")) or bool(vm.get("quantized"))

    def _optimized_active(self) -> bool:
        """True once the loaded model is a backend artifact (not a bare .pt),
        or the TPU backend which consumes .pt via XLA."""
        if getattr(self, "model", None) is None:
            return False
        if getattr(self.model, "model_type", "") == "tpu":
            return True
        path = str(getattr(self, "yolo_model_file", "") or "")
        return bool(path) and not path.lower().endswith(".pt")

    def is_ready(self) -> tuple[bool, str]:
        ready, status = self._readiness()
        self._set_status(status)
        return ready, status

    def _readiness(self) -> tuple[bool, str]:
        job = self._current_job()
        in_flight = job is not None and job["thread"].is_alive()

        # Optimization configured: only "ready" once the optimized build
        # finishes. Kicks the build off as a background job if none is
        # running. A finished-but-failed build still leaves the camera
        # usable on its fallback, reported with the error status.
        if self._optimization_requested():
            if self._optimized_active():
                return True, "ready"
            if in_flight:
                return False, self._optimizing_status()
            if job is not None and job["status"].startswith("error: optimize failed"):
                return True, job["status"]
            if self.model is None and not self._has_fallback():
                return False, "error: no model configured/found"
            status = self.request_optimize()
            if status.startswith("optimizing"):
                return False, status
            return True, status

        if in_flight:
            if self.model is not None or self._has_fallback():
                return True, self._optimizing_status()
            return False, self._optimizing_status()

        if self.model is not None:
            if job is not None and job["status"].startswith("error: optimize failed"):
                return True, job["status"]
            if job is not None and job["status"] == "ready":
                return True, "ready"
            if self._model_file_is_pt():
                return True, "using unoptimized .pt fallback"
            return True, "ready"

        if job is not None and job["status"].startswith("error: optimize failed"):
            if self._has_fallback():
                return True, job["status"]
            return False, job["status"]

        if self._has_fallback():
            return True, "using unoptimized .pt fallback"
        return False, "error: no model configured/found"

    def get_optimization_options(self) -> dict:
        schema = self.config_schema()
        return {
            key: schema[key]
            for key in ("auto_opt", "target_format", "quantized", "quantization_dataset")
            if key in schema
        }

    def optimize(self, **kwargs) -> str:
        """Start a backend build of this camera's source .pt as a background
        job (generic entry point over request_optimize())."""
        return self.request_optimize()

    def request_optimize(self) -> str:
        """Start a backend build (rknn/onnx/engine/...) of this camera's
        source .pt as a background job (via the existing boot conversion
        subprocess). Target format comes from target_format or
        recommend_format(). No-op if a job is already in flight for this
        camera. Never blocks."""
        with _OPTIMIZE_LOCK:
            existing = _OPTIMIZE_JOBS.get(self._cam_name)
            if existing is not None and existing["thread"].is_alive():
                return existing["status"]

            vm_cfg = self.config.get("vision_model") or {}
            source_pt = vm_cfg.get("source_pt") or vm_cfg.get("file_path")
            pt_path = self._resolve_model_path(str(source_pt or ""))
            if pt_path is None or not pt_path.exists() or pt_path.suffix.lower() != ".pt":
                return "error: optimize failed - source .pt not found, using fallback"
            if self.model is None and not self._has_fallback():
                return "error: no model configured/found"

            job = {"status": self._optimizing_status(), "thread": None}
            thread = threading.Thread(
                target=self._optimize_worker,
                args=(str(pt_path),),
                daemon=True,
                name=f"Optimize-{self._cam_name}",
            )
            job["thread"] = thread
            _OPTIMIZE_JOBS[self._cam_name] = job
            thread.start()
            return job["status"]

    def _optimize_worker(self, pt_path: str):
        try:
            from iSpy.vision.optimizer import _convert_model_subprocess

            input_size = list(getattr(self, "input_size", (640, 640)))
            vm_cfg = self.config.get("vision_model") or {}
            dataset = vm_cfg.get("quantization_dataset") or None
            target = self._target_format_cached()

            if target == "tpu":
                # TPU consumes the .pt directly via torch_xla at runtime, but
                # only when GenericYolo loads it with device="tpu" (model_type
                # "tpu"). Without that it loads as a plain CPU/GPU YOLO and
                # _optimized_active() stays False, so is_ready() would keep
                # restarting the "optimize" thread on every poll. vm_extra
                # pushes device="tpu" into the runtime load AND the persisted
                # config for subsequent boots.
                self.logger.info(
                    "Camera '%s': TPU backend - keeping .pt, no conversion.",
                    self._cam_name,
                )
                self._activate_optimized_model(pt_path, vm_extra={"device": "tpu"})
                self._set_optimize_status("ready (tpu)")
                return

            converted = _convert_model_subprocess(
                pt_path,
                target,
                input_size,
                quantize=bool(self.quantized),
                force=True,
                dataset_path=dataset,
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
            self._set_optimize_status("ready")
        except Exception as exc:
            self.logger.exception(
                "Optimization failed for camera '%s': %s", self._cam_name, exc
            )
            self._set_optimize_status(
                f"error: optimize failed - {exc}, using fallback"
            )

    def _set_optimize_status(self, status: str):
        with _OPTIMIZE_LOCK:
            job = _OPTIMIZE_JOBS.get(self._cam_name)
            if job is not None:
                job["status"] = status

    def _current_vm_config(self) -> dict:
        vm = self.config.get("vision_model")
        if isinstance(vm, dict):
            return json.loads(json.dumps(vm))
        return {
            "file_path": getattr(self, "yolo_model_file", ""),
            "input_size": list(getattr(self, "input_size", (640, 640))),
        }

    def _activate_optimized_model(self, artifact_path: str, vm_extra: dict | None = None):
        """Swap this camera onto the freshly built backend artifact: reload the
        model, start the pipeline worker if needed, and persist the per-camera
        file_path. Raises RuntimeError if the optimized model fails to load -
        the camera then stays on its fallback. vm_extra is merged into the
        persisted vision_model (e.g. {"device": "tpu"}) so the artifact is
        loaded the same way on subsequent boots."""
        vm = self._current_vm_config()
        vm["file_path"] = artifact_path
        vm["quantized"] = True
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
        self.quantized = True
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

        vm_out = self.config.get("vision_model")
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
                    entry_vm = entry.get("vision_model")
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

            if not self.frame_sync and not self._preproc_q.empty():
                time.sleep(0.001)
                continue

            last_ts = ts
            orig_shape = frame.shape
            preprocessed = self.model._preprocess_frame(frame)
            self._preproc_q.put((preprocessed, frame, orig_shape))

    def _focal_length_px_fov(self, img_w: int) -> float:
        # FOV-derived intrinsic - independent of any specific game piece's
        # known size, unlike self.focal_length_pixels.
        if self.fov and self.fov > 0:
            return (img_w / 2.0) / math.tan(math.radians(self.fov / 2.0))
        return self.focal_length_pixels  # fallback if FOV isn't calibrated

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
        # Unified depth model: cast the bottom-center ray and intersect it
        # with the ground plane (objects are assumed to sit on the ground).
        # This is geometrically exact and consistent for every pitch; the
        # old size-based two-zone heuristic is kept only as a fallback for
        # the degenerate case where the ray runs parallel to the ground.
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
        # solvePnP output is in the units of `pnp.object_points`, which the
        # config documents as meters (the ~1.8 m tall COCO skeleton).  The rest
        # of the pipeline (camera offsets, _camera_point_to_robot) works in the
        # codebase's internal inch convention, so convert meters -> inches
        # first.  Without this a 3 m deep object/keypoint set renders ~39x too
        # small - a skeleton the size of a speck of dots in the viewer.
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
            # Both depth models assume the object sits on the ground, so the
            # reported z is the ground-plane intersection (0). z_mode is kept
            # as an accepted config key for backwards compatibility; both
            # "size_based" and "ground_plane" now use the same ray+plane math.
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
        data, frame = self.get_yolo_data()
        if data is None or frame is None:
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
        return objects, frame
 
    def run_with_supplied_data(self, data: Results) -> list[Object]:
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