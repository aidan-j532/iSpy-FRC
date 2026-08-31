import logging
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from iSpy.vision.pipelines.base import BackgroundPreparedPipeline
from iSpy.vision.pipelines.optimizable import OptimizableModelPipeline
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision.Object import Object

_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
_DEPTH_INPUT_SIZE = 518
_DEPTH_ARTIFACT_STEM = "depth_anything_v2_small"
_DEPTH_HF_CACHE_DIR = Path(__file__).resolve().parents[3] / "YoloModels" / "huggingface"
_YOLO_MODELS_DIR = Path(__file__).resolve().parents[3] / "YoloModels"

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class DepthAnythingPipeline(OptimizableModelPipeline, BackgroundPreparedPipeline):
    plugin_name = "depth_anything"
    # monocular depth maps need no calibration - disable the default ChArUco tab
    calibration_sections = []

    _OPT_OPTIONS_EXTRA = ("input_size", "model_size")

    @classmethod
    def show_calibration(cls) -> bool:
        return False

    def is_ready(self) -> tuple[bool, str]:
        # pure status report - never triggers/blocks on optimization
        if not self.estimate_depth:
            self._set_status("ready")
            return True, "ready"

        if not self._optimization_requested():
            if self._preparing():
                self._set_status("downloading (model weights)")
                return False, "downloading (model weights)"
            if self._model is not None:
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

    @classmethod
    def config_schema(cls) -> dict:
        schema = {
            "model_size": {
                "type": "select",
                "label": "Model Size",
                "options": ["small"],
                "default": "small",
                "help": "Depth Anything V2 Small is downloaded automatically from Hugging Face.",
            },
            "estimate_depth": {
                "type": "toggle",
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
        schema.update(cls._optimization_schema(
            target_formats=("auto", "onnx"),
            input_size_default=_DEPTH_INPUT_SIZE,
            input_size_help="Square resolution used for the optimized ONNX export "
                            "and inference.",
        ))
        return schema

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
        self._infer = None
        self._load_error = None
        self._optimizing = False
        self._optimize_error: str | None = None
        self._frame_count = 0
        self._every = 5
        self._last_depth = None
        self._last_objects = []
        self._last_annotated = None

        self.unit = config.get("unit", "meter")
        self.max_depth = float(camera_config.get_pipeline_setting("max_depth", 10.0))
        self.estimate_depth = bool(camera_config.get_pipeline_setting("estimate_depth", True))

        raw_quantize = camera_config.get_pipeline_setting("quantize")
        if raw_quantize is None:
            raw_quantize = camera_config.get_pipeline_setting("quantized")  # legacy key
        if raw_quantize is None:
            raw_quantize = False
        if isinstance(raw_quantize, str):
            raw_quantize = raw_quantize.strip().lower() in ("1", "true", "yes", "on")
        self.quantize = bool(raw_quantize)

        self._requested_format = str(camera_config.get_pipeline_setting("target_format") or "auto").lower()
        self._quantization_dataset = camera_config.get_pipeline_setting("quantization_dataset") or None
        try:
            self._input_size = int(camera_config.get_pipeline_setting("input_size") or _DEPTH_INPUT_SIZE)
        except (TypeError, ValueError):
            self._input_size = _DEPTH_INPUT_SIZE
        self._target_format: str | None = None

        # max_depth is in meters; scale z into the configured unit so
        # every pipeline emits the same unit.
        self._z_scale = {
            "meter": 1.0, "meters": 1.0,
            "inch": 39.37007874, "inches": 39.37007874,
            "foot": 3.280839895, "feet": 3.280839895,
            "centimeter": 100.0, "centimeters": 100.0,
            # FRC/WPILib convention: meters out
            "frc": 1.0,
        }.get(self.unit, 1.0)
        self._unit_label = {
            "meter": "m", "meters": "m",
            "inch": "in", "inches": "in",
            "foot": "ft", "feet": "ft",
            "centimeter": "cm", "centimeters": "cm",
            "frc": "m",
        }.get(self.unit, self.unit)

        raw_optimize = camera_config.get_pipeline_setting("optimize")
        if raw_optimize is None:
            raw_optimize = camera_config.get_pipeline_setting("auto_opt")  # legacy key
        if raw_optimize is None:
            raw_optimize = False
        self._auto_opt = self._normalize_auto_opt(raw_optimize)

        try:
            self._every = max(1, int(camera_config.get_pipeline_setting("process_every", 5)))
        except (TypeError, ValueError):
            self._every = 5

        super().__init__(
            camera_config,
            (640, 480),
            camera_config.get("grayscale", False),
        )

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
                name=f"Optimize-{self.config.get('name', 'depth_anything')}",
            ).start()
        else:
            self.prepare()

    def _prepare(self):
        self._load_model()

    def optimize(self, **kwargs) -> str:
        if not self._optimization_requested():
            return "optimization disabled for this camera (set 'Optimize' in camera settings)"
        if self._optimizing:
            return "optimizing"

        if kwargs.pop("force", False):
            return self._optimize_forced()

        self._optimizing = True
        self._set_status("optimizing (onnx build)")
        try:
            # reuse cached artifact; only a full rebuild re-exports
            self._load_optimized(force=False)
            if not self._optimized_active():
                self._load_optimized(force=True)
        except Exception as exc:
            self._optimize_error = f"optimized ONNX build failed: {exc}"
            self._load_error = self._optimize_error
            self._optimizing = False
            return f"error: optimized ONNX build failed - {exc}"
        self._optimizing = False

        if self._optimized_active():
            self._load_error = None
            self._optimize_error = None
            self._set_status("ready")
            return "ready"
        status = self._optimize_error or "error: optimized ONNX build failed - no artifact produced"
        self._optimize_error = status
        self._set_status(status)
        return status

    def _optimize_forced(self) -> str:
        self._optimizing = True
        self._set_status("optimizing (onnx build)")
        try:
            self._load_optimized(force=True)
        except Exception as exc:
            self._optimize_error = f"optimized ONNX build failed: {exc}"
            self._load_error = self._optimize_error
            self._optimizing = False
            return f"error: optimized ONNX build failed - {exc}"
        self._optimizing = False

        if self._optimized_active():
            self._load_error = None
            self._optimize_error = None
            self._set_status("ready")
            return "ready"
        status = self._optimize_error or "error: optimized ONNX build failed - no artifact produced"
        self._optimize_error = status
        self._set_status(status)
        return status

    # _resolve_target_format is inherited from OptimizableModelPipeline -
    # this pipeline has no override-worthy behaviour (unlike yolo_world,
    # whose recommend_format() call deliberately keeps dependency checks).

    def _optimization_requested(self) -> bool:
        # depth estimation can be disabled entirely - no model, no build
        return bool(getattr(self, "estimate_depth", True)) and (
            super()._optimization_requested()
        )

    def _optimized_active(self) -> bool:
        return getattr(self, "_session", None) is not None

    # ------------------------------------------------------------------
    # stale-artifact resync hooks (see OptimizableModelPipeline)
    #
    # Depth Anything ships exactly one Hugging Face checkpoint and every
    # artifact path is derived from config at boot (fixed stem + input
    # size + target format), so there is no user-picked source model and
    # no persisted path that could drift. The hooks exist so the boot
    # guard is callable on every model-backed pipeline instead of raising
    # AttributeError if a future refactor adds persisted state.
    # ------------------------------------------------------------------

    def _resolve_model_path(self, path: str) -> Path | None:
        if not path:
            return None
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[3] / p
        return p

    def _source_model_path(self) -> Path | None:
        # single fixed HF checkpoint - no authoritative user-picked .pt
        # exists, so there is never anything to resync against (yet)
        return None

    def _persist_file_path(self, file_path: str, config: iSpyConfig | None):
        # artifact locations are derived, not stored - nothing to persist
        pass

    def _optimized_active(self) -> bool:
        return getattr(self, "_session", None) is not None

    def _is_processable(self) -> bool:
        if getattr(self, "_optimizing", False):
            return False
        if self._optimization_requested():
            return self._optimized_active()
        return (
            getattr(self, "_model", None) is not None
            or getattr(self, "_session", None) is not None
        )

    def _load_model(self):
        if not self.estimate_depth:
            self.logger.info("Depth estimation disabled by config.")
            return

        self._load_pipeline()

    def _load_optimized(self, force: bool = False):
        target = self._target_format_cached()

        if target == "onnx":
            return self._load_onnx(force)
        if target == "openvino":
            return self._load_openvino(force)
        if target == "engine":
            return self._load_tensorrt(force)
        if target == "coreml":
            return self._load_coreml(force)
        if target == "tflite":
            return self._load_tflite(force)
        if target == "rknn":
            return self._load_rknn(force)
        if target == "tpu":
            return self._load_tpu(force)

        self.logger.warning(
            "Unknown optimized target format %r - falling back to onnx", target,
        )
        return self._load_onnx(force)

    def _export_onnx_source(self, force: bool = False, quantize: bool | None = None):
        from iSpy.vision.QuantizedModel import ensure_onnx_model

        def build():
            import torch.nn as nn

            from transformers import AutoModelForDepthEstimation

            class _DepthModule(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, pixel_values):
                    return self.model(pixel_values=pixel_values).predicted_depth

            self.logger.info(
                "Loading Depth Anything V2 Small weights from Hugging Face..."
            )
            model = AutoModelForDepthEstimation.from_pretrained(
                _DEPTH_MODEL_ID, cache_dir=str(_DEPTH_HF_CACHE_DIR)
            )
            model.eval()
            return _DepthModule(model)

        return ensure_onnx_model(
            build,
            _DEPTH_ARTIFACT_STEM,
            input_size=(self._input_size, self._input_size),
            quantize=self.quantize if quantize is None else quantize,
            force=force,
            dataset_path=self._quantization_dataset,
        )

    def _artifact_path(self, target: str) -> Path:
        stem = f"{_DEPTH_ARTIFACT_STEM}_{self._input_size}x{self._input_size}"
        if target == "openvino":
            # real OpenVINO IR folder, same convention as object_detection:
            # YoloModels/openvino/<stem>_openvino_model/<stem>_openvino_model.xml
            return _YOLO_MODELS_DIR / "openvino" / f"{stem}_openvino_model"
        ext = {  # every other backend keeps a single artifact file
            "engine": ".engine", "coreml": ".mlpackage",
            "rknn": ".rknn", "tflite": ".tflite",
        }.get(target, f".{target}")
        return _YOLO_MODELS_DIR / target / f"{stem}{ext}"

    def _load_onnx(self, force: bool = False):
        artifact, converted = self._export_onnx_source(force)
        if not converted or not artifact:
            self._session = None
            self._load_error = "optimized ONNX conversion produced no artifact"
            return

        import onnxruntime as ort

        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self._session = ort.InferenceSession(artifact, providers=providers)
        self._infer = self._infer_depth_onnx
        self._load_error = None
        self.logger.info("Loaded optimized Depth Anything ONNX from %s", artifact)

    def _load_openvino(self, force: bool = False):
        # OpenVINO converts the fp32 ONNX export into a real OpenVINO IR
        # (.xml + .bin) cached under YoloModels/openvino/, then runs it on
        # the best available device (GPU/NPU, else CPU). compiled blobs are
        # cached too, so boot does not re-compile the model every time.
        artifact, converted = self._export_onnx_source(force, quantize=False)
        if not converted or not artifact:
            self._session = None
            self._load_error = "OpenVINO build produced no ONNX source"
            return

        try:
            import openvino as ov

            from iSpy.config.AutoOpt import resolve_openvino_device

            device = resolve_openvino_device()
            if device.startswith("intel:"):
                device = device.split(":", 1)[1]  # OpenVINO wants 'GPU', not 'intel:gpu'
            core = ov.Core()
            core.set_property({"CACHE_DIR": str(_YOLO_MODELS_DIR / "openvino" / ".cache")})

            ir_dir = self._artifact_path("openvino")
            ir_xml = ir_dir / f"{ir_dir.name}.xml"
            if not ir_xml.exists() or force:
                self.logger.info(
                    "Converting Depth Anything ONNX -> OpenVINO IR in %s ...", ir_dir,
                )
                ir_dir.mkdir(parents=True, exist_ok=True)
                ov.save_model(core.read_model(str(artifact)), str(ir_xml))
            if not ir_xml.exists():
                raise RuntimeError(f"OpenVINO conversion produced no IR: {ir_dir}")

            registered = next(
                (d for d in core.available_devices if d.lower() == device.lower()),
                None,
            )
            if registered is None:
                registered = next(
                    (d for d in core.available_devices if device.lower() in d.lower()),
                    None,
                )
            if registered is None:
                self.logger.warning(
                    "OpenVINO device %r not registered (%s) - using AUTO",
                    device, core.available_devices,
                )
                registered = "AUTO"
            self._session = self._compile_openvino(core, ir_xml, registered)
            self._infer = self._infer_depth_openvino
            self._load_error = None
            self.logger.info(
                "Loaded optimized Depth Anything OpenVINO (%s) from %s", registered, ir_dir,
            )
        except Exception as exc:
            self.logger.warning(
                "OpenVINO backend failed (%s) - falling back to onnx", exc,
            )
            self._load_onnx(force)

    def _compile_openvino(self, core, ir_xml: Path, device: str):
        cache_dir = _YOLO_MODELS_DIR / "openvino" / ".cache"
        for attempt in range(2):
            try:
                return core.compile_model(str(ir_xml), device)
            except Exception as exc:
                if attempt == 0:
                    self.logger.warning(
                        "OpenVINO compile failed (%s) - clearing blob cache and retrying",
                        exc,
                    )
                    import shutil
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    continue
                raise

    def _load_tensorrt(self, force: bool = False):
        try:
            import tensorrt as trt  # noqa: F401
        except ImportError:
            self.logger.warning(
                "TensorRT not installed - 'engine' backend unavailable, falling back to onnx"
            )
            return self._load_onnx(force)

        artifact, converted = self._export_onnx_source(force, quantize=False)
        if not converted or not artifact:
            self._session = None
            self._load_error = "TensorRT build produced no ONNX source"
            return

        engine_path = self._artifact_path("engine")
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not engine_path.exists() or force:
                self.logger.info("Building TensorRT engine from %s ...", artifact)
                trt_logger = trt.Logger(trt.Logger.WARNING)
                builder = trt.Builder(trt_logger)
                network = builder.create_network(
                    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                )
                parser = trt.OnnxParser(network, trt_logger)
                with open(artifact, "rb") as f:
                    if not parser.parse(f.read()):
                        raise RuntimeError(parser.get_error(0).desc())
                config = builder.create_builder_config()
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
                serialized = builder.build_serialized_network(network, config)
                if serialized is None:
                    raise RuntimeError("TensorRT engine build returned None")
                engine_path.write_bytes(serialized)

            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self._session = runtime.deserialize_cuda_engine(engine_path.read_bytes())
            self._infer = self._infer_depth_engine
            self._load_error = None
            self.logger.info("Loaded optimized Depth Anything TensorRT from %s", engine_path)
        except Exception as exc:
            self.logger.warning(
                "TensorRT build/load failed (%s) - falling back to onnx", exc,
            )
            self._session = None
            self._load_onnx(force)

    def _load_coreml(self, force: bool = False):
        try:
            import coremltools as ct  # noqa: F401
        except ImportError:
            self.logger.warning(
                "coremltools not installed - 'coreml' backend unavailable, falling back to onnx"
            )
            return self._load_onnx(force)

        artifact, converted = self._export_onnx_source(force, quantize=False)
        if not converted or not artifact:
            self._session = None
            self._load_error = "CoreML build produced no ONNX source"
            return

        model_path = self._artifact_path("coreml")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not model_path.exists() or force:
                self.logger.info("Converting %s -> CoreML ...", artifact)
                mlmodel = ct.convert(artifact, source="onnx", compute_units=ct.ComputeUnit.ALL)
                mlmodel.save(str(model_path))
            self._session = ct.models.MLModel(str(model_path))
            self._infer = self._infer_depth_coreml
            self._load_error = None
            self.logger.info("Loaded optimized Depth Anything CoreML from %s", model_path)
        except Exception as exc:
            self.logger.warning(
                "CoreML build/load failed (%s) - falling back to onnx", exc,
            )
            self._session = None
            self._load_onnx(force)

    def _load_tflite(self, force: bool = False):
        try:
            import onnx2tf  # noqa: F401
        except ImportError:
            self.logger.warning(
                "onnx2tf not installed - 'tflite' backend unavailable, falling back to onnx"
            )
            return self._load_onnx(force)

        artifact, converted = self._export_onnx_source(force, quantize=False)
        if not converted or not artifact:
            self._session = None
            self._load_error = "TFLite build produced no ONNX source"
            return

        out_dir = _YOLO_MODELS_DIR / "tflite"
        out_dir.mkdir(parents=True, exist_ok=True)
        tflite_path = out_dir / f"{_DEPTH_ARTIFACT_STEM}_{self._input_size}x{self._input_size}.tflite"
        try:
            if not tflite_path.exists() or force:
                self.logger.info("Converting %s -> TFLite ...", artifact)
                onnx2tf.convert(
                    input_onnx_file_path=artifact,
                    output_folder_path=str(out_dir),
                    non_verbose=True,
                )
                generated = out_dir / f"{Path(artifact).stem}_float32.tflite"
                if generated.exists():
                    generated.replace(tflite_path)
            interpreter = self._tflite_interpreter(tflite_path)
            interpreter.allocate_tensors()
            self._session = interpreter
            self._infer = self._infer_depth_tflite
            self._load_error = None
            self.logger.info("Loaded optimized Depth Anything TFLite from %s", tflite_path)
        except Exception as exc:
            self.logger.warning(
                "TFLite build/load failed (%s) - falling back to onnx", exc,
            )
            self._session = None
            self._load_onnx(force)

    @staticmethod
    def _tflite_interpreter(model_path: Path):
        for mod in ("tflite_runtime.interpreter", "ai_edge_litert"):
            try:
                if mod == "tflite_runtime.interpreter":
                    from tflite_runtime.interpreter import Interpreter
                else:
                    from ai_edge_litert import Interpreter
                return Interpreter(model_path=str(model_path))
            except Exception:
                continue
        import tensorflow as tf

        return tf.lite.Interpreter(model_path=str(model_path))

    def _load_rknn(self, force: bool = False):
        try:
            from rknn.api import RKNN  # noqa: F401
        except ImportError:
            self.logger.warning(
                "rknn-toolkit2 not installed - 'rknn' backend unavailable, falling back to onnx"
            )
            return self._load_onnx(force)

        artifact, converted = self._export_onnx_source(force, quantize=False)
        if not converted or not artifact:
            self._session = None
            self._load_error = "RKNN build produced no ONNX source"
            return

        rknn_path = self._artifact_path("rknn")
        rknn_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not rknn_path.exists() or force:
                self.logger.info("Building RKNN model from %s ...", artifact)
                from rknn.api import RKNN

                rknn = RKNN(verbose=False)
                try:
                    rknn.config(target_platform="rk3588")
                    rknn.load_onnx(model=artifact, input_size_list=[[1, 3, self._input_size, self._input_size]])
                    dataset = self._rknn_calibration_txt()
                    rknn.build(do_quantization=self.quantize, dataset=dataset)
                    rknn.export_rknn(str(rknn_path))
                finally:
                    rknn.release()

            from rknnlite.api import RKNNLite

            rknn_lite = RKNNLite(verbose=False)
            if rknn_lite.load_rknn(str(rknn_path)) != 0:
                raise RuntimeError(f"Failed to load RKNN model: {rknn_path}")
            if rknn_lite.init_runtime() != 0:
                raise RuntimeError("Failed to init RKNN runtime")
            self._session = rknn_lite
            self._infer = self._infer_depth_rknn
            self._load_error = None
            self.logger.info("Loaded optimized Depth Anything RKNN from %s", rknn_path)
        except Exception as exc:
            self.logger.warning(
                "RKNN build/load failed (%s) - falling back to onnx", exc,
            )
            self._session = None
            self._load_onnx(force)

    def _rknn_calibration_txt(self) -> str | None:
        if not self._quantization_dataset:
            return None
        ds = Path(self._quantization_dataset)
        if not ds.exists():
            return None
        images = sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp") for p in ds.rglob(ext)
        )
        if not images:
            return None
        txt = ds / "rknn_calibration.txt"
        txt.write_text("\n".join(str(p) for p in images))
        return str(txt)

    def _load_tpu(self, force: bool = False):
        try:
            import torch_xla.core.xla_model as xm  # noqa: F401
        except ImportError:
            self.logger.warning(
                "torch_xla not installed - 'tpu' backend unavailable, falling back to onnx"
            )
            return self._load_onnx(force)

        try:
            import torch
            from transformers import AutoModelForDepthEstimation

            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
            self._model = (
                AutoModelForDepthEstimation.from_pretrained(
                    _DEPTH_MODEL_ID, cache_dir=str(_DEPTH_HF_CACHE_DIR)
                )
                .to(device)
                .eval()
            )
            self._session = self._model
            self._infer = self._infer_depth_tpu
            self._load_error = None
            self.logger.info("Loaded Depth Anything on TPU device %s", device)
        except Exception as exc:
            self.logger.warning(
                "TPU backend failed (%s) - falling back to onnx", exc,
            )
            self._model = None
            self._session = None
            self._load_onnx(force)

    def _preprocess_depth(self, frame: np.ndarray) -> np.ndarray:
        size = getattr(self, "_input_size", _DEPTH_INPUT_SIZE)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = (
            cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
            .astype(np.float32)
            / 255.0
        )
        return (
            (img.transpose(2, 0, 1) - _IMAGENET_MEAN) / _IMAGENET_STD
        ).astype(np.float32)[None]

    def _postprocess_depth(self, depth, frame: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth)
        while depth.ndim > 2:
            depth = depth[0]
        return cv2.resize(
            depth.astype(np.float32),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

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
                cache_dir=str(_DEPTH_HF_CACHE_DIR),
            )

            self._infer = self._infer_depth_pipeline
            self._load_error = None
            self.logger.info("Loaded Depth Anything V2 Small.")

        except Exception as exc:
            self._load_error = f"failed to load Depth Anything V2 from Hugging Face: {exc}"
            self.logger.exception(
                "Failed to load Depth Anything V2 from Hugging Face."
            )
            self._model = None

    def _infer_depth(self, frame: np.ndarray):
        # runtime picked once at load time - no re-deciding per frame
        if self._infer is not None:
            return self._infer(frame)
        return self._infer_depth_pipeline(frame)

    def _infer_depth_pipeline(self, frame: np.ndarray):
        image = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        result = self._model(image)

        # some transformers versions expose depth as either key
        depth = result.get("predicted_depth")

        if depth is not None:
            if hasattr(depth, "detach"):
                depth = depth.detach().cpu().numpy()

            depth = np.asarray(depth)

        # strip batch/channel dims
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
        pixel_values = self._preprocess_depth(frame)
        depth = self._session.run(None, {"pixel_values": pixel_values})[0]
        return self._postprocess_depth(depth, frame)

    def _infer_depth_openvino(self, frame: np.ndarray) -> np.ndarray:
        pixel_values = self._preprocess_depth(frame)
        outputs = self._session({"pixel_values": pixel_values})
        depth = next(iter(outputs.values()))
        return self._postprocess_depth(depth, frame)

    def _infer_depth_engine(self, frame: np.ndarray) -> np.ndarray:
        import numpy as np

        import tensorrt as trt

        pixel_values = self._preprocess_depth(frame)
        engine = self._session
        context = engine.create_execution_context()
        for idx in range(engine.num_io_tensors):
            name = engine.get_tensor_name(idx)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                context.set_tensor_shape(name, pixel_values.shape)
        bindings = []
        output = None
        for idx in range(engine.num_io_tensors):
            name = engine.get_tensor_name(idx)
            shape = context.get_tensor_shape(name)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                bindings.append(np.array(pixel_values).ctypes.data)
            else:
                out = np.empty(tuple(shape), dtype=np.float32)
                bindings.append(out.ctypes.data)
                output = out
        context.execute_v2(bindings)
        return self._postprocess_depth(output, frame)

    def _infer_depth_coreml(self, frame: np.ndarray) -> np.ndarray:
        pixel_values = self._preprocess_depth(frame)
        out = self._session.predict({"pixel_values": pixel_values})
        depth = next(iter(out.values()))
        return self._postprocess_depth(depth, frame)

    def _infer_depth_tflite(self, frame: np.ndarray) -> np.ndarray:
        interpreter = self._session
        details = interpreter.get_input_details()[0]
        pixel_values = self._preprocess_depth(frame).astype(details["dtype"])
        interpreter.set_tensor(details["index"], pixel_values)
        interpreter.invoke()
        depth = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
        return self._postprocess_depth(depth, frame)

    def _infer_depth_rknn(self, frame: np.ndarray) -> np.ndarray:
        size = getattr(self, "_input_size", _DEPTH_INPUT_SIZE)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        out = self._session.inference(inputs=[img])
        depth = out[0]
        return self._postprocess_depth(depth, frame)

    def _infer_depth_tpu(self, frame: np.ndarray) -> np.ndarray:
        import torch

        import torch_xla.core.xla_model as xm

        pixel_values = torch.from_numpy(self._preprocess_depth(frame))
        with torch.no_grad():
            pred = self._session(pixel_values=pixel_values.to(self._session.device)).predicted_depth
        depth = xm.send_cpu(pred).cpu().numpy()
        return self._postprocess_depth(depth, frame)

    def _distance_from_depth(self, raw: float) -> float:
        norm = float(np.clip(raw, 0.0, 1e6))

        d_min = getattr(self, "_dmin", 0.0)
        d_max = getattr(self, "_dmax", 1.0)

        span = d_max - d_min
        if span <= 1e-9:
            return self.max_depth

        closeness = (norm - d_min) / span

        # Depth Anything gives relative depth, not real meters
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

        # nearest point = highest relative inverse-depth value
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

        # resize if the model output resolution differs from the frame
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

        # map the center pixel onto the depth map
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

        label = f"Depth {center_d:.2f} {self._unit_label}"

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

    def run(self):
        frame = self.get_frame()

        if frame is None:
            return [], None

        if not self._is_processable():
            return [], frame

        model = getattr(self, "_model", None)
        session = getattr(self, "_session", None)

        self._frame_count = getattr(self, "_frame_count", 0) + 1

        every = max(1, self._every)
        last_depth = self._last_depth

        # reuse the last depth map between inference frames
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
                return [], frame

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

        depth = getattr(self, "_last_depth", None)
        if depth is None or not getattr(depth, "size", 0):
            return frame

        try:
            h, w = frame.shape[:2]

            normalized = cv2.normalize(
                depth,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
            ).astype(np.uint8)

            if normalized.ndim == 2:
                map_img = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
            else:
                map_img = normalized

            if map_img.shape[:2] != (h, w):
                map_img = cv2.resize(
                    map_img,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )

            cx, cy = w // 2, h // 2
            radius = max(
                6,
                min(w, h) // 30,
            )

            depth_x = min(
                depth.shape[1] - 1,
                int(cx * depth.shape[1] / w),
            )
            depth_y = min(
                depth.shape[0] - 1,
                int(cy * depth.shape[0] / h),
            )
            center_d = self._distance_from_depth(
                float(depth[depth_y, depth_x])
            )

            cv2.circle(
                map_img,
                (cx, cy),
                radius,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                map_img,
                f"Depth {center_d:.2f} {self._unit_label}",
                (cx - radius * 2, cy + radius + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                map_img,
                "Depth Map",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            return map_img

        except Exception:
            return frame

    def destroy(self):
        self._model = None
        self._session = None
        self._infer = None
        self._last_depth = None
        self._last_objects = []
        self._last_annotated = None

        if hasattr(super(), "destroy"):
            super().destroy()

