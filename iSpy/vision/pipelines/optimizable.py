"""Shared optimization/model-backend machinery for model-backed pipelines.

Model-backed pipelines (object_detection, yolo_world, depth_anything) all
need the same pieces: optimization settings in their config schema, target
format resolution, a "should we build an optimized artifact" check and a
background build runner. This module extracts that machinery into one mixin
so a fix lands everywhere at once instead of being hand-duplicated three
times.

Subclasses set these attributes in __init__ BEFORE calling anything here:

    self.logger            logging.Logger
    self.quantize          bool   - int8 quantization requested
    self._auto_opt         bool   - optimize requested (legacy auto_opt folded in)
    self._requested_format str    - explicit target format or "auto"
    self._target_format    str|None    - resolved lazily, leave None
    self._optimizing       bool        - build in flight, leave False
    self._optimize_error   str|None    - last build failure, leave None
"""

import logging
from pathlib import Path

#: every backend any model-backed pipeline can build; 'auto' resolution must
#: pick from this set or fall back to onnx
SUPPORTED_TARGET_FORMATS = ("onnx", "rknn", "tflite", "openvino", "engine", "coreml", "tpu")


class OptimizableModelPipeline:
    """Mixin for pipelines backed by a convertible/quantizable model."""

    #: extra config-schema keys surfaced by get_optimization_options()
    _OPT_OPTIONS_EXTRA: tuple[str, ...] = ()

    #: hardware targets a model-backed pipeline can route its inference onto.
    #: The active one is resolved from the loaded model at runtime by
    #: active_hardware() (RKNN->NPU, TPU->TPU, TensorRT/CoreML/OpenVINO->GPU,
    #: ONNX/TFLite/pytorch->CPU or GPU).
    hardware: tuple[str, ...] = ("cpu", "gpu", "npu", "tpu")

    #: resolved-format -> hardware label. 'format' is the _path_format() token
    #: for the active model artifact/file.
    _HARDWARE_BY_FORMAT = {
        "rknn": "npu",
        "tpu": "tpu",
        "engine": "gpu",    # NVIDIA TensorRT
        "coreml": "gpu",    # Apple GPU
        "openvino": "gpu",  # Intel GPU/VPU
        "tflite": "cpu",
        "onnx": "cpu",
        "pytorch": "cpu",
    }

    @classmethod
    def needs_model_backend(cls) -> bool:
        return True

    def active_hardware(self) -> str | None:
        """Resolve the hardware this pipeline's loaded model is running on.

        Priority:
          1. unambiguous runtime descriptors (model_type tpu/rknn)
          2. the active model file's format (engine/coreml/openvino/tflite/
             onnx/pytorch)
          3. the raw-model device (yolo / bare torch model on CUDA -> gpu)

        Returns None when no dedicated accelerator is identifiable (its CPU
        usage is already reported by the system CPU reading).
        """
        model = getattr(self, "model", None)
        mt = getattr(model, "model_type", None)
        if mt == "tpu":
            return "tpu"
        if mt == "rknn":
            return "npu"

        path = None
        for attr in ("yolo_model_file", "_model_path", "model_file"):
            value = getattr(self, attr, None)
            if not value:
                value = getattr(model, attr, None)
            if value:
                path = value
                break
        if path:
            fmt = self._path_format(str(path))
            # pytorch/onnx can run on either CPU or CUDA - check the loaded
            # model's device rather than hard-coding CPU.
            if fmt in ("pytorch", "onnx"):
                device = getattr(model, "device", "cpu")
                return "gpu" if (device is not None and str(device) != "cpu") else "cpu"
            hw = self._HARDWARE_BY_FORMAT.get(fmt)
            if hw:
                return hw

        device = getattr(model, "device", "cpu")
        if device is not None and str(device) != "cpu":
            return "gpu"
        if mt == "yolo":
            return "cpu"
        return None

    # ------------------------------------------------------------------
    # config schema
    # ------------------------------------------------------------------

    @classmethod
    def _optimization_schema(
        cls,
        target_formats: tuple[str, ...] = ("auto", "onnx"),
        input_size_default: int | None = None,
        input_size_help: str = "Letterbox resolution used for the optimized "
                               "model conversion and inference.",
    ) -> dict:
        """Common optimization fields shared by every model-backed pipeline.

        model_size stays per-pipeline (each downloads different weight sets);
        everything here has identical semantics across pipelines.
        """
        schema = {
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
                "options": list(target_formats),
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
                "nullable": True,
                "browse_root": "QuantizeDataset",
                "quantization": True,
                "gated_by": "quantize",
                "help": "Optional folder of calibration images used for "
                        "quantization. Leave empty to auto-download images "
                        "from the model's calibration keywords.",
            },
        }
        if input_size_default is not None:
            schema["input_size"] = {
                "type": "number",
                "label": "Input Size",
                "default": input_size_default,
                "quantization": True,
                "help": input_size_help,
            }
        return schema

    @classmethod
    def recommended_format(cls) -> str:
        try:
            from iSpy.config.AutoOpt import recommend_format

            return recommend_format(ignore_dependencies=True)
        except Exception:
            logging.getLogger(__name__).warning(
                "AutoOpt.recommend_format did NOT work for your device, "
                "falling back to ONNX!"
            )
            return "onnx"

    def get_optimization_options(self) -> dict:
        schema = self.config_schema()
        keys = ("optimize", "target_format", "quantize", "quantization_dataset") \
            + self._OPT_OPTIONS_EXTRA
        return {key: schema[key] for key in keys if key in schema}

    # ------------------------------------------------------------------
    # target format resolution
    # ------------------------------------------------------------------

    def _resolve_target_format(self) -> str:
        explicit = str(getattr(self, "_requested_format", "") or "").strip().lower()
        if explicit and explicit != "auto":
            target = explicit
        else:
            target = self.recommended_format()
        if target not in SUPPORTED_TARGET_FORMATS:
            self.logger.warning(
                "Recommended target format %r unsupported - using onnx", target,
            )
            return "onnx"
        return target

    def _target_format_cached(self) -> str:
        if self._target_format is None:
            self._target_format = self._resolve_target_format()
        return self._target_format

    # ------------------------------------------------------------------
    # build state
    # ------------------------------------------------------------------

    def _optimization_requested(self) -> bool:
        """True when an optimized artifact should exist and be active.

        Honors both current keys (quantize/optimize) and legacy aliases
        (quantized/auto_opt); subclasses reading a vision_model block get
        that block checked too.
        """
        if bool(getattr(self, "quantize", False)):
            return True
        if getattr(self, "_auto_opt", False):
            return True
        vm_getter = getattr(self, "_current_vm_config", None)
        if vm_getter is not None:
            vm = vm_getter()
            if isinstance(vm, dict):
                return bool(vm.get("optimize") or vm.get("auto_opt")) or bool(
                    vm.get("quantize") or vm.get("quantized")
                )
        return False

    def _is_processable(self) -> bool:
        if getattr(self, "_optimizing", False):
            return False
        if not self._calibration_processable():
            return False
        if self.model is None:
            return False
        if self._optimization_requested():
            return self._optimized_active()
        return True

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

    def _optimize_runner(self):
        status = self.optimize()
        if not self._optimized_active():
            self._optimize_error = status
        self._set_status(status)

    # ------------------------------------------------------------------
    # stale-artifact guard (resync-on-boot)
    #
    # vision_model.file_path can drift from source_pt when the model was
    # re-picked in the UI: it may point at an artifact built for an older
    # model, which would silently keep that old model running. The source
    # .pt is authoritative - load its already-built artifact when one
    # exists, else the .pt itself (the background optimizer swaps file_path
    # once its fresh build lands).
    #
    # Only pipelines with a persisted, user-picked source model have a
    # resync that can actually fire (object_detection today). The other
    # model-backed pipelines still implement the three helpers - their
    # paths are derived from config at boot, so _source_model_path()
    # resolves to None and the guard exits immediately - so this stays
    # safely callable everywhere instead of an AttributeError landmine.
    # ------------------------------------------------------------------

    def _resync_stale_model_file_path(self, config) -> None:
        source = self._source_model_path()
        current = str(getattr(self, "yolo_model_file", "") or "")
        current_stem = Path(current).stem if current else None
        source_stem = source.stem if source is not None else None
        if source is None or not current or current_stem == source_stem:
            return
        from iSpy.vision.optimizer import existing_artifact_for
        artifact = existing_artifact_for(source, self._target_format_cached())
        preferred = artifact or str(self._resolve_model_path(source) or source)
        if self._resolve_model_path(current) != self._resolve_model_path(preferred):
            self.logger.warning(
                "Camera '%s': vision_model.file_path (%s) doesn't match "
                "source_pt (%s) - correcting to %s and persisting.",
                getattr(self, "_cam_name", "?"), current, source, preferred,
            )
            self.yolo_model_file = preferred
            self._persist_file_path(preferred, config)
