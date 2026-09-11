import logging
from pathlib import Path

#: every backend any model-backed pipeline can build; 'auto' resolution must
#: pick from this set or fall back to onnx
SUPPORTED_TARGET_FORMATS = (
    "onnx",
    "rknn",
    "tflite",
    "openvino",
    "engine",
    "coreml",
    "tpu",
    "hailo",
    "qnn",
)


class OptimizableModelPipeline:
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
        "hailo": "npu",  # Hailo NPU
        "qnn": "npu",  # Qualcomm NPU (ONNX Runtime QNN EP)
        "tpu": "tpu",
        "engine": "gpu",  # NVIDIA TensorRT
        "coreml": "gpu",  # Apple GPU
        "openvino": "gpu",  # Intel GPU/VPU
        "tflite": "cpu",
        "onnx": "cpu",
        "pytorch": "cpu",
    }

    @classmethod
    def needs_model_backend(cls) -> bool:
        return True

    def active_hardware(self) -> str | None:
        model = getattr(self, "model", None)
        mt = getattr(model, "model_type", None)
        if mt == "tpu":
            return "tpu"
        if mt in ("rknn", "hailo", "qnn"):
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
        schema = {
            "optimize": {
                "type": "select",
                "label": "Optimize/Convert",
                # HAILO DISABLED - see <reason>
                # original: ["auto", "onnx", "hef", "off"]
                "options": ["auto", "onnx", "off"],
                "default": "off",
                "optimize_toggle": True,
                "help": "Auto-detect and build the best backend artifact for this "
                "device (rknn on Rockchip NPU, engine on "
                "NVIDIA, onnx elsewhere, etc.) in the background. "
                "'auto' picks the best format via recommend_format(). "
                "Set 'onnx' to force a specific backend. "
                "'off' disables optimization. Falls back to the top-level "
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
        keys = (
            "optimize",
            "target_format",
            "quantize",
            "quantization_dataset",
        ) + self._OPT_OPTIONS_EXTRA
        return {key: schema[key] for key in keys if key in schema}

    # ------------------------------------------------------------------
    # target format resolution
    # ------------------------------------------------------------------

    def _resolve_target_format(self) -> str:
        explicit = str(getattr(self, "_requested_format", "") or "").strip().lower()
        auto_opt_fmt = self._auto_opt_to_target_format()
        if explicit and explicit != "auto":
            target = explicit
        elif auto_opt_fmt:
            target = auto_opt_fmt
        else:
            target = self.recommended_format()
        if target not in SUPPORTED_TARGET_FORMATS:
            self.logger.warning(
                "Recommended target format %r unsupported - using onnx",
                target,
            )
            return "onnx"
        return target

    def _target_format_cached(self) -> str:
        requested_format = getattr(self, "_requested_format", "")
        auto_opt = getattr(self, "_auto_opt", False)
        vm_getter = getattr(self, "_current_vm_config", None)
        if vm_getter is not None:
            vm = vm_getter()
            if isinstance(vm, dict):
                if "target_format" in vm:
                    requested_format = vm.get("target_format")
                auto_opt = vm.get("optimize", vm.get("auto_opt", auto_opt))
        requested_format = str(requested_format or "auto").strip().lower()
        cache_key = (requested_format, str(auto_opt).strip().lower())
        if cache_key != getattr(self, "_target_format_request", None):
            self._requested_format = requested_format
            self._auto_opt = self._normalize_auto_opt(auto_opt)
            self._target_format_request = cache_key
            self._target_format = None
        if self._target_format is None:
            self._target_format = self._resolve_target_format()
        return self._target_format

    # ------------------------------------------------------------------
    # build state
    # ------------------------------------------------------------------

    def _optimization_requested(self) -> bool:
        vm_getter = getattr(self, "_current_vm_config", None)
        if vm_getter is not None:
            vm = vm_getter()
            if isinstance(vm, dict):
                opt_val = vm.get("optimize", vm.get("auto_opt", False))
                if isinstance(opt_val, str):
                    if opt_val.lower().strip() in ("off", "false", "0", ""):
                        opt_val = False
                    else:
                        opt_val = True
                q_val = vm.get("quantize", vm.get("quantized", False))
                return bool(opt_val) or bool(q_val)

        if bool(getattr(self, "quantize", False)):
            return True
        auto_opt = getattr(self, "_auto_opt", False)
        if isinstance(auto_opt, str):
            if auto_opt.lower().strip() in ("off", "false", "0", ""):
                return False
            return True
        if auto_opt:
            return True
        return False

    @staticmethod
    def _normalize_auto_opt(raw_optimize) -> bool | str:
        if raw_optimize is None:
            return False
        if isinstance(raw_optimize, str):
            val = raw_optimize.strip().lower()
            if val in ("off", "false", "0", "no", "none", ""):
                return False
            if val in ("true", "yes", "on", "1", "auto"):
                return True
            if val in SUPPORTED_TARGET_FORMATS:
                return val
            return True
        return bool(raw_optimize)

    def _auto_opt_to_target_format(self) -> str | None:
        auto_opt = getattr(self, "_auto_opt", False)
        if isinstance(auto_opt, str):
            val = auto_opt.lower().strip()
            if val in ("off", "false", "0", "", "auto", "true", "1", "yes", "on"):
                return None
            if val in SUPPORTED_TARGET_FORMATS:
                return val
            return None
        return None

    def _is_processable(self) -> bool:
        if not self.calibration_ready():
            return False
        if getattr(self, "_optimizing", False):
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
        # QNN artifacts are .onnx files committed under a 'qnn' output dir -
        # sniff the directory before the generic .onnx suffix, or they'd be
        # mislabelled as plain onnx and the format wouldn't round-trip.
        if "\\qnn\\" in p or "/qnn/" in p:
            return "qnn"
        for ext, fmt in (
            (".pt", "pytorch"),
            (".onnx", "onnx"),
            (".rknn", "rknn"),
            (".hef", "hailo"),
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
                getattr(self, "_cam_name", "?"),
                current,
                source,
                preferred,
            )
            self.yolo_model_file = preferred
            self._persist_file_path(preferred, config)
