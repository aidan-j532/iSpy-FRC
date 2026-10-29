import os
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Fake rknnlite (import-order-dependent: monkeypatch sys.modules;
# works if this file is the FIRST import of iSpy.vision.genericYolo)
rknnlite_mod = types.ModuleType("rknnlite")
rknnlite_api = types.ModuleType("rknnlite.api")


class FakeRKNNLite:
    NPU_CORE_0 = 0
    NPU_CORE_0_1 = 1
    NPU_CORE_0_1_2 = 2

    def __init__(self, *args, **kwargs):
        self.release = MagicMock()

    def load_rknn(self, path):
        return 0

    def init_runtime(self, core_mask=0):
        return 0

    def inference(self, inputs):
        return None


rknnlite_api.RKNNLite = FakeRKNNLite
rknnlite_mod.api = rknnlite_api
rknnlite_mod.__file__ = "fake_rknnlite/__init__.py"
sys.modules["rknnlite"] = rknnlite_mod
sys.modules["rknnlite.api"] = rknnlite_api

# Remove the module-level patch - instead, each test that needs RKNN fake
# should patch genericYolo.RKNNLite directly using unittest.mock.patch


# Fake tflite_runtime
tflite_mod = types.ModuleType("tflite_runtime")
tflite_interp = types.ModuleType("tflite_runtime.interpreter")
tflite_interp.Interpreter = MagicMock()
tflite_interp.load_delegate = MagicMock(return_value=[])
tflite_mod.interpreter = tflite_interp
sys.modules["tflite_runtime"] = tflite_mod
sys.modules["tflite_runtime.interpreter"] = tflite_interp

# Fake torch (pulled in transitively by some imports)
torch_mod = types.ModuleType("torch")
torch_mod.device = MagicMock()
torch_mod.cuda = MagicMock()
torch_mod.cuda.is_available = MagicMock(return_value=False)
torch_mod.cuda.device_count = MagicMock(return_value=0)
sys.modules["torch"] = torch_mod

# Fake scipy
scipy_mod = types.ModuleType("scipy")
scipy_spatial = types.ModuleType("scipy.spatial")
scipy_transform = types.ModuleType("scipy.spatial.transform")
scipy_transform.Rotation = MagicMock()
scipy_spatial.transform = scipy_transform
scipy_mod.spatial = scipy_spatial
sys.modules["scipy"] = scipy_mod
sys.modules["scipy.spatial"] = scipy_spatial
sys.modules["scipy.spatial.transform"] = scipy_transform

# Fake tensorrt (for .engine runtime tests)
trt_mod = types.ModuleType("tensorrt")


class _FakeTensorIOMode:
    INPUT = "input"
    OUTPUT = "output"


class _FakeEngineContext:
    def __init__(self, shapes, output_name):
        self._shapes = dict(shapes)
        self._output_name = output_name
        self._addresses = {}

    def set_input_shape(self, name, shape):
        self._shapes[name] = tuple(shape)
        return True

    def get_tensor_shape(self, name):
        return self._shapes.get(name)

    def set_tensor_address(self, name, ptr):
        self._addresses[name] = ptr
        return True

    def execute_v2(self, bindings):
        self.bindings = bindings
        return True

    def execute_async_v3(self, stream_handle=None):
        self.bindings = None
        return True


class _FakeEngine:
    def __init__(self, input_name, output_name, feat_w):
        self._input_name = input_name
        self._output_name = output_name
        self._feat_w = feat_w
        self.num_io_tensors = 2

    def get_tensor_name(self, idx):
        return self._input_name if idx == 0 else self._output_name

    def get_tensor_mode(self, name):
        return (
            _FakeTensorIOMode.INPUT
            if name == self._input_name
            else _FakeTensorIOMode.OUTPUT
        )

    def get_tensor_dtype(self, name):
        return np.float32

    def create_execution_context(self):
        shapes = {
            self._input_name: (1, 3, 640, 640),
            self._output_name: (1, 0, self._feat_w),
        }
        return _FakeEngineContext(shapes, self._output_name)


class _FakeTensorRTRuntime:
    def __init__(self, *args, **kwargs):
        self.engine = _FakeEngine("images", "output0", 5)

    def deserialize_cuda_engine(self, data):
        return self.engine


trt_mod.Runtime = _FakeTensorRTRuntime
trt_mod.Logger = MagicMock()
trt_mod.TensorIOMode = _FakeTensorIOMode
trt_mod.nptype = lambda dtype: dtype
sys.modules["tensorrt"] = trt_mod


class _FakeTorchTensor:
    def __init__(self, array):
        self.array = array
        self.dtype = array.dtype
        self.shape = array.shape

    def to(self, device):
        return self

    def data_ptr(self):
        return id(self.array)

    def cpu(self):
        return self

    def numpy(self):
        return self.array
    
    def copy_(self, other):
        return self


torch_mod.from_numpy = lambda array: _FakeTorchTensor(array)
torch_mod.empty = lambda shape, dtype, device: _FakeTorchTensor(
    np.empty(shape, dtype=dtype)
)

# Fake openvino (for .xml runtime tests)
ov_mod = types.ModuleType("openvino")


class _FakeOutput:
    def __init__(self, name):
        self._name = name

    def get_any_name(self):
        return self._name


class _FakeCompiledModel:
    def __init__(self, feat_w):
        self._feat_w = feat_w
        self.inputs = [_FakeOutput("images")]

    def __call__(self, feed):
        return {"output0": np.empty((1, 0, self._feat_w), dtype=np.float32)}


class _FakeOpenVINOCore:
    available_devices = ["CPU"]

    def __init__(self, *args, **kwargs):
        pass

    def compile_model(self, model_path, device="AUTO", *args, **kwargs):
        return _FakeCompiledModel(5)


ov_mod.Core = _FakeOpenVINOCore
sys.modules["openvino"] = ov_mod

from iSpy.config.AutoOpt import SUPPORTED_FORMATS
from iSpy.vision.genericYolo import (
    Box,
    GenericYolo,
    ModelFileError,
    Results,
    normalize_model_config,
)
from iSpy.vision.metadata import (
    derive_format_metadata,
    metadata_path_for,
    read_metadata,
    write_metadata,
)


def make_frame(w=320, h=320):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 1] = np.linspace(0, 255, w, dtype=np.uint8)  # green gradient
    return frame


def _no_hw_flags():
    return {
        "has_rockchip_npu": False,
        "has_hailo_npu": False,
        "has_qualcomm_npu": False,
        "has_edge_tpu": False,
        "has_apple_silicon": False,
        "has_tpu": False,
        "has_tpu_hardware": False,
        "has_nvidia": False,
        "has_tensorrt": False,
        "has_intel_vpu": False,
        "has_intel_gpu": False,
        "has_amd_gpu": False,
        "has_arm": False,
    }


def _recommend(**overrides):
    import iSpy.config.AutoOpt as ao

    flags = _no_hw_flags()
    runtime_supported = overrides.pop("runtime_supported", True)
    ignore_dependencies = overrides.pop("ignore_dependencies", False)
    flags.update(overrides)
    with ExitStack() as stack:
        for name, value in flags.items():
            stack.enter_context(patch.object(ao, name, return_value=value))
        return ao.recommend_format(
            ignore_dependencies=ignore_dependencies,
            runtime_supported=runtime_supported,
        )


def _pipeline_recommended_format(**overrides):
    """OptimizableModelPipeline.recommended_format() with every hardware probe
    stubbed out. The pipeline asks recommend_format(runtime_supported=False), so
    a real arm host answers 'tflite' and a real intel one 'openvino' - patching
    only the coreml flag would make the assertion machine-dependent.
    Loaded by path to dodge the eager imports in pipelines/__init__.py (scipy
    etc.) that may be missing."""
    import importlib.util

    import iSpy.config.AutoOpt as ao

    module_path = (
        Path(__file__).resolve().parents[1] / "vision" / "pipelines" / "optimizable.py"
    )
    spec = importlib.util.spec_from_file_location("_optimizable", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    flags = _no_hw_flags()
    flags.update(overrides)
    with ExitStack() as stack:
        for name, value in flags.items():
            stack.enter_context(patch.object(ao, name, return_value=value))
        module._recommended_format_cached.cache_clear()
        try:
            return module.OptimizableModelPipeline.recommended_format()
        finally:
            module._recommended_format_cached.cache_clear()


class TestAutoOpt(unittest.TestCase):
    def test_returns_string(self):
        self.assertIsInstance(_recommend(), str)

    def test_returns_known_format(self):
        known = SUPPORTED_FORMATS
        self.assertIn(_recommend(), known, f"Unknown format: {_recommend()}")

    def test_rknn_wins_when_rockchip_npu(self):
        self.assertEqual(_recommend(has_rockchip_npu=True), "rknn")

    def test_rockchip_preserves_priority_over_hailo(self):
        # A board with both a Rockchip and a Hailo NPU stays on rknn - Rockchip
        # is the primary supported target.
        self.assertEqual(_recommend(has_rockchip_npu=True, has_hailo_npu=True), "rknn")

    def test_hailo_npu_uses_hailo_format(self):
        self.assertEqual(_recommend(has_hailo_npu=True), "hailo")

    def test_qualcomm_npu_uses_qnn_format(self):
        self.assertEqual(_recommend(has_qualcomm_npu=True), "qnn")

    def test_hailo_preserves_priority_over_qualcomm(self):
        # Order between the NPU vendors only matters on multi-NPU boards; hailo
        # is checked before qnn so it wins when both are present.
        self.assertEqual(_recommend(has_hailo_npu=True, has_qualcomm_npu=True), "hailo")

    def test_edge_tpu_uses_tflite(self):
        self.assertEqual(_recommend(has_edge_tpu=True), "tflite")

    def test_nvidia_without_tensorrt_falls_back_to_onnx(self):
        self.assertEqual(_recommend(has_nvidia=True, has_tensorrt=False), "onnx")

    def test_nvidia_with_tensorrt_uses_engine(self):
        self.assertEqual(_recommend(has_nvidia=True, has_tensorrt=True), "engine")

    def test_tpu_uses_tpu_format(self):
        self.assertEqual(_recommend(has_tpu=True), "tpu")

    def test_tpu_runtime_detected_without_host_markers(self):
        import iSpy.config.AutoOpt as ao

        torch_xla = types.ModuleType("torch_xla")
        torch_xla_core = types.ModuleType("torch_xla.core")
        xla_model = types.ModuleType("torch_xla.core.xla_model")
        xla_model.xla_device = MagicMock(return_value="xla:0")
        xla_runtime = types.ModuleType("torch_xla.runtime")
        xla_runtime.device_type = MagicMock(return_value="TPU")
        modules = {
            "torch_xla": torch_xla,
            "torch_xla.core": torch_xla_core,
            "torch_xla.core.xla_model": xla_model,
            "torch_xla.runtime": xla_runtime,
        }

        with (
            patch.dict(sys.modules, modules),
            patch.object(ao, "has_tpu_hardware", return_value=False),
        ):
            ao.has_tpu.cache_clear()
            try:
                self.assertTrue(ao.has_tpu())
            finally:
                ao.has_tpu.cache_clear()

    def test_tpu_hardware_without_torch_xla_installs_tpu(self):
        # torch_xla missing but the hardware is there - installer has to be told
        # tpu so it can pull torch_xla in, same escape hatch tensorrt gets.
        self.assertEqual(
            _recommend(has_tpu=False, has_tpu_hardware=True, ignore_dependencies=True),
            "tpu",
        )

    def test_tpu_hardware_without_torch_xla_ignored_by_default(self):
        self.assertEqual(_recommend(has_tpu=False, has_tpu_hardware=True), "onnx")

    def test_intel_gpu_uses_openvino(self):
        self.assertEqual(_recommend(has_intel_gpu=True), "openvino")

    def test_amd_gpu_uses_onnx(self):
        self.assertEqual(_recommend(has_amd_gpu=True), "onnx")

    def test_arm_edge_uses_tflite(self):
        self.assertEqual(_recommend(has_arm=True), "tflite")

    def test_no_special_hardware_defaults_to_onnx(self):
        self.assertEqual(_recommend(), "onnx")

    def test_runtime_unsupported_skips_coreml_on_apple_silicon(self):
        # Bug 7: CoreML has no runtime inference, so a load-and-run caller
        # must NOT be handed 'coreml' even on Apple hardware.
        self.assertEqual(
            _recommend(has_apple_silicon=True, runtime_supported=False), "onnx"
        )
        self.assertEqual(
            _recommend(has_apple_silicon=True, runtime_supported=True), "coreml"
        )

    def test_optimizable_recommended_format_skips_coreml_on_apple_silicon(self):
        # OptimizableModelPipeline.recommended_format() defaults the build
        # target, so it must never pick coreml when the runtime can't load the
        # artifact that pick would produce. The fallback may vary by host, e.g.
        # TFLite on ARM, so assert only the invariant this test is about.
        self.assertNotEqual(
            _pipeline_recommended_format(has_apple_silicon=True), "coreml"
        )

    def test_runtime_unsupported_skips_engine_on_nvidia(self):
        self.assertEqual(
            _recommend(has_nvidia=True, has_tensorrt=True, runtime_supported=False),
            "onnx",
        )
        self.assertEqual(
            _recommend(has_nvidia=True, has_tensorrt=True, runtime_supported=True),
            "engine",
        )

    def test_runtime_unsupported_keeps_openvino_on_intel(self):
        # OpenVINO IR is loadable at runtime by GenericYolo, so it must remain
        # a valid recommendation even when runtime_supported=False (the build-target
        # path used by auto-opt / ispy-optimize).
        self.assertEqual(
            _recommend(has_intel_gpu=True, runtime_supported=False), "openvino"
        )
        self.assertEqual(
            _recommend(has_intel_gpu=True, runtime_supported=True), "openvino"
        )

    def test_runtime_unsupported_keeps_cpu_backends(self):
        # rknn/tflite/tpu/hailo/qnn remain valid runtime formats and shouldn't
        # be skipped.
        self.assertEqual(
            _recommend(has_rockchip_npu=True, runtime_supported=False), "rknn"
        )
        self.assertEqual(
            _recommend(has_edge_tpu=True, runtime_supported=False), "tflite"
        )
        self.assertEqual(
            _recommend(has_hailo_npu=True, runtime_supported=False), "hailo"
        )
        self.assertEqual(
            _recommend(has_qualcomm_npu=True, runtime_supported=False), "qnn"
        )


class TestBoxResults(unittest.TestCase):
    def test_box_stores_values(self):
        b = Box([10, 20, 50, 60], 0.95)
        self.assertEqual(b.xyxy, [10, 20, 50, 60])
        self.assertAlmostEqual(b.conf, 0.95)

    def test_results_plot_returns_frame(self):
        frame = make_frame()
        boxes = [Box([10, 10, 100, 100], 0.9), Box([150, 150, 200, 200], 0.7)]
        r = Results(boxes, frame.shape)
        out = r.plot(frame.copy())
        self.assertEqual(out.shape, frame.shape)

    def test_results_plot_empty_boxes(self):
        frame = make_frame()
        r = Results([], frame.shape)
        out = r.plot(frame.copy())
        self.assertEqual(out.shape, frame.shape)

    def test_results_str(self):
        r = Results([Box([0, 0, 10, 10], 0.5)], (320, 320))
        self.assertIn("boxes=1", str(r))

    def test_results_plot_draws_rectangle(self):
        frame = make_frame()
        r = Results([Box([10, 10, 50, 50], 0.9)], frame.shape)
        out = r.plot(frame.copy())
        self.assertFalse(np.array_equal(out, frame))


class TestGenericYoloModelSelection(unittest.TestCase):
    def setUp(self):
        # Patch genericYolo.RKNNLite and RKNN_FOUND directly so tests work regardless of import order
        from iSpy import vision

        self._original_rknnlite = vision.genericYolo.RKNNLite
        self._original_rknn_found = vision.genericYolo.RKNN_FOUND
        vision.genericYolo.RKNNLite = FakeRKNNLite
        vision.genericYolo.RKNN_FOUND = True

    def tearDown(self):
        # Restore original RKNNLite and RKNN_FOUND
        from iSpy import vision

        vision.genericYolo.RKNNLite = self._original_rknnlite
        vision.genericYolo.RKNN_FOUND = self._original_rknn_found

    def _dummy_model(self, suffix=".rknn", size=4096):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "model" + suffix)
        with open(path, "wb") as f:
            f.write(b"\x00" * size)
        return path

    def test_missing_model_file_raises_model_file_error(self):
        # graceful-degradation contract: boot catches ModelFileError and runs
        # the camera without detection instead of crashing the whole app
        with self.assertRaises(ModelFileError):
            GenericYolo({"file_path": "does/not/exist.rknn", "task": "detect"})

    def test_empty_model_file_raises_model_file_error(self):
        path = self._dummy_model(size=0)
        with self.assertRaises(ModelFileError):
            GenericYolo({"file_path": path, "task": "detect"})

    def test_truncated_model_file_raises_model_file_error(self):
        path = self._dummy_model(size=512)  # under the 1 KiB floor
        with self.assertRaises(ModelFileError):
            GenericYolo({"file_path": path, "task": "detect"})

    def test_loads_rknn_when_rknnlite_present(self):
        path = self._dummy_model(".rknn")
        w = GenericYolo({"file_path": path, "task": "detect", "num_classes": 1})
        self.assertEqual(w.model_type, "rknn")

    def test_rknn_release_calls_model_release(self):
        path = self._dummy_model(".rknn")
        w = GenericYolo({"file_path": path, "task": "detect", "num_classes": 1})
        w.model.release = MagicMock()
        w.release()
        w.model.release.assert_called_once()


class TestCompiledFormatGenericYolo(unittest.TestCase):
    def _complete_cfg(self, path):
        return {
            "file_path": path,
            "task": "detect",
            "num_classes": 1,
            "input_size": [640, 640],
            "frame_batches": 1,
            "min_conf": 0.25,
            "input": {
                "layout": "nchw",
                "dtype": "float32",
                "letterbox": False,
                "normalize": True,
                "scale": 255.0,
            },
            "output": {
                "format": "raw",
                "layout": "features_first",
                "quantization": "none",
                "box_format": "cxcywh",
                "score_mode": "multi_class",
                "scores_are_logits": False,
                "apply_software_nms": False,
                "nms_iou": 0.45,
            },
        }

    def test_engine_constructs_and_predicts_results(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        eng = os.path.join(tmp.name, "model.engine")
        with open(eng, "wb") as f:
            f.write(b"\x00" * 4096)

        with patch.object(torch_mod.cuda, "is_available", return_value=True), patch.object(
            torch_mod.cuda, "device_count", return_value=1
        ):
            w = GenericYolo(self._complete_cfg(eng))
            self.assertEqual(w.model_type, "engine")
            res = w.predict(make_frame())
        self.assertIsInstance(res, Results)
        self.assertEqual(res.boxes, [])

    def test_engine_requires_tensorrt(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        eng = os.path.join(tmp.name, "model.engine")
        with open(eng, "wb") as f:
            f.write(b"\x00" * 4096)

        # tensorrt is faked at module load above; null it out in sys.modules so
        # the in-method `import tensorrt` fails, and assert GenericYolo surfaces
        # the ImportError from _load_engine.
        with self.assertRaises(ImportError):
            with patch.dict(sys.modules, {"tensorrt": None}):
                GenericYolo(self._complete_cfg(eng))

    def test_engine_plan_header_is_skipped_from_file(self):
        from iSpy.vision.engine_utils import read_engine_plan

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.engine"
            header = b'{"producer": "ultralytics"}'
            plan = b"serialized-plan"
            path.write_bytes(len(header).to_bytes(4, "little") + header + plan)
            self.assertEqual(read_engine_plan(path), plan)

    def test_engine_plan_unrecognized_header_falls_back_to_raw(self):
        from iSpy.vision.engine_utils import read_engine_plan

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.engine"
            raw = b"raw-plan-bytes" * 4
            path.write_bytes(raw)
            self.assertEqual(read_engine_plan(path), raw)

    def test_tpu_batch_padding_repeats_last_frame(self):
        frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (1, 2, 3)]

        batch = GenericYolo._pad_tpu_batch(frames, 4)

        self.assertEqual(batch.shape, (4, 2, 2, 3))
        self.assertEqual(batch[:, 0, 0, 0].tolist(), [1, 2, 3, 3])
        with self.assertRaises(ValueError):
            GenericYolo._pad_tpu_batch(frames, 2)

    def test_tpu_batch_and_dtype_config_defaults(self):
        config = normalize_model_config(self._complete_cfg("unused.onnx"))
        self.assertEqual(config["batch_size"], 1)
        self.assertEqual(config["tpu_dtype"], "bf16")

        config = self._complete_cfg("unused.onnx")
        config.update(batch_size=4, tpu_dtype="fp32")
        normalized = normalize_model_config(config)
        self.assertEqual(normalized["batch_size"], 4)
        self.assertEqual(normalized["tpu_dtype"], "fp32")

        config["tpu_dtype"] = "float16"
        with self.assertRaisesRegex(ValueError, "tpu_dtype"):
            normalize_model_config(config)

    def test_tpu_predict_preserves_order_across_batches(self):
        wrapper = GenericYolo.__new__(GenericYolo)
        wrapper.model_type = "tpu"
        wrapper.batch_size = 2
        wrapper._preprocess_tpu_frame = lambda frame: frame
        batch_lengths = []

        def forward(batch):
            batch_lengths.append(len(batch))
            return np.asarray([frame[0, 0, 0] for frame in batch])

        wrapper.forward_tpu_batch = forward
        wrapper.postprocess_tpu_output = lambda output, shape: (int(output), shape[0])
        frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
        shapes = [(10, 11, 3), (20, 21, 3), (30, 31, 3)]

        results = wrapper.predict(frames, orig_shape=shapes)

        self.assertEqual(batch_lengths, [2, 1])
        self.assertEqual(results, [(10, 10), (20, 20), (30, 30)])

    def test_tpu_output_uses_software_nms(self):
        wrapper = GenericYolo.__new__(GenericYolo)
        wrapper.num_classes = 1
        wrapper.output = {
            "format": "raw",
            "layout": "features_first",
            "box_format": "cxcywh",
            "score_mode": "objectness",
            "apply_software_nms": False,
            "nms_iou": 0.45,
            "scores_are_logits": False,
        }
        wrapper._configure_tpu_output()
        wrapper.model_type = "tpu"
        wrapper.task = "detect"
        wrapper.input_size = (640, 640)
        wrapper.min_conf = 0.25
        wrapper._output_verified = False
        wrapper._feat_width = 0
        duplicate_boxes = np.array(
            [[320, 320, 100, 100, 0.95], [322, 320, 100, 100, 0.85]],
            dtype=np.float32,
        ).T

        result = wrapper.postprocess([duplicate_boxes], (640, 640, 3))

        self.assertTrue(wrapper.output["apply_software_nms"])
        self.assertEqual(wrapper.output["score_mode"], "objectness")
        self.assertEqual(len(result.boxes), 1)

    def test_cpu_onnx_uses_only_cpu_provider(self):
        session_options = types.SimpleNamespace()
        used = {}

        class FakeSession:
            def get_inputs(self):
                return [types.SimpleNamespace(name="images")]

            def get_outputs(self):
                return [types.SimpleNamespace(name="output0")]

            def get_providers(self):
                return ["CPUExecutionProvider"]

        ort = types.ModuleType("onnxruntime")
        ort.set_default_logger_severity = lambda level: None
        ort.get_available_providers = lambda: [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        ort.SessionOptions = lambda: session_options
        ort.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL=1)

        def create_session(path, sess_options, providers):
            used["providers"] = providers
            return FakeSession()

        ort.InferenceSession = create_session
        wrapper = GenericYolo.__new__(GenericYolo)
        wrapper.device = "cpu"
        wrapper.logger = MagicMock()

        with patch.dict(sys.modules, {"onnxruntime": ort, "onnxruntime_qnn": None}):
            wrapper._load_onnx("cpu-model.onnx")

        self.assertEqual(used["providers"], ["CPUExecutionProvider"])
        self.assertEqual(wrapper.provider, "CPUExecutionProvider")

    def test_openvino_dir_constructs_and_predicts_results(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ir_dir = os.path.join(tmp.name, "model_openvino_model")
        os.makedirs(ir_dir, exist_ok=True)
        with open(os.path.join(ir_dir, "model_openvino_model.xml"), "wb") as f:
            f.write(b"\x00" * 4096)

        w = GenericYolo(self._complete_cfg(ir_dir))
        self.assertEqual(w.model_type, "openvino")

        res = w.predict(make_frame())
        self.assertIsInstance(res, Results)
        self.assertEqual(res.boxes, [])

    def test_openvino_single_xml_constructs(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        xml_path = os.path.join(tmp.name, "net_openvino_model.xml")
        with open(xml_path, "wb") as f:
            f.write(b"\x00" * 4096)

        w = GenericYolo(self._complete_cfg(xml_path))
        self.assertEqual(w.model_type, "openvino")


class TestTpuQueueFreshness(unittest.TestCase):
    def test_drop_oldest_keeps_newest_frames_in_order(self):
        import queue

        from iSpy.vision.pipelines.object_detection import _put_latest

        frames = queue.Queue(maxsize=2)
        for sequence in (1, 2, 3):
            _put_latest(frames, sequence)

        self.assertEqual([frames.get_nowait(), frames.get_nowait()], [2, 3])


class TestBenchmarkValidity(unittest.TestCase):
    def test_missing_model_is_a_benchmark_error(self):
        from types import SimpleNamespace

        from iSpy.validations.benchmarking import _require_benchmark_model

        with self.assertRaisesRegex(RuntimeError, "model failed to load"):
            _require_benchmark_model(SimpleNamespace(model=None))

    def test_zero_completed_inferences_is_a_benchmark_error(self):
        from iSpy.validations.benchmarking import _require_benchmark_count

        with self.assertRaisesRegex(RuntimeError, "no inference completed"):
            _require_benchmark_count(0)

    def test_tpu_core_probe_skips_when_torch_xla_is_missing(self):
        from iSpy.validations.benchmarking import _local_tpu_core_count

        with patch.dict(sys.modules, {"torch_xla": None}):
            self.assertEqual(_local_tpu_core_count(), 0)

    def test_cpu_system_info_does_not_import_torch_xla(self):
        import builtins

        from iSpy.validations.bench_to_matplotlib import collect_system_info

        real_import = builtins.__import__

        def import_without_xla(name, *args, **kwargs):
            if name == "torch_xla" or name.startswith("torch_xla."):
                raise AssertionError("CPU benchmark imported torch_xla")
            return real_import(name, *args, **kwargs)

        with (
            patch("iSpy.config.AutoOpt.has_tpu_hardware", return_value=False),
            patch("builtins.__import__", side_effect=import_without_xla),
        ):
            info = collect_system_info()

        self.assertNotIn("torch_xla", info)


class TestModelMetadata(unittest.TestCase):
    def test_rknn_metadata_contract(self):
        m = derive_format_metadata(
            {"task": "detect", "nc": 1, "input_size": [640, 640]}, "rknn"
        )
        self.assertEqual(m["output_format"], "raw")
        self.assertEqual(m["input_layout"], "nhwc")
        self.assertEqual(m["input_dtype"], "uint8")
        self.assertEqual(m["quantization"], "int8")
        self.assertFalse(m["scores_are_logits"])

    def test_onnx_metadata_contract(self):
        m = derive_format_metadata({"task": "detect", "nc": 3}, "onnx")
        self.assertEqual(m["output_format"], "raw")
        self.assertEqual(m["input_layout"], "nchw")
        self.assertEqual(m["input_dtype"], "float32")
        self.assertEqual(m["input_normalize"], True)
        self.assertEqual(m["quantization"], "none")
        self.assertEqual(m["score_mode"], "multi_class")

    def test_metadata_write_read_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "model.rknn"
            write_metadata(
                metadata_path_for(model),
                {
                    "nc": 2,
                    "names": {0: "coral", 1: "reef"},
                    "input_size": [320, 320],
                    "scores_are_logits": False,
                    "nms_iou": 0.45,
                },
            )
            self.assertTrue(metadata_path_for(model).exists())
            back = read_metadata(model)
            self.assertEqual(back["nc"], 2)
            self.assertEqual(back["names"], {0: "coral", 1: "reef"})
            self.assertAlmostEqual(back["nms_iou"], 0.45)

    def test_missing_metadata_round_trip_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(read_metadata(Path(d) / "nope.rknn"))


class TestPosePtRegression(unittest.TestCase):
    _yolo_pt = None

    @classmethod
    def _locate_pose_model(cls):
        repo_root = Path(__file__).resolve().parents[2]
        candidate = repo_root / "YoloModels" / "pytorch" / "_default_pose.pt"
        return candidate if candidate.exists() else None

    @classmethod
    def setUpClass(cls):
        if cls._locate_pose_model() is None:
            raise unittest.SkipTest("Required test model _default_pose.pt not found")

        import importlib

        # Remove fake/import-time-only torch modules so the real torch can be used.
        # The real torch is needed for actual inference testing.
        for _m in [
            m for m in list(sys.modules) if m == "torch" or m.startswith("torch.")
        ]:
            sys.modules.pop(_m, None)
        try:
            real_torch = importlib.import_module("torch")
            if getattr(real_torch, "nn", None) is None:
                raise RuntimeError("real torch missing torch.nn")
            cls._yolo_pt = importlib.import_module("iSpy.vision.yolo_pt")
            importlib.reload(cls._yolo_pt)
        except Exception as exc:  # no real torch on this machine
            raise unittest.SkipTest(f"real torch unavailable: {exc}")

    # No tearDownClass - the torch module removal is intentional for this test class
    # and is acceptable since this test specifically needs the real torch

    def test_pose_pt_loads_and_infers(self):
        model_path = self._locate_pose_model()
        model = self._yolo_pt.load_yolo_pt(str(model_path), task="pose")
        self.assertEqual(model.task, "pose")
        self.assertEqual((model.num_keypoints, model.keypoint_dims), (17, 3))

        rng = np.random.RandomState(20260901)
        frame = (rng.rand(480, 640, 3) * 255).astype(np.uint8)
        results = model([frame], imgsz=640, conf=0.001)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertIsNotNone(
            r.keypoints,
            "low-threshold run produced no candidate - pose decode never exercised",
        )
        kd = r.keypoints.data
        self.assertEqual(kd.ndim, 3)
        self.assertEqual(kd.shape[1:], (17, 3))


class TestValidateSystemRegression(unittest.TestCase):
    def test_broken_model_file_path_flips_validate_system_to_false(self):
        from iSpy.validations.validate_system import validate_system

        old_cwd = os.getcwd()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(os.chdir, old_cwd)

        bad_dir = Path(tmp.name) / "YoloModels" / "pytorch"
        bad_dir.mkdir(parents=True)
        # NOT a .txt file - validate_model_files exempts .txt from the
        # invalid-path check, so a .txt "corrupt model" would slip through and
        # validate_system() would recurse into run_unit_tests() forever.
        (bad_dir / "corrupt model.svb").write_bytes(b"junk")
        os.chdir(tmp.name)

        # validate_model_files() raises on the invalid path -> validate_system()
        # returns False (never reaches run_unit_tests -> no recursion).
        self.assertFalse(validate_system())


class _QuietLogging:
    def __init__(self):
        self._stack = ExitStack()

    def __enter__(self):
        import logging

        logger = logging.getLogger("iSpy")
        self._stack.__enter__()
        self._stack.enter_context(patch.object(logger, "debug", MagicMock()))
        self._stack.enter_context(patch.object(logger, "info", MagicMock()))
        self._stack.enter_context(patch.object(logger, "warning", MagicMock()))
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)


class TestObjectDetectionFillMissingConfigRegression(unittest.TestCase):
    @staticmethod
    def _restore_scipy_optimize(had_optimize, prior_optimize):
        if had_optimize:
            sys.modules["scipy.optimize"] = prior_optimize
        else:
            sys.modules.pop("scipy.optimize", None)

    def _build_pipeline(self, model_path: str):
        # unit_tests.py fakes scipy with only scipy.spatial at import time, but
        # the object_detection -> calibration import chain needs scipy.optimize.
        # Register a fake submodule (and clean it up) so the pipeline module
        # imports on machines where real scipy is absent.

        scipy_optimize = types.ModuleType("scipy.optimize")
        scipy_optimize.least_squares = MagicMock()
        had_optimize = "scipy.optimize" in sys.modules
        prior_optimize = sys.modules.get("scipy.optimize")
        sys.modules["scipy.optimize"] = scipy_optimize
        self.addCleanup(self._restore_scipy_optimize, had_optimize, prior_optimize)

        from iSpy.config.iSpyConfig import iSpyCameraConfig, iSpyConfig
        from iSpy.vision.pipelines.object_detection import ObjectDetectionPipeline

        config = iSpyConfig()
        cam_entry = {
            "name": "bug6_cam",
            "source": 99,
            "fps_cap": 1000,
            "yaw": 0,
            "pitch": 0,
            "height": 1.0,
            "x": 0,
            "y": 0,
            "grayscale": False,
            "subsystem": "test",
            "calibration": {
                "distance": 1.0,
                "game_piece_size": 1.0,
                "size": 100,
                "fov": 90,
            },
            "pipeline": {
                "name": "object_detection",
                "settings": {
                    "vision_model": {"file_path": model_path, "task": "detect"}
                },
            },
        }
        config.set("camera_configs", {"bug6_cam": cam_entry})
        cam_cfg = iSpyCameraConfig(cam_entry)
        with _QuietLogging():
            return ObjectDetectionPipeline(cam_cfg, config)

    def test_malformed_sidecar_valueerror_still_constructs_with_model_none(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_dir = Path(tmp.name)

        weight = tmp_dir / "corrupt.pt"
        weight.write_bytes(
            b"\x00" * 512
        )  # under the load floor -> ModelFileError later

        sidecar = tmp_dir / "corrupt_metadata.yaml"
        sidecar.write_text("nc: not_an_int\n", encoding="utf-8")

        try:
            camera = self._build_pipeline(str(weight))
        except ValueError as e:
            self.fail(
                "ObjectDetectionPipeline.__init__ must not propagate the "
                f"fill_missing_config ValueError (BUG 6); got: {e}"
            )

        # the camera should surface as errored, not disappear entirely
        self.assertIsNone(camera.model)
        try:
            ready, status = camera.is_ready()
        finally:
            # the pipeline starts a background _reader thread that keeps
            # retrying the bogus source (99) - stop it so it doesn't spam
            # stderr / keep the boot log alive on a nonexistent device
            camera.destroy()
        self.assertFalse(ready)
        self.assertTrue(status.startswith("error:"), f"unexpected status: {status!r}")

    def test_good_sidecar_constructs_ready(self):
        # sanity: with NO sidecar and a missing file (fill_missing_config short
        # circuits on os.path.exists -> returns config as-is), the pipeline
        # still constructs with model None + error status, so a missing model
        # does not take down the camera either.
        camera = self._build_pipeline("does/not/exist.pt")
        self.assertIsNone(camera.model)
        try:
            ready, status = camera.is_ready()
        finally:
            camera.destroy()
        self.assertFalse(ready)
        self.assertTrue(status.startswith("error:"))

    def test_destroy_stops_reader_thread_promptly(self):
        # Regression for the boot-time leak: a pipeline built on a bogus source
        # (99) must not leave its daemon _reader thread retrying /dev/video99
        # after destroy(). On the RKNN board that leaked thread's cv2 polling
        # raced native teardown at interpreter shutdown -> Segmentation fault.
        camera = self._build_pipeline("does/not/exist.pt")
        reader = getattr(camera, "_reader_thread", None)
        self.assertIsNotNone(reader, "reader thread should have been started")
        self.assertTrue(reader.is_alive())
        camera.destroy()
        reader.join(timeout=2.0)
        self.assertFalse(
            reader.is_alive(),
            "destroy() must stop the reader thread (polling must not survive "
            "into interpreter shutdown)",
        )


class TestEKFTracker(unittest.TestCase):
    def _make_tracker(self, distance_threshold=1.0):
        from iSpy.plugins.trackers.BuiltIn.EKFTracker import EKFTracker

        return EKFTracker(
            {
                "config": {
                    "process_noise": 0.5,
                    "measurement_noise": 0.1,
                    "distance_threshold": distance_threshold,
                    "stale_threshold": 2.0,
                },
                "global_config": None,
            }
        )

    def test_ekf_smoothes_better_than_raw_measurements(self):
        import importlib
        from types import SimpleNamespace

        from iSpy.vision.Object import Object

        rng = np.random.default_rng(7)
        tracker = self._make_tracker(distance_threshold=3.0)
        ekf_module = importlib.import_module(
            "iSpy.plugins.trackers.BuiltIn.EKFTracker"
        )
        clock_value = 0.0

        def advance_clock():
            nonlocal clock_value
            clock_value += 0.05
            return clock_value

        # constant-velocity ground truth: start at origin, move +1.0 m/s in x
        velocity = np.array([1.0, 0.0, 0.0])
        dt = 0.1
        n = 30
        true_positions = []
        raw_sq = 0.0
        # collect smoothed positions as they are produced
        smoothed_path = []

        with patch.object(
            ekf_module,
            "time",
            SimpleNamespace(monotonic=advance_clock),
        ):
            for i in range(n):
                t_i = i * dt
                true = velocity * t_i
                true_positions.append(true)
                noise = rng.normal(0.0, 0.5, size=3)
                raw = true + noise
                raw_sq += float(np.linalg.norm(noise) ** 2)

                obj = Object(
                    x=float(raw[0]), y=float(raw[1]), z=float(raw[2]), name="ball"
                )
                tracker.update([obj], 0.0, 0.0, 0.0, 0.0)
                smoothed_path.append(
                    np.array(tracker.tracked_objects[0].get_position())
                )

        true_arr = np.array(true_positions)
        sm_arr = np.array(smoothed_path)

        raw_rmse = float(np.sqrt(raw_sq / n))
        smoothed_rmse = float(
            np.sqrt(np.mean(np.sum((sm_arr - true_arr) ** 2, axis=1)))
        )

        self.assertLess(
            smoothed_rmse,
            raw_rmse,
            msg=f"EKF ({smoothed_rmse:.4f}) must beat raw noise ({raw_rmse:.4f})",
        )

    def test_same_identity_gating(self):
        # a cone must never merge into a robot track, mirroring ObjectTracker
        from iSpy.vision.Object import Object

        tracker = self._make_tracker()
        robot = Object(x=0.0, y=0.0, z=0.0, name="robot")
        cone = Object(x=0.05, y=0.0, z=0.0, name="cone")
        tracker.update([robot], 0.0, 0.0, 0.0, 0.0)
        tracker.update([cone], 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(len(tracker.tracked_objects), 2)

    def test_run_returns_tracked_objects(self):
        tracker = self._make_tracker()
        self.assertEqual(tracker.run(), tracker.tracked_objects)


class TestSelectionState(unittest.TestCase):
    def test_basic_lifecycle(self):
        from iSpy.plugins.selection import SelectionState

        s = SelectionState()
        self.assertIsNone(s.selected_id)
        self.assertIsNone(s.age_s())
        s.select(7)
        self.assertEqual(s.selected_id, 7)
        self.assertIsNotNone(s.age_s())
        s.clear()
        self.assertIsNone(s.selected_id)
        self.assertIsNone(s.age_s())

    def test_addon_base_exposes_selection(self):
        from iSpy.plugins.bases import AddonBase

        self.assertTrue(hasattr(AddonBase, "selection"))
        ctx = {"selection": object()}
        inst = AddonBase.__new__(AddonBase)
        inst.context = ctx
        self.assertIs(inst.selection, ctx["selection"])


class TestTargetSelector(unittest.TestCase):
    def _make_selector(
        self, reacquire_timeout_s: float = 1.0, output_key: str = "selected_target"
    ):
        from iSpy.plugins.selection import SelectionState
        from iSpy.plugins.utilities.BuiltIn.TargetSelector import TargetSelector

        selection = SelectionState()
        selector = TargetSelector(
            {
                "config": {
                    "reacquire_timeout_s": reacquire_timeout_s,
                    "output_key": output_key,
                },
                "flask_app": None,
                "selection": selection,
            }
        )
        return selector, selection

    def test_nothing_selected_publishes_none(self):
        from iSpy.vision.Object import Object

        selector, _selection = self._make_selector()
        frame_data = {"detections": [Object(x=1.0, y=2.0, z=3.0, id=9)]}
        selector.update(frame_data)
        self.assertEqual(frame_data["addon_data"]["selected_target"], None)

    def test_publishes_selected_object(self):
        from iSpy.vision.Object import Object

        selector, selection = self._make_selector()
        obj = Object(x=1.0, y=2.0, z=3.0, id=5, name="cone")
        selection.select(5)
        frame_data = {"detections": [obj]}
        selector.update(frame_data)
        published = frame_data["addon_data"]["selected_target"]
        self.assertIsNotNone(published)
        self.assertEqual(published["id"], 5)
        self.assertEqual(published["name"], "cone")

    def test_holds_lock_inside_reacquire_timeout(self):
        from iSpy.vision.Object import Object

        selector, selection = self._make_selector(reacquire_timeout_s=5.0)
        selection.select(5)
        frame_data = {"detections": [Object(x=1.0, y=2.0, z=3.0, id=5)]}
        selector.update(frame_data)
        # id drops out briefly but we are still inside the timeout
        frame_data2 = {"detections": []}
        selector.update(frame_data2)
        # selection must be retained, and nothing new published this tick
        self.assertEqual(selection.selected_id, 5)
        self.assertNotIn("selected_target", frame_data2.get("addon_data", {}))

    def test_idless_objects_noop_no_raise(self):
        selector, selection = self._make_selector()
        selection.select(5)
        # Object always has an id; simulate a plain id-less fallback dict
        obj_no_id = {"x": 1.0, "y": 2.0, "z": 3.0}
        frame_data = {"detections": [obj_no_id]}
        selector.update(frame_data)  # must not raise
        published = frame_data.get("addon_data", {}).get("selected_target")
        self.assertIsNone(published)


class TestCalibrationGating(unittest.TestCase):
    @staticmethod
    def _restore_scipy_optimize(had_optimize, prior_optimize):
        if had_optimize:
            sys.modules["scipy.optimize"] = prior_optimize
        else:
            sys.modules.pop("scipy.optimize", None)

    def _calibration(self, **over):
        calib = {"distance": 0.0, "game_piece_size": 0.0, "size": 0, "fov": 0}
        calib.update(over)
        return calib

    def _build_detect_pipeline(self, task):
        scipy_optimize = types.ModuleType("scipy.optimize")
        scipy_optimize.least_squares = MagicMock()
        had = "scipy.optimize" in sys.modules
        prior = sys.modules.get("scipy.optimize")
        sys.modules["scipy.optimize"] = scipy_optimize
        self.addCleanup(self._restore_scipy_optimize, had, prior)

        from iSpy.config.iSpyConfig import iSpyCameraConfig, iSpyConfig
        from iSpy.vision.pipelines.object_detection import ObjectDetectionPipeline

        config = iSpyConfig()
        cam_entry = {
            "name": "calib_cam",
            "source": 99,
            "fps_cap": 1000,
            "yaw": 0,
            "pitch": 0,
            "height": 1.0,
            "x": 0,
            "y": 0,
            "grayscale": False,
            "subsystem": "test",
            "calibration": self._calibration(),
            "pipeline": {
                "name": "object_detection",
                "settings": {
                    "vision_model": {"file_path": "does/not/exist.pt", "task": task}
                },
            },
        }
        config.set("camera_configs", {"calib_cam": cam_entry})
        cam_cfg = iSpyCameraConfig(cam_entry)
        with _QuietLogging():
            return ObjectDetectionPipeline(cam_cfg, config)

    def test_detect_uncalibrated_is_yellow_and_runs(self):
        camera = self._build_detect_pipeline("detect")
        try:
            level, msg = camera.calibration_status()
            self.assertEqual(level, "yellow")
            self.assertIn("Needs Calibration for Better Accuracy", msg)
            # _gate_uncalibrated must NOT block a detect pipeline
            self.assertIsNone(
                camera._gate_uncalibrated(np.zeros((10, 10, 3), dtype=np.uint8))
            )
        finally:
            camera.destroy()

    def test_pose_uncalibrated_is_red_and_blocks(self):
        camera = self._build_detect_pipeline("pose")
        try:
            level, msg = camera.calibration_status()
            self.assertEqual(level, "red")
            self.assertEqual(msg, "Needs Calibration")
            # _gate_uncalibrated blocks a pose pipeline
            gated = camera._gate_uncalibrated(np.zeros((10, 10, 3), dtype=np.uint8))
            self.assertIsNotNone(gated)
            objs, frame = gated
            self.assertEqual(objs, [])
            self.assertIsNotNone(frame)
        finally:
            camera.destroy()

    def test_calibrated_detect_is_green(self):
        scipy_optimize = types.ModuleType("scipy.optimize")
        scipy_optimize.least_squares = MagicMock()
        had = "scipy.optimize" in sys.modules
        prior = sys.modules.get("scipy.optimize")
        sys.modules["scipy.optimize"] = scipy_optimize
        self.addCleanup(self._restore_scipy_optimize, had, prior)

        from iSpy.config.iSpyConfig import iSpyCameraConfig, iSpyConfig
        from iSpy.vision.pipelines.object_detection import ObjectDetectionPipeline

        config = iSpyConfig()
        cam_entry = {
            "name": "calib_cam",
            "source": 99,
            "fps_cap": 1000,
            "yaw": 0,
            "pitch": 0,
            "height": 1.0,
            "x": 0,
            "y": 0,
            "grayscale": False,
            "subsystem": "test",
            "calibration": self._calibration(fov=90),
            "pipeline": {
                "name": "object_detection",
                "settings": {
                    "vision_model": {"file_path": "does/not/exist.pt", "task": "detect"}
                },
            },
        }
        config.set("camera_configs", {"calib_cam": cam_entry})
        cam_cfg = iSpyCameraConfig(cam_entry)
        with _QuietLogging():
            camera = ObjectDetectionPipeline(cam_cfg, config)
        try:
            level, msg = camera.calibration_status()
            self.assertEqual(level, "ready")
            # calibrated gate lets the pipeline through (model missing -> not
            # processable, but that's a separate concern from calibration)
            self.assertIsNone(
                camera._gate_uncalibrated(np.zeros((10, 10, 3), dtype=np.uint8))
            )
        finally:
            camera.destroy()


class TestObjectResetRegression(unittest.TestCase):
    def test_reset_time_clears_destroyed_and_alive(self):
        # Bug 3: age an Object past alive_time so update() marks it destroyed,
        # then reset_time() must clear destroyed/alive so the next update()
        # does not immediately re-destroy it (a persistent pipeline reuses one
        # Object forever).
        import time

        from iSpy.vision.Object import Object

        obj = Object(x=0.0, y=0.0, z=0.0, alive_time=0.01)
        obj.start_time = time.perf_counter() - 5.0
        obj.update()
        self.assertTrue(obj.destroyed)

        obj.reset_time()
        self.assertFalse(obj.destroyed)
        self.assertEqual(obj.alive, 0)

        obj.update()
        self.assertFalse(obj.destroyed, "reset object must not re-destroy")

    def test_tracker_revives_stale_detection(self):
        # Bug 3: after an object ages out (destroyed -> filtered -> re-appended),
        # _merge must restart its age clock instead of leaving it permanently
        # destroyed.
        import time

        from iSpy.plugins.trackers.BuiltIn.ObjectTracker import ObjectTracker
        from iSpy.vision.Object import Object

        tracker = ObjectTracker(
            {
                "config": {"distance_threshold": 1.0, "stale_threshold": 0.05},
                "global_config": None,
            }
        )
        obj = Object(x=1.0, y=0.0, z=0.0, name="voxel_world", id=7)
        tracker.update([obj], 0.0, 0.0, 0.0)
        self.assertEqual(len(tracker.tracked_objects), 1)

        # age the object far past its stale threshold so the next tick drops it
        obj.start_time = time.perf_counter() - 5.0
        obj.update()
        self.assertTrue(obj.destroyed)

        tracker.update([obj], 0.0, 0.0, 0.0)
        self.assertEqual(
            len(tracker.tracked_objects),
            1,
            "stale-absent detection must be revived, not left destroyed",
        )
        # and the revived track must survive the following tick too
        tracker.update([obj], 0.0, 0.0, 0.0)
        self.assertEqual(len(tracker.tracked_objects), 1)

    def test_ekf_tracker_revives_stale_detection(self):
        import time

        from iSpy.plugins.trackers.BuiltIn.EKFTracker import EKFTracker
        from iSpy.vision.Object import Object

        tracker = EKFTracker(
            {
                "config": {
                    "process_noise": 0.5,
                    "measurement_noise": 0.1,
                    "distance_threshold": 1.0,
                    "stale_threshold": 0.05,
                },
                "global_config": None,
            }
        )
        obj = Object(x=1.0, y=0.0, z=0.0, name="voxel_world", id=7)
        tracker.update([obj], 0.0, 0.0, 0.0)
        self.assertEqual(len(tracker.tracked_objects), 1)

        obj.start_time = time.perf_counter() - 5.0
        obj.update()
        self.assertTrue(obj.destroyed)

        tracker.update([obj], 0.0, 0.0, 0.0)
        self.assertEqual(len(tracker.tracked_objects), 1)
        tracker.update([obj], 0.0, 0.0, 0.0)
        self.assertEqual(len(tracker.tracked_objects), 1)


class TestOfflineNetworkPolicy(unittest.TestCase):
    def tearDown(self):
        from iSpy.core import network_policy

        network_policy.set_offline(False)

    def test_address_policy_truth_table(self):
        from iSpy.core.network_policy import is_private_address

        for host in (
            "127.0.0.1",
            "10.0.0.2",
            "172.16.0.1",
            "172.31.255.254",
            "192.168.10.1",
            "169.254.10.7",
            "::1",
            "fe80::1",
            "fd12::1",
            "::ffff:127.0.0.1",
            "localhost",
        ):
            self.assertTrue(is_private_address(host), host)

        for host in (
            "8.8.8.8",
            "172.15.0.1",  # just below the 172.16/12 block
            "172.32.0.1",  # just above it
            "192.169.0.1",
            "2001:4860:4860::8888",
            None,
            "",
        ):
            self.assertFalse(is_private_address(host), host)

    def test_unresolvable_hostname_counts_as_blocked(self):
        from iSpy.core.network_policy import is_private_address

        # while offline a name we cannot resolve might be anywhere - blocked
        self.assertFalse(is_private_address("offline-policy-test.invalid"))

    def test_socket_guard_blocks_public_and_clears(self):
        import socket

        from iSpy.core import network_policy

        original_connect = network_policy._original_connect
        try:
            network_policy.set_offline(True, "unit-test")
            self.assertTrue(network_policy.is_offline())
            self.assertEqual(network_policy.offline_reason(), "unit-test")
            self.assertIsNot(socket.socket.connect, original_connect)

            with self.assertRaises(network_policy.OfflineError):
                socket.getaddrinfo("8.8.8.8", 443)
            with socket.socket() as probe:
                with self.assertRaises(network_policy.OfflineError):
                    probe.connect(("8.8.8.8", 443))

            # loopback + LAN pass the guard. nothing listens on port 9, so a
            # connection error is expected - an OfflineError is not.
            for target in (("127.0.0.1", 9), ("10.0.0.2", 9)):
                try:
                    socket.create_connection(target, timeout=0.1)
                except network_policy.OfflineError:
                    self.fail(f"{target} must not be blocked while offline")
                except OSError:
                    pass
        finally:
            network_policy.set_offline(False)

        self.assertFalse(network_policy.is_offline())
        self.assertIs(socket.socket.connect, original_connect)
        # and the original lookup works again
        self.assertTrue(socket.getaddrinfo("127.0.0.1", 9))

    def test_report_dedupes_and_clears_on_flip(self):
        from iSpy.core import network_policy

        network_policy.set_offline(True, "unit-test")
        network_policy.record_skip("tracker", "probe", "needs internet")
        network_policy.record_skip("tracker", "probe", "needs internet")
        network_policy.record_degraded("pipeline", "cam", "feature", "reason")

        report = network_policy.get_report()
        self.assertTrue(report["offline"])
        self.assertEqual(len(report["skipped"]), 1)
        self.assertEqual(len(report["degraded"]), 1)
        self.assertEqual(report["skipped"][0]["name"], "probe")

        # flipping offline off must not leak the old run's report
        network_policy.set_offline(False)
        report = network_policy.get_report()
        self.assertFalse(report["offline"])
        self.assertEqual(report["skipped"], [])
        self.assertEqual(report["degraded"], [])

    def test_set_offline_is_idempotent_and_keeps_first_reason(self):
        import logging

        from iSpy.core import network_policy

        network_policy.set_offline(True, "config")

        records = []

        class _Handler(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Handler()
        policy_logger = logging.getLogger("iSpy.core.network_policy")
        policy_logger.addHandler(handler)
        try:
            # game mode arming on top of an already-offline boot must not
            # re-log or change the reason (no double-warn)
            network_policy.set_offline(True, "game_mode")
        finally:
            policy_logger.removeHandler(handler)

        self.assertEqual(network_policy.offline_reason(), "config")
        on_lines = [
            r.getMessage()
            for r in records
            if r.getMessage().startswith("Offline mode ON")
        ]
        self.assertEqual(on_lines, [])


class TestOfflineGate(unittest.TestCase):
    def tearDown(self):
        from iSpy.core import network_policy

        network_policy.set_offline(False)

    def _gate(self, cls, name="probe", group="tracker", logger=None):
        import logging

        from iSpy.iSpy import iSpy

        inst = iSpy.__new__(iSpy)
        inst.logger = logger or logging.getLogger("iSpy.validations.offline_test")
        return iSpy._gate_addon(inst, group, name, cls, {})

    def _requirement_class(self):
        from iSpy.core.network_policy import NetworkRequirement
        from iSpy.plugins.bases import AddonBase

        class _Needy(AddonBase):
            plugin_name = "test/needy"

            @classmethod
            def network_requirements(cls, settings):
                return [
                    NetworkRequirement("weights", "downloads weights", critical=True)
                ]

        return _Needy

    def test_critical_requirement_is_skipped_not_raised(self):
        from iSpy.core import network_policy

        needy = self._requirement_class()
        network_policy.set_offline(True, "unit-test")

        # boot validation runs this suite with logging.disable(ERROR), which
        # would silently drop the warning before assertLogs could see it.
        # Spy on the logger the gate is handed instead - independent of the
        # global logging state, so this passes both standalone and at boot.
        gate_logger = MagicMock()
        allowed = self._gate(needy, logger=gate_logger)

        self.assertFalse(allowed, "a critical need must keep the add-on unconstructed")
        warnings = [
            call.args[0] % call.args[1:]
            for call in gate_logger.warning.call_args_list
        ]
        self.assertTrue(
            any("skipping tracker 'probe'" in message for message in warnings),
            warnings,
        )

        report = network_policy.get_report()
        self.assertEqual([s["name"] for s in report["skipped"]], ["probe"])
        self.assertIn("downloads weights", report["skipped"][0]["reason"])

        # online, the very same add-on loads as normal
        network_policy.set_offline(False)
        self.assertTrue(self._gate(needy))

    def test_optional_requirement_degrades_but_keeps_the_addon(self):
        from iSpy.core import network_policy
        from iSpy.core.network_policy import NetworkRequirement
        from iSpy.plugins.bases import AddonBase

        class _Partial(AddonBase):
            plugin_name = "test/partial"

            @classmethod
            def network_requirements(cls, settings):
                return [
                    NetworkRequirement("sync", "online sync only", critical=False)
                ]

        network_policy.set_offline(True, "unit-test")
        self.assertTrue(self._gate(_Partial, name="partial", group="utility"))

        report = network_policy.get_report()
        self.assertEqual(report["skipped"], [])
        self.assertEqual(len(report["degraded"]), 1)
        self.assertEqual(report["degraded"][0]["feature"], "sync")

    def test_broken_requirements_do_not_break_boot(self):
        from iSpy.core import network_policy
        from iSpy.plugins.bases import AddonBase

        class _Broken(AddonBase):
            plugin_name = "test/broken"

            @classmethod
            def network_requirements(cls, settings):
                raise RuntimeError("boom")

        network_policy.set_offline(True, "unit-test")
        # raising here would take the whole boot down - it must not
        self.assertTrue(self._gate(_Broken, name="broken"))

    def test_needs_internet_shortcut(self):
        from iSpy.core import network_policy
        from iSpy.plugins.bases import AddonBase

        class _Flagged(AddonBase):
            plugin_name = "test/flagged"
            needs_internet = True

        reqs = _Flagged.network_requirements({})
        self.assertEqual(len(reqs), 1)
        self.assertTrue(reqs[0].critical)
        self.assertEqual(AddonBase.network_requirements({}), [])

        network_policy.set_offline(True, "unit-test")
        self.assertFalse(self._gate(_Flagged, name="flagged"))

    def test_conditional_requirement_reads_settings(self):
        from iSpy.core.network_policy import NetworkRequirement
        from iSpy.plugins.bases import AddonBase

        class _Conditional(AddonBase):
            plugin_name = "test/conditional"

            @classmethod
            def network_requirements(cls, settings):
                if settings.get("remote_log"):
                    return [
                        NetworkRequirement("remote log", "posts to a server", True)
                    ]
                return []

        self.assertEqual(_Conditional.network_requirements({}), [])
        self.assertEqual(_Conditional.network_requirements({"remote_log": False}), [])
        reqs = _Conditional.network_requirements({"remote_log": True})
        self.assertEqual(len(reqs), 1)

    def test_try_online_falls_back_offline_and_recovers(self):
        from iSpy.core import network_policy
        from iSpy.plugins.bases import AddonBase

        class _Helper(AddonBase):
            plugin_name = "test/helper"

        helper = _Helper.__new__(_Helper)
        network_policy.set_offline(True, "unit-test")

        calls = []
        self.assertEqual(
            helper.try_online("lookup", lambda: calls.append(1), fallback="fb"),
            "fb",
        )
        self.assertEqual(calls, [], "fn must not run while offline")
        self.assertFalse(helper.network_allowed)
        with self.assertRaises(network_policy.OfflineError):
            helper.require_network("downloading a thing")

        report = network_policy.get_report()
        self.assertEqual(report["degraded"][0]["feature"], "lookup")

        network_policy.set_offline(False)
        self.assertTrue(helper.network_allowed)
        self.assertEqual(helper.try_online("lookup", lambda: "ok"), "ok")

    def test_depth_anything_requirement_is_cache_conditional(self):
        from iSpy.vision.pipelines import depth_anything as da

        with (
            patch.object(da, "_depth_weights_cached", return_value=False),
            patch(
                "iSpy.vision.pipelines.optimizable.offline_build_requirements",
                return_value=[],
            ),
        ):
            reqs = da.DepthAnythingPipeline.network_requirements({})
        self.assertTrue(any(r.critical for r in reqs), reqs)

        with (
            patch.object(da, "_depth_weights_cached", return_value=True),
            patch(
                "iSpy.vision.pipelines.optimizable.offline_build_requirements",
                return_value=[],
            ),
        ):
            reqs = da.DepthAnythingPipeline.network_requirements({})
        self.assertEqual(reqs, [])

    def test_yolo_world_requires_weights_on_disk(self):
        from iSpy.vision.pipelines import yolo_world as yw

        asset = Path(yw.__file__).resolve().parents[2] / "assets" / "yolo-world.pt"
        if asset.exists():
            self.skipTest("bundled yolo-world asset is present")

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(yw, "_WORLD_MODEL_DIR", Path(tmp)):
                reqs = yw.YoloWorldPipeline.network_requirements({"model_size": "s"})

        self.assertTrue(any(r.critical for r in reqs), reqs)


class TestOfflineWebSurface(unittest.TestCase):
    def tearDown(self):
        from iSpy.core import network_policy

        network_policy.set_offline(False)

    def test_network_endpoint_payload_shape(self):
        from flask import Flask

        from iSpy.core import network_policy
        from iSpy.web.Backend.PluginStatus import PluginStatusModule

        network_policy.set_offline(True, "unit-test")
        network_policy.record_skip("tracker", "probe", "needs internet")
        network_policy.record_degraded("pipeline", "cam", "feature", "why")

        app = Flask(__name__)
        PluginStatusModule({"config": None}).register_routes(app)
        resp = app.test_client().get("/api/network")
        self.assertEqual(resp.status_code, 200)

        payload = resp.get_json()
        self.assertTrue(payload["offline"])
        self.assertEqual(payload["reason"], "unit-test")
        self.assertEqual(payload["skipped"][0]["name"], "probe")
        self.assertEqual(payload["degraded"][0]["feature"], "feature")

    def test_offline_declaration_warning_scan(self):
        from iSpy.web.Backend.PluginStatus import _offline_declaration_warning

        sneaky = "import requests\n\n\nclass X:\n    pass\n"
        self.assertIsNotNone(_offline_declaration_warning(sneaky))

        flagged = "import aiohttp\n\n\nclass X:\n    needs_internet = True\n"
        self.assertIsNone(_offline_declaration_warning(flagged))

        declaring = (
            "import urllib.request\n\n\nclass X:\n"
            "    @classmethod\n"
            "    def network_requirements(cls, settings):\n"
            "        return []\n"
        )
        self.assertIsNone(_offline_declaration_warning(declaring))

        clean = "import numpy as np\n\n\nclass X:\n    pass\n"
        self.assertIsNone(_offline_declaration_warning(clean))

        self.assertIsNone(_offline_declaration_warning("def broken(:\n"))


class TestGameModeOfflineOverride(unittest.TestCase):
    def tearDown(self):
        from iSpy.core import network_policy

        network_policy.set_offline(False)

    def test_game_mode_forces_offline_without_persisting_it(self):
        import json

        from iSpy.boot import boot
        from iSpy.config.iSpyConfig import iSpyConfig
        from iSpy.core import network_policy

        self.assertFalse(network_policy.is_offline())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with _QuietLogging():
                config = iSpyConfig(str(path), create=True)
            config.set("game_mode_web_off_next_run", True)
            config.save(quiet=True)

            boot._consume_game_mode_web_off(config)

            # the one-shot run is offline in this process...
            self.assertTrue(network_policy.is_offline())
            self.assertEqual(network_policy.offline_reason(), "game_mode")
            # ...but nothing about it reaches config.json, so a crash can
            # never leave offline_mode stuck on
            self.assertFalse(bool(config.get("offline_mode", False)))
            saved = json.loads(path.read_text(encoding="utf-8-sig"))
            self.assertFalse(bool(saved.get("offline_mode", False)))
            self.assertFalse(bool(saved.get("game_mode_web_off_next_run", True)))
            self.assertFalse(bool(saved.get("app_mode", True)))
            self.assertTrue(bool(saved.get("game_mode_app_mode_backup", False)))


class TestPerformanceProfiles(unittest.TestCase):
    def setUp(self):
        from iSpy.core import performance

        os.environ.pop("ISPY_FORCE_WEB", None)
        performance.set_level_for_run(performance.DEFAULT_LEVEL)

    def tearDown(self):
        from iSpy.core import performance

        os.environ.pop("ISPY_FORCE_WEB", None)
        performance.set_level_for_run(performance.DEFAULT_LEVEL)

    @staticmethod
    def _temp_config(tmp):
        from iSpy.config.iSpyConfig import iSpyConfig

        with _QuietLogging():
            return iSpyConfig(str(Path(tmp) / "config.json"), create=True)

    def test_clamp_accepts_only_real_levels(self):
        from iSpy.core import performance

        for level in range(0, 5):
            self.assertEqual(performance.clamp(level), level)
            self.assertEqual(performance.clamp(str(level)), level)
        for bad in (None, -1, 5, 99, "banana", "2.5", 2.5, True, False, [], {}):
            self.assertIsNone(performance.clamp(bad), f"{bad!r} must not be a level")

    def test_invalid_config_falls_back_to_full_features(self):
        from iSpy.core import performance

        with _QuietLogging():
            self.assertEqual(
                performance.arm_from_config({"performance_level": 99}), 4
            )
            self.assertEqual(
                performance.arm_from_config({"performance_level": "banana"}), 4
            )
            self.assertEqual(performance.arm_from_config({}), 4)
            self.assertEqual(
                performance.arm_from_config({"performance_level": "2"}), 2
            )
        self.assertEqual(performance.get_level(), 2)

    def test_profiles_get_monotonically_cheaper(self):
        from iSpy.core import performance

        levels = list(range(4, -1, -1))  # 4 -> 0 = full features -> max fps

        multipliers = [performance.PROFILES[l].ui_poll_multiplier for l in levels]
        self.assertEqual(multipliers, sorted(multipliers))

        caches = [performance.PROFILES[l].system_metrics_cache_s for l in levels]
        self.assertEqual(caches, sorted(caches))

        qualities = [performance.PROFILES[l].mjpeg_quality for l in levels]
        self.assertEqual(qualities, sorted(qualities, reverse=True))

        strides = [performance.PROFILES[l].rollback_downsample_min for l in levels]
        self.assertEqual(strides, sorted(strides))

        # 0 means "every tick"; the rated levels must throttle, never speed up
        hz = [performance.PROFILES[l].web_update_hz for l in levels if l < 4]
        self.assertTrue(hz)
        self.assertEqual(hz, sorted(hz, reverse=True))

        # the two "everything off" facts of level 0
        self.assertFalse(performance.PROFILES[0].web_enabled)
        self.assertFalse(performance.PROFILES[0].rollback_recorder)
        self.assertFalse(performance.PROFILES[0].metrics_enabled)
        # the default level keeps today's behavior
        full = performance.PROFILES[4]
        self.assertTrue(full.web_enabled)
        self.assertTrue(full.rollback_recorder)
        self.assertTrue(full.metrics_enabled)
        self.assertEqual(full.ui_poll_multiplier, 1.0)
        self.assertEqual(full.web_update_hz, 0)
        self.assertFalse(full.skip_web_without_viewers)
        self.assertFalse(full.annotate_only_when_watched)
        self.assertTrue(full.fps_console_print)

    def test_effective_app_mode_truth_table(self):
        from iSpy.core import performance

        on = {"app_mode": True}
        off = {"app_mode": False}

        performance.set_level_for_run(4)
        self.assertTrue(performance.effective_app_mode(on))
        self.assertFalse(performance.effective_app_mode(off))

        # level 0 turns the web off even though app_mode still says True
        performance.set_level_for_run(0)
        self.assertFalse(performance.effective_app_mode(on))
        self.assertFalse(performance.effective_app_mode(off))

        # the recovery hatch wins for one run - and for the whole run only
        with patch.dict(os.environ, {"ISPY_FORCE_WEB": "1"}):
            self.assertTrue(performance.effective_app_mode(on))
            self.assertTrue(performance.effective_app_mode(off))
        os.environ.pop("ISPY_FORCE_WEB", None)
        self.assertFalse(performance.effective_app_mode(on))

    def test_effective_app_mode_never_writes_app_mode(self):
        import json

        from iSpy.core import performance

        with tempfile.TemporaryDirectory() as tmp:
            config = self._temp_config(tmp)
            path = Path(tmp) / "config.json"
            before_dict = json.loads(json.dumps(config.config))
            before_file = path.read_bytes()

            performance.set_level_for_run(0)
            with _QuietLogging():
                self.assertFalse(performance.effective_app_mode(config))
                self.assertEqual(
                    performance.arm_from_config(config), 4
                )  # the test config has no level 0 - still 4

            self.assertEqual(config.config, before_dict)
            self.assertEqual(path.read_bytes(), before_file)
            self.assertTrue(bool(config.get("app_mode", True)))

    def test_api_rejects_level_zero_without_confirm(self):
        from iSpy.core import performance
        from iSpy.web.Backend.WebApp import create_app

        with tempfile.TemporaryDirectory() as tmp:
            config = self._temp_config(tmp)
            with _QuietLogging():
                app = create_app(cameras=[], config=config)
            client = app.flask_app.test_client()

            resp = client.post("/api/performance", json={"level": 0})
            self.assertEqual(resp.status_code, 400)
            self.assertEqual(int(config.get("performance_level")), 4)

            resp = client.post("/api/performance", json={"level": 0, "confirm_max": True})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(int(config.get("performance_level")), 0)
            self.assertTrue(resp.get_json().get("restart_pending"))
            self.assertTrue(performance.restart_pending(config))

            for bad in (5, -1, "banana", None):
                resp = client.post("/api/performance", json={"level": bad})
                self.assertEqual(resp.status_code, 400, f"level {bad!r}")

            resp = client.get("/api/performance")
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload["level"], 4)  # running level is unchanged
            self.assertEqual(payload["configured_level"], 0)
            self.assertTrue(payload["restart_pending"])
            self.assertEqual(len(payload["levels"]), 5)
            self.assertEqual(payload["levels"][0]["label"], "Max FPS")
            self.assertEqual(payload["levels"][4]["label"], "Full Features")

    def test_health_entry_reflects_the_level(self):
        from iSpy.core import performance
        from iSpy.web.modules.health import HealthModule

        mod = HealthModule({"config": {"health_stale_threshold": 1.0}})

        performance.set_level_for_run(4)
        entry = mod._performance_entry()
        self.assertEqual(entry["name"], "Performance profile")
        self.assertEqual(entry["color"], "green")
        self.assertIn("Full Features (4)", entry["state"])

        performance.set_level_for_run(0)
        entry = mod._performance_entry()
        self.assertEqual(entry["color"], "yellow")
        self.assertIn("web UI off for this run", entry["state"])

    def test_web_scheduler_throttles_and_always_runs_health(self):
        import time

        from iSpy.core import performance
        from iSpy.web.Backend.WebApp import create_app

        with tempfile.TemporaryDirectory() as tmp:
            config = self._temp_config(tmp)
            with _QuietLogging():
                app = create_app(cameras=[], config=config)

        class _Fake:
            def __init__(self):
                self.calls = 0

            def update(self, frame_data):
                self.calls += 1

        health, dashboard = _Fake(), _Fake()
        app.modules = {"health": health, "dashboard": dashboard}

        # level 4 updates every module on every tick, like before the profile
        performance.set_level_for_run(4)
        for _ in range(10):
            app.update({})
        self.assertEqual((health.calls, dashboard.calls), (10, 10))

        # level 1 caps the payload builders at 2 Hz, health never skipped
        performance.set_level_for_run(1)
        app._last_update.clear()
        health.calls = dashboard.calls = 0
        ticks = 0
        deadline = time.perf_counter() + 0.6
        while time.perf_counter() < deadline:
            app.update({})
            ticks += 1
            time.sleep(0.005)
        self.assertGreater(ticks, 10)
        self.assertEqual(health.calls, ticks)
        self.assertGreaterEqual(dashboard.calls, 1)
        self.assertLessEqual(dashboard.calls, 3)


class TestHSVDetectionPipeline(unittest.TestCase):
    _DEFAULT_HSV = {
        "hue_range": [8, 20],
        "sat_range": [120, 255],
        "val_range": [120, 255],
    }

    def _base_frame(self, color=(0, 128, 255), radius=40):
        import cv2

        image = np.full((480, 640, 3), 128, dtype=np.uint8)
        cv2.circle(image, (320, 240), radius, color, -1)
        return image

    def _write_image(self, image):
        import cv2

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = str(Path(tmp.name) / "frame.png")
        self.assertTrue(cv2.imwrite(path, image))
        return path

    def _build(self, settings=None, image=None, name="hsv_cam"):
        from iSpy.config.iSpyConfig import iSpyCameraConfig, iSpyConfig
        from iSpy.vision.pipelines.hsv_detection import HSVDetectionPipeline

        image = self._base_frame() if image is None else image
        cam_entry = {
            "name": name,
            "source": self._write_image(image),
            "yaw": 0,
            "pitch": 0,
            "height": 24.0,
            "x": 0,
            "y": 0,
            "grayscale": False,
            "subsystem": "test",
            "calibration": {
                "distance": 0.0,
                "game_piece_size": 0.0,
                "size": 0,
                "fov": 60,
                "focal_length_pixels": 500,
            },
            "pipeline": {
                "name": "hsv_detection",
                "settings": dict(settings or {}),
            },
        }
        config = iSpyConfig()
        config.set("camera_configs", {name: cam_entry})
        cam_cfg = iSpyCameraConfig(cam_entry)
        camera = HSVDetectionPipeline(cam_cfg, config)
        self.addCleanup(camera.destroy)
        return camera, config, cam_entry

    def _make_camera(self, settings=None, image=None):
        return self._build(settings, image)[0]

    def test_detects_orange_circle(self):
        camera = self._make_camera(dict(self._DEFAULT_HSV, shape_hint="circle"))

        objects, frame = camera.run()

        self.assertIsNotNone(frame)
        self.assertEqual(len(objects), 1)
        obj = objects[0]
        self.assertEqual(obj.name, "hsv_object")
        self.assertGreater(obj.confidence, 0.5)
        self.assertEqual(obj.vis_meta["kind"], "hsv")
        self.assertIn(obj.depth_source, ("ground_plane", "size_based"))
        box = obj.vis_meta["bbox"]
        self.assertAlmostEqual(box[0] + box[2] / 2.0, 320, delta=6)
        self.assertAlmostEqual(box[1] + box[3] / 2.0, 240, delta=6)

    def test_hue_wrap_detects_red(self):
        red = self._base_frame(color=(0, 0, 255))
        camera = self._make_camera(
            {
                "hue_range": [170, 10],
                "sat_range": [120, 255],
                "val_range": [80, 255],
            },
            image=red,
        )

        objects, _ = camera.run()

        self.assertEqual(len(objects), 1)

    def test_lighting_normalize_keeps_dark_and_bright_blobs(self):
        import cv2

        base = self._base_frame(color=(0, 128, 255))
        hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)

        dark_hsv = hsv.copy()
        dark_hsv[:, :, 2] = (dark_hsv[:, :, 2].astype(np.float32) * 0.35).astype(
            np.uint8
        )
        dark = cv2.cvtColor(dark_hsv, cv2.COLOR_HSV2BGR)

        bright_hsv = hsv.copy()
        bright_hsv[:, :, 2] = np.clip(
            bright_hsv[:, :, 2].astype(np.int16) + 60, 0, 255
        ).astype(np.uint8)
        bright = cv2.cvtColor(bright_hsv, cv2.COLOR_HSV2BGR)

        for label, image in (("dark", dark), ("bright", bright)):
            camera = self._make_camera(
                dict(
                    self._DEFAULT_HSV,
                    val_range=[20, 255],
                    lighting_normalize="clahe",
                ),
                image=image,
            )
            objects, _ = camera.run()
            self.assertEqual(len(objects), 1, label)

    def test_filters_reject_noise_and_elongated_blobs(self):
        import cv2

        noisy = self._base_frame(color=(0, 128, 255), radius=40)
        cv2.circle(noisy, (60, 60), 1, (0, 128, 255), -1)
        camera = self._make_camera(dict(self._DEFAULT_HSV, min_area_px=150), image=noisy)
        objects, _ = camera.run()
        self.assertEqual(len(objects), 1)
        self.assertGreaterEqual(objects[0].vis_meta["area_px"], 150)

        strip = np.full((480, 640, 3), 128, dtype=np.uint8)
        cv2.rectangle(strip, (170, 230), (470, 250), (0, 128, 255), -1)
        camera = self._make_camera(
            dict(self._DEFAULT_HSV, min_area_px=150, max_aspect=6),
            image=strip,
        )
        objects, _ = camera.run()
        self.assertEqual(objects, [])

    def test_garbage_settings_clamp_without_raising(self):
        camera = self._make_camera(
            {
                "hue_range": [999, -5],
                "sat_range": ["nope", 2],
                "val_range": [300, -10],
                "process_scale": 5,
                "max_aspect": 0,
                "min_area_px": -50,
                "blur_kernel": 4,
                "morph_kernel_size": 2,
                "preset": "not_a_color",
                "z_mode": "sideways",
                "max_detections": 0,
            }
        )

        params = camera._runtime.params
        self.assertEqual(params.preset, "custom")
        self.assertEqual(params.hue_range, (179, 0))  # clamped, wrap kept
        self.assertEqual(params.sat_range, (110, 255))  # fell back to default
        self.assertEqual(params.val_range, (80, 255))  # non-wrap lo>hi -> default
        self.assertEqual(params.process_scale, 1.0)
        self.assertEqual(params.max_aspect, 1.0)
        self.assertEqual(params.min_area_px, 0.0)
        self.assertEqual(params.blur_kernel, 5)  # even size pushed odd
        self.assertEqual(params.morph_kernel_size, 3)
        self.assertEqual(params.z_mode, "ground_plane")
        self.assertEqual(params.max_detections, 1)

        objects, _ = camera.run()  # must not raise
        self.assertIsInstance(objects, list)

    def test_range_setting_coercion(self):
        from iSpy.web.Backend.PluginStatus import _coerce_setting_value

        defn = {"type": "range", "min": 0, "max": 179, "step": 1, "default": [5, 25]}
        self.assertEqual(_coerce_setting_value([10, 999], defn), [10, 179])
        self.assertEqual(_coerce_setting_value(["8", "20"], defn), [8, 20])
        self.assertEqual(
            _coerce_setting_value([170, 10], dict(defn, wrap=True)), [170, 10]
        )
        with self.assertRaises(ValueError):
            _coerce_setting_value([20, 10], defn)
        with self.assertRaises(ValueError):
            _coerce_setting_value([1], defn)
        with self.assertRaises(ValueError):
            _coerce_setting_value(["x", 3], defn)

    def test_mask_view_action_switches_the_returned_frame(self):
        camera = self._make_camera(dict(self._DEFAULT_HSV))

        objects, frame = camera.run()
        self.assertEqual(len(objects), 1)
        self.assertGreater(int(frame[240, 320].sum()), 0)

        result = camera.invoke_action("mask_view", True)
        self.assertTrue(result["ok"])
        actions = {action["id"]: action for action in result["actions"]}
        self.assertTrue(actions["mask_view"]["value"])

        objects, frame = camera.run()
        self.assertEqual(len(objects), 1)
        self.assertEqual(int(frame[5, 5].sum()), 0)  # background masked out
        self.assertGreater(int(frame[240, 320].sum()), 0)  # blob kept

        result = camera.invoke_action("cycle_mask_style")
        self.assertTrue(result["ok"])
        self.assertEqual(camera._runtime.params.mask_style, "binary")

        self.assertFalse(camera.invoke_action("not_a_real_action")["ok"])

    def test_action_route_rejects_undeclared_actions(self):
        from flask import Flask

        from iSpy.config.iSpyConfig import get_pipeline_settings
        from iSpy.web.modules.cameras import CamerasModule

        camera, config, cam_entry = self._build(dict(self._DEFAULT_HSV))
        saved = []
        config.save = lambda quiet=False: saved.append(bool(quiet))

        module = CamerasModule({"config": config})
        module.live_cameras["hsv_cam"] = camera
        app = Flask(__name__)

        with app.test_request_context("/", method="POST", json={"value": True}):
            _response, status = module._camera_action("hsv_cam", "no_such_action")
        self.assertEqual(status, 404)

        with app.test_request_context("/", method="POST", json={"value": True}):
            response = module._camera_action("hsv_cam", "mask_view")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])
        self.assertTrue(saved)
        self.assertEqual(
            get_pipeline_settings(cam_entry).get("view_mode"), "mask"
        )
        self.assertEqual(camera.take_pending_settings(), {})

    def test_update_settings_applies_to_next_run(self):
        camera = self._make_camera(dict(self._DEFAULT_HSV))
        before = camera._runtime

        applied = camera.update_settings({"min_area_px": 100000})
        self.assertEqual(applied["min_area_px"], 100000)
        self.assertIsNot(camera._runtime, before)
        objects, _ = camera.run()
        self.assertEqual(objects, [])

        camera.update_settings({"min_area_px": 10})
        objects, _ = camera.run()
        self.assertEqual(len(objects), 1)

        self.assertEqual(camera.update_settings({"not_a_key": 1}), {})

    def test_sample_hsv_returns_median(self):
        from flask import Flask

        from iSpy.web.modules.cameras import CamerasModule

        camera, config, _cam_entry = self._build(dict(self._DEFAULT_HSV))
        config.save = lambda quiet=False: None

        module = CamerasModule({"config": config})
        module.live_cameras["hsv_cam"] = camera
        app = Flask(__name__)

        with app.test_request_context("/?x=320&y=240&w=640&h=480"):
            response = module._sample_hsv("hsv_cam")

        payload = response.get_json()
        self.assertEqual(payload["hsv"], [15, 255, 255])


if __name__ == "__main__":
    unittest.main(verbosity=2)
