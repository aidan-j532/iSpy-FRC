"""Restored boot-time unit-test subset (Day 5).

The regression subset that runs during `validate_system` / boot. Modules
are covered by `tests/` in the pytest suite; this file keeps a real,
fast, dependency-light subset runnable in the `iSpy.validations`
discovery used by `ez.unit_tests()` and `validate_system.run_unit_tests()`.

Hardware backends (rknnlite, tflite_runtime, torch, scipy) are faked at
import time so tests behave identically on any machine. Ultralytics is
not faked: it is never a runtime dependency (only an optional build-time
exporter), so nothing imports it at module load.
"""
import os
import sys
import threading
import time
import types
import unittest
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# ─── Fake out hardware imports so tests run on any machine ───────────────────

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
    """Mimics trt.IExecutionContext just enough for GenericYolo._run_engine."""

    def __init__(self, shapes, output_name):
        self._shapes = dict(shapes)
        self._output_name = output_name

    def set_tensor_shape(self, name, shape):
        self._shapes[name] = tuple(shape)

    def get_tensor_shape(self, name):
        return self._shapes.get(name)

    def execute_v2(self, bindings):
        # bindings layout is [input_ptr, output_ptr]; the output buffer was
        # pre-initialized to an empty (1, 0, F) tensor, so nothing to write.
        pass


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

    def create_execution_context(self):
        shapes = {
            self._input_name: (1, 3, 640, 640),
            self._output_name: (1, 0, self._feat_w),
        }
        return _FakeEngineContext(shapes, self._output_name)


class _FakeTensorRTRuntime:
    def __init__(self, *args, **kwargs):
        self.engine = _FakeEngine("images", "output0", 5)

    def deserialize_cuda_engine(self, data):  # noqa: ARG001
        return self.engine


trt_mod.Runtime = _FakeTensorRTRuntime
trt_mod.Logger = MagicMock()
trt_mod.TensorIOMode = _FakeTensorIOMode
sys.modules["tensorrt"] = trt_mod

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

    def __call__(self, feed):  # noqa: ARG001
        return {"output0": np.empty((1, 0, self._feat_w), dtype=np.float32)}


class _FakeOpenVINOCore:
    available_devices = ["CPU"]

    def __init__(self, *args, **kwargs):
        pass

    def compile_model(self, model_path, device="AUTO", *args, **kwargs):  # noqa: ARG001
        return _FakeCompiledModel(5)


ov_mod.Core = _FakeOpenVINOCore
sys.modules["openvino"] = ov_mod

# ─── Modules under test ──────────────────────────────────────────────────────

from iSpy.config.AutoOpt import SUPPORTED_FORMATS, recommend_format  # noqa: E402
from iSpy.vision.genericYolo import Box, GenericYolo, ModelFileError, Results  # noqa: E402
from iSpy.vision.metadata import (  # noqa: E402
    derive_format_metadata,
    metadata_path_for,
    read_metadata,
    write_metadata,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_frame(w=320, h=320):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 1] = np.linspace(0, 255, w, dtype=np.uint8)  # green gradient
    return frame


def _no_hw_flags():
    return {
        "has_rockchip_npu": False,
        "has_hailo_npu": False,
        "has_edge_tpu": False,
        "has_apple_silicon": False,
        "has_tpu": False,
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
    flags.update(overrides)
    with ExitStack() as stack:
        for name, value in flags.items():
            stack.enter_context(patch.object(ao, name, return_value=value))
        return ao.recommend_format(runtime_supported=runtime_supported)


# ─── AutoOpt tests ───────────────────────────────────────────────────────────

class TestAutoOpt(unittest.TestCase):

    def test_returns_string(self):
        self.assertIsInstance(_recommend(), str)

    def test_returns_known_format(self):
        # HAILO DISABLED - "hef" removed from the search space (see AutoOpt)
        known = SUPPORTED_FORMATS
        self.assertIn(_recommend(), known, f"Unknown format: {_recommend()}")

    def test_rknn_wins_when_rockchip_npu(self):
        self.assertEqual(_recommend(has_rockchip_npu=True), "rknn")

    def test_hailo_disabled_never_selects_hef(self):
        # HAILO DISABLED - has_hailo_npu() is neutered (returns False) and the
        # recommend_format() hef branch is commented out, so even simulated
        # Hailo hardware must fall through to the next best backend (onnx here).
        self.assertEqual(_recommend(has_hailo_npu=True), "onnx")
        self.assertNotEqual(_recommend(has_hailo_npu=True), "hef")

    def test_edge_tpu_uses_tflite(self):
        self.assertEqual(_recommend(has_edge_tpu=True), "tflite")

    def test_nvidia_without_tensorrt_falls_back_to_onnx(self):
        self.assertEqual(
            _recommend(has_nvidia=True, has_tensorrt=False), "onnx"
        )

    def test_nvidia_with_tensorrt_uses_engine(self):
        self.assertEqual(
            _recommend(has_nvidia=True, has_tensorrt=True), "engine"
        )

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

    def test_runtime_unsupported_skips_engine_on_nvidia(self):
        self.assertEqual(
            _recommend(has_nvidia=True, has_tensorrt=True, runtime_supported=False),
            "onnx",
        )
        self.assertEqual(
            _recommend(has_nvidia=True, has_tensorrt=True, runtime_supported=True),
            "engine",
        )

    def test_runtime_unsupported_skips_openvino_on_intel(self):
        self.assertEqual(
            _recommend(has_intel_gpu=True, runtime_supported=False), "onnx"
        )
        self.assertEqual(
            _recommend(has_intel_gpu=True, runtime_supported=True), "openvino"
        )

    def test_runtime_unsupported_keeps_cpu_backends(self):
        # rknn/tflite/tpu remain valid runtime formats and shouldn't be skipped.
        self.assertEqual(
            _recommend(has_rockchip_npu=True, runtime_supported=False), "rknn"
        )
        self.assertEqual(
            _recommend(has_edge_tpu=True, runtime_supported=False), "tflite"
        )


# ─── Box / Results tests ─────────────────────────────────────────────────────

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


# ─── GenericYolo model handling (missing / broken / valid) ──────────────────

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
            GenericYolo(
                {"file_path": "does/not/exist.rknn", "task": "detect"}
            )

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


# ─── Compiled formats (.engine / .xml) runtime (Bug 7) ──────────────────────

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


# ─── RKNN / ONNX metadata round-trip ─────────────────────────────────────────

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


# ─── Pose-checkpoint regression (BUG 1) ──────────────────────────────────────

class TestPosePtRegression(unittest.TestCase):
    """Real _default_pose.pt load + inference regression.

    The import-time fake 'torch' shim above cannot load a real ``.pt``, so
    this class swaps the REAL torch package back in (when the machine has it
    installed) and re-imports yolo_pt against it. Guards the C3k topology
    regression: a v26 pose checkpoint's C3k instances must forward through the
    C3-style graph (cv1/cv2 -> c_ channels, cv3 fuses), not a C2f chunk.
    Machines without torch skip the test.
    """

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

        # Restore the REAL torch (the fake above only satisfies import-time).
        for _m in [m for m in list(sys.modules) if m == "torch" or m.startswith("torch.")]:
            sys.modules.pop(_m, None)
        try:
            real_torch = importlib.import_module("torch")
            if getattr(real_torch, "nn", None) is None:
                raise RuntimeError("real torch missing torch.nn")
            cls._yolo_pt = importlib.import_module("iSpy.vision.yolo_pt")
            importlib.reload(cls._yolo_pt)
        except Exception as exc:  # no real torch on this machine
            raise unittest.SkipTest(f"real torch unavailable: {exc}")

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
            r.keypoints, "low-threshold run produced no candidate - pose decode never exercised"
        )
        kd = r.keypoints.data
        self.assertEqual(kd.ndim, 3)
        self.assertEqual(kd.shape[1:], (17, 3))


# ─── validate_system() regression (BUG 2) ────────────────────────────────────

class TestValidateSystemRegression(unittest.TestCase):
    """A failing validator must flip validate_system() to False."""

    def test_broken_model_file_path_flips_validate_system_to_false(self):
        from iSpy.validations.validate_system import validate_system

        old_cwd = os.getcwd()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(os.chdir, old_cwd)

        bad_dir = Path(tmp.name) / "YoloModels" / "pytorch"
        bad_dir.mkdir(parents=True)
        (bad_dir / "corrupt model!.txt").write_bytes(b"junk")
        os.chdir(tmp.name)

        # validate_model_files() raises on the invalid path -> validate_system()
        # returns False (never reaches run_unit_tests -> no recursion).
        self.assertFalse(validate_system())


# ─── object_detection pipeline config-normalization regression (BUG 6) ───────

class _QuietLogging:
    """Temporarily silence INFO/DEBUG logs during pipeline construction."""

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
    """BUG 6: fill_missing_config raising ValueError must not skip the camera.

    A malformed metadata sidecar (e.g. ``nc: not_an_int``) makes
    fill_missing_config raise ValueError before the GenericYolo block. The
    pipeline must still construct (self.model is None) so the camera shows an
    error status instead of silently vanishing from self.cameras.
    """

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
        from iSpy import vision as _vision

        scipy_optimize = types.ModuleType("scipy.optimize")
        scipy_optimize.least_squares = MagicMock()
        had_optimize = "scipy.optimize" in sys.modules
        prior_optimize = sys.modules.get("scipy.optimize")
        sys.modules["scipy.optimize"] = scipy_optimize
        self.addCleanup(self._restore_scipy_optimize, had_optimize, prior_optimize)

        from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
        from iSpy.vision.pipelines.object_detection import ObjectDetectionPipeline

        config = iSpyConfig()
        cam_entry = {
            "name": "bug6_cam",
            "source": 99,
            "fps_cap": 1000,
            "yaw": 0, "pitch": 0, "height": 1.0,
            "x": 0, "y": 0,
            "grayscale": False,
            "subsystem": "test",
            "calibration": {"distance": 1.0, "game_piece_size": 1.0, "size": 100, "fov": 90},
            "pipeline": {
                "name": "object_detection",
                "settings": {"vision_model": {"file_path": model_path, "task": "detect"}},
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
        weight.write_bytes(b"\x00" * 512)  # under the load floor -> ModelFileError later

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
    """F1: the EKF tracker smooths a constant-velocity target's position.

    Feeds a synthetic target moving at constant velocity with Gaussian
    position noise and asserts the Kalman-smoothed track's RMSE to the true
    path is lower than the raw noisy measurements' RMSE.
    """

    def _make_tracker(self):
        from iSpy.plugins.trackers.BuiltIn.EKFTracker import EKFTracker
        return EKFTracker({
            "config": {
                "process_noise": 0.5,
                "measurement_noise": 0.1,
                "distance_threshold": 1.0,
                "stale_threshold": 2.0,
            },
            "global_config": None,
        })

    def test_ekf_smoothes_better_than_raw_measurements(self):
        from iSpy.vision.Object import Object

        rng = np.random.default_rng(7)
        tracker = self._make_tracker()

        # constant-velocity ground truth: start at origin, move +1.0 m/s in x
        velocity = np.array([1.0, 0.0, 0.0])
        dt = 0.1
        n = 30
        true_positions = []
        raw_sq = 0.0
        # collect smoothed positions as they are produced
        smoothed_path = []

        for i in range(n):
            t_i = i * dt
            true = velocity * t_i
            true_positions.append(true)
            noise = rng.normal(0.0, 0.5, size=3)  # fairly noisy measurements
            raw = true + noise
            raw_sq += float(np.linalg.norm(noise) ** 2)

            obj = Object(x=float(raw[0]), y=float(raw[1]), z=float(raw[2]), name="ball")
            tracker.update([obj], 0.0, 0.0, 0.0, 0.0)
            smoothed_path.append(np.array(tracker.tracked_objects[0].get_position()))

        true_arr = np.array(true_positions)
        sm_arr = np.array(smoothed_path)

        raw_rmse = float(np.sqrt(raw_sq / n))
        smoothed_rmse = float(np.sqrt(np.mean(np.sum((sm_arr - true_arr) ** 2, axis=1))))

        self.assertLess(
            smoothed_rmse, raw_rmse,
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
    """F2a: the shared selection primitive must be a standalone, modular class."""

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
    """F2a: target_selector reads/publishes the selected tracked Object.

    Selection state lives on the shared context (SelectionState), not on the
    utility - the utility only publishes and owns the web routes.
    """

    def _make_selector(self, reacquire_timeout_s: float = 1.0,
                       output_key: str = "selected_target"):
        from iSpy.plugins.selection import SelectionState
        from iSpy.plugins.utilities.BuiltIn.TargetSelector import TargetSelector
        selection = SelectionState()
        selector = TargetSelector({
            "config": {
                "reacquire_timeout_s": reacquire_timeout_s,
                "output_key": output_key,
            },
            "flask_app": None,
            "selection": selection,
        })
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
        from iSpy.vision.Object import Object
        selector, selection = self._make_selector()
        selection.select(5)
        # Object always has an id; simulate a plain id-less fallback dict
        obj_no_id = {"x": 1.0, "y": 2.0, "z": 3.0}
        frame_data = {"detections": [obj_no_id]}
        selector.update(frame_data)  # must not raise
        published = frame_data.get("addon_data", {}).get("selected_target")
        self.assertIsNone(published)


# ─── calibration gating: yellow vs red (detection vs pose) ──────────────────

class TestCalibrationGating(unittest.TestCase):
    """Uncalibrated detect pipelines run with a yellow warning; pose pipelines
    block (red) until calibration exists."""

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
        from iSpy import vision as _vision
        scipy_optimize = types.ModuleType("scipy.optimize")
        scipy_optimize.least_squares = MagicMock()
        had = "scipy.optimize" in sys.modules
        prior = sys.modules.get("scipy.optimize")
        sys.modules["scipy.optimize"] = scipy_optimize
        self.addCleanup(self._restore_scipy_optimize, had, prior)

        from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
        from iSpy.vision.pipelines.object_detection import ObjectDetectionPipeline

        config = iSpyConfig()
        cam_entry = {
            "name": "calib_cam",
            "source": 99,
            "fps_cap": 1000,
            "yaw": 0, "pitch": 0, "height": 1.0,
            "x": 0, "y": 0,
            "grayscale": False,
            "subsystem": "test",
            "calibration": self._calibration(),
            "pipeline": {
                "name": "object_detection",
                "settings": {"vision_model": {"file_path": "does/not/exist.pt", "task": task}},
            },
        }
        config.set("camera_configs", {"calib_cam": cam_entry})
        cam_cfg = iSpyCameraConfig(cam_entry)
        with _QuietLogging():
            return ObjectDetectionPipeline(cam_cfg, config)

    def test_detect_uncalibrated_is_yellow_and_runs(self):
        """Detect-task model without calibration -> yellow warning, not blocked."""
        camera = self._build_detect_pipeline("detect")
        try:
            level, msg = camera.calibration_status()
            self.assertEqual(level, "yellow")
            self.assertIn("Needs Calibration for Better Accuracy", msg)
            # _gate_uncalibrated must NOT block a detect pipeline
            self.assertIsNone(camera._gate_uncalibrated(np.zeros((10, 10, 3), dtype=np.uint8)))
        finally:
            camera.destroy()

    def test_pose_uncalibrated_is_red_and_blocks(self):
        """Pose-task model without calibration -> red warning, pipeline blocked."""
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
        """A calibrated pipeline (fov > 0) reports ready, not yellow."""
        from iSpy import vision as _vision
        scipy_optimize = types.ModuleType("scipy.optimize")
        scipy_optimize.least_squares = MagicMock()
        had = "scipy.optimize" in sys.modules
        prior = sys.modules.get("scipy.optimize")
        sys.modules["scipy.optimize"] = scipy_optimize
        self.addCleanup(self._restore_scipy_optimize, had, prior)

        from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
        from iSpy.vision.pipelines.object_detection import ObjectDetectionPipeline

        config = iSpyConfig()
        cam_entry = {
            "name": "calib_cam",
            "source": 99,
            "fps_cap": 1000,
            "yaw": 0, "pitch": 0, "height": 1.0,
            "x": 0, "y": 0,
            "grayscale": False,
            "subsystem": "test",
            "calibration": self._calibration(fov=90),
            "pipeline": {
                "name": "object_detection",
                "settings": {"vision_model": {"file_path": "does/not/exist.pt", "task": "detect"}},
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
            self.assertIsNone(camera._gate_uncalibrated(np.zeros((10, 10, 3), dtype=np.uint8)))
        finally:
            camera.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=2)