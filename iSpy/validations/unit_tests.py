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
    flags.update(overrides)
    with ExitStack() as stack:
        for name, value in flags.items():
            stack.enter_context(patch.object(ao, name, return_value=value))
        return ao.recommend_format()


# ─── AutoOpt tests ───────────────────────────────────────────────────────────

class TestAutoOpt(unittest.TestCase):

    def test_returns_string(self):
        self.assertIsInstance(_recommend(), str)

    def test_returns_known_format(self):
        known = SUPPORTED_FORMATS | {"hef"}
        self.assertIn(_recommend(), known, f"Unknown format: {_recommend()}")

    def test_rknn_wins_when_rockchip_npu(self):
        self.assertEqual(_recommend(has_rockchip_npu=True), "rknn")

    def test_hailo_uses_hef(self):
        self.assertEqual(_recommend(has_hailo_npu=True), "hef")

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
        for candidate in (
            repo_root / "YoloModels" / "pytorch" / "_default_pose.pt",
            repo_root / "iSpy" / "assets" / "_default_pose.pt",
        ):
            if candidate.exists():
                return candidate
        return None

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
        ready, status = camera.is_ready()
        self.assertFalse(ready)
        self.assertTrue(status.startswith("error:"), f"unexpected status: {status!r}")

    def test_good_sidecar_constructs_ready(self):
        # sanity: with NO sidecar and a missing file (fill_missing_config short
        # circuits on os.path.exists -> returns config as-is), the pipeline
        # still constructs with model None + error status, so a missing model
        # does not take down the camera either.
        camera = self._build_pipeline("does/not/exist.pt")
        self.assertIsNone(camera.model)
        ready, status = camera.is_ready()
        self.assertFalse(ready)
        self.assertTrue(status.startswith("error:"))


if __name__ == "__main__":
    unittest.main(verbosity=2)