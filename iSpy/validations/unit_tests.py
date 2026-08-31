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


if __name__ == "__main__":
    unittest.main(verbosity=2)