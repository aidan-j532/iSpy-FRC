import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from iSpy.vision.yolo_pt import load_yolo_pt, register_shim, YoloPT
from iSpy.vision.genericYolo import GenericYolo, torch_load
from iSpy.vision.metadata import metadata_from_pt
from iSpy.vision.ModelInspector import _inspect_ultralytics


_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_DETECT_PT = _REPO / "iSpy" / "assets" / "_default_detect.pt"


def _make_test_frame() -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), -1)
    return frame


class UltralyticsRemovalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _DEFAULT_DETECT_PT.exists():
            raise unittest.SkipTest(f"Required test model not found: {_DEFAULT_DETECT_PT}")

    def test_load_yolo_pt_without_ultralytics_import(self):
        """load_yolo_pt must work even when the real ultralytics package is unavailable."""
        with mock.patch.dict(sys.modules, {"ultralytics": None, "ultralytics.nn": None}):
            for mod in list(sys.modules):
                if mod.startswith("ultralytics"):
                    sys.modules.pop(mod, None)
            register_shim()
            model = load_yolo_pt(str(_DEFAULT_DETECT_PT), task="detect")
            self.assertEqual(model.task, "detect")
            self.assertEqual(model.nc, 80)
            self.assertIn("person", model.names.values())
            self.assertTrue(hasattr(model.model, "eval"))

    def test_yolo_pt_inference_returns_expected_structure(self):
        """YoloPT.__call__ returns results with boxes.xyxy, boxes.conf, boxes.cls."""
        register_shim()
        model = load_yolo_pt(str(_DEFAULT_DETECT_PT), task="detect")
        frame = _make_test_frame()
        results = model(frame, verbose=False)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertTrue(hasattr(r, "boxes"))
        self.assertTrue(hasattr(r.boxes, "xyxy"))
        self.assertTrue(hasattr(r.boxes, "conf"))
        self.assertTrue(hasattr(r.boxes, "cls"))
        self.assertEqual(r.boxes.xyxy.shape[1], 4)
        self.assertEqual(r.boxes.conf.shape[0], r.boxes.xyxy.shape[0])

    def test_metadata_from_pt_works_without_ultralytics(self):
        """metadata_from_pt reads task/nc/names/input_size without ultralytics."""
        with mock.patch.dict(sys.modules, {"ultralytics": None}):
            for mod in list(sys.modules):
                if mod.startswith("ultralytics"):
                    sys.modules.pop(mod, None)
            register_shim()
            meta = metadata_from_pt(_DEFAULT_DETECT_PT)
            self.assertEqual(meta.get("task"), "detect")
            self.assertEqual(meta.get("nc"), 80)
            self.assertIn("person", meta.get("names", {}).values())
            self.assertEqual(meta.get("input_size"), [640, 640])

    def test_generic_yolo_yolo_model_type_predict(self):
        """GenericYolo with model_type='yolo' runs inference via load_yolo_pt."""
        config = {
            "task": "detect",
            "file_path": str(_DEFAULT_DETECT_PT),
            "min_conf": 0.5,
            "input_size": (640, 640),
            "margin": 0,
            "target_format": "pytorch",
            "optimize": False,
        }
        gy = GenericYolo(config)
        self.assertEqual(gy.model_type, "yolo")
        self.assertTrue(hasattr(gy, "model"))
        frame = _make_test_frame()
        results = gy.predict(frame)
        self.assertTrue(hasattr(results, "boxes"))
        self.assertIsInstance(results.boxes, list)

    def test_torch_load_registers_shim_and_loads_checkpoint(self):
        """genericYolo.torch_load registers shim and unpickles checkpoint."""
        with mock.patch.dict(sys.modules, {"ultralytics": None}):
            for mod in list(sys.modules):
                if mod.startswith("ultralytics"):
                    sys.modules.pop(mod, None)
            checkpoint = torch_load(_DEFAULT_DETECT_PT)
            self.assertIn("model", checkpoint)
            self.assertIn("train_args", checkpoint)

    def test_model_inspector_inspect_ultralytics(self):
        """ModelInspector._inspect_ultralytics reads metadata without ultralytics."""
        with mock.patch.dict(sys.modules, {"ultralytics": None}):
            for mod in list(sys.modules):
                if mod.startswith("ultralytics"):
                    sys.modules.pop(mod, None)
            register_shim()
            meta = _inspect_ultralytics(str(_DEFAULT_DETECT_PT), "detect")
            self.assertEqual(meta.get("task"), "detect")
            self.assertEqual(meta.get("num_classes"), 80)
            self.assertEqual(meta.get("input_size"), [640, 640])

    def test_yolopt_to_device_moves_model(self):
        """YoloPT.to(device) moves the model and returns self."""
        register_shim()
        model = load_yolo_pt(str(_DEFAULT_DETECT_PT), task="detect")
        m2 = model.to("cpu")
        self.assertIs(m2, model)
        self.assertEqual(model._device, "cpu")

    def test_yolopt_names_and_nc_are_accessible(self):
        """YoloPT exposes .names and .nc consistently."""
        register_shim()
        model = load_yolo_pt(str(_DEFAULT_DETECT_PT), task="detect")
        self.assertIsInstance(model.names, dict)
        self.assertEqual(len(model.names), model.nc)
        self.assertEqual(model.nc, 80)


class CameraRestructureTests(unittest.TestCase):
    """Sanity-check that the camera-source restructure imports work."""

    def test_cameras_package_exports(self):
        from iSpy.vision.Cameras import CameraBase, OpenCVCamera, TelloCamera, create_camera, get_camera_classes
        self.assertTrue(callable(CameraBase))
        self.assertTrue(callable(OpenCVCamera))
        self.assertTrue(callable(TelloCamera))
        self.assertTrue(callable(create_camera))
        self.assertTrue(callable(get_camera_classes))
        classes = get_camera_classes()
        self.assertIn("opencv", classes)
        self.assertIn("tello", classes)

    def test_camera_facade_reexports(self):
        from iSpy.vision.Camera import Camera, CameraBase as FacadeBase, create_camera as FacadeCreate
        from iSpy.vision.Cameras import CameraBase as RealBase, create_camera as RealCreate
        self.assertIs(FacadeBase, RealBase)
        self.assertIs(FacadeCreate, RealCreate)

    def test_pipeline_aliases_present(self):
        from iSpy.vision.pipelines import get_pipeline_classes
        classes = get_pipeline_classes()
        self.assertIn("object_detection", classes)
        self.assertIn("april_tag", classes)
        self.assertIn("qr_code", classes)
        self.assertIn("yolo_world", classes)

    def test_backward_aliases_exist(self):
        from iSpy.vision.pipelines.object_detection import ObjectDetectionPipeline, ObjectDetectionCamera
        from iSpy.vision.pipelines.april_tag import AprilTagPipeline, AprilTagCamera
        from iSpy.vision.pipelines.yolo_world import YoloWorldPipeline, YoloWorldCamera
        self.assertIs(ObjectDetectionCamera, ObjectDetectionPipeline)
        self.assertIs(AprilTagCamera, AprilTagPipeline)
        self.assertIs(YoloWorldCamera, YoloWorldPipeline)


if __name__ == "__main__":
    unittest.main()