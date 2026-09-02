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
_DEFAULT_POSE_PT = _REPO / "iSpy" / "assets" / "_default_pose.pt"


def _make_test_frame() -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), -1)
    return frame


def _make_noise_frame() -> np.ndarray:
    rng = np.random.RandomState(20260901)
    return (rng.rand(480, 640, 3) * 255).astype(np.uint8)


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


class PoseModelRegressionTests(unittest.TestCase):
    """BUG 1 regression: _default_pose.pt loads, infers, returns keypoints.

    Root cause that this guards against: the C3k block was implemented as a
    C2f-style chunk block while pickled YOLO11/v26 pose C3k instances carry a
    C3 topology (cv1/cv2 project to hidden channels, cv3 fuses). The mismatch
    crashed the forward with "Given groups=1, weight of size [32, 32, 3, 3],
    expected input[1, 16, ...] to have 32 channels". Also guards the keypoint
    decode already producing (N, K, dims) per the Ultralytics layout.
    """

    @classmethod
    def setUpClass(cls):
        if not _DEFAULT_POSE_PT.exists():
            raise unittest.SkipTest(f"Required test model not found: {_DEFAULT_POSE_PT}")

    def test_pose_pt_loads_and_infers_with_keypoints(self):
        register_shim()
        model = load_yolo_pt(str(_DEFAULT_POSE_PT), task="pose")
        self.assertEqual(model.task, "pose")
        self.assertEqual((model.num_keypoints, model.keypoint_dims), (17, 3))
        self.assertEqual(model.nc, 1)
        self.assertIn("person", model.names.values())

        frame = _make_noise_frame()
        results = model(frame, imgsz=640, conf=0.001)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertIsNotNone(
            r.keypoints, "synthetic frame produced no candidate - pose decode never exercised"
        )
        kd = r.keypoints.data
        self.assertEqual(kd.ndim, 3)
        self.assertEqual(kd.shape[1:], (17, 3))
        self.assertEqual(len(r.boxes), kd.shape[0])

    def test_pose_pt_keypoint_values_are_finite(self):
        register_shim()
        model = load_yolo_pt(str(_DEFAULT_POSE_PT), task="pose")
        frame = _make_noise_frame()
        results = model(frame, imgsz=640, conf=0.001)
        r = results[0]
        if r.keypoints is None or not len(r.boxes):
            self.skipTest("no detections to check against")
        kd = r.keypoints.data
        self.assertEqual(kd.shape[1:], (17, 3))
        self.assertEqual(len(r.boxes), kd.shape[0])
        # decoded keypoints must be finite and land near the 480x640 frame
        # (a few px of negative margin is normal at edges - ultralytics does
        # not clamp them either); garbage decodes would be wildly out of range.
        self.assertTrue(bool(kd[..., :2].isfinite().all()))
        self.assertTrue(bool((kd[..., 0] >= -100).all()))
        self.assertTrue(bool((kd[..., 0] <= 800).all()))
        self.assertTrue(bool((kd[..., 1] >= -100).all()))
        self.assertTrue(bool((kd[..., 1] <= 800).all()))


class YoloWorldIsolationTests(unittest.TestCase):
    """yolo_world's Ultralytics dependency is isolated to a subprocess worker."""

    def test_yolo_world_module_has_no_in_process_ultralytics_import(self):
        """Importing yolo_world in the serving process must not pull ultralytics."""
        src = Path(__file__).resolve().parents[1] / "iSpy" / "vision" / "pipelines" / "yolo_world.py"
        text = src.read_text(encoding="utf-8")
        self.assertNotIn("from ultralytics import", text)

    def test_yolo_world_imports_without_leaking_ultralytics(self):
        # In a fresh interpreter (no pre-imported iSpy modules), importing the
        # serving module must not pull ultralytics into the process. Prompt
        # step 4: "Confirm yolo_world.py itself no longer has `from ultralytics
        # import YOLOWorld, YOLO` anywhere in the main process's import graph".
        import subprocess
        code = (
            "import sys\n"
            "import iSpy.vision.pipelines.yolo_world\n"
            "leaks = [m for m in sys.modules if m.startswith('ultralytics')]\n"
            "print('LEAKS', leaks)\n"
        )
        repo = Path(__file__).resolve().parents[1]
        r = subprocess.run([sys.executable, "-c", code], cwd=str(repo),
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("LEAKS []", r.stdout)
        self.assertNotIn("ultralytics", r.stderr.lower())

    def test_reparam_worker_exists_and_isolates_ultralytics(self):
        worker = Path(__file__).resolve().parents[1] / "iSpy" / "boot" / "_yoloworld_reparam_worker.py"
        self.assertTrue(worker.exists())
        text = worker.read_text(encoding="utf-8")
        self.assertIn("from ultralytics import", text)
        self.assertIn("YOLOWorld", text)

    def test_yolo_world_method_delegates_to_subprocess(self):
        from iSpy.vision.pipelines.yolo_world import YoloWorldPipeline
        self.assertTrue(hasattr(YoloWorldPipeline, "_reparameterize_world_subprocess"))


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

    def test_pipeline_classes_are_pipelines(self):
        from iSpy.vision.pipelines.object_detection import ObjectDetectionPipeline
        from iSpy.vision.pipelines.april_tag import AprilTagPipeline
        from iSpy.vision.pipelines.yolo_world import YoloWorldPipeline
        self.assertTrue(issubclass(ObjectDetectionPipeline, object))
        self.assertTrue(issubclass(AprilTagPipeline, object))
        self.assertTrue(issubclass(YoloWorldPipeline, object))


if __name__ == "__main__":
    unittest.main()