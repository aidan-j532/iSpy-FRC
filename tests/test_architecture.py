"""Tests for the generic vision pipeline architecture: camera -> pipeline
configuration, the common pipeline lifecycle (prepare / get_ready / state),
generalized boot behavior, and the reusable quantization dataset system."""

import json
import logging
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import flask

from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision.pipelines.base import (
    VisionPipeline,
    BackgroundPreparedPipeline,
)
from iSpy.vision.pipelines.april_tag import AprilTagCamera
from iSpy.vision.pipelines.qr_code import QRCodeCamera
from iSpy.vision.pipelines.line_tracking import LineTrackingCamera
from iSpy.vision.pipelines.object_detection import ObjectDetectionCamera
from iSpy.vision.pipelines.depth_anything import DepthAnythingCamera
from iSpy.vision.pipelines.yolo_world import YoloWorldCamera
from iSpy.vision.pipelines import get_pipeline_classes
from iSpy.vision.optimizer import default_quantization_dataset_dir
from iSpy.web.modules.datasets import DatasetsModule


class ConcretePipeline(VisionPipeline):
    def run(self):
        return [], None

    def destroy(self):
        pass


# ---------------------------------------------------------------------------
# Configuration: camera -> pipeline -> pipeline config
# ---------------------------------------------------------------------------

class PipelineConfigTests(unittest.TestCase):
    def test_default_config_is_object_detection_with_bundled_pose_model(self):
        cfg = iSpyConfig()
        cam = cfg.default_config["camera_configs"]["default_cam"]
        self.assertEqual(cam["pipeline"], "object_detection")
        self.assertEqual(
            cam["vision_model"]["file_path"],
            "YoloModels/pytorch/_default_pose.pt",
        )
        self.assertEqual(
            cam["vision_model"]["source_pt"],
            "YoloModels/pytorch/_default_pose.pt",
        )

    def test_new_camera_pipeline_config_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            data = {
                "camera_configs": {
                    "cam_detect": {
                        "name": "cam_detect", "source": 0,
                        "pipeline": "object_detection",
                        "vision_model": {"file_path": "YoloModels/pytorch/a.pt",
                                         "min_conf": 0.3},
                    },
                    "cam_tags": {
                        "name": "cam_tags", "source": 1,
                        "pipeline": "april_tag",
                        "tag_size_inches": 4.5,
                    },
                }
            }
            path.write_text(json.dumps(data))
            cfg = iSpyConfig(str(path), create=False)
            detect = cfg.camera_config("cam_detect")
            self.assertEqual(detect["pipeline"], "object_detection")
            self.assertEqual(detect["vision_model"]["min_conf"], 0.3)
            tags = cfg.camera_config("cam_tags")
            self.assertEqual(tags["pipeline"], "april_tag")
            self.assertEqual(tags["tag_size_inches"], 4.5)

    def test_object_detection_settings_not_globally_required(self):
        # An AprilTag camera needs no vision_model/optimization config at all.
        cam_cfg = iSpyCameraConfig({
            "name": "cam_tags", "source": 1,
            "pipeline": "april_tag", "tag_size_inches": 6.5,
        })
        self.assertNotIn("vision_model", cam_cfg)
        self.assertEqual(cam_cfg.get("tag_size_inches"), 6.5)

    def test_no_legacy_migration_happens(self):
        # Legacy top-level vision_model must NOT be migrated into cameras.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            data = {
                "vision_model": {"file_path": "YoloModels/pytorch/legacy.pt"},
                "camera_configs": {
                    "cam_tags": {"name": "cam_tags", "source": 1,
                                 "pipeline": "april_tag"},
                },
            }
            path.write_text(json.dumps(data))
            cfg = iSpyConfig(str(path), create=False)
            self.assertIsNone(cfg.camera_config("cam_tags").get("vision_model"))
            self.assertFalse(hasattr(cfg, "_migrate_legacy_vision_model"))

    def test_missing_config_raises_instead_of_silent_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nope.json"
            with self.assertRaises(RuntimeError):
                iSpyConfig(str(path), create=False)

    def test_corrupt_config_raises_with_boot_f_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{not json!!")
            with self.assertRaises(RuntimeError) as ctx:
                iSpyConfig(str(path), create=False)
            self.assertIn("boot -f", str(ctx.exception))

    def test_pipeline_own_optimization_config(self):
        # Optimization settings belong to the object_detection pipeline config
        # (its per-camera vision_model block) - not the global config.
        cfg = iSpyConfig()
        self.assertIsNone(cfg.get("vision_model"))
        cam = cfg.camera_config("default_cam")
        self.assertTrue(cam["vision_model"]["quantized"])


# ---------------------------------------------------------------------------
# Pipeline lifecycle: prepare / get_ready / state
# ---------------------------------------------------------------------------

class PipelineLifecycleTests(unittest.TestCase):
    def test_all_pipelines_inherit_the_generic_base(self):
        for name, cls in get_pipeline_classes().items():
            self.assertTrue(
                issubclass(cls, VisionPipeline),
                f"{name} does not inherit VisionPipeline",
            )
            self.assertTrue(hasattr(cls, "prepare"))
            self.assertTrue(hasattr(cls, "is_ready"))
            self.assertTrue(hasattr(cls, "get_state"))
            self.assertTrue(hasattr(cls, "process"))

    def test_no_prep_pipeline_is_ready_and_ready_state(self):
        pipeline = ConcretePipeline.__new__(ConcretePipeline)
        ready, status = pipeline.is_ready()
        self.assertTrue(ready)
        self.assertEqual(status, "ready")
        self.assertEqual(pipeline.get_state(), "ready")

    def test_state_mapping(self):
        cases = [
            ("ready", "ready"),
            ("using unoptimized .pt fallback", "ready"),
            ("optimizing (rknn build)", "optimizing"),
            ("downloading (model weights)", "downloading"),
            ("initializing", "initializing"),
            ("error: no model configured/found", "error"),
        ]
        for status, expected in cases:
            pipeline = ConcretePipeline.__new__(ConcretePipeline)
            pipeline._set_status(status)
            self.assertEqual(pipeline.get_state(), expected, status)

    def test_background_prepared_pipeline_runs_prep_once(self):
        class FakeBackground(BackgroundPreparedPipeline):
            def run(self):
                return [], None

            def destroy(self):
                pass

            def _prepare(self):
                self.done = True

        pipeline = FakeBackground.__new__(FakeBackground)
        pipeline._prep_thread = None
        pipeline._prep_started = False
        pipeline._prep_lock = threading.Lock()
        pipeline._status = "initializing"

        pipeline.prepare()
        prep_thread = pipeline._prep_thread
        prep_thread.join()
        self.assertFalse(pipeline._preparing())
        self.assertTrue(pipeline.done)

        pipeline.prepare()  # idempotent - must not start a second thread
        self.assertIs(pipeline._prep_thread, prep_thread)

    def test_failed_preparation_enters_error_state(self):
        class BrokenPipeline(ConcretePipeline):
            def is_ready(self):
                ready, status = self._readiness()
                self._set_status(status)
                return ready, status

            def _readiness(self):
                return False, "error: something broke"

        pipeline = BrokenPipeline.__new__(BrokenPipeline)
        pipeline._status = "initializing"
        ready, status = pipeline.is_ready()
        self.assertFalse(ready)
        self.assertEqual(pipeline.get_state(), "error")
        self.assertIn("something broke", status)

    def test_user_model_selection_is_pipeline_driven(self):
        self.assertTrue(ObjectDetectionCamera.uses_user_model())
        for cls in (AprilTagCamera, QRCodeCamera, LineTrackingCamera,
                    DepthAnythingCamera, YoloWorldCamera):
            self.assertFalse(cls.uses_user_model(), cls.__name__)

    def test_only_object_detection_offers_optimization_options(self):
        pipeline = ObjectDetectionCamera.__new__(ObjectDetectionCamera)
        options = pipeline.get_optimization_options()
        self.assertIn("auto_opt", options)
        self.assertIn("target_format", options)
        self.assertIn("quantized", options)
        self.assertFalse(ConcretePipeline.__new__(ConcretePipeline).get_optimization_options())
        self.assertEqual(ConcretePipeline.__new__(ConcretePipeline).optimize(),
                         "not supported")

    def test_model_backed_pipelines_offer_optimization_options(self):
        # YOLO World and Depth Anything are optimizable pipelines too - the
        # generic optimize endpoint must not be object-detection-only.
        for cls, keys in (
            (YoloWorldCamera, ("quantize", "target_format", "quantization_dataset", "input_size")),
            (DepthAnythingCamera, ("optimize", "model_size")),
        ):
            options = cls.__new__(cls).get_optimization_options()
            for key in keys:
                self.assertIn(key, options, f"{cls.__name__} missing {key}")

    def test_optimize_disabled_returns_explanation(self):
        # Quantize/Optimize disabled in config -> request_optimize explains
        # instead of pretending to start a build.
        yw = YoloWorldCamera.__new__(YoloWorldCamera)
        yw.quantize = False
        yw._optimizing = False
        self.assertIn("disabled", yw.request_optimize())

        da = DepthAnythingCamera.__new__(DepthAnythingCamera)
        da.optimize = False
        da._optimizing = False
        self.assertIn("disabled", da.request_optimize())

    def test_optimize_starts_background_build(self):
        import time as _time

        da = DepthAnythingCamera.__new__(DepthAnythingCamera)
        da.optimize = True
        da._optimizing = False
        da._prep_thread = None
        da._prep_started = False
        da._load_error = None
        da._session = None
        da.logger = logging.getLogger("test")
        da._load_optimized = lambda force=False: None
        status = da.request_optimize()
        self.assertTrue(status.startswith("optimizing"), status)
        deadline = _time.monotonic() + 5
        while getattr(da, "_optimizing", False) and _time.monotonic() < deadline:
            _time.sleep(0.01)
        self.assertFalse(da._optimizing)
        self.assertTrue(da.get_status().startswith("error"))

    def test_no_prep_pipelines_report_ready_immediately(self):
        for cls in (AprilTagCamera, QRCodeCamera, LineTrackingCamera):
            pipeline = cls.__new__(cls)
            ready, status = pipeline.is_ready()
            self.assertTrue(ready, f"{cls.__name__}: {status}")
            self.assertEqual(status, "ready")


# ---------------------------------------------------------------------------
# Boot: readiness wait behavior
# ---------------------------------------------------------------------------

class BootTests(unittest.TestCase):
    def _fake_config(self):
        cfg = iSpyConfig()
        cfg.config["camera_configs"] = {
            "cam_0": {"name": "cam_0", "source": 0, "pipeline": "fake"},
        }
        return cfg

    def test_boot_fails_fast_on_pipeline_error(self):
        from iSpy.boot.boot import _wait_for_pipeline_ready

        class FakePipeline:
            def __init__(self, camera_config, config, core_mask=None):
                pass

            def is_ready(self):
                return False, "error: model weights failed to load"

        with self.assertRaises(RuntimeError) as ctx:
            _wait_for_pipeline_ready(self._fake_config(), {"fake": FakePipeline})
        self.assertIn("cam_0", str(ctx.exception))
        self.assertIn("model weights failed to load", str(ctx.exception))

    def test_boot_fails_on_unknown_pipeline(self):
        from iSpy.boot.boot import _wait_for_pipeline_ready

        with self.assertRaises(RuntimeError) as ctx:
            _wait_for_pipeline_ready(self._fake_config(), {})
        self.assertIn("unknown pipeline", str(ctx.exception))

    def test_boot_completes_when_all_ready(self):
        from iSpy.boot.boot import _wait_for_pipeline_ready

        class FakeReadyPipeline:
            def __init__(self, camera_config, config, core_mask=None):
                pass

            def is_ready(self):
                return True, "ready"

        _wait_for_pipeline_ready(self._fake_config(), {"fake": FakeReadyPipeline})

    def test_boot_flag_is_fresh_not_first_boot(self):
        import inspect
        import iSpy.boot.boot as boot
        sig = inspect.signature(boot.on_boot)
        self.assertIn("fresh", sig.parameters)
        self.assertNotIn("first_boot", sig.parameters)


# ---------------------------------------------------------------------------
# Quantization dataset: reusable, not model-attached
# ---------------------------------------------------------------------------

class QuantizeDatasetTests(unittest.TestCase):
    def test_default_dataset_dir_is_reusable_default(self):
        self.assertEqual(default_quantization_dataset_dir().name, "default")
        self.assertEqual(default_quantization_dataset_dir().parent.name, "QuantizeDataset")

    def test_datasets_module_creates_default_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("iSpy.web.modules.datasets.Path.cwd",
                            return_value=Path(tmp)):
                mod = DatasetsModule({"config": None})
                self.assertTrue((Path(tmp) / "QuantizeDataset" / "default" / "images").exists())
                self.assertEqual(mod._active, "default")

    def test_users_can_create_new_dataset_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("iSpy.web.modules.datasets.Path.cwd",
                            return_value=Path(tmp)):
                mod = DatasetsModule({"config": None})

                class FakeRequest:
                    def get_json(self, force=True):
                        return {"name": "competition"}

                app = flask.Flask(__name__)
                with app.app_context():
                    with mock.patch("iSpy.web.modules.datasets.request", FakeRequest()):
                        resp = mod._create()
                    result, status = (resp[0], resp[1]) if isinstance(resp, tuple) else (resp, 200)
                    self.assertEqual(status, 200)
                    self.assertTrue(
                        (Path(tmp) / "QuantizeDataset" / "competition" / "images").exists()
                    )

                    resp = mod._list()
                    names = {d["name"] for d in resp.get_json()["datasets"]}
                    self.assertIn("competition", names)
                    self.assertIn("default", names)

    def test_duplicate_dataset_folder_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("iSpy.web.modules.datasets.Path.cwd",
                            return_value=Path(tmp)):
                mod = DatasetsModule({"config": None})

                class FakeRequest:
                    def get_json(self, force=True):
                        return {"name": "default"}

                app = flask.Flask(__name__)
                with app.app_context():
                    with mock.patch("iSpy.web.modules.datasets.request", FakeRequest()):
                        resp = mod._create()
                    self.assertEqual(resp[1], 409)


if __name__ == "__main__":
    unittest.main()
