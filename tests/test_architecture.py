import json
import logging
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import cv2
import flask

from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision.pipelines.base import (
    VisionPipeline,
    BackgroundPreparedPipeline,
)
from iSpy.vision.pipelines.april_tag import AprilTagCamera
from iSpy.vision.pipelines.qr_code import QRCodeCamera
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


class _FakeCamConfig:
    def __init__(self, calibration=None):
        self._calibration = calibration or {}

    def get(self, key, default=None):
        if key == "calibration":
            return self._calibration
        return default


class _SpyDetector:
    def __init__(self):
        self.called = False

    def detectMarkers(self, gray):
        self.called = True
        return [], None, []


# ---------------------------------------------------------------------------
# Configuration: camera -> pipeline -> pipeline config
# ---------------------------------------------------------------------------

class PipelineConfigTests(unittest.TestCase):
    def test_default_config_is_object_detection_with_bundled_pose_model(self):
        cfg = iSpyConfig()
        cam = cfg.default_config["camera_configs"]["default_cam"]
        self.assertEqual(cam["pipeline"]["name"], "object_detection")
        self.assertEqual(
            cam["pipeline"]["settings"]["vision_model"]["file_path"],
            "YoloModels/pytorch/_default_pose.pt",
        )
        self.assertEqual(
            cam["pipeline"]["settings"]["vision_model"]["source_pt"],
            "YoloModels/pytorch/_default_pose.pt",
        )

    def test_legacy_flat_camera_config_migrates_to_nested_pipeline(self):
        # old configs kept settings flat on the camera entry - fold every
        # non-camera key into pipeline.settings so old config.json files still load
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            data = {
                "camera_configs": {
                    "cam_detect": {
                        "name": "cam_detect", "source": 0,
                        "pipeline": "object_detection",
                        "vision_model": {"file_path": "YoloModels/pytorch/a.pt",
                                         "min_conf": 0.3},
                        "target_format": "onnx",
                        "stale_flat_junk": 1,
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
            self.assertEqual(detect.pipeline_name(), "object_detection")
            settings = detect.pipeline_settings()
            self.assertEqual(settings["vision_model"]["min_conf"], 0.3)
            self.assertEqual(settings["target_format"], "onnx")
            # Non-camera keys were folded out of the flat layout.
            self.assertNotIn("target_format", detect.data)
            tags = cfg.camera_config("cam_tags")
            self.assertEqual(tags.pipeline_name(), "april_tag")
            self.assertEqual(tags.get_pipeline_setting("tag_size_inches"), 4.5)

    def test_new_camera_pipeline_config_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            data = {
                "camera_configs": {
                    "cam_detect": {
                        "name": "cam_detect", "source": 0,
                        "pipeline": {"name": "object_detection", "settings": {
                            "vision_model": {"file_path": "YoloModels/pytorch/a.pt",
                                             "min_conf": 0.3},
                        }},
                    },
                    "cam_tags": {
                        "name": "cam_tags", "source": 1,
                        "pipeline": {"name": "april_tag", "settings": {
                            "tag_size_inches": 4.5,
                        }},
                    },
                }
            }
            path.write_text(json.dumps(data))
            cfg = iSpyConfig(str(path), create=False)
            detect = cfg.camera_config("cam_detect")
            self.assertEqual(detect.pipeline_name(), "object_detection")
            self.assertEqual(
                detect.get_pipeline_setting("vision_model")["min_conf"], 0.3
            )
            tags = cfg.camera_config("cam_tags")
            self.assertEqual(tags.pipeline_name(), "april_tag")
            self.assertEqual(tags.get_pipeline_setting("tag_size_inches"), 4.5)

    def test_object_detection_settings_not_globally_required(self):
        # An AprilTag camera needs no vision_model/optimization config at all.
        cam_cfg = iSpyCameraConfig({
            "name": "cam_tags", "source": 1,
            "pipeline": {"name": "april_tag", "settings": {"tag_size_inches": 6.5}},
        })
        self.assertNotIn("vision_model", cam_cfg)
        self.assertEqual(
            cam_cfg.get_pipeline_setting("tag_size_inches"), 6.5
        )

    def test_legacy_top_level_vision_model_migrates_to_cameras(self):
        # legacy top-level vision_model folds into model-backed cameras (no
        # boot -f required, no data loss); the top-level key is dropped.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            data = {
                "vision_model": {"file_path": "YoloModels/pytorch/legacy.pt",
                                 "source_pt": "YoloModels/pytorch/legacy.pt",
                                 "min_conf": 0.7},
                "camera_configs": {
                    "cam_detect": {"name": "cam_detect", "source": 0,
                                   "pipeline": "object_detection"},
                    "cam_tags": {"name": "cam_tags", "source": 1,
                                 "pipeline": "april_tag"},
                },
            }
            path.write_text(json.dumps(data))
            cfg = iSpyConfig(str(path), create=False)
            self.assertIsNone(cfg.get("vision_model"))
            detect = cfg.camera_config("cam_detect")
            self.assertEqual(detect.pipeline_name(), "object_detection")
            vm = detect.get_pipeline_setting("vision_model")
            self.assertEqual(vm["file_path"], "YoloModels/pytorch/legacy.pt")
            self.assertEqual(vm["min_conf"], 0.7)
            # Non-model-backed cameras never receive a vision_model block.
            tags = cfg.camera_config("cam_tags")
            self.assertNotIn("vision_model", tags.pipeline_settings())

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

    def test_save_is_atomic_and_thread_safe(self):
        # concurrent savers (web handlers + bg optimizer thread) must never be
        # able to corrupt config.json; temp file + os.replace guarantees every
        # on-disk state is a complete, valid JSON document.
        import threading

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            cfg = iSpyConfig(str(path))
            errors = []

            def writer(n):
                try:
                    for _ in range(15):
                        cfg.set("camera_configs", {
                            "cam": {"name": "cam", "source": n,
                                    "pipeline": "april_tag"},
                        })
                        cfg.save()
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=writer, args=(n,)) for n in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("camera_configs", data)
            # no temp-file leftovers after a successful atomic write
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_pipeline_own_optimization_config(self):
        # optimization settings live in the pipeline config now, opt-in -
        # no background build on first boot
        cfg = iSpyConfig()
        self.assertIsNone(cfg.get("vision_model"))
        cam = cfg.camera_config("default_cam")
        self.assertFalse(
            cam.get_pipeline_setting("vision_model")["quantize"]
        )
        self.assertFalse(cfg.get("optimize", True))


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
        for cls in (AprilTagCamera, QRCodeCamera,
                    DepthAnythingCamera, YoloWorldCamera):
            self.assertFalse(cls.uses_user_model(), cls.__name__)

    def test_only_object_detection_offers_optimization_options(self):
        pipeline = ObjectDetectionCamera.__new__(ObjectDetectionCamera)
        options = pipeline.get_optimization_options()
        self.assertIn("optimize", options)
        self.assertIn("target_format", options)
        self.assertIn("quantize", options)
        self.assertFalse(ConcretePipeline.__new__(ConcretePipeline).get_optimization_options())
        self.assertEqual(ConcretePipeline.__new__(ConcretePipeline).optimize(),
                         "not supported")

    def test_model_backed_pipelines_offer_optimization_options(self):
        # yolo world + depth anything are optimizable too - optimize() mustnt be od-only
        for cls, keys in (
            (YoloWorldCamera, ("optimize", "quantize", "target_format", "quantization_dataset", "input_size")),
            (DepthAnythingCamera, ("optimize", "quantize", "target_format", "quantization_dataset", "input_size", "model_size")),
        ):
            options = cls.__new__(cls).get_optimization_options()
            for key in keys:
                self.assertIn(key, options, f"{cls.__name__} missing {key}")

    def test_stale_artifact_resync_hooks_exist_on_all_model_backed_pipelines(self):
        # _resync_stale_model_file_path() needs three pipeline-provided
        # helpers; every model-backed pipeline must define them so the
        # boot guard is callable everywhere (a missing hook would turn the
        # guard into an AttributeError and silently drop the protection)
        from iSpy.vision.pipelines.optimizable import OptimizableModelPipeline

        hooks = ("_source_model_path", "_resolve_model_path", "_persist_file_path")
        for cls in (ObjectDetectionCamera, YoloWorldCamera, DepthAnythingCamera):
            self.assertTrue(issubclass(cls, OptimizableModelPipeline), cls.__name__)
            for hook in hooks:
                owner = next(
                    (k for k in cls.__mro__ if hook in vars(k)), None
                )
                self.assertIsNotNone(owner, f"{cls.__name__} missing resync hook {hook}")
                self.assertIsNot(
                    owner, OptimizableModelPipeline,
                    f"{cls.__name__} inherits {hook} instead of defining it",
                )

    def test_resync_stale_model_file_path_is_safe_on_every_model_backed_pipeline(self):
        # calling the guard on a bare instance must never raise - pipelines
        # without persisted model state exit cleanly via their hooks
        yw = YoloWorldCamera.__new__(YoloWorldCamera)
        yw.logger = logging.getLogger("test.yw")
        yw.model_size = "s"
        yw.classes = ["object"]
        yw._resync_stale_model_file_path(None)

        da = DepthAnythingCamera.__new__(DepthAnythingCamera)
        da.logger = logging.getLogger("test.da")
        da._resync_stale_model_file_path(None)  # fixed checkpoint -> no-op

        od = ObjectDetectionCamera.__new__(ObjectDetectionCamera)
        od.logger = logging.getLogger("test.od")

        def _no_vm():
            return {}

        od._current_vm_config = _no_vm

        def _resolve(path):
            return Path(path) if path else None

        od._resolve_model_path = _resolve
        persisted = []
        od._persist_file_path = lambda fp, cfg: persisted.append(fp)
        od._resync_stale_model_file_path(None)  # no vision_model block -> no-op
        self.assertEqual(persisted, [])

    def test_optimize_disabled_returns_explanation(self):
        # optimize disabled in config -> explain instead of pretending to build
        yw = YoloWorldCamera.__new__(YoloWorldCamera)
        yw.quantize = False
        yw._optimizing = False
        self.assertIn("disabled", yw.optimize())

        da = DepthAnythingCamera.__new__(DepthAnythingCamera)
        da._auto_opt = False
        da.estimate_depth = True
        da._optimizing = False
        self.assertIn("disabled", da.optimize())

    def test_optimize_runs_synchronously_and_gates_readiness(self):
        # failed build -> not-ready + gated to raw feed; success -> ready
        da = DepthAnythingCamera.__new__(DepthAnythingCamera)
        da._auto_opt = True
        da.estimate_depth = True
        da._optimizing = False
        da._optimize_error = None
        da._load_error = None
        da._session = None
        da.model = None
        da.logger = logging.getLogger("test")
        da._load_optimized = lambda force=False: None  # stub out the real build

        taken = da.optimize()
        self.assertIn("error", taken)
        self.assertFalse(da._optimizing)
        ready, status = da.is_ready()
        self.assertFalse(ready)
        self.assertIn("error", status)
        self.assertFalse(da._is_processable())

        ok = DepthAnythingCamera.__new__(DepthAnythingCamera)
        ok._auto_opt = True
        ok.estimate_depth = True
        ok._optimizing = False
        ok._optimize_error = None
        ok._load_error = None
        ok._session = object()
        ok._model = None
        ok.logger = logging.getLogger("test")
        ok._load_optimized = lambda force=False: None
        ok._optimized_active = lambda: True

        self.assertEqual(ok.optimize(), "ready")
        self.assertEqual(ok.is_ready(), (True, "ready"))
        self.assertTrue(ok._is_processable())

    def test_no_prep_pipelines_report_ready_immediately(self):
        for cls in (AprilTagCamera, QRCodeCamera):
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

    def test_broken_camera_does_not_block_healthy_cameras(self):
        # a single camera whose pipeline fails to construct is skipped w/ a log;
        # the healthy cameras still boot (Day 2 resilience).
        from unittest import mock
        import iSpy.iSpy as ispy_module
        from iSpy.config.iSpyConfig import iSpyConfig

        class BrokenPipeline:
            def __init__(self, *a, **k):
                raise RuntimeError("model weights missing")

        class GoodPipeline:
            def __init__(self, *a, **k):
                self.ok = True

        def fake_classes():
            return {"broken": BrokenPipeline, "good": GoodPipeline}

        cfg = iSpyConfig()
        cfg.config["camera_configs"] = {
            "bad": {"name": "bad", "source": 0, "pipeline": "broken"},
            "ok": {"name": "ok", "source": 1, "pipeline": "good"},
        }
        cfg._rebuild_camera_configs()
        with mock.patch(
            "iSpy.vision.pipelines.get_pipeline_classes", fake_classes
        ):
            cams = ispy_module.iSpy._build_cameras_from_config(cfg)
        self.assertEqual(len(cams), 1)
        self.assertTrue(cams[0].ok)

    def test_missing_model_degrades_gracefully(self):
        # Day 5: a camera pointing at a missing model still constructs and
        # runs. GenericYolo's ModelFileError is caught at boot, the pipeline
        # reports model=None / no detection, and the whole boot survives.
        with tempfile.TemporaryDirectory() as tmp:
            config = iSpyConfig(file_path=str(Path(tmp) / "config.json"))
            config.config["app_mode"] = False
            missing = str(Path(tmp) / "does_not_exist.rknn")
            cam_cfg = iSpyCameraConfig({
                "name": "cam", "source": 99, "fps_cap": 1000,
                "yaw": 0, "pitch": 0, "height": 1.0, "x": 0, "y": 0,
                "grayscale": False, "subsystem": "bench",
                "calibration": {"distance": 1.0, "game_piece_size": 1.0,
                                "size": 100, "fov": 90},
                "pipeline": {"name": "object_detection", "settings": {
                    "vision_model": {"file_path": missing,
                                     "source_pt": missing}}},
            })
            cam = ObjectDetectionCamera(cam_cfg, config)
        self.assertIsNone(cam.model)
        self.assertFalse(cam._use_pipeline)

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
                self.assertTrue((Path(tmp) / "QuantizeDataset" / "default").exists())
                # No global "active dataset" state file should be written.
                self.assertFalse((Path(tmp) / "Config" / "active_dataset.json").exists())
                self.assertFalse(hasattr(mod, "_active"))

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
                        (Path(tmp) / "QuantizeDataset" / "competition").exists()
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


# ---------------------------------------------------------------------------
# Day 6: cross-platform discovery / runtime checks
# ---------------------------------------------------------------------------

class CaptureBackendDetectionTests(unittest.TestCase):
    def test_windows_prefers_msmf(self):
        from iSpy.vision.Camera import Camera

        self.assertEqual(
            Camera._get_capture_backend_candidates("Windows"), [cv2.CAP_MSMF]
        )

    def test_linux_prefers_v4l2_then_any(self):
        from iSpy.vision.Camera import Camera

        self.assertEqual(
            Camera._get_capture_backend_candidates("Linux"), [cv2.CAP_V4L2, cv2.CAP_ANY]
        )

    def test_other_platforms_fall_back_to_any(self):
        from iSpy.vision.Camera import Camera

        self.assertEqual(
            Camera._get_capture_backend_candidates("Darwin"), [cv2.CAP_ANY]
        )


class CameraOpenBoundedTests(unittest.TestCase):
    """Regression: _open_capture_bounded must use the module global properly
    (UnboundLocalError on first real open) and always release its slot."""

    class _FakeCap:
        def __init__(self, *a, **k):
            self._opened = True

        def isOpened(self):
            return self._opened

        def release(self):
            self._opened = False

    def _mk_cam(self):
        from iSpy.vision.Camera import Camera

        return Camera({"name": "open_test", "source": "missing_frame_test.png"},
                      (640, 480), False)

    def test_live_slot_released_after_success(self):
        from iSpy.vision import Camera as cam_mod

        cam = self._mk_cam()
        with mock.patch.object(cam_mod.cv2, "VideoCapture", side_effect=self._FakeCap):
            cap = cam._open_capture_bounded(cam_mod.cv2.CAP_ANY)
        self.assertTrue(cap.isOpened())
        self.assertEqual(cam_mod._open_worker_live, 0)

    def test_live_slot_released_after_failure(self):
        from iSpy.vision import Camera as cam_mod

        cam = self._mk_cam()

        def boom(*a, **k):
            raise RuntimeError("driver wedged")

        with mock.patch.object(cam_mod.cv2, "VideoCapture", side_effect=boom):
            with self.assertRaises(RuntimeError):
                cam._open_capture_bounded(cam_mod.cv2.CAP_ANY)
        self.assertEqual(cam_mod._open_worker_live, 0)

    def test_cap_blocks_oversubscription_and_releases_slot(self):
        from iSpy.vision import Camera as cam_mod

        cam = self._mk_cam()
        with cam_mod._open_worker_guard:
            cam_mod._open_worker_live = cam_mod._OPEN_WORKER_MAX
        try:
            with self.assertRaises(cam_mod.CameraOpenTimeout):
                cam._open_capture_bounded(cam_mod.cv2.CAP_ANY)
        finally:
            with cam_mod._open_worker_guard:
                cam_mod._open_worker_live = 0
        self.assertEqual(cam_mod._open_worker_live, 0)


class MDNSHostnameTests(unittest.TestCase):
    def test_default_hostname_has_ispy_prefix_and_stable_suffix(self):
        from iSpy.boot.setup_service import default_mdns_hostname

        name = default_mdns_hostname()
        self.assertTrue(name.startswith("ispy-"))
        self.assertEqual(len(name), len("ispy-") + 6)
        self.assertEqual(default_mdns_hostname(), name)

    def test_env_override_wins(self):
        from iSpy.boot.setup_service import MDNS_HOSTNAME_ENV
        from iSpy.boot.setup_service import default_mdns_hostname

        with mock.patch.dict("os.environ", {MDNS_HOSTNAME_ENV: "my-coproc"}, clear=False):
            self.assertEqual(default_mdns_hostname(), "my-coproc")


class ServiceUserTests(unittest.TestCase):
    def test_sudo_invocation_uses_sudo_user(self):
        from iSpy.boot.setup_service import _service_user

        with mock.patch("iSpy.boot.setup_service.os.geteuid", return_value=0, create=True):
            with mock.patch.dict(os.environ, {"SUDO_USER": "orangepi"}, clear=False):
                self.assertEqual(_service_user(), "orangepi")

    def test_root_without_sudo_user_runs_as_root(self):
        from iSpy.boot.setup_service import _service_user

        with mock.patch("iSpy.boot.setup_service.os.geteuid", return_value=0, create=True):
            with mock.patch.dict(os.environ, {}, clear=False):
                self.assertEqual(_service_user(), "root")

    def test_non_root_uses_invoking_user(self):
        from iSpy.boot.setup_service import _service_user

        with mock.patch("iSpy.boot.setup_service.os.geteuid", side_effect=OSError("n/a"), create=True):
            with mock.patch("iSpy.boot.setup_service.getpass.getuser", return_value="aidan"):
                self.assertEqual(_service_user(), "aidan")


class RKNNPlatformTests(unittest.TestCase):
    def test_undetected_platform_falls_back_with_warning_stamp(self):
        from iSpy.vision import optimizer as opt

        with mock.patch.dict("os.environ", {}, clear=False):
            with mock.patch.object(opt, "_detect_rknn_target_platform", return_value=None):
                target, detected = opt._resolve_rknn_target_platform()
        self.assertEqual(target, "rk3588")
        self.assertFalse(detected)

    def test_env_override_is_visible_and_authoritative(self):
        from iSpy.vision import optimizer as opt

        with mock.patch.dict(
            "os.environ", {"ISPY_RKNN_TARGET_PLATFORM": "rk3576"}, clear=False
        ):
            target, detected = opt._resolve_rknn_target_platform()
        self.assertEqual(target, "rk3576")
        self.assertTrue(detected)


class RKNNWheelUrlTests(unittest.TestCase):
    def test_full_wheel_base_points_at_live_release_tag(self):
        from iSpy.vision import optimizer as opt

        base = opt._RKNN_FULL_BASE
        self.assertEqual(
            base,
            "https://github.com/aidan-j532/iSpy-FRC/releases/download/RKNN_Wheels",
        )
        for (arch, tag), fn in opt._RKNN_FULL_WHEELS.items():
            self.assertTrue(fn.endswith(".whl"), fn)
            self.assertIn(arch, fn)
            self.assertIn(tag, fn)
            self.assertIn("rknn_toolkit2-2.3.2", fn)
        for (arch, tag), fn in opt._RKNN_LITE_FILENAMES.items():
            self.assertEqual(arch, "aarch64")
            self.assertIn(tag, fn)
            self.assertIn("rknn_toolkit_lite2-2.3.2", fn)


class UserCalibrationDataYamlTests(unittest.TestCase):
    def test_data_yaml_written_with_metadata_classes(self):
        from iSpy.vision.optimizer import _user_calibration_data_yaml
        import iSpy.vision.metadata as meta

        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "userds"
            ds.mkdir()
            pt_meta = Path(tmp) / "model.pt"
            pt_meta.write_bytes(b"fake-model")
            with mock.patch.object(
                meta, "read_metadata", return_value={"nc": 3, "names": {0: "a", 1: "b", 2: "c"}}
            ):
                yaml_path = Path(_user_calibration_data_yaml(str(pt_meta), ds))
            content = yaml_path.read_text()
            self.assertIn(f"nc: 3", content)
            self.assertIn('"a"', content)
            self.assertTrue(yaml_path.exists())


class PipelineCalibrationGateTests(unittest.TestCase):
    """Pipelines declare whether they need calibration to run via
    requires_calibration()/calibration_sections; while uncalibrated their run()
    degrades to a plain frame pass-through (the same [], frame pattern used
    while optimizing / when the model is unavailable)."""

    def _april_cam(self, calibration):
        cam = AprilTagCamera.__new__(AprilTagCamera)
        cam.config = _FakeCamConfig(calibration)
        cam.detector = _SpyDetector()
        cam.get_frame = lambda: __import__("numpy").zeros((40, 40, 3), dtype=__import__("numpy").uint8)
        cam.logger = None
        return cam

    def test_requires_calibration_reflects_declared_sections(self):
        self.assertTrue(ObjectDetectionCamera.requires_calibration())
        self.assertTrue(AprilTagCamera.requires_calibration())
        self.assertTrue(QRCodeCamera.requires_calibration())
        self.assertFalse(DepthAnythingCamera.requires_calibration())
        self.assertFalse(YoloWorldCamera.requires_calibration())

    def test_section_satisfied_rules(self):
        self.assertTrue(VisionPipeline._section_satisfied("charuco", {
            "camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "dist_coeffs": [0, 0, 0, 0, 0],
        }))
        self.assertFalse(VisionPipeline._section_satisfied("charuco", {"fov": 68.0}))
        self.assertFalse(VisionPipeline._section_satisfied("charuco", {}))
        self.assertTrue(VisionPipeline._section_satisfied("focal", {"focal_length_pixels": 500.0}))
        self.assertTrue(VisionPipeline._section_satisfied("focal", {"fov": 68.0}))
        self.assertFalse(VisionPipeline._section_satisfied("focal", {}))
        self.assertTrue(VisionPipeline._section_satisfied("pnp", {"pnp": {"objects": []}}))
        self.assertFalse(VisionPipeline._section_satisfied("pnp", {}))

    def test_any_declared_section_satisfies_pipeline(self):
        class MultiSectionPipeline(VisionPipeline):
            calibration_sections = ["charuco", "focal"]

            def run(self):
                return [], None

            def destroy(self):
                pass

        cam = MultiSectionPipeline.__new__(MultiSectionPipeline)
        cam.config = _FakeCamConfig({"fov": 68.0})
        self.assertTrue(cam.calibration_ready())
        cam.config = _FakeCamConfig({})
        self.assertFalse(cam.calibration_ready())

    def test_uncalibrated_april_tag_returns_raw_frame_without_detection(self):
        cam = self._april_cam({})
        objects, frame = cam.run()
        self.assertEqual(objects, [])
        self.assertIsNotNone(frame)
        self.assertFalse(cam.detector.called, "detector must not run uncalibrated")

    def test_focal_only_does_not_satisfy_charuco_pipeline(self):
        cam = self._april_cam({"fov": 68.0})
        objects, frame = cam.run()
        self.assertEqual(objects, [])
        self.assertFalse(cam.detector.called)

    def test_charuco_calibrated_april_tag_runs_detection(self):
        cam = self._april_cam({
            "camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "dist_coeffs": [0, 0, 0, 0, 0],
        })
        objects, frame = cam.run()
        self.assertEqual(objects, [])  # no markers present
        self.assertTrue(cam.detector.called)

    def test_calibration_mode_bypasses_gate(self):
        import time
        cam = self._april_cam({})
        cam.calibration_active = True
        cam.calibration_last_seen = time.monotonic()
        cam._CALIBRATION_HEARTBEAT_TIMEOUT = 10.0
        self.assertTrue(cam._calibration_processable())

    def test_object_detection_gate_via_is_processable(self):
        cam = ObjectDetectionCamera.__new__(ObjectDetectionCamera)
        cam.config = _FakeCamConfig({})
        cam._optimizing = False
        cam._optimization_requested = lambda: False
        cam.model = object()
        self.assertFalse(cam._is_processable())

        cam.config = _FakeCamConfig({
            "camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "dist_coeffs": [0, 0, 0, 0, 0],
        })
        self.assertTrue(cam._is_processable())

    def test_pipeline_payload_exposes_requires_calibration(self):
        from iSpy.web.Backend.PluginStatus import _build_vision_pipeline_payloads

        by_name = {p["name"]: p for p in _build_vision_pipeline_payloads()}
        self.assertTrue(by_name["object_detection"]["requires_calibration"])
        self.assertFalse(by_name["yolo_world"]["requires_calibration"])
        self.assertIn("calibration_sections", by_name["object_detection"])


if __name__ == "__main__":
    unittest.main()

