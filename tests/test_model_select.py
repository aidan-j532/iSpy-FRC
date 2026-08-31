import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import flask

from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision.pipelines.object_detection import ObjectDetectionPipeline
from iSpy.vision.optimizer import existing_artifact_for
from iSpy.web.modules.cameras import _resolve_vision_model_files
from iSpy.web.modules.models import ModelsModule


_REPO = Path(__file__).resolve().parents[1]


def _make_source_tree(tmp: Path, names=("pose",)) -> Path:
    tmp = tmp.resolve()
    (tmp / "YoloModels" / "pytorch").mkdir(parents=True, exist_ok=True)
    (tmp / "YoloModels" / "rknn").mkdir(parents=True, exist_ok=True)
    (tmp / "YoloModels" / "onnx").mkdir(parents=True, exist_ok=True)
    for name in names:
        (tmp / "YoloModels" / "pytorch" / f"{name}.pt").write_bytes(b"fake pt")
        (tmp / "YoloModels" / "rknn" / f"{name}.rknn").write_bytes(b"fake rknn")
    return tmp


class _RootPatched(unittest.TestCase):

    def setUp(self):
        self.tmp = _make_source_tree(Path(tempfile.mkdtemp()))
        self._root_patch = mock.patch(
            "iSpy.vision.optimizer._PROJECT_ROOT", self.tmp
        )
        self._root_patch.start()

    def tearDown(self):
        self._root_patch.stop()


class ExistingArtifactTests(_RootPatched):
    def test_none_when_nothing_built(self):
        fresh = self.tmp / "YoloModels" / "pytorch" / "fresh.pt"
        fresh.write_bytes(b"fake pt")
        self.assertIsNone(existing_artifact_for(fresh, "engine"))

    def test_prefers_requested_format(self):
        pt = self.tmp / "YoloModels" / "pytorch" / "pose.pt"
        self.assertEqual(
            existing_artifact_for(pt, "rknn"),
            "YoloModels/rknn/pose.rknn",
        )

    def test_falls_back_to_any_built_format(self):
        # no onnx/pose.onnx, but the rknn build exists - that runs fine
        pt = self.tmp / "YoloModels" / "pytorch" / "pose.pt"
        self.assertEqual(
            existing_artifact_for(pt, "onnx"),
            "YoloModels/rknn/pose.rknn",
        )

    def test_ignores_non_pt_input(self):
        pt = self.tmp / "YoloModels" / "pytorch" / "pose.pt"
        self.assertIsNone(existing_artifact_for(self.tmp / "YoloModels" / "rknn" / "pose.rknn", "rknn"))
        self.assertIsNone(existing_artifact_for(None, "rknn"))
        self.assertIsNone(existing_artifact_for("", "rknn"))
        self.assertEqual(existing_artifact_for(pt, None), "YoloModels/rknn/pose.rknn")


class ResolveVisionModelFilesTests(_RootPatched):
    def test_points_at_existing_artifact(self):
        settings = {
            "target_format": "rknn",
            "vision_model": {
                "file_path": "YoloModels/pytorch/pose.pt",
                "source_pt": "YoloModels/pytorch/pose.pt",
            },
        }
        _resolve_vision_model_files(settings)
        vm = settings["vision_model"]
        self.assertEqual(vm["source_pt"], "YoloModels/pytorch/pose.pt")
        self.assertEqual(vm["file_path"], "YoloModels/rknn/pose.rknn")

    def test_falls_back_to_source_pt(self):
        settings = {
            "target_format": "engine",
            "vision_model": {
                "file_path": "YoloModels/pytorch/fresh.pt",
                "source_pt": "YoloModels/pytorch/fresh.pt",
            },
        }
        (self.tmp / "YoloModels" / "pytorch" / "fresh.pt").write_bytes(b"fake pt")
        _resolve_vision_model_files(settings)
        vm = settings["vision_model"]
        self.assertEqual(vm["file_path"], "YoloModels/pytorch/fresh.pt")

    def test_leaves_non_pt_models_alone(self):
        settings = {
            "vision_model": {
                "file_path": "YoloModels/rknn/pose.rknn",
                "source_pt": "YoloModels/rknn/pose.rknn",
            },
        }
        _resolve_vision_model_files(settings)
        self.assertEqual(settings["vision_model"]["file_path"], "YoloModels/rknn/pose.rknn")


class ModelsSelectTests(unittest.TestCase):

    _MODEL = "YoloModels/pytorch/_boot_test_pose.pt"
    _ARTIFACT = "YoloModels/rknn/_boot_test_pose.rknn"

    def setUp(self):
        (self._repo_root / "YoloModels" / "pytorch").mkdir(parents=True, exist_ok=True)
        (self._repo_root / "YoloModels" / "rknn").mkdir(parents=True, exist_ok=True)
        (self._repo_root / self._MODEL).write_bytes(b"fake pt")

    def tearDown(self):
        for rel in (self._MODEL, self._ARTIFACT):
            f = self._repo_root / rel
            if f.exists():
                f.unlink()
        for d in (self._repo_root / "YoloModels" / "rknn", self._repo_root / "YoloModels" / "pytorch"):
            if d.exists() and not any(d.iterdir()):
                d.rmdir()

    @property
    def _repo_root(self) -> Path:
        return _REPO

    def _config(self, target_format):
        tmp = Path(tempfile.mkdtemp())
        cfg = iSpyConfig(file_path=str(tmp / "config.json"))
        cfg.config["app_mode"] = False
        cfg.config["camera_configs"] = {
            "cam_0": {
                "name": "cam_0",
                "source": 0,
                "pipeline": {
                    "name": "object_detection",
                    "settings": {
                        "target_format": target_format,
                        "vision_model": {"file_path": self._MODEL, "source_pt": self._MODEL},
                    },
                },
                "calibration": {"distance": 1.0, "game_piece_size": 1.0, "size": 100, "fov": 90},
            }
        }
        return cfg

    def _select(self, cfg):
        app = flask.Flask(__name__)
        pt = self._repo_root / self._MODEL
        with app.test_request_context("/api/models/select", json={"file_path": str(pt)}):
            mod = ModelsModule({"config": cfg})
            return mod._select()

    def test_select_persists_artifact_file_path(self):
        (self._repo_root / self._ARTIFACT).write_bytes(b"fake rknn")
        cfg = self._config("rknn")
        resp = self._select(cfg)
        self.assertTrue(resp.json["success"])
        vm = cfg.config["camera_configs"]["cam_0"]["pipeline"]["settings"]["vision_model"]
        self.assertEqual(vm["source_pt"], self._MODEL)
        self.assertEqual(vm["file_path"], self._ARTIFACT)

    def test_select_falls_back_to_source_pt(self):
        cfg = self._config("engine")
        self._select(cfg)
        vm = cfg.config["camera_configs"]["cam_0"]["pipeline"]["settings"]["vision_model"]
        self.assertEqual(vm["source_pt"], self._MODEL)
        self.assertEqual(vm["file_path"], self._MODEL)


class _FakeModel:
    model_type = "rknn"

    def _preprocess_frame(self, frame):
        return frame


class OptimizedActiveTests(unittest.TestCase):

    _PT = "YoloModels/pytorch/_boot_test_pose.pt"
    _ARTIFACT = "YoloModels/rknn/_boot_test_pose.rknn"
    _STALE = "YoloModels/rknn/_boot_test_stale_fuel.rknn"

    def setUp(self):
        (_REPO / "YoloModels" / "pytorch").mkdir(parents=True, exist_ok=True)
        (_REPO / "YoloModels" / "rknn").mkdir(parents=True, exist_ok=True)
        (_REPO / self._PT).write_bytes(b"fake pt")

    def tearDown(self):
        for rel in (self._PT, self._ARTIFACT, self._STALE):
            f = _REPO / rel
            if f.exists():
                f.unlink()
        for d in (_REPO / "YoloModels" / "rknn", _REPO / "YoloModels" / "pytorch"):
            if d.exists() and not any(d.iterdir()):
                d.rmdir()

    def _cam(self, file_path, source_pt):
        cam_cfg = iSpyCameraConfig({
            "name": "cam", "source": 0,
            "pipeline": {"name": "object_detection", "settings": {"vision_model": {
                "file_path": file_path, "source_pt": source_pt}}},
            "calibration": {"distance": 1.0, "game_piece_size": 1.0, "size": 100, "fov": 90},
        })
        cam = ObjectDetectionPipeline.__new__(ObjectDetectionPipeline)
        cam.config = cam_cfg
        cam.model = SimpleNamespace(model_type="rknn")
        cam.yolo_model_file = file_path
        cam._requested_format = "rknn"
        cam._target_format = "rknn"
        cam.logger = logging.getLogger("test")
        return cam

    def test_rejects_stale_artifact_with_built_source(self):
        (_REPO / self._ARTIFACT).write_bytes(b"fake artifact")
        (_REPO / self._STALE).write_bytes(b"fake stale")
        cam = self._cam(str(_REPO / self._STALE), str(_REPO / self._PT))
        self.assertFalse(cam._optimized_active())

    def test_rejects_stale_artifact_when_source_artifact_missing(self):
        (_REPO / self._STALE).write_bytes(b"fake stale")
        cam = self._cam(str(_REPO / self._STALE), str(_REPO / self._PT))
        self.assertFalse(cam._optimized_active())

    def test_accepts_artifact_named_for_current_source(self):
        (_REPO / self._ARTIFACT).write_bytes(b"fake artifact")
        cam = self._cam(str(_REPO / self._ARTIFACT), str(_REPO / self._PT))
        self.assertTrue(cam._optimized_active())


class BootLoadTests(unittest.TestCase):

    _PT = "YoloModels/pytorch/_boot_test_pose.pt"
    _ARTIFACT = "YoloModels/rknn/_boot_test_pose.rknn"
    _STALE = "YoloModels/rknn/_boot_test_stale_fuel.rknn"

    def setUp(self):
        (_REPO / "YoloModels" / "pytorch").mkdir(parents=True, exist_ok=True)
        (_REPO / "YoloModels" / "rknn").mkdir(parents=True, exist_ok=True)
        (_REPO / self._PT).write_bytes(b"fake pt")

    def tearDown(self):
        for rel in (self._PT, self._ARTIFACT, self._STALE):
            f = _REPO / rel
            if f.exists():
                f.unlink()
        for d in (_REPO / "YoloModels" / "rknn", _REPO / "YoloModels" / "pytorch"):
            if d.exists() and not any(d.iterdir()):
                d.rmdir()

    def _build(self, vm_cfg):
        tmp = Path(tempfile.mkdtemp())
        config = iSpyConfig(file_path=str(tmp / "config.json"))
        config.config["app_mode"] = False
        cam_cfg = iSpyCameraConfig({
            "name": "cam", "source": 99, "fps_cap": 1000,
            "yaw": 0, "pitch": 0, "height": 1.0, "x": 0, "y": 0,
            "grayscale": False, "subsystem": "bench",
            "calibration": {"distance": 1.0, "game_piece_size": 1.0, "size": 100, "fov": 90},
            "pipeline": {"name": "object_detection", "settings": {"vision_model": dict(vm_cfg)}},
        })
        with mock.patch("iSpy.vision.pipelines.object_detection.GenericYolo", return_value=_FakeModel()), \
             mock.patch("iSpy.vision.ModelInspector.fill_missing_config", side_effect=lambda m: dict(m)), \
             mock.patch.object(ObjectDetectionPipeline, "_optimize_runner", lambda self: None):
            cam = ObjectDetectionPipeline(cam_cfg, config)
            return cam

    def test_prefers_existing_artifact_over_stale_file_path(self):
        (_REPO / self._ARTIFACT).write_bytes(b"fake artifact")
        (_REPO / self._STALE).write_bytes(b"fake stale")
        cam = self._build({
            "file_path": self._STALE,
            "source_pt": self._PT,
            "optimize": True, "target_format": "rknn",
            "min_conf": 0.5, "input_size": (640, 640), "margin": 0,
        })
        self.assertTrue(Path(cam.yolo_model_file).name == "_boot_test_pose.rknn")

    def test_falls_back_to_source_pt_when_no_artifact(self):
        (_REPO / self._STALE).write_bytes(b"fake stale")
        cam = self._build({
            "file_path": self._STALE,
            "source_pt": self._PT,
            "optimize": True, "target_format": "rknn",
            "min_conf": 0.5, "input_size": (640, 640), "margin": 0,
        })
        self.assertTrue(Path(cam.yolo_model_file).name == "_boot_test_pose.pt")


if __name__ == "__main__":
    unittest.main()
