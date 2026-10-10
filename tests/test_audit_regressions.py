import inspect
import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from iSpy.boot import setup_service
from iSpy.config.iSpyConfig import iSpyConfig
from iSpy.utilities.MultipleCameraHandler import MultipleCameraHandler
from iSpy.vision.Cameras.ReplayCamera import ReplayCamera
from iSpy.vision.Object import Object
from iSpy.vision.genericYolo import GenericYolo
from iSpy.web.modules import dashboard, rollback


def write_replay_files(session: Path, camera="Left"):
    session.mkdir(parents=True, exist_ok=True)
    stem = "rollback_20261008_220000"
    (session / f"{stem}.json").write_text(
        json.dumps({"stem": stem, "session": session.name, "cams": [camera]}),
        encoding="utf-8",
    )
    clip = session / f"{stem}_{camera}.avi"
    clip.write_bytes(b"")
    return clip


class ServiceSetupRegressions(unittest.TestCase):
    @mock.patch.object(setup_service, "_service_user", return_value="robot")
    @mock.patch.object(setup_service, "_make_python", return_value="/venv/bin/python")
    @mock.patch.object(setup_service, "_systemd_is_running", return_value=False)
    @mock.patch.object(setup_service, "run")
    @mock.patch.object(
        setup_service.subprocess,
        "run",
        return_value=SimpleNamespace(returncode=0, stderr=""),
    )
    def test_service_runs_the_live_vision_cli_and_is_chroot_safe(
        self, _write, run, *_mocks
    ):
        setup_service.setup_systemd("ignored", project_root="/opt/ispy")

        unit = _write.call_args.kwargs["input"]
        self.assertIn("ExecStart=/venv/bin/python -m iSpy.cli start", unit)
        self.assertNotIn("-m iSpy.boot.boot", unit)
        self.assertEqual(run.call_args_list, [mock.call(["sudo", "systemctl", "enable", "iSpy"])])

    @mock.patch.object(setup_service, "_make_python", return_value="/venv/bin/python")
    @mock.patch.object(setup_service, "_systemd_is_running", return_value=False)
    @mock.patch.object(setup_service, "run")
    @mock.patch.object(
        setup_service.subprocess,
        "run",
        return_value=SimpleNamespace(returncode=0, stderr=""),
    )
    def test_first_boot_service_is_enabled_without_starting_in_a_chroot(
        self, _write, run, *_mocks
    ):
        setup_service.setup_first_boot_service(project_root="/opt/ispy")
        self.assertEqual(
            run.call_args_list,
            [mock.call(["sudo", "systemctl", "enable", "ispy-first-boot"])],
        )


class TriangulationOutputUnitTests(unittest.TestCase):
    def test_triangulated_points_are_converted_from_inches_to_frc_meters(self):
        handler = MultipleCameraHandler(
            [],
            {
                "unit": "frc",
                "triangulation_match_distance": 2.0,
                "triangulation_max_residual": 0.5,
            },
        )
        a = Object(0.0, 0.0, name="fuel", ray_origin=np.zeros(3), ray_direction=np.ones(3))
        b = Object(0.0, 0.0, name="fuel", ray_origin=np.zeros(3), ray_direction=np.ones(3))
        inch_point = np.array([39.37007874, 78.74015748, 118.11023622])
        with mock.patch(
            "iSpy.utilities.MultipleCameraHandler.triangulation.closest_point_between_rays",
            return_value=(inch_point, 0.1),
        ):
            result = handler._merge_with_triangulation([[a], [b]])
        np.testing.assert_allclose(result[0].get_position(), [1.0, 2.0, 3.0], atol=1e-7)
        self.assertEqual(result[0].depth_source, "triangulated")
        handler.destroy()


class LoopTimingRegressions(unittest.TestCase):
    @staticmethod
    def _fake_pose():
        from wpimath.geometry import Pose2d

        return Pose2d()

    def test_solo_loop_time_includes_vision_before_tracker_updates(self):
        import time

        from iSpy.iSpy import iSpy

        class Camera:
            plugin_name = "test"
            _camera_config = {}

            def get_frame_age(self):
                return 0.0

            def get_code_times(self):
                return {}

            def in_calibration_mode(self):
                return False

        class Tracker:
            def update(self, detections, *_pose):
                time.sleep(0.01)
                return detections

        vision = iSpy.__new__(iSpy)
        vision.cameras = []
        vision.trackers = {"tracker": Tracker()}
        vision._raw_frames = lambda _cameras: {}
        vision._reset_frame_processor_times = lambda: None
        vision.run_solo_vision = lambda _camera: (
            time.sleep(0.04) or ([], np.zeros((4, 4, 3), dtype=np.uint8))
        )
        vision._merge_frame_processor_times = lambda _times: 0.0
        vision._get_pose = lambda _times=None: self._fake_pose()
        vision._addon_breakdown_parts = lambda _plugin: {}

        result = vision._run_loop_body_solo(Camera())
        self.assertGreaterEqual(result["loop_s"], 0.045)

    def test_multi_loop_time_includes_vision_before_tracker_updates(self):
        import time

        from iSpy.iSpy import iSpy

        class Camera:
            plugin_name = "test"

            def get_frame_age(self):
                return 0.0

            def get_code_times(self):
                return {}

        class Handler:
            cameras = [Camera()]

            @staticmethod
            def get_camera_frames():
                return {}

        class Tracker:
            def update(self, detections, *_pose):
                time.sleep(0.01)
                return detections

        vision = iSpy.__new__(iSpy)
        vision.cameras = []
        vision.trackers = {"tracker": Tracker()}
        vision._raw_frames = lambda _cameras: {}
        vision._reset_frame_processor_times = lambda: None
        vision.run_multi_vision = lambda _handler: (
            time.sleep(0.04) or ([], np.zeros((4, 4, 3), dtype=np.uint8))
        )
        vision._merge_frame_processor_times = lambda _times: 0.0
        vision._get_pose = lambda _times=None: self._fake_pose()
        vision._addon_breakdown_parts = lambda _plugin: {}

        result = vision._run_loop_body_multi(Handler())
        self.assertGreaterEqual(result["loop_s"], 0.045)


class OpenVinoOutputAlignmentTests(unittest.TestCase):
    def test_raw_activated_scores_are_nms_filtered(self):
        model = GenericYolo.__new__(GenericYolo)
        model.output = {
            "format": "hardware_nms",
            "layout": "anchors_first",
            "score_mode": "classwise",
            "nms_iou": 0.45,
        }
        model.logger = logging.getLogger(__name__)
        model._align_openvino_output((1, 5, 6))
        self.assertFalse(model.output["scores_are_logits"])
        self.assertTrue(model.output["apply_software_nms"])
        self.assertEqual(model.output["nms_iou"], 0.45)

        model.model_type = "openvino"
        model.task = "detect"
        model.num_classes = 1
        model.min_conf = 0.5
        model.input_size = (640, 640)
        model._feat_width = 5
        model._output_verified = True
        tensor = np.zeros((1, 5, 6), dtype=np.float32)
        tensor[0, :, 0] = [100, 100, 40, 40, 0.9]
        tensor[0, :, 1] = [100, 100, 40, 40, 0.8]
        tensor[0, :, 2] = [300, 300, 40, 40, 0.7]
        tensor[0, :, 3] = [500, 500, 30, 30, 0.1]
        result = model.postprocess([tensor], (640, 640))
        self.assertEqual(len(result.boxes), 2)
        self.assertAlmostEqual(max(box.conf for box in result.boxes), 0.9)


class ReplayIntegrationRegressions(unittest.TestCase):
    def test_replay_camera_uses_the_camera_base_get_frame_contract(self):
        params = inspect.signature(ReplayCamera.get_frame).parameters
        self.assertEqual(list(params), ["self"])

    def test_boot_replay_uses_recorded_camera_and_disables_robot_outputs(self):
        from iSpy.boot.boot import _configure_replay

        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "session_2026-10-08_22-00-00"
            write_replay_files(session)
            (session / "session.json").write_text(
                json.dumps(
                    {
                        "cameras": {
                            "Left": {
                                "pipeline": "object_detection",
                                "x": 1,
                                "y": 2,
                                "height": 3,
                                "yaw": 4,
                                "pitch": 5,
                                "calibration": {"fov": 60},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = iSpyConfig()
            config.config["plugins"]["utilities"]["FRC/network_table_handler"] = {}
            _configure_replay(config, str(session), "Left", 2.0, loop=False)

            cameras = config.get("camera_configs")
            self.assertEqual(list(cameras), ["replay_Left"])
            replay = cameras["replay_Left"]
            self.assertEqual(replay["camera_type"], "replay")
            self.assertEqual(replay["replay_cam"], "Left")
            self.assertEqual(replay["replay_speed"], 2.0)
            self.assertFalse(replay["replay_loop"])
            self.assertEqual(replay["height"], 3)
            self.assertNotIn(
                "FRC/network_table_handler",
                config.get("plugins")["utilities"],
            )
            self.assertFalse(config.get("rollback")["enabled"])

    def test_rollback_api_generates_command_for_the_selected_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            session = data_dir / "session_2026-10-08_22-00-00"
            write_replay_files(session)
            module = rollback.RollbackModule(
                {"config": {"rollback": {"data_dir": str(data_dir)}}}
            )
            from flask import Flask

            app = Flask(__name__)
            module.register_routes(app)
            response = app.test_client().get(
                f"/api/rollback/{session.name}/replay?cam=Left"
            )
            payload = response.get_json()
            self.assertIn("ispy-boot --replay", payload["command"])
            self.assertIn("--replay-cam Left", payload["command"])

            with mock.patch.dict("os.environ", {"ISPY_REPLAY_MODE": "1"}):
                status = app.test_client().get("/api/rollback/status").get_json()
            self.assertTrue(status["replay_mode"])


class DashboardTemperatureUnitTests(unittest.TestCase):
    def test_temperature_unit_is_returned_even_without_psutil(self):
        module = dashboard.DashboardModule.__new__(dashboard.DashboardModule)
        module.context = {"config": {"temperature_unit": "fahrenheit"}}
        module._get_hardware = lambda: []
        with mock.patch.object(dashboard, "PSUTIL_AVAILABLE", False):
            result = module._get_system_metrics()
        self.assertEqual(result["temperature_unit"], "fahrenheit")

    def test_temperature_unit_is_returned_with_psutil_metrics(self):
        module = dashboard.DashboardModule.__new__(dashboard.DashboardModule)
        module.context = {"config": {"temperature_unit": "fahrenheit"}}
        module._get_hardware = lambda: []
        fake_psutil = SimpleNamespace(
            cpu_percent=lambda interval=None: 12,
            virtual_memory=lambda: SimpleNamespace(percent=50, used=1000, total=2000),
            sensors_temperatures=lambda: {},
        )
        with (
            mock.patch.object(dashboard, "PSUTIL_AVAILABLE", True),
            mock.patch.object(dashboard, "psutil", fake_psutil, create=True),
        ):
            result = module._get_system_metrics()
        self.assertEqual(result["temperature_unit"], "fahrenheit")


if __name__ == "__main__":
    unittest.main()
