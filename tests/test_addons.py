import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from iSpy.config.iSpyConfig import iSpyAddonConfig, iSpyConfig
from iSpy.plugins._loader import load_plugins
from iSpy.plugins.bases import (
    TrackerBase, UtilityBase, FrameProcessorBase, AddonBase,
)
from iSpy.vision.Object import Object

PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "iSpy" / "plugins"

BARE_CONTEXT = {
    "config": iSpyAddonConfig({}),
    "global_config": None,
    "cameras": [],
    "flask_app": None,
    "vision_instance": None,
}


def addon_context(cls, settings: dict | None = None) -> dict:
    ctx = dict(BARE_CONTEXT)
    ctx["config"] = iSpyAddonConfig(
        settings or {}, defaults=cls.default_settings()
    )
    return ctx


class AddonDiscoveryTests(unittest.TestCase):
    def test_all_builtin_addons_are_discovered(self):
        trackers = load_plugins(PLUGIN_ROOT / "trackers", TrackerBase)
        utilities = load_plugins(PLUGIN_ROOT / "utilities", UtilityBase)
        fps = load_plugins(PLUGIN_ROOT / "frame_processors", FrameProcessorBase)
        self.assertTrue({"object_tracker", "path_planner"} <= set(trackers))
        self.assertTrue({"network_table_handler", "rollback"} <= set(utilities))
        self.assertIn("example_frame_processor", fps)
        self.assertIn("example_tracker", trackers)
        self.assertIn("example_utility", utilities)

    def test_every_addon_is_an_addonbase_subclass(self):
        for subdir, base in (("trackers", TrackerBase),
                             ("utilities", UtilityBase),
                             ("frame_processors", FrameProcessorBase)):
            for name, cls in load_plugins(PLUGIN_ROOT / subdir, base).items():
                self.assertTrue(issubclass(cls, AddonBase), name)

    def test_addon_schemas_are_valid(self):
        valid_types = {"text", "number", "toggle", "list"}
        for subdir, base in (("trackers", TrackerBase),
                             ("utilities", UtilityBase),
                             ("frame_processors", FrameProcessorBase)):
            for name, cls in load_plugins(PLUGIN_ROOT / subdir, base).items():
                schema = cls.config_schema()
                self.assertIsInstance(schema, dict, name)
                for key, defn in schema.items():
                    self.assertIsInstance(defn, dict, f"{name}.{key}")
                    self.assertIn("type", defn, f"{name}.{key}")
                    self.assertIn(defn["type"], valid_types, f"{name}.{key}")
                    if defn["type"] == "number":
                        self.assertIsInstance(defn.get("default", 0), (int, float))
                    elif defn["type"] == "toggle":
                        self.assertIsInstance(defn.get("default", False), bool)
                    elif defn["type"] == "list":
                        self.assertIn("fields", defn, f"{name}.{key}")
                    else:
                        self.assertIsInstance(defn.get("default", ""), str)
                    self.assertIn("label", defn, f"{name}.{key}")

    def test_addon_default_settings_match_schemas(self):
        for subdir, base in (("trackers", TrackerBase),
                             ("utilities", UtilityBase),
                             ("frame_processors", FrameProcessorBase)):
            for name, cls in load_plugins(PLUGIN_ROOT / subdir, base).items():
                for key, value in cls.default_settings().items():
                    self.assertEqual(
                        cls.config_schema()[key]["default"], value, name
                    )

    def test_empty_plugin_name_guard(self):
        # Every discovered add-on must carry a unique, string plugin_name.
        seen = set()
        for subdir, base in (("trackers", TrackerBase),
                             ("utilities", UtilityBase),
                             ("frame_processors", FrameProcessorBase)):
            for name in load_plugins(PLUGIN_ROOT / subdir, base):
                self.assertIsInstance(name, str)
                self.assertNotIn(name, seen)
                seen.add(name)


class AddonBaseTests(unittest.TestCase):
    def test_base_default_schema_is_empty(self):
        self.assertEqual(AddonBase.config_schema(), {})

    def test_base_context_normalizes_config(self):
        inst = AddonBase({"config": {"a": 1}})
        self.assertIsInstance(inst.config, iSpyAddonConfig)
        self.assertEqual(inst.config.get("a"), 1)
        inst2 = AddonBase({})
        self.assertIsInstance(inst2.config, iSpyAddonConfig)
        self.assertEqual(inst2.config.to_dict(), {})

    def test_status_mixin_contract(self):
        inst = AddonBase({})
        self.assertEqual(inst.get_status(), "idle")
        inst.update_status("running")
        self.assertEqual(inst.get_status(), "running")


class ObjectTrackerTests(unittest.TestCase):
    def _make(self, settings=None):
        return load_plugins(PLUGIN_ROOT / "trackers", TrackerBase)[
            "object_tracker"](addon_context(
                load_plugins(PLUGIN_ROOT / "trackers", TrackerBase)["object_tracker"],
                settings))

    def test_schema_defaults_apply(self):
        tracker = self._make()
        self.assertEqual(tracker.distance_threshold, 0.5)
        self.assertEqual(tracker.stale_threshold, 1.0)

    def test_custom_settings_apply(self):
        tracker = self._make(
            {"distance_threshold": 0.9, "stale_threshold": 2.5}
        )
        self.assertEqual(tracker.distance_threshold, 0.9)
        self.assertEqual(tracker.stale_threshold, 2.5)

    def test_invalid_distance_threshold_falls_back(self):
        tracker = self._make({"distance_threshold": -4})
        self.assertEqual(tracker.distance_threshold, 0.5)

    def test_nearby_detections_merge(self):
        tracker = self._make()
        fuels = tracker.update(
            [Object(0.0, 0.0), Object(0.1, 0.05)], 0, 0, 0)
        self.assertEqual(len(fuels), 1)
        first_id = fuels[0].get_id()
        # same spot again -> same object, not a new one
        fuels = tracker.update([Object(0.03, -0.02)], 0, 0, 0)
        self.assertEqual(len(fuels), 1)
        self.assertEqual(fuels[0].get_id(), first_id)

    def test_distant_detections_stay_separate(self):
        tracker = self._make({"distance_threshold": 0.3})
        fuels = tracker.update([Object(0.0, 0.0), Object(5.0, 5.0)], 0, 0, 0)
        self.assertEqual(len(fuels), 2)

    def test_different_names_never_merge_even_when_close(self):
        # a cone 10cm from a robot must not merge into the robot
        cone = Object(0.0, 0.0)
        cone.name = "cone"
        robot = Object(0.1, 0.05)
        robot.name = "robot"
        tracked = self._make().update([cone, robot], 0, 0, 0)
        self.assertEqual(len(tracked), 2)
        names = {o.name for o in tracked}
        self.assertEqual(names, {"cone", "robot"})

    def test_same_name_close_detections_still_merge(self):
        tracker = self._make()
        a = Object(0.0, 0.0)
        a.name = "cone"
        b = Object(0.03, -0.02)
        b.name = "cone"
        tracked = tracker.update([a], 0, 0, 0)
        first_id = tracked[0].get_id()
        tracked = tracker.update([b], 0, 0, 0)
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].get_id(), first_id)

    def test_empty_names_are_treated_as_equal(self):
        # legacy pipelines that never set .name keep old merge behavior
        tracker = self._make()
        tracked = tracker.update([Object(0.0, 0.0)], 0, 0, 0)
        first_id = tracked[0].get_id()
        tracked = tracker.update([Object(0.03, -0.02)], 0, 0, 0)
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].get_id(), first_id)

    def test_stale_objects_are_dropped(self):
        tracker = self._make({"stale_threshold": 0.05})
        tracker.update([Object(0.0, 0.0)], 0, 0, 0)
        time.sleep(0.1)
        fuels = tracker.update([], 0, 0, 0)
        self.assertEqual(fuels, [])

    def test_robot_transform_applied(self):
        tracker = self._make()
        fuels = tracker.update([Object(1.0, 0.0)], 10, 20, 0)
        pos = fuels[0].get_position_normally()
        self.assertAlmostEqual(pos[0], 11.0)
        self.assertAlmostEqual(pos[1], 20.0)

    def test_stop_clears_tracked_list(self):
        tracker = self._make()
        tracker.update([Object(0.0, 0.0)], 0, 0, 0)
        self.assertTrue(tracker.tracked_objects)
        tracker.stop()
        self.assertEqual(tracker.tracked_objects, [])


class PathPlannerTests(unittest.TestCase):
    def _make(self, settings=None):
        cls = load_plugins(PLUGIN_ROOT / "trackers", TrackerBase)["path_planner"]
        return cls(addon_context(cls, settings))

    def test_schema_defaults_apply(self):
        planner = self._make()
        self.assertEqual(planner.epsilon, 0.3)
        self.assertEqual(planner.min_samples, 3)

    def test_custom_settings_apply(self):
        planner = self._make({"epsilon": 1.0, "min_samples": 2})
        self.assertEqual(planner.epsilon, 1.0)
        self.assertEqual(planner.min_samples, 2)

    def test_clusters_and_noise_are_separated(self):
        planner = self._make({"epsilon": 0.3, "min_samples": 3})
        detections = [
            Object(0.0, 0.0), Object(0.1, 0.0), Object(0.2, 0.0),  # cluster
            Object(9.0, 9.0),  # noise
        ]
        cleaned = planner.update(detections, 0, 0, 0)
        self.assertIs(cleaned, planner.cluster_positions)
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(len(planner.get_noise_positions()), 1)

    def test_empty_input(self):
        planner = self._make()
        self.assertEqual(planner.update([], 0, 0, 0), [])
        self.assertEqual(planner.get_noise_positions(), [])

    def test_run_returns_clusters(self):
        planner = self._make({"epsilon": 0.3, "min_samples": 3})
        planner.update([Object(0.0, 0.0), Object(0.1, 0.0), Object(0.2, 0.0)],
                       0, 0, 0)
        self.assertEqual(len(planner.run()), 3)

    def test_stop_does_not_raise(self):
        self._make().stop()


class RollBackTests(unittest.TestCase):
    def test_schema_defaults_apply(self):
        cls = load_plugins(PLUGIN_ROOT / "utilities", UtilityBase)["rollback"]
        with tempfile.TemporaryDirectory() as tmp:
            rec = cls(addon_context(cls, {"data_dir": tmp}))
            self.assertEqual(rec._video_output_dir, tmp)
            self.assertEqual(rec._fps, 30.0)
            self.assertEqual(rec._max_queue, 300)
            self.assertEqual(rec._downsample, 1)

    def test_enabled_by_presence_records_frames_to_disk(self):
        cls = load_plugins(PLUGIN_ROOT / "utilities", UtilityBase)["rollback"]
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("iSpy.plugins.utilities.BuiltIn.RollBack.time.sleep"):
                rec = cls(addon_context(cls, {"data_dir": tmp, "downsample": 1}))
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                for _ in range(5):
                    rec.update({"frame": frame})
                rec.stop()
            files = list(Path(tmp).glob("recording_*"))
            self.assertEqual(len(files), 1)
            self.assertGreater(files[0].stat().st_size, 24,
                               "recording file must contain frames")

    def test_records_only_every_nth_frame(self):
        cls = load_plugins(PLUGIN_ROOT / "utilities", UtilityBase)["rollback"]
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("iSpy.plugins.utilities.BuiltIn.RollBack.time.sleep"):
                rec = cls(addon_context(cls, {"data_dir": tmp, "downsample": 3}))
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                for _ in range(6):
                    rec.update({"frame": frame})
                rec.stop()
            self.assertGreaterEqual(rec._frame_counter, 6)
            files = list(Path(tmp).glob("recording_*"))
            self.assertEqual(len(files), 1)

    def test_skips_missing_frames(self):
        cls = load_plugins(PLUGIN_ROOT / "utilities", UtilityBase)["rollback"]
        with tempfile.TemporaryDirectory() as tmp:
            rec = cls(addon_context(cls, {"data_dir": tmp}))
            rec.update({"frame": None})  # must not raise / start a writer
            self.assertFalse(rec._started)
            rec.update({})  # missing key entirely
            self.assertFalse(rec._started)
            rec.stop()

    def test_stop_before_start_is_safe(self):
        cls = load_plugins(PLUGIN_ROOT / "utilities", UtilityBase)["rollback"]
        with tempfile.TemporaryDirectory() as tmp:
            rec = cls(addon_context(cls, {"data_dir": tmp}))
            rec.stop()  # no-op, no crash


class NetworkTableHandlerTests(unittest.TestCase):

    def _fresh_module(self, is_connected=True):
        import importlib
        import sys
        sys.modules.pop("iSpy.plugins.utilities.BuiltIn.NetworkHandler", None)
        fake = mock.Mock()
        fake.isConnected.return_value = is_connected
        fake.getTable = mock.Mock(return_value=fake)
        fake.getStructArrayTopic = mock.Mock(return_value=fake)
        fake.getStructTopic = mock.Mock(return_value=fake)
        fake.getDoubleTopic = mock.Mock(return_value=fake)
        fake.getBooleanTopic = mock.Mock(return_value=fake)
        fake.getStringTopic = mock.Mock(return_value=fake)
        fake.subscribe = mock.Mock(return_value=fake)
        fake.publish = mock.Mock(return_value=fake)
        fake.get = mock.Mock(return_value={"x": 1.0, "y": 2.0})
        self._fake_inst = fake
        ntcore_fake = mock.Mock()
        ntcore_fake.NetworkTableInstance.getDefault.return_value = fake
        with mock.patch.dict(sys.modules, {"ntcore": ntcore_fake}):
            return importlib.import_module(
                "iSpy.plugins.utilities.BuiltIn.NetworkHandler")

    def _handler(self, settings=None):
        mod = self._fresh_module()
        cls = mod.NetworkTableHandler
        return cls(addon_context(cls, settings))

    def test_ip_comes_from_addon_settings(self):
        handler = self._handler({"network_tables_ip": "10.1.2.3"})
        self._fake_inst.setServer.assert_called_with("10.1.2.3")
        self._fake_inst.startClient4.assert_called_once()

    def test_default_ip_applies(self):
        handler = self._handler({})
        self._fake_inst.setServer.assert_called_with("10.0.0.2")

    def test_connection_retries_are_non_blocking(self):
        # connecting must never stall boot: startClient4 resolves in the
        # background and __init__ returns immediately with not-connected state.
        mod = self._fresh_module(is_connected=False)
        with mock.patch("time.sleep") as sleep:
            handler = mod.NetworkTableHandler(addon_context(mod.NetworkTableHandler))
            self.assertEqual(sleep.call_count, 0)  # no polling sleep anywhere
            self.assertFalse(handler.isConnected())

    def test_update_publishes_vision_data_when_connected(self):
        handler = self._handler({})
        handler.update({
            "detections": [], "fps": 30.5, "detection_count": 2,
            "camera_lag_s": 0.04, "cameras": [],
        })
        self.assertGreaterEqual(self._fake_inst.flush.call_count, 1)
        self._fake_inst.getStructArrayTopic.assert_called()

    def test_update_with_connected_false_is_noop(self):
        mod = self._fresh_module(is_connected=False)
        handler = mod.NetworkTableHandler(addon_context(mod.NetworkTableHandler))
        handler.update({"detections": []})
        self.assertEqual(self._fake_inst.flush.call_count, 0)

    def test_update_with_detection_objects_publishes_structs(self):
        handler = self._handler({})
        det = Object(1.0, 2.0, 3.0)
        handler.update({"detections": [det], "fps": 10, "detection_count": 1,
                        "camera_lag_s": 0.0, "cameras": []})
        self._fake_inst.getStructArrayTopic.assert_called()

    def test_update_is_pipeline_agnostic(self):
        # detections from ANY pipeline (april tag, qr, depth, custom) flow
        # through unchanged - the publisher only touches generic Object
        # accessors, never YOLO-specific fields
        handler = self._handler({})
        tag = Object(0.5, -1.25, 2.0)
        tag.name = "april_tag_3"
        tag.roll = 0.1
        tag.pitch = -0.2
        tag.yaw = 3.0
        qr = Object(9.0, 8.0, 7.0)
        qr.name = "qr_code"
        handler.update({"detections": [tag, qr], "fps": 60,
                        "detection_count": 2, "camera_lag_s": 0.0,
                        "cameras": []})
        self._fake_inst.getStructArrayTopic.assert_called()
        pub = handler._subscribers["pub/VisionData/vision_data"]
        structs = pub.set.call_args[0][0]
        self.assertEqual(len(structs), 2)
        self.assertAlmostEqual(structs[0].x, 0.5)
        self.assertAlmostEqual(structs[0].y, -1.25)
        self.assertAlmostEqual(structs[1].z, 7.0)

    def test_get_robot_pose(self):
        handler = self._handler({})
        result = handler.get_robot_pose()
        self._fake_inst.getStructTopic.assert_called_once()
        self.assertEqual(result, {"x": 1.0, "y": 2.0})

    def test_stop_is_safe(self):
        handler = self._handler({})
        handler.stop()
        handler.update({"detections": []})


class HealthModuleTests(unittest.TestCase):
    """The always-on HealthModule (iSpy/web/modules/health.py) is the single
    canonical health implementation since health_reporter/status_reporter
    were merged into it (PROMPT 5)."""

    class FakeCamera:
        def __init__(self, name, age):
            self.source = name
            self.config = {"name": name}
            self._age = age

        def get_frame_age(self):
            return self._age

    def _make(self, threshold=None, cameras=None, vision=None):
        from iSpy.web.modules.health import HealthModule
        cfg = iSpyConfig()
        if threshold is not None:
            cfg.config["health_stale_threshold"] = threshold
        ctx = {"config": cfg, "cameras": cameras or [], "vision_instance": vision}
        return HealthModule(ctx)

    def test_stale_threshold_from_top_level_config(self):
        # used to live under utilities.health_reporter.stale_threshold;
        # now a plain top-level config key read by the core module
        self.assertEqual(self._make()._stale_threshold, 1.0)
        self.assertEqual(self._make(threshold=2.5)._stale_threshold, 2.5)

    def test_payload_reflects_updates(self):
        mod = self._make()
        mod.update({"fps": 25.0, "vision_s": 0.02, "detection_count": 7})
        payload, _healthy = mod._build_payload()
        self.assertEqual(payload["fps"], 25.0)
        self.assertEqual(payload["vision_ms"], 20.0)
        self.assertEqual(payload["detections"], 7)
        self.assertGreaterEqual(payload["loop_count"], 1)
        self.assertIn("cameras", payload)
        self.assertIn("addon_health", payload)

    def test_camera_status_marks_stale(self):
        mod = self._make(
            cameras=[self.FakeCamera("a", 0.1), self.FakeCamera("b", 5.0)]
        )
        payload, healthy = mod._build_payload()
        by_name = {c["name"]: c for c in payload["cameras"]}
        self.assertTrue(by_name["a"]["ok"])
        self.assertFalse(by_name["b"]["ok"])
        self.assertFalse(healthy)

    def test_addon_health_from_network_table_handler(self):
        vision = mock.Mock()
        vision.trackers = {}
        vision.utilities = {
            "network_table_handler": mock.Mock(
                get_health=lambda: {
                    "ok": True,
                    "title": "NetworkTables",
                    "info": "Connected",
                    "rows": [{"label": "Robot IP", "value": "10.1.2.3"}],
                }
            ),
        }
        vision.frame_processors = {}
        mod = self._make(vision=vision)
        payload, healthy = mod._build_payload()
        self.assertTrue(healthy)
        self.assertEqual(len(payload["addon_health"]), 1)
        entry = payload["addon_health"][0]
        self.assertEqual(entry["name"], "network_table_handler")
        self.assertEqual(entry["type"], "utility")
        self.assertTrue(entry["ok"])
        self.assertEqual(entry["title"], "NetworkTables")

    def test_unhealthy_addon_degrades_banner(self):
        vision = mock.Mock()
        vision.trackers = {}
        vision.utilities = {
            "network_table_handler": mock.Mock(
                get_health=lambda: {"ok": False, "title": "NetworkTables", "info": "Disconnected", "rows": []}
            ),
        }
        vision.frame_processors = {}
        mod = self._make(vision=vision)
        payload, healthy = mod._build_payload()
        self.assertFalse(healthy)
        self.assertEqual(payload["status"], "degraded")

    def test_broken_camera_counts_as_bad(self):
        class BrokenCam:
            source = "broken"

            def get_frame_age(self):
                raise RuntimeError("boom")

        mod = self._make(cameras=[BrokenCam()])
        payload, healthy = mod._build_payload()
        self.assertFalse(healthy)
        self.assertFalse(payload["cameras"][0]["ok"])

    def test_plugin_statuses_pulled_from_vision_instance(self):
        vision = mock.Mock()
        vision.trackers = {"object_tracker": mock.Mock(get_status=lambda: "running")}
        vision.utilities = {}
        vision.frame_processors = {}
        mod = self._make(vision=vision)
        statuses = mod._plugin_statuses()
        self.assertEqual(statuses,
                         [{"name": "object_tracker", "type": "tracker",
                           "status": "running"}])

    def test_stop_is_safe(self):
        self._make().stop()


class ExampleAddonTests(unittest.TestCase):
    def test_example_tracker_counts_updates(self):
        cls = load_plugins(PLUGIN_ROOT / "trackers", TrackerBase)[
            "example_tracker"]
        tracker = cls(addon_context(cls, {"count_start": 5}))
        self.assertEqual(tracker.count, 5)
        out = tracker.update([], 0, 0, 0)
        self.assertEqual(out, [])
        self.assertEqual(tracker.count, 6)

    def test_example_frame_processor_blackens(self):
        cls = load_plugins(PLUGIN_ROOT / "frame_processors", FrameProcessorBase)[
            "example_frame_processor"]
        fp = cls(addon_context(cls))
        frame = np.full((10, 10, 3), 255, dtype=np.uint8)
        out = fp.process(frame)
        self.assertTrue(np.all(out == 0))
        fp.stop()  # safe

    def test_example_utility_ignores_missing_flask(self):
        cls = load_plugins(PLUGIN_ROOT / "utilities", UtilityBase)[
            "example_utility"]
        util = cls(addon_context(cls))
        self.assertIsNone(util.flask_app)
        self.assertIsNone(util.get_robot_pose())
        util.stop()

    def test_example_utility_registers_route_with_flask(self):
        import flask
        cls = load_plugins(PLUGIN_ROOT / "utilities", UtilityBase)[
            "example_utility"]
        app = flask.Flask(__name__)
        ctx = addon_context(cls)
        ctx["flask_app"] = app
        util = cls(ctx)
        with app.test_client() as client:
            resp = client.get("/ispy-example")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("hello", resp.get_data(as_text=True))

    def test_example_utility_declares_and_publishes_output(self):
        # template utilities demonstrate the addon output system:
        # output_key declared in schema -> counter published every tick
        cls = load_plugins(PLUGIN_ROOT / "utilities", UtilityBase)[
            "example_utility"]
        self.assertIn("output_key", cls.config_schema())
        util = cls(addon_context(cls))
        frame_data = {}
        util.update(frame_data)
        util.update(frame_data)
        self.assertEqual(
            frame_data["addon_data"]["example_output"], 2)


if __name__ == "__main__":
    unittest.main()