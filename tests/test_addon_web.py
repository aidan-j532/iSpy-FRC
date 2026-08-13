"""add-on web layer (/addons API) + full iSpy integration tests:
schemas/settings/enabled state, toggling (presence == enabled), settings
validation + coercion, disable-before-delete, iSpy loading every add-on.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import flask

from iSpy.config.iSpyConfig import iSpyConfig
from iSpy.web.Backend.PluginStatus import (
    PluginStatusModule,
    _coerce_setting_value,
)
from iSpy.plugins._loader import load_plugins
from iSpy.plugins.bases import TrackerBase, UtilityBase, FrameProcessorBase

PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "iSpy" / "plugins"


def _app_context():
    return flask.Flask(__name__).app_context()


class FakeVision:
    def __init__(self):
        self.trackers = {}
        self.utilities = {}
        self.frame_processors = {}
        self.config = iSpyConfig()


class PluginStatusModuleTests(unittest.TestCase):
    def _module(self, plugins=None):
        cfg = iSpyConfig()
        cfg.config["app_mode"] = False
        if plugins is not None:
            cfg.config["plugins"] = plugins
        vision = FakeVision()
        vision.config = cfg
        mod = PluginStatusModule({
            "config": cfg,
            "vision_instance": vision,
        })
        return mod, cfg

    # ---------- available ----------

    def test_available_lists_all_addons_with_schemas(self):
        mod, cfg = self._module()
        with _app_context():
            payload = mod._available().get_json()
        kinds = {p["type"] for p in payload["available"]}
        self.assertTrue({"tracker", "utility", "frame_processor"} <= kinds)
        by_name = {(p["type"], p["name"]): p for p in payload["available"]}

        obj = by_name[("tracker", "object_tracker")]
        self.assertIn("distance_threshold", obj["config_schema"])
        self.assertEqual(obj["config_schema"]["distance_threshold"]["default"], 0.5)
        self.assertFalse(obj["enabled"])
        self.assertEqual(obj["settings"], {})

        nt = by_name[("utility", "network_table_handler")]
        self.assertEqual(
            nt["config_schema"]["network_tables_ip"]["default"], "10.0.0.2")

        # Built-in vision pipelines remain read-only listings.
        self.assertTrue(any(p["builtin"] and p["type"] == "vision_pipeline"
                            for p in payload["available"]))

    def test_builtin_flag_marks_bundled_addons(self):
        mod, cfg = self._module()
        with _app_context():
            payload = mod._available().get_json()
        by_name = {(p["type"], p["name"]): p for p in payload["available"]}
        # <type>/BuiltIn/ add-ons are builtin - toggleable + configurable but never deletable
        self.assertTrue(by_name[("tracker", "object_tracker")]["builtin"])
        self.assertTrue(by_name[("utility", "video_recorder")]["builtin"])
        # ...user-authored ones arent
        self.assertFalse(by_name[("tracker", "example_tracker")]["builtin"])

    def test_source_serves_bundled_addon_file(self):
        mod, cfg = self._module()
        with _app_context():
            resp = mod._source("utility", "health_reporter")
        payload = resp.get_json()
        self.assertIn("class HealthReporter", payload["source"])
        self.assertEqual(payload["filename"], "BuiltIn/HealthReporter.py")

    def test_source_serves_custom_addon_file(self):
        mod, cfg = self._module()
        with _app_context():
            resp = mod._source("tracker", "example_tracker")
        payload = resp.get_json()
        self.assertIn("plugin_name = \"example_tracker\"", payload["source"])

    def test_source_unknown_addon_404(self):
        mod, cfg = self._module()
        with _app_context():
            resp = mod._source("tracker", "nope")
        self.assertEqual(resp[1], 404)

    def test_available_reflects_enabled_settings_from_config(self):
        mod, cfg = self._module({
            "trackers": {"object_tracker": {"distance_threshold": 0.9}},
            "utilities": {"network_table_handler": {"network_tables_ip": "10.1.1.1"}},
            "frame_processors": {},
        })
        with _app_context():
            payload = mod._available().get_json()
        by_name = {(p["type"], p["name"]): p for p in payload["available"]}
        self.assertTrue(by_name[("tracker", "object_tracker")]["enabled"])
        self.assertEqual(
            by_name[("tracker", "object_tracker")]["settings"],
            {"distance_threshold": 0.9},
        )
        nt = by_name[("utility", "network_table_handler")]
        self.assertTrue(nt["enabled"])
        self.assertEqual(nt["settings"], {"network_tables_ip": "10.1.1.1"})

    # ---------- toggle ----------

    def test_toggle_enable_adds_dict_entry(self):
        mod, cfg = self._module()
        with _app_context(), mock.patch(
            "iSpy.web.Backend.PluginStatus.request",
            _FakeRequest.get_json({"name": "object_tracker", "type": "tracker",
                                   "enable": True})):
            resp = mod._toggle()
        self.assertTrue(resp.get_json()["success"])
        self.assertIn("object_tracker", cfg.config["plugins"]["trackers"])
        self.assertEqual(
            cfg.config["plugins"]["trackers"]["object_tracker"],
            {"distance_threshold": 0.5, "stale_threshold": 1.0},
        )

    def test_toggle_disable_removes_dict_entry(self):
        mod, cfg = self._module({
            "trackers": {"object_tracker": {"distance_threshold": 0.7}},
            "utilities": {}, "frame_processors": {},
        })
        with _app_context(), mock.patch(
            "iSpy.web.Backend.PluginStatus.request",
            _FakeRequest.get_json({"name": "object_tracker", "type": "tracker",
                                   "enable": False})):
            resp = mod._toggle()
        self.assertTrue(resp.get_json()["success"])
        self.assertNotIn("object_tracker", cfg.config["plugins"]["trackers"])

    def test_toggle_unknown_addon_404(self):
        mod, cfg = self._module()
        with _app_context(), mock.patch(
            "iSpy.web.Backend.PluginStatus.request",
            _FakeRequest.get_json({"name": "nope", "type": "tracker",
                                   "enable": True})):
            resp = mod._toggle()
        self.assertEqual(resp[1], 404)

    def test_toggle_vision_pipeline_rejected(self):
        mod, cfg = self._module()
        with _app_context(), mock.patch(
            "iSpy.web.Backend.PluginStatus.request",
            _FakeRequest.get_json({"name": "april_tag", "type": "vision_pipeline",
                                   "enable": True})):
            resp = mod._toggle()
        self.assertEqual(resp[1], 400)

    # ---------- settings ----------

    def test_save_settings_merges_into_enabled_addon(self):
        mod, cfg = self._module({
            "trackers": {"object_tracker": {"distance_threshold": 0.5}},
            "utilities": {}, "frame_processors": {},
        })
        with _app_context(), mock.patch(
            "iSpy.web.Backend.PluginStatus.request",
            _FakeRequest.get_json({"name": "object_tracker", "type": "tracker",
                                   "settings": {"stale_threshold": 2.0}})):
            resp = mod._save_settings()
        self.assertTrue(resp.get_json()["success"])
        self.assertEqual(
            cfg.get_addon_settings("trackers", "object_tracker"),
            {"distance_threshold": 0.5, "stale_threshold": 2.0},
        )

    def test_save_settings_coerces_types(self):
        mod, cfg = self._module({
            "trackers": {"object_tracker": {}},
            "utilities": {"network_table_handler": {}},
            "frame_processors": {},
        })
        with _app_context(), mock.patch(
            "iSpy.web.Backend.PluginStatus.request",
            _FakeRequest.get_json({"name": "object_tracker", "type": "tracker",
                                   "settings": {"distance_threshold": "0.75"}})):
            resp = mod._save_settings()
        saved = cfg.get_addon_settings("trackers", "object_tracker")
        self.assertIsInstance(saved["distance_threshold"], float)
        self.assertEqual(saved["distance_threshold"], 0.75)

    def test_save_settings_rejects_unknown_keys(self):
        mod, cfg = self._module({
            "trackers": {"object_tracker": {}},
            "utilities": {}, "frame_processors": {},
        })
        with _app_context(), mock.patch(
            "iSpy.web.Backend.PluginStatus.request",
            _FakeRequest.get_json({"name": "object_tracker", "type": "tracker",
                                   "settings": {"bogus_key": 1}})):
            resp = mod._save_settings()
        self.assertEqual(resp[1], 400)
        self.assertNotIn("bogus_key", cfg.get_addon_settings("trackers",
                                                             "object_tracker"))

    def test_save_settings_rejects_bad_numbers(self):
        mod, cfg = self._module({
            "trackers": {"object_tracker": {}},
            "utilities": {}, "frame_processors": {},
        })
        with _app_context(), mock.patch(
            "iSpy.web.Backend.PluginStatus.request",
            _FakeRequest.get_json({"name": "object_tracker", "type": "tracker",
                                   "settings": {"distance_threshold": "abc"}})):
            resp = mod._save_settings()
        self.assertEqual(resp[1], 400)

    def test_save_settings_requires_enabled_addon(self):
        mod, cfg = self._module()
        with _app_context(), mock.patch(
            "iSpy.web.Backend.PluginStatus.request",
            _FakeRequest.get_json({"name": "object_tracker", "type": "tracker",
                                   "settings": {"distance_threshold": 0.5}})):
            resp = mod._save_settings()
        self.assertEqual(resp[1], 409)

    def test_save_settings_unknown_addon_404(self):
        mod, cfg = self._module()
        with _app_context(), mock.patch(
            "iSpy.web.Backend.PluginStatus.request",
            _FakeRequest.get_json({"name": "nope", "type": "tracker",
                                   "settings": {}})):
            resp = mod._save_settings()
        self.assertEqual(resp[1], 404)

    # ---------- status ----------

    def test_status_reports_loaded_instances(self):
        vision = FakeVision()
        vision.trackers["object_tracker"] = mock.Mock(get_status=lambda: "running")
        vision.utilities["health_reporter"] = mock.Mock(get_status=lambda: "idle")
        mod = PluginStatusModule({"config": iSpyConfig(),
                                  "vision_instance": vision})
        with _app_context():
            payload = mod._status().get_json()
        plugins = payload["plugins"]
        self.assertEqual(len(plugins), 2)
        by_name = {p["name"]: p for p in plugins}
        self.assertEqual(by_name["object_tracker"]["type"], "tracker")
        self.assertEqual(by_name["object_tracker"]["status"], "running")
        self.assertEqual(by_name["health_reporter"]["status"], "idle")

    # ---------- delete ----------

    def test_delete_disables_then_removes_file(self):
        mod, cfg = self._module({
            "trackers": {"example_tracker": {}},
            "utilities": {}, "frame_processors": {},
        })
        # create a disposable add-on file so deletion has something to delete
        target = PLUGIN_ROOT / "trackers" / "delete_me_test.py"
        target.write_text("from iSpy.plugins.bases import TrackerBase\n"
                          "class DeleteMe(TrackerBase):\n"
                          "    plugin_name = \"delete_me_test\"\n")
        try:
            # disable the example, delete the new one
            with _app_context():
                resp = mod._delete("tracker", "delete_me_test")
            self.assertTrue(resp.get_json()["success"])
            self.assertFalse(target.exists())
            self.assertNotIn("delete_me_test",
                             cfg.config["plugins"]["trackers"])
        finally:
            target.unlink(missing_ok=True)

    def test_delete_builtin_rejected(self):
        mod, cfg = self._module()
        # builtins resolve to non-existent plain files (reserved dir unreachable
        # by filename) so deletion always 404s, never touching built-in code
        with _app_context():
            resp = mod._delete("tracker", "BuiltIn/ObjectTracker")
        self.assertEqual(resp[1], 404)
        with _app_context():
            resp2 = mod._delete("tracker", "object_tracker")
        self.assertEqual(resp2[1], 404)

    def test_resolve_safe_path_never_reaches_builtin_dir(self):
        mod, cfg = self._module()
        for filename in ("BuiltIn/ObjectTracker", "BuiltIn/ObjectTracker.py",
                         "../BuiltIn/HealthReporter"):
            path = mod._resolve_safe_path("utilities", filename)
            self.assertIsNotNone(path)
            self.assertNotIn("BuiltIn", path.parts, filename)
            self.assertTrue(path.is_relative_to(
                Path(__file__).resolve().parents[1] / "iSpy" / "plugins"))

    def test_create_and_upload_are_schema_checked(self):
        mod, cfg = self._module()
        app = flask.Flask(__name__)
        mod.register_routes(app)
        mod.context = {"config": cfg, "vision_instance": FakeVision()}
        with app.test_client() as client:
            r = client.post("/api/plugins/create", json={
                "type": "tracker",
                "filename": "bad_addon",
                "code": "class NoBase:\n    pass\n",
            })
            self.assertEqual(r.status_code, 400)
            r2 = client.post("/api/plugins/create", json={
                "type": "vision_pipeline",
                "filename": "nope",
                "code": "x = 1\n",
            })
            self.assertEqual(r2.status_code, 400)


class CoerceSettingValueTests(unittest.TestCase):
    def test_number_coercion(self):
        self.assertEqual(_coerce_setting_value("7", {"type": "number"}), 7.0)
        self.assertEqual(_coerce_setting_value(7, {"type": "number"}), 7)
        self.assertEqual(_coerce_setting_value(7.5, {"type": "number"}), 7.5)
        with self.assertRaises(ValueError):
            _coerce_setting_value("abc", {"type": "number"})
        with self.assertRaises(ValueError):
            _coerce_setting_value(True, {"type": "number"})
        with self.assertRaises(ValueError):
            _coerce_setting_value(None, {"type": "number"})

    def test_text_coercion(self):
        self.assertEqual(_coerce_setting_value(123, {"type": "text"}), "123")

    def test_toggle_coercion(self):
        for truthy in (True, "true", "1", "yes", "on"):
            self.assertTrue(_coerce_setting_value(truthy, {"type": "toggle"}))
        for falsy in (False, "false", "0", "no", "off", ""):
            self.assertFalse(_coerce_setting_value(falsy, {"type": "toggle"}))

    def test_unknown_types_pass_through(self):
        self.assertEqual(_coerce_setting_value({"a": 1}, {}), {"a": 1})
        self.assertEqual(_coerce_setting_value("x", None), "x")


class _FakeRequest:
    @staticmethod
    def get_json(data):
        fake = mock.Mock()
        fake.get_json.return_value = data
        return fake


class iSpyAddonLoadingTests(unittest.TestCase):
    def _build_config(self):
        cfg = iSpyConfig()
        cfg.config["app_mode"] = False
        cfg.config["plugins"] = {
            "trackers": {"example_tracker": {"count_start": 5}},
            "utilities": {"health_reporter": {"stale_threshold": 2.0}},
            "frame_processors": {"example_frame_processor": {}},
        }
        return cfg

    def test_ispy_instantiates_all_enabled_addons_with_their_settings(self):
        from iSpy.iSpy import iSpy
        cfg = self._build_config()
        ispy = iSpy(cameras=[], config=cfg)
        try:
            self.assertIn("example_tracker", ispy.trackers)
            self.assertIn("health_reporter", ispy.utilities)
            self.assertIn("example_frame_processor", ispy.frame_processors)

            tracker = ispy.trackers["example_tracker"]
            self.assertEqual(tracker.count, 5)  # own settings applied

            reporter = ispy.utilities["health_reporter"]
            self.assertEqual(reporter._stale_threshold, 2.0)

            fp = ispy.frame_processors["example_frame_processor"]
            self.assertEqual(fp.process(123), 0)
        finally:
            ispy._stop_all_plugins()

    def test_ispy_skips_unknown_addons_gracefully(self):
        from iSpy.iSpy import iSpy
        cfg = iSpyConfig()
        cfg.config["app_mode"] = False
        cfg.config["plugins"] = {
            "trackers": {"not_a_real_tracker": {}},
            "utilities": {}, "frame_processors": {},
        }
        ispy = iSpy(cameras=[], config=cfg)
        try:
            self.assertEqual(ispy.trackers, {})
        finally:
            ispy._stop_all_plugins()

    def test_ispy_full_pipeline_builtin_addons(self):
        """all real builtin add-ons load through iSpy with defaults"""
        import tempfile
        from iSpy.iSpy import iSpy
        cfg = iSpyConfig()
        cfg.config["app_mode"] = False
        cfg.config["plugins"] = {
            "trackers": {
                "object_tracker": {},
                "path_planner": {},
            },
            "utilities": {
                "health_reporter": {},
            },
            "frame_processors": {},
        }
        ispy = iSpy(cameras=[], config=cfg)
        try:
            self.assertEqual(
                ispy.trackers["object_tracker"].distance_threshold, 0.5)
            self.assertEqual(ispy.trackers["path_planner"].epsilon, 0.3)
            self.assertEqual(
                ispy.utilities["health_reporter"]._stale_threshold, 1.0)
        finally:
            ispy._stop_all_plugins()


if __name__ == "__main__":
    unittest.main()