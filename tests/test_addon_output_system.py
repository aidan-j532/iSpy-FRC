"""Addon output system: utility output_key -> frame_data["addon_data"] ->
NetworkTables publishing (auto type detection, JSON fallback, selectable
sources). See DOCS.d/02-plugins.md."""

import json
import unittest
from unittest import mock

from iSpy.config.iSpyConfig import iSpyAddonConfig, iSpyConfig
from iSpy.plugins.bases import (
    UtilityBase,
    validate_output_key,
    find_duplicate_output_keys,
)
from iSpy.plugins._loader import load_plugins
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "iSpy" / "plugins"

BARE_CONTEXT = {
    "config": iSpyAddonConfig({}),
    "global_config": None,
    "cameras": [],
    "flask_app": None,
    "vision_instance": None,
}


def addon_context(cls, settings=None):
    ctx = dict(BARE_CONTEXT)
    ctx["config"] = iSpyAddonConfig(settings or {}, defaults=cls.default_settings())
    return ctx


def _temp_recordings_dir(self):
    import shutil
    import tempfile
    path = tempfile.mkdtemp(prefix="ispy_rollback_")
    self.addCleanup(shutil.rmtree, path, ignore_errors=True)
    return path


class _OutputUtility(UtilityBase):
    plugin_name = "_test_output_utility"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "output_key": {
                "type": "text",
                "label": "Output Key",
                "default": "test_output",
            },
        }


class ValidateOutputKeyTests(unittest.TestCase):
    def test_valid_key_normalizes_whitespace(self):
        self.assertEqual(validate_output_key("  robot_speed \n"), ("robot_speed", None))

    def test_empty_rejected(self):
        key, err = validate_output_key("")
        self.assertIsNone(key)
        self.assertIn("empty", err)

    def test_whitespace_only_rejected(self):
        key, err = validate_output_key("   ")
        self.assertIsNone(key)
        self.assertIn("empty", err)

    def test_non_string_rejected(self):
        key, err = validate_output_key(42)
        self.assertIsNone(key)
        self.assertIsNotNone(err)

    def test_none_means_unset(self):
        key, err = validate_output_key(None)
        self.assertIsNone(key)
        self.assertIsNone(err)

    def test_dots_rejected(self):
        # dots are reserved for nested source paths (addon_data.<key>)
        key, err = validate_output_key("a.b")
        self.assertIsNone(key)
        self.assertIn("dots", err)


class UtilityPublishOutputTests(unittest.TestCase):
    def test_value_lands_under_addon_data(self):
        inst = _OutputUtility(addon_context(_OutputUtility))
        frame_data = {"fps": 30.0}
        written = inst.publish_output(frame_data, 123)
        self.assertTrue(written)
        self.assertEqual(frame_data["addon_data"]["test_output"], 123)

    def test_core_top_level_keys_are_never_overwritten(self):
        # even a hostile/mistaken output_key like "fps" stays namespaced
        inst = _OutputUtility(addon_context(_OutputUtility, {"output_key": "fps"}))
        frame_data = {"fps": 30.0, "detections": [1]}
        inst.publish_output(frame_data, 9999)
        self.assertEqual(frame_data["fps"], 30.0)
        self.assertEqual(frame_data["detections"], [1])
        self.assertEqual(frame_data["addon_data"]["fps"], 9999)

    def test_explicit_key_overrides_setting(self):
        inst = _OutputUtility(addon_context(_OutputUtility))
        frame_data = {}
        inst.publish_output(frame_data, "x", output_key="explicit_key")
        self.assertEqual(frame_data["addon_data"], {"explicit_key": "x"})

    def test_no_output_key_is_a_safe_noop(self):
        class Quiet(UtilityBase):
            plugin_name = "_quiet"

        inst = Quiet(addon_context(Quiet))
        self.assertIsNone(inst.declared_output_key())
        frame_data = {"detections": []}
        self.assertFalse(inst.publish_output(frame_data, 1))
        self.assertNotIn("addon_data", frame_data)

    def test_blank_configured_key_is_ignored_at_runtime(self):
        inst = _OutputUtility(
            addon_context(_OutputUtility, {"output_key": "   "}))
        frame_data = {}
        self.assertFalse(inst.publish_output(frame_data, 1))
        self.assertNotIn("addon_data", frame_data)


class DuplicateOutputKeyTests(unittest.TestCase):
    def test_duplicates_detected_with_both_names(self):
        a = _OutputUtility(addon_context(_OutputUtility, {"output_key": "shared"}))
        b = _OutputUtility(addon_context(_OutputUtility, {"output_key": "shared"}))
        c = _OutputUtility(addon_context(_OutputUtility, {"output_key": "unique"}))

        dupes = find_duplicate_output_keys({"util_a": a, "util_b": b, "util_c": c})
        self.assertEqual(dupes, {"shared": ["util_a", "util_b"]})

    def test_no_duplicates_when_keys_differ(self):
        a = _OutputUtility(addon_context(_OutputUtility))
        dupes = find_duplicate_output_keys({"only": a})
        self.assertEqual(dupes, {})

    def test_vision_instance_warns_at_boot(self):
        from iSpy.iSpy import iSpy
        cfg = iSpyConfig()
        cfg.config["app_mode"] = False
        cfg.config["plugins"] = {
            "trackers": {},
            "utilities": {
                "example_utility": {"output_key": "clash"},
                "rollback": {"output_key": "clash",
                             "data_dir": _temp_recordings_dir(self)},
            },
            "frame_processors": {},
        }
        with self.assertLogs("iSpy.iSpy", level="WARNING") as captured:
            vision = iSpy(cameras=[], config=cfg)
        try:
            clash_logs = [m for m in captured.output if "'clash'" in m]
            self.assertTrue(clash_logs, f"no collision warning in {captured.output}")
            joined = " ".join(clash_logs)
            self.assertIn("example_utility", joined)
            self.assertIn("rollback", joined)
        finally:
            vision._stop_all_plugins()


class NetworkHandlerSourceResolutionTests(unittest.TestCase):
    _fake_inst = None

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

    def test_existing_top_level_source_resolves(self):
        handler = self._handler()
        self.assertEqual(
            handler._resolve_source("fps", {"fps": 42.5}), 42.5)

    def test_special_case_detections_still_works(self):
        handler = self._handler()
        dets = [object()]
        self.assertIs(
            handler._resolve_source("detections", {"detections": dets}), dets)
        self.assertEqual(handler._resolve_source("detections", {}), [])

    def test_addon_data_dotted_source_resolves(self):
        handler = self._handler()
        frame_data = {"addon_data": {"robot_speed": 3.4}}
        self.assertEqual(
            handler._resolve_source("addon_data.robot_speed", frame_data), 3.4)

    def test_missing_sources_fail_safely_as_none(self):
        handler = self._handler()
        self.assertIsNone(handler._resolve_source("addon_data.nope", {}))
        self.assertIsNone(handler._resolve_source("totally_missing", {}))
        # missing addon_data namespace entirely must not raise either
        self.assertIsNone(handler._resolve_source("addon_data.anything", {"fps": 1}))

    def _auto_handler(self, key):
        return self._handler({"publish": [
            {"name": key, "data_type": "auto",
             "source": f"addon_data.{key}", "nt_topic": key},
        ]})

    def test_auto_publishes_bool_as_boolean_not_number(self):
        handler = self._auto_handler("flag")
        handler.update({"addon_data": {"flag": True}})
        pub = handler._subscribers["pub/VisionData/flag"]
        pub.set.assert_called_with(True)
        handler.inst.getBooleanTopic.assert_called_with("flag")

    def test_auto_publishes_false_correctly(self):
        handler = self._auto_handler("flag")
        handler.update({"addon_data": {"flag": False}})
        pub = handler._subscribers["pub/VisionData/flag"]
        pub.set.assert_called_with(False)

    def test_auto_publishes_int_and_float_as_number(self):
        handler = self._auto_handler("count")
        handler.update({"addon_data": {"count": 42}})
        pub = handler._subscribers["pub/VisionData/count"]
        pub.set.assert_called_with(42.0)
        handler.inst.getDoubleTopic.assert_called_with("count")

        handler2 = self._auto_handler("ratio")
        handler2.update({"addon_data": {"ratio": 3.14}})
        pub2 = handler2._subscribers["pub/VisionData/ratio"]
        pub2.set.assert_called_with(3.14)

    def test_auto_publishes_str_as_string(self):
        handler = self._auto_handler("status")
        handler.update({"addon_data": {"status": "hello"}})
        pub = handler._subscribers["pub/VisionData/status"]
        pub.set.assert_called_with("hello")
        handler.inst.getStringTopic.assert_called_with("status")

    def test_structured_fallback_serializes_to_json_string(self):
        handler = self._auto_handler("blob")
        payload = {"x": 1, "items": [1, 2, 3]}
        handler.update({"addon_data": {"blob": payload}})
        pub = handler._subscribers["pub/VisionData/blob"]
        published = pub.set.call_args[0][0]
        self.assertIsInstance(published, str)
        self.assertEqual(json.loads(published), payload)
        handler.inst.getStringTopic.assert_called_with("blob")

    def test_list_fallback_serializes_to_json_string(self):
        handler = self._auto_handler("points")
        handler.update({"addon_data": {"points": [1, 2, 3]}})
        pub = handler._subscribers["pub/VisionData/points"]
        self.assertEqual(json.loads(pub.set.call_args[0][0]), [1, 2, 3])

    def test_unserializable_fallback_fails_safely(self):
        handler = self._auto_handler("bad")

        class Unserializable:
            pass

        # default=str makes most things serializable; simulate total failure
        with mock.patch(
            "iSpy.plugins.utilities.BuiltIn.NetworkHandler.json.dumps",
            side_effect=TypeError("boom"),
        ):
            handler.update({"addon_data": {"bad": Unserializable()}})  # must not raise

    def test_manual_types_still_publish(self):
        # existing manually configured behavior is untouched
        handler = self._handler({"publish": [
            {"name": "fps", "data_type": "number", "source": "fps", "nt_topic": "fps"},
        ]})
        handler.update({"fps": 60.5, "cameras": []})
        pub = handler._subscribers["pub/VisionData/fps"]
        pub.set.assert_called_with(60.5)

    def test_struct_array_publishing_preserved(self):
        from iSpy.vision.Object import Object
        handler = self._handler()
        det = Object(1.0, 2.0, 3.0)
        handler.update({"detections": [det], "fps": 10,
                        "detection_count": 1, "camera_lag_s": 0.0, "cameras": []})
        self._fake_inst.getStructArrayTopic.assert_called()


class PublishSourcesApiTests(unittest.TestCase):
    def _module(self, plugins=None):
        import flask
        from iSpy.web.Backend.PluginStatus import PluginStatusModule
        cfg = iSpyConfig()
        cfg.config["app_mode"] = False
        if plugins is not None:
            cfg.config["plugins"] = plugins
        vision = mock.Mock()
        vision.trackers = {}
        vision.utilities = {}
        vision.frame_processors = {}
        mod = PluginStatusModule({"config": cfg, "vision_instance": vision})
        return mod, cfg, flask.Flask(__name__).app_context()

    def test_core_sources_always_present(self):
        mod, _cfg, ctx = self._module()
        with ctx:
            payload = mod._publish_sources().get_json()
        sources = [s["source"] for s in payload["sources"]]
        for expected in ("fps", "detection_count", "camera_lag_s", "detections"):
            self.assertIn(expected, sources)

    def test_enabled_utility_outputs_declared_even_before_first_frame(self):
        mod, cfg, ctx = self._module({
            "trackers": {}, "utilities": {"example_utility": {}},
            "frame_processors": {},
        })
        with ctx:
            payload = mod._publish_sources().get_json()
        entry = next(s for s in payload["sources"]
                     if s["source"] == "addon_data.example_output")
        self.assertEqual(entry["utility"], "example_utility")

    def test_disabled_utilities_excluded(self):
        mod, _cfg, ctx = self._module({
            "trackers": {}, "utilities": {}, "frame_processors": {},
        })
        with ctx:
            payload = mod._publish_sources().get_json()
        self.assertFalse(
            any(s["source"].startswith("addon_data.") for s in payload["sources"]))

    def test_customized_output_key_used_from_config(self):
        mod, cfg, ctx = self._module({
            "trackers": {},
            "utilities": {"example_utility": {"output_key": "my_counter"}},
            "frame_processors": {},
        })
        with ctx:
            payload = mod._publish_sources().get_json()
        sources = [s["source"] for s in payload["sources"]]
        self.assertIn("addon_data.my_counter", sources)
        self.assertNotIn("addon_data.example_output", sources)

    def test_duplicate_declared_outputs_flagged(self):
        mod, cfg, ctx = self._module({
            "trackers": {},
            "utilities": {
                "example_utility": {"output_key": "clash"},
                "rollback": {"output_key": "clash",
                             "data_dir": _temp_recordings_dir(self)},
            },
            "frame_processors": {},
        })
        with mock.patch(
            "iSpy.web.Backend.PluginStatus.load_plugins",
            return_value={
                "example_utility": _OutputUtility,
                "rollback": _OutputUtility,
            },
        ):
            with ctx:
                payload = mod._publish_sources().get_json()
        clashes = [s for s in payload["sources"] if s["source"] == "addon_data.clash"]
        self.assertEqual(len(clashes), 2)
        self.assertTrue(all(s.get("duplicate") for s in clashes))

    def test_route_registered(self):
        import flask
        from iSpy.web.Backend.PluginStatus import PluginStatusModule
        app = flask.Flask(__name__)
        mod = PluginStatusModule({"config": iSpyConfig()})
        mod.register_routes(app)
        rules = [r.rule for r in app.url_map.iter_rules()]
        self.assertIn("/api/plugins/publish-sources", rules)


class SaveSettingsOutputKeyValidationTests(unittest.TestCase):
    def _save(self, settings):
        import flask
        from unittest.mock import Mock
        from iSpy.web.Backend.PluginStatus import PluginStatusModule
        cfg = iSpyConfig()
        cfg.config["app_mode"] = False
        cfg.config["plugins"] = {
            "trackers": {}, "utilities": {"example_utility": {}},
            "frame_processors": {},
        }
        mod = PluginStatusModule({"config": cfg, "vision_instance": Mock()})
        req = Mock()
        req.get_json.return_value = {
            "name": "example_utility", "type": "utility", "settings": settings,
        }
        with flask.Flask(__name__).app_context():
            with mock.patch("iSpy.web.Backend.PluginStatus.request", req):
                resp = mod._save_settings()
        return resp, cfg

    def test_valid_key_saved_normalized(self):
        resp, cfg = self._save({"output_key": "  spaced_key "})
        self.assertEqual(resp.status_code, 200)
        saved = cfg.get_addon_settings("utilities", "example_utility")
        self.assertEqual(saved["output_key"], "spaced_key")

    def test_empty_key_rejected_400(self):
        resp, cfg = self._save({"output_key": ""})
        self.assertEqual(resp[1], 400)
        self.assertNotIn("output_key",
                         cfg.get_addon_settings("utilities", "example_utility"))

    def test_whitespace_only_key_rejected_400(self):
        resp, _cfg = self._save({"output_key": "   \n\t"})
        self.assertEqual(resp[1], 400)

    def test_dotted_key_rejected_400(self):
        resp, _cfg = self._save({"output_key": "a.b"})
        self.assertEqual(resp[1], 400)


if __name__ == "__main__":
    unittest.main()


