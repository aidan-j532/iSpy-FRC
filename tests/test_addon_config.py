"""add-on (plugin) config restructure tests.

add-ons are dicts keyed by enabled add-on; presence == enabled, no flag.
settings that used to live at the top level (dbscan, distance_threshold, ...)
now live in their add-on; legacy list-format configs migrate automatically.
"""

import json
import tempfile
import unittest
from pathlib import Path

from iSpy.config.iSpyConfig import (
    iSpyConfig,
    iSpyAddonConfig,
)


def _write_config(data: dict) -> str:
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / "config.json"
    path.write_text(json.dumps(data))
    return str(path)


class AddonDefaultConfigTests(unittest.TestCase):
    def test_default_plugins_are_dicts_not_lists(self):
        cfg = iSpyConfig()
        for addon_type in ("trackers", "utilities", "frame_processors"):
            self.assertIsInstance(
                cfg.config["plugins"][addon_type], dict,
                f"plugins.{addon_type} must be a dict",
            )
            self.assertEqual(cfg.config["plugins"][addon_type], {})

    def test_default_config_has_no_legacy_global_keys(self):
        cfg = iSpyConfig()
        for key in (
            "dbscan", "distance_threshold", "stale_threshold", "record_mode",
            "record_dir", "use_network_tables", "network_tables_ip",
        ):
            self.assertNotIn(key, cfg.config, f"legacy key '{key}' still present")

    def test_default_config_still_has_shared_global_keys(self):
        cfg = iSpyConfig()
        for key in ("num_gpus", "device", "unit", "max_fps", "app_mode",
                    "camera_configs", "log_level", "log_file"):
            self.assertIn(key, cfg.config)


class AddonMigrationTests(unittest.TestCase):
    def _legacy_config(self, extra=None):
        data = {
            "plugins": {
                "trackers": ["object_tracker", "path_planner"],
                "utilities": ["video_recorder", "network_table_handler"],
                "frame_processors": [],
            },
            "dbscan": {"epsilon": 0.42, "min_samples": 7},
            "distance_threshold": 0.88,
            "stale_threshold": 2.25,
            "record_mode": True,
            "record_dir": "CustomDir",
            "use_network_tables": True,
            "network_tables_ip": "10.6.6.6",
        }
        if extra:
            data.update(extra)
        return data

    def _load(self, data: dict) -> iSpyConfig:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(data))
            return iSpyConfig(str(path), create=False)

    def test_legacy_lists_become_dicts(self):
        data = {"plugins": {
            "trackers": ["object_tracker", "path_planner"],
            "utilities": ["video_recorder", "network_table_handler"],
            "frame_processors": [],
        }}
        cfg = self._load(data)
        self.assertEqual(
            cfg.config["plugins"]["trackers"],
            {"object_tracker": {}, "path_planner": {}},
        )
        self.assertEqual(
            cfg.config["plugins"]["utilities"],
            {"video_recorder": {}, "network_table_handler": {}},
        )
        self.assertEqual(cfg.config["plugins"]["frame_processors"], {})

    def test_legacy_global_settings_fold_into_enabled_addons(self):
        cfg = self._load(self._legacy_config())
        self.assertEqual(
            cfg.get_addon_settings("trackers", "object_tracker"),
            {"distance_threshold": 0.88, "stale_threshold": 2.25},
        )
        self.assertEqual(
            cfg.get_addon_settings("trackers", "path_planner"),
            {"epsilon": 0.42, "min_samples": 7},
        )
        self.assertEqual(
            cfg.get_addon_settings("utilities", "network_table_handler"),
            {"network_tables_ip": "10.6.6.6"},
        )
        self.assertEqual(
            cfg.get_addon_settings("utilities", "video_recorder"),
            {"record_dir": "CustomDir"},
        )

    def test_legacy_enabled_flags_become_presence(self):
        # record_mode/use_network_tables are gone - the add-ons they enabled are just present
        cfg = self._load(self._legacy_config())
        self.assertTrue(cfg.is_addon_enabled("utilities", "video_recorder"))
        self.assertTrue(cfg.is_addon_enabled("utilities", "network_table_handler"))

    def test_legacy_global_keys_are_removed(self):
        cfg = self._load(self._legacy_config())
        for key in (
            "dbscan", "distance_threshold", "stale_threshold", "record_mode",
            "record_dir", "use_network_tables", "network_tables_ip",
        ):
            self.assertNotIn(key, cfg.config)

    def test_disabled_addons_get_no_migrated_settings(self):
        # settings must never enable an add-on - stays absent even with legacy values around
        data = self._legacy_config()
        data["plugins"]["trackers"] = ["path_planner"]  # object_tracker disabled
        cfg = self._load(data)
        self.assertFalse(cfg.is_addon_enabled("trackers", "object_tracker"))
        self.assertIsNone(cfg.get_addon_settings("trackers", "object_tracker"))

    def test_disabled_flags_do_not_enable_addons(self):
        # a False flag keeps the add-on disabled even if legacy top-level settings exist
        data = {
            "plugins": {"trackers": [], "utilities": [], "frame_processors": []},
            "use_network_tables": False,
            "network_tables_ip": "10.6.6.6",
            "record_mode": False,
            "record_dir": "CustomDir",
        }
        cfg = self._load(data)
        self.assertFalse(cfg.is_addon_enabled("utilities", "network_table_handler"))
        self.assertFalse(cfg.is_addon_enabled("utilities", "video_recorder"))

    def test_migration_is_idempotent(self):
        # loading twice mustnt duplicate or clobber anything
        data = self._legacy_config()
        cfg1 = self._load(data)
        cfg2 = self._load(json.loads(json.dumps(cfg1.config)))
        self.assertEqual(cfg1.config["plugins"], cfg2.config["plugins"])

    def test_new_dict_layout_loads_untouched(self):
        data = {
            "plugins": {
                "trackers": {"object_tracker": {"distance_threshold": 0.9}},
                "utilities": {"health_reporter": {"stale_threshold": 0.7}},
                "frame_processors": {},
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(data))
            cfg = iSpyConfig(str(path), create=False)
        self.assertEqual(
            cfg.get_addon_settings("trackers", "object_tracker"),
            {"distance_threshold": 0.9},
        )
        self.assertEqual(
            cfg.get_addon_settings("utilities", "health_reporter"),
            {"stale_threshold": 0.7},
        )

    def test_malformed_addon_values_do_not_crash(self):
        data = {
            "plugins": {
                "trackers": ["object_tracker", 3, None],
                "utilities": {"network_table_handler": "not-a-dict"},
                "frame_processors": 42,
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(data))
            cfg = iSpyConfig(str(path), create=False)
        self.assertIn("object_tracker", cfg.config["plugins"]["trackers"])
        self.assertNotIn(3, cfg.config["plugins"]["trackers"])
        self.assertNotIn(None, cfg.config["plugins"]["trackers"])
        # malformed value still counts as enabled (presence == enabled), just no settings
        self.assertTrue(cfg.is_addon_enabled("utilities", "network_table_handler"))
        self.assertEqual(
            cfg.get_addon_settings("utilities", "network_table_handler"), {}
        )
        self.assertEqual(cfg.config["plugins"]["frame_processors"], {})


class AddonConfigHelperTests(unittest.TestCase):
    def setUp(self):
        self.cfg = iSpyConfig()
        self.cfg.config["plugins"] = {
            "trackers": {"object_tracker": {"distance_threshold": 0.6}},
            "utilities": {},
            "frame_processors": {},
        }

    def test_enable_addon_adds_entry(self):
        self.cfg.enable_addon("utilities", "video_recorder", save=False)
        self.assertTrue(self.cfg.is_addon_enabled("utilities", "video_recorder"))
        self.assertEqual(
            self.cfg.get_addon_settings("utilities", "video_recorder"), {}
        )

    def test_enable_addon_keeps_existing_settings(self):
        self.cfg.enable_addon("utilities", "video_recorder", save=False)
        self.cfg.update_addon_settings("utilities", "video_recorder",
                                       {"record_dir": "Clips"}, save=False)
        self.cfg.enable_addon("utilities", "video_recorder", save=False)
        self.assertEqual(
            self.cfg.get_addon_settings("utilities", "video_recorder"),
            {"record_dir": "Clips"},
        )

    def test_enable_addon_with_settings(self):
        self.cfg.enable_addon(
            "utilities", "network_table_handler",
            settings={"network_tables_ip": "1.2.3.4"}, save=False,
        )
        self.assertEqual(
            self.cfg.get_addon_setting(
                "utilities", "network_table_handler", "network_tables_ip"),
            "1.2.3.4",
        )

    def test_disable_addon_removes_entry(self):
        self.cfg.disable_addon("trackers", "object_tracker", save=False)
        self.assertFalse(self.cfg.is_addon_enabled("trackers", "object_tracker"))
        self.assertIsNone(self.cfg.get_addon_settings("trackers", "object_tracker"))

    def test_set_addon_settings_requires_enabled(self):
        self.cfg.set_addon_settings("utilities", "nope", {"a": 1}, save=False)
        self.assertIsNone(self.cfg.get_addon_settings("utilities", "nope"))
        self.cfg.set_addon_settings("trackers", "object_tracker",
                                    {"distance_threshold": 1.0}, save=False)
        self.assertEqual(
            self.cfg.get_addon_settings("trackers", "object_tracker"),
            {"distance_threshold": 1.0},
        )

    def test_update_addon_settings_merges(self):
        self.cfg.update_addon_settings("trackers", "object_tracker",
                                       {"stale_threshold": 3.0}, save=False)
        self.assertEqual(
            self.cfg.get_addon_settings("trackers", "object_tracker"),
            {"distance_threshold": 0.6, "stale_threshold": 3.0},
        )

    def test_get_addon_setting_with_enabled_and_disabled(self):
        self.assertEqual(
            self.cfg.get_addon_setting(
                "trackers", "object_tracker", "distance_threshold", 0.5),
            0.6,
        )
        self.assertEqual(
            self.cfg.get_addon_setting(
                "trackers", "missing", "distance_threshold", 0.5),
            0.5,
        )

    def test_addon_entries_rejects_unknown_types(self):
        self.assertEqual(self.cfg.addon_entries("vision_pipelines"), {})
        self.assertEqual(self.cfg.addon_entries("nope"), {})

    def test_save_persists_dict_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            cfg = iSpyConfig(str(path), create=True)
            cfg.enable_addon("trackers", "object_tracker",
                             {"distance_threshold": 1.1})
            loaded = iSpyConfig(str(path), create=False)
            self.assertEqual(
                loaded.get_addon_settings("trackers", "object_tracker"),
                {"distance_threshold": 1.1},
            )

    def test_unknown_addon_types_are_noops(self):
        self.cfg.enable_addon("bogus", "x", save=False)
        self.cfg.disable_addon("bogus", "x", save=False)
        self.cfg.set_addon_settings("bogus", "x", {}, save=False)
        self.cfg.update_addon_settings("bogus", "x", {}, save=False)


class iSpyAddonConfigTests(unittest.TestCase):
    def test_empty_settings_apply_defaults(self):
        ac = iSpyAddonConfig({}, defaults={"distance_threshold": 0.5,
                                           "stale_threshold": 1.0})
        self.assertEqual(ac.get("distance_threshold"), 0.5)
        self.assertEqual(ac.get("missing", "fallback"), "fallback")

    def test_explicit_settings_win_over_defaults(self):
        ac = iSpyAddonConfig({"distance_threshold": 9.0},
                             defaults={"distance_threshold": 0.5})
        self.assertEqual(ac.get("distance_threshold"), 9.0)

    def test_raw_dict_wrapping_and_mutation(self):
        ac = iSpyAddonConfig({"a": 1})
        ac.set("b", 2)
        ac.setdefault("a", 99)  # existing key untouched
        ac.setdefault("c", 3)
        self.assertEqual(ac.to_dict(), {"a": 1, "b": 2, "c": 3})
        self.assertIn("a", ac)
        self.assertEqual(ac["a"], 1)
        self.assertEqual(ac.get_nested("nope", "deep", default="d"), "d")
        self.assertEqual(list(ac.keys()), ["a", "b", "c"])

    def test_non_dict_input_handled(self):
        ac = iSpyAddonConfig(None)
        self.assertEqual(ac.to_dict(), {})
        ac2 = iSpyAddonConfig("junk")
        self.assertEqual(ac2.to_dict(), {})


if __name__ == "__main__":
    unittest.main()