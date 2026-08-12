"""Regression tests for the settings page field list.

The moved global keys (distance_threshold, stale_threshold, dbscan.*,
record_mode, record_dir, use_network_tables, network_tables_ip) now live as
add-on settings on the Add-ons page. If they reappear as data-key fields in
settings.html, saving the form would resurrect them as top-level config keys.
"""

import re
import unittest
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[1] / "iSpy" / "web" / "templates"
SETTINGS_HTML = (TEMPLATES / "settings.html").read_text()

REMOVED_KEYS = [
    "distance_threshold",
    "stale_threshold",
    "dbscan.epsilon",
    "dbscan.min_samples",
    "record_mode",
    "record_dir",
    "use_network_tables",
    "network_tables_ip",
]


def data_keys(html: str):
    return [
        key
        for key in re.findall(r'data-key="([^"]+)"', html)
        if not key.startswith("${")
    ]


class TestSettingsPageKeys(unittest.TestCase):
    def test_no_removed_keys_as_data_key(self):
        keys = data_keys(SETTINGS_HTML)
        for key in REMOVED_KEYS:
            self.assertNotIn(key, keys, f"removed global key {key!r} still a settings field")

    def test_all_settings_fields_are_still_valid_global_keys(self):
        valid = {"optimize", "unit", "frame_sync", "metrics", "debug_mode", "log_level"}
        self.assertTrue(set(data_keys(SETTINGS_HTML)) <= valid, data_keys(SETTINGS_HTML))

    def test_bool_key_set_no_longer_contains_removed_keys(self):
        for key in ("record_mode", "use_network_tables"):
            self.assertNotIn(f'"{key}"', SETTINGS_HTML)

    def test_settings_page_points_to_addons_page(self):
        self.assertIn("/addons", SETTINGS_HTML)


if __name__ == "__main__":
    unittest.main()