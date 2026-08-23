import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from iSpy.boot.boot import cleanup_missing_cameras
from iSpy.config.iSpyConfig import iSpyConfig
from iSpy.web.Backend import save_store

# windows-style ids - same physical cam, two different usb ports
DEV_PORT_1 = "##?#USB#VID_046D&PID_082D&MI_00#7&25a1ca30&0&0000"
DEV_PORT_2 = "##?#USB#VID_046D&PID_082D&MI_00#6&1b3d9f21&1&0002"
DEV_OTHER = "##?#USB#VID_1BCF&PID_2283&MI_00#8&2c4e0a11&0&0003"


def _cam(name, source=0, device_id=None, **extra):
    cfg = {
        "name": name,
        "source": source,
        "pipeline": {"name": "april_tag", "settings": {"tag_size_inches": 6.5}},
    }
    cfg.update(extra)
    if device_id:
        cfg["device_id"] = device_id
    return cfg


def _dev(path, device_id=None):
    d = {"path": path}
    if device_id:
        d["device_id"] = device_id
    return d


@contextmanager
def _probed(devices):
    with mock.patch(
        "iSpy.web.modules.cameras.CamerasModule._probe_devices",
        return_value=devices,
    ):
        yield


@contextmanager
def _isolated_save_dir():
    with tempfile.TemporaryDirectory() as tmp:
        original = save_store._SAVE_DIR
        save_store._SAVE_DIR = Path(tmp)
        try:
            yield Path(tmp)
        finally:
            save_store._SAVE_DIR = original


class BootCameraCleanupTests(unittest.TestCase):
    def _make_config(self, tmp, cams):
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps({"camera_configs": cams}))
        return iSpyConfig(str(path), create=False)

    def _saved_keys(self, config):
        data = json.loads(Path(config.file_path).read_text())
        return set(data["camera_configs"])

    def test_missing_device_is_retired_with_full_settings_saved(self):
        with tempfile.TemporaryDirectory() as tmp, \
                _isolated_save_dir() as save_dir, \
                _probed([_dev("1", DEV_OTHER)]):
            cfg = self._make_config(tmp, {
                "gone_cam": _cam("gone_cam", source=7, device_id=DEV_PORT_1,
                                 yaw=12, height=2.5,
                                 calibration={"fov": 68, "distance": 1.5}),
                "live_cam": _cam("live_cam", source=1, device_id=DEV_OTHER),
            })
            cleanup_missing_cameras(cfg)

            self.assertEqual(self._saved_keys(cfg), {"live_cam"})
            data = json.loads(Path(cfg.file_path).read_text())
            self.assertIn("live_cam", data["camera_configs"])
            # nothing else was configured, so nothing new should be invented
            self.assertNotIn("gone_cam", data["camera_configs"])

            profiles = save_store.read("camera_profiles", {})
            saved = profiles.get(DEV_PORT_1)
            self.assertIsNotNone(saved)
            # full entry stashed so re-creating restores everything
            self.assertEqual(saved["name"], "gone_cam")
            self.assertEqual(saved["yaw"], 12)
            self.assertEqual(saved["height"], 2.5)
            self.assertEqual(saved["calibration"]["fov"], 68)
            self.assertEqual(saved["pipeline"]["settings"]["tag_size_inches"], 6.5)

    def test_present_device_and_port_moved_device_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp, \
                _isolated_save_dir(), \
                _probed([
                    _dev("0", DEV_PORT_2),   # same vid/pid, different port
                    _dev("1", DEV_OTHER),
                ]):
            cfg = self._make_config(tmp, {
                "moved_cam": _cam("moved_cam", source="0", device_id=DEV_PORT_1),
                "other_cam": _cam("other_cam", source="1", device_id=DEV_OTHER),
            })
            cleanup_missing_cameras(cfg)
            self.assertEqual(self._saved_keys(cfg), {"moved_cam", "other_cam"})
            self.assertEqual(save_store.read("camera_profiles", {}), {})

    def test_url_source_never_retired(self):
        with tempfile.TemporaryDirectory() as tmp, \
                _isolated_save_dir(), \
                _probed([_dev("0", DEV_PORT_1)]):
            cfg = self._make_config(tmp, {
                "stream_cam": _cam("stream_cam", source="rtsp://10.0.0.2/video"),
                "local_cam": _cam("local_cam", source="0", device_id=DEV_PORT_1),
            })
            cleanup_missing_cameras(cfg)
            self.assertEqual(self._saved_keys(cfg), {"stream_cam", "local_cam"})

    def test_image_source_never_retired(self):
        with tempfile.TemporaryDirectory() as tmp, \
                _isolated_save_dir(), \
                _probed([_dev("0", DEV_PORT_1)]):
            cfg = self._make_config(tmp, {
                "img_cam": _cam("img_cam", source="clips/test.png"),
                "local_cam": _cam("local_cam", source="0", device_id=DEV_PORT_1),
            })
            cleanup_missing_cameras(cfg)
            self.assertEqual(self._saved_keys(cfg), {"img_cam", "local_cam"})

    def test_dead_path_like_source_is_retired(self):
        with tempfile.TemporaryDirectory() as tmp, \
                _isolated_save_dir(), \
                _probed([_dev("0", DEV_PORT_1)]):
            dead = str(Path(tmp) / "does_not_exist")
            live = Path(tmp) / "real_node"
            live.write_text("")
            cfg = self._make_config(tmp, {
                "dead_cam": _cam("dead_cam", source=dead),
                "live_cam": _cam("live_cam", source=str(live)),
            })
            cleanup_missing_cameras(cfg)
            self.assertEqual(self._saved_keys(cfg), {"live_cam"})

    def test_bare_index_without_device_id_is_kept(self):
        # index order shifts when cams unplug - no trustworthy signal, keep it
        with tempfile.TemporaryDirectory() as tmp, \
                _isolated_save_dir(), \
                _probed([_dev("0")]):
            cfg = self._make_config(tmp, {
                "default_cam": _cam("default_cam", source=0),
                "maybe_gone": _cam("maybe_gone", source=3),
                "known": _cam("known", source="0"),
            })
            cleanup_missing_cameras(cfg)
            self.assertEqual(
                self._saved_keys(cfg), {"default_cam", "maybe_gone", "known"},
            )

    def test_probe_failure_skips_cleanup_entirely(self):
        with tempfile.TemporaryDirectory() as tmp, \
                _isolated_save_dir(), \
                mock.patch(
                    "iSpy.web.modules.cameras.CamerasModule._probe_devices",
                    side_effect=RuntimeError("v4l2 exploded"),
                ):
            cfg = self._make_config(tmp, {
                "a": _cam("a", source=0),
                "b": _cam("b", source=1),
            })
            cleanup_missing_cameras(cfg)
            self.assertEqual(self._saved_keys(cfg), {"a", "b"})

    def test_no_devices_detected_skips_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp, \
                _isolated_save_dir(), \
                _probed([]):
            cfg = self._make_config(tmp, {
                "a": _cam("a", source=0, device_id=DEV_PORT_1),
                "b": _cam("b", source=1, device_id=DEV_OTHER),
            })
            cleanup_missing_cameras(cfg)
            self.assertEqual(self._saved_keys(cfg), {"a", "b"})

    def test_all_missing_keeps_one_so_boot_survives(self):
        with tempfile.TemporaryDirectory() as tmp, \
                _isolated_save_dir() as save_dir, \
                _probed([_dev("0", DEV_OTHER)]):
            cfg = self._make_config(tmp, {
                "first": _cam("first", source=0, device_id=DEV_PORT_1),
                "second": _cam("second", source=1, device_id="##?#USB#VID_AAAA&PID_BBBB&MI_00#1&x"),
            })
            cleanup_missing_cameras(cfg)

            keys = self._saved_keys(cfg)
            self.assertEqual(keys, {"first"})
            data = json.loads(Path(cfg.file_path).read_text())
            self.assertIn("first", data["camera_configs"])

            profiles = save_store.read("camera_profiles", {})
            self.assertIn("second", [p["name"] for p in profiles.values()
                                     if p["device_id"] == "##?#USB#VID_AAAA&PID_BBBB&MI_00#1&x"])

    def test_single_camera_config_is_untouched_without_probing(self):
        with tempfile.TemporaryDirectory() as tmp, \
                _isolated_save_dir(), \
                mock.patch(
                    "iSpy.web.modules.cameras.CamerasModule._probe_devices",
                    side_effect=AssertionError("must not probe for a single cam"),
                ):
            cfg = self._make_config(tmp, {
                "only": _cam("only", source=0, device_id=DEV_PORT_1),
            })
            cleanup_missing_cameras(cfg)
            self.assertEqual(self._saved_keys(cfg), {"only"})


if __name__ == "__main__":
    unittest.main()
