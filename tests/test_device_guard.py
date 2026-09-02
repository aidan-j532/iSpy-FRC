import os
import signal
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from iSpy.vision.Cameras import _device_guard as dg


class DeviceGuardPredicateTests(unittest.TestCase):
    def test_is_ispy_process_matches_our_repo_root(self):
        self.assertTrue(dg._is_ispy_process("python3 " + dg._REPO_ROOT + "/iSpy/boot/boot.py"))
        self.assertTrue(dg._is_ispy_process("python3 /opt/" + "ispy" + "/run.py"))

    def test_is_ispy_process_rejects_stray_grabbber(self):
        self.assertFalse(dg._is_ispy_process("python3 /usr/bin/motion"))

    def test_devnode_normalizes_source(self):
        self.assertEqual(dg._devnode("/dev/video99"), Path("/dev/video99"))
        self.assertEqual(dg._devnode(0), Path("/dev/video0"))
        self.assertEqual(dg._devnode(("v4l2", "/dev/video3")), Path("/dev/video3"))
        self.assertIsNone(dg._devnode("rtsp://10.0.0.2/video"))
        self.assertIsNone(dg._devnode("clips/a.png"))

    def test_own_pids_contains_current_process(self):
        self.assertIn(os.getpid(), dg._own_pids())

    def test_free_camera_non_linux_is_noop(self):
        with mock.patch.object(dg, "_is_linux", return_value=False):
            with mock.patch.object(dg, "_kill_holder") as kill:
                self.assertEqual(dg.free_camera_device("/dev/video0"), [])
                kill.assert_not_called()

    def test_free_camera_non_devnode_is_noop(self):
        with mock.patch.object(dg, "_is_linux", return_value=True):
            with mock.patch.object(dg, "_kill_holder") as kill:
                self.assertEqual(dg.free_camera_device("rtsp://x/video"), [])
                kill.assert_not_called()


class DeviceGuardKillTests(unittest.TestCase):
    @staticmethod
    def _char_dev():
        m = mock.MagicMock()
        m.is_char_device.return_value = True
        return m

    def test_frees_stray_holder_but_never_ispy_or_self(self):
        with mock.patch.object(dg, "_is_linux", return_value=True), \
                mock.patch.object(dg, "_devnode") as dn, \
                mock.patch.object(dg, "shutil") as sh, \
                mock.patch.object(dg, "_holders_via_fuser",
                                  return_value=[111, 222]), \
                mock.patch.object(dg, "_own_pids", return_value={222}), \
                mock.patch.object(dg, "_process_cmdline",
                                  side_effect={
                                      111: "/usr/bin/motion",
                                      222: "python3 /opt/ispy/run.py",
                                  }.get), \
                mock.patch.object(dg, "_kill_holder") as kill:
            sh.which.return_value = "/usr/bin/fuser"
            dn.return_value = self._char_dev()
            self.assertEqual(dg.free_camera_device("/dev/video0"), [111])
            kill.assert_called_once_with(111)

    def test_kill_holder_terms_then_kills_when_stubborn(self):
        sigkill = getattr(signal, "SIGKILL", 9)
        fake_signal = SimpleNamespace(SIGTERM=signal.SIGTERM, SIGKILL=sigkill)
        with mock.patch.object(dg, "_is_ispy_process", return_value=False), \
                mock.patch.object(dg, "_own_pids", return_value=set()), \
                mock.patch.object(dg, "_process_cmdline", return_value="/usr/bin/motion"), \
                mock.patch.object(dg, "_killable",
                                  side_effect=[True, True, False]) as alive, \
                mock.patch.object(dg, "signal", fake_signal), \
                mock.patch.object(dg, "time") as tm, \
                mock.patch.object(dg.os, "kill") as okill:
            tm.monotonic.side_effect = [0.0, 0.5, 2.5]
            dg._kill_holder(999)
            okill.assert_any_call(999, signal.SIGTERM)
            okill.assert_any_call(999, sigkill)


if __name__ == "__main__":
    unittest.main()
