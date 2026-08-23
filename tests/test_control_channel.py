import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from iSpy.core import control_channel
from iSpy.core.control_channel import (
    ControlServer,
    SupervisorControlClient,
    send_command,
    state_file_path,
)


class _TmpCwd:
    """Chdir into a fresh temp dir; control-channel state stays hermetic."""

    def __init__(self):
        self._tmp = None
        self._old = None

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        return Path(self._tmp.name)

    def __exit__(self, *exc):
        os.chdir(self._old)
        self._tmp.cleanup()


class ControlServerTests(unittest.TestCase):
    def test_server_publishes_port_in_state_file(self):
        with _TmpCwd():
            server = ControlServer({"PAUSE": lambda: None})
            try:
                port = server.start()
                self.assertIsInstance(port, int)
                self.assertGreater(port, 0)
                state = json.loads(state_file_path().read_text())
                self.assertEqual(state.get("control_port"), port)
            finally:
                server.stop()
            state = json.loads(state_file_path().read_text())
            self.assertIsNone(state.get("control_port"))

    def test_send_command_roundtrip(self):
        with _TmpCwd():
            called = threading.Event()
            server = ControlServer({"PAUSE": called.set})
            try:
                server.start()
                ok, detail = send_command("PAUSE", server.port, timeout=2)
                self.assertTrue(ok, detail)
                self.assertTrue(called.wait(timeout=2))
            finally:
                server.stop()

    def test_unknown_command_rejected(self):
        with _TmpCwd():
            server = ControlServer({"PAUSE": lambda: None})
            try:
                server.start()
                ok, detail = send_command("EXPLODE", server.port, timeout=2)
                self.assertFalse(ok)
                self.assertIn("unknown command", detail)
            finally:
                server.stop()

    def test_send_to_dead_port_fails_cleanly(self):
        ok, detail = send_command("PAUSE", 1, timeout=0.5)  # nothing listens
        self.assertFalse(ok)

    def test_start_is_idempotent(self):
        with _TmpCwd():
            server = ControlServer({"PAUSE": lambda: None})
            try:
                first = server.start()
                second = server.start()
                self.assertEqual(first, second)
            finally:
                server.stop()


class SupervisorControlClientTests(unittest.TestCase):
    def test_no_port_reports_not_running(self):
        with _TmpCwd():
            client = SupervisorControlClient()
            ok, detail = client.send("PAUSE")
            self.assertFalse(ok)
            self.assertEqual(detail, "not running")

    def test_stale_port_invalidates_and_recovers(self):
        with _TmpCwd():
            client = SupervisorControlClient()
            # point the cache at a dead port without a server behind it
            state_file_path().parent.mkdir(parents=True, exist_ok=True)
            state_file_path().write_text(json.dumps({"control_port": 1}))
            ok, _ = client.send("RESUME")
            self.assertFalse(ok)
            # child republishes a real port -> next send must re-read file
            server = ControlServer({"RESUME": lambda: None})
            try:
                server.start()
                ok, detail = client.send("RESUME")
                self.assertTrue(ok, detail)
            finally:
                server.stop()

    def test_merge_write_preserves_other_keys(self):
        with _TmpCwd():
            control_channel.update_service_state(status="running", pid=42)
            control_channel.update_service_state(control_port=1234)
            state = json.loads(state_file_path().read_text())
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["pid"], 42)
            self.assertEqual(state["control_port"], 1234)


_CHILD_SCRIPT = """import sys, time, threading
sys.path.insert(0, {root!r})
from iSpy.core.control_channel import ControlServer

shutdown = threading.Event()
server = ControlServer({{
    "PAUSE": lambda: print("PAUSED", flush=True),
    "RESUME": lambda: print("RESUMED", flush=True),
    "SHUTDOWN": shutdown.set,
}})
server.start()
print("READY", flush=True)
while not shutdown.is_set():
    time.sleep(0.05)
print("BYE", flush=True)
"""


class VisionSupervisorIntegrationTests(unittest.TestCase):
    """End-to-end: supervisor drives pause/resume/stop through the channel."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_cwd = os.getcwd()
        os.chdir(self._tmp.name)
        script = Path(self._tmp.name) / "child.py"
        script.write_text(_CHILD_SCRIPT.format(root=str(REPO_ROOT)))
        from iSpy.boot.service_daemon import VisionSupervisor

        self.supervisor_cls = VisionSupervisor
        self.entry = str(script)

    def tearDown(self):
        os.chdir(self._old_cwd)
        # Windows can briefly hold the state file after process exit
        for attempt in range(5):
            try:
                self._tmp.cleanup()
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2)

    def _wait_for_control_port(self, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            path = state_file_path()
            if path.exists():
                try:
                    port = json.loads(path.read_text()).get("control_port")
                    if isinstance(port, int):
                        return port
                except (json.JSONDecodeError, OSError):
                    pass
            time.sleep(0.05)
        self.fail("child never published its control port")

    def test_pause_resume_stop_via_control_channel(self):
        sup = self.supervisor_cls(self.entry)
        r = sup.start()
        self.assertTrue(r["ok"])
        try:
            self._wait_for_control_port()

            r = sup.pause()
            self.assertTrue(r["ok"], r.get("error"))
            self.assertEqual(sup.status, "paused")

            r = sup.resume()
            self.assertTrue(r["ok"], r.get("error"))
            self.assertEqual(sup.status, "running")

            r = sup.stop(timeout=10)
            self.assertTrue(r["ok"])
            self.assertEqual(sup.status, "stopped")
            self.assertEqual(sup.proc.poll(), 0, "child should exit cleanly on SHUTDOWN")
        finally:
            if sup.proc and sup.proc.poll() is None:
                sup.proc.kill()

    def test_pause_before_channel_ready_reports_not_running_error(self):
        sup = self.supervisor_cls(self.entry)
        sup.start()
        try:
            # child is booting - no control_port in the state file yet
            r = sup.pause()
            self.assertFalse(r["ok"])
            self.assertEqual(r["error"], "not running")
        finally:
            if sup.proc and sup.proc.poll() is None:
                sup.proc.kill()

    def test_send_when_not_started_is_not_running(self):
        sup = self.supervisor_cls(self.entry)
        r = sup.pause()
        self.assertFalse(r["ok"])
        self.assertEqual(r["error"], "not running")


if __name__ == "__main__":
    unittest.main()
