"""Supervisor tests: cheap imports, state machine, hand-off, messages, auth."""

import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from iSpy.core import control_channel
from iSpy.core.request_auth import is_lan_address, is_loopback, restart_allowed
from iSpy.supervisor import config as supervisor_config
from iSpy.supervisor import messages as supervisor_messages
from iSpy.supervisor.child import bind_with_retry
from iSpy.supervisor.process import Supervisor
from iSpy.supervisor.state import (
    DEFAULT_MAX_ATTEMPTS,
    child_snapshot,
    read_log_tail,
    snapshot,
)
from iSpy.supervisor.status_server import StatusServer, render_page

REPO_ROOT = Path(__file__).resolve().parents[1]


class _TmpCwd:
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


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def _get_json(url: str):
    with contextlib.closing(urllib.request.urlopen(url, timeout=3)) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _post_json(url: str):
    request = urllib.request.Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with contextlib.closing(urllib.request.urlopen(request, timeout=3)) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


class ImportCheapnessTests(unittest.TestCase):
    def test_importing_the_supervisor_pulls_in_no_heavy_modules(self):
        code = (
            "import sys\n"
            "import iSpy.supervisor\n"
            "heavy = [m for m in ('torch','cv2','flask','PIL','iSpy.boot.boot')"
            " if m in sys.modules]\n"
            "print(','.join(heavy))\n"
            "raise SystemExit(1 if heavy else 0)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class MessageLoaderTests(unittest.TestCase):
    def test_missing_file_uses_built_in_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "ui_messages.json"
            missing_image = Path(td) / "missing.png"
            with mock.patch.object(supervisor_messages, "MESSAGES_FILE", missing):
                with mock.patch.object(
                    supervisor_messages, "_DEFAULT_IMAGE", missing_image
                ):
                    loaded = supervisor_messages.load_messages()
        self.assertEqual(
            loaded["restarting"], supervisor_messages.DEFAULT_MESSAGES["restarting"]
        )
        self.assertGreaterEqual(len(loaded["restarting"]), 10)
        self.assertGreaterEqual(len(loaded["fresh_boot"]), 10)
        self.assertIsNone(loaded["image"])

    def test_user_file_is_reread_and_can_override_the_image(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "ui_messages.json"
            image = root / "team.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
            path.write_text(
                json.dumps(
                    {
                        "restarting": ["One", "Two"],
                        "fresh_boot": ["Setup"],
                        "image": str(image),
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                supervisor_messages, "MESSAGES_FILE", path
            ):
                first = supervisor_messages.load_messages()
                path.write_text(
                    json.dumps({"restarting": ["Changed"], "fresh_boot": ["Again"]}),
                    encoding="utf-8",
                )
                second = supervisor_messages.load_messages()
        self.assertEqual(first["restarting"], ["One", "Two"])
        self.assertEqual(first["image"], str(image))
        self.assertEqual(second["restarting"], ["Changed"])


class StateSnapshotTests(unittest.TestCase):
    def test_child_running_upgrades_booting_and_preserves_attempts(self):
        with _TmpCwd():
            control_channel.update_supervisor_state(
                state="booting", pid=123, attempt=2, max_attempts=5
            )
            control_channel.update_service_state(
                status="running",
                phase="running",
                pid=123,
                web_enabled=False,
                web_off_reason="performance level 0",
            )
            state = snapshot()
        self.assertEqual(state["state"], "running")
        self.assertFalse(state["web_enabled"])
        self.assertEqual(state["attempt"], 2)
        self.assertEqual(state["attempt_label"], "2/5")
        self.assertEqual(state["phase"], "running")
        self.assertIn("last_error", state)

    def test_stale_child_state_from_another_pid_is_ignored(self):
        with _TmpCwd():
            control_channel.update_supervisor_state(state="booting", pid=222)
            control_channel.update_service_state(
                status="running", phase="old child", pid=111
            )
            state = snapshot()
        self.assertEqual(state["state"], "booting")
        self.assertNotEqual(state["phase"], "old child")

    def test_standalone_child_reports_running_even_with_stale_files(self):
        with _TmpCwd():
            control_channel.update_supervisor_state(
                state="crash_loop_stopped", pid=None
            )
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ISPY_SUPERVISOR", None)
                state = child_snapshot()
        self.assertEqual(state["state"], "running")
        self.assertFalse(state["managed"])
        self.assertEqual(state["served_by"], "child")

    def test_supervisor_snapshot_is_marked_as_served_by_supervisor(self):
        with _TmpCwd():
            control_channel.update_supervisor_state(state="booting", pid=123)
            state = snapshot()
        self.assertEqual(state["served_by"], "supervisor")

    def test_log_tail_is_bounded_and_tolerates_missing_file(self):
        with _TmpCwd():
            out = Path("Outputs")
            out.mkdir()
            (out / "log.txt").write_text(
                "\n".join(f"line {i}" for i in range(30)), encoding="utf-8"
            )
            tail = read_log_tail(max_lines=20)
            self.assertEqual(len(tail), 20)
            self.assertEqual(tail[-1], "line 29")
            (out / "log.txt").unlink()
            self.assertEqual(read_log_tail(), [])


class RestartAuthTests(unittest.TestCase):
    def test_loopback_always_allowed(self):
        self.assertTrue(is_loopback("127.0.0.1"))
        self.assertTrue(is_loopback("::1"))
        self.assertTrue(is_loopback("::ffff:127.0.0.1"))
        self.assertTrue(restart_allowed("127.0.0.1", {}, allow_remote=False))

    def test_lan_allowed_by_default_but_not_public(self):
        self.assertTrue(is_lan_address("192.168.1.50"))
        self.assertTrue(is_lan_address("10.0.0.4"))
        self.assertTrue(is_lan_address("172.16.5.2"))
        self.assertTrue(restart_allowed("192.168.1.50", {}, allow_remote=True))
        self.assertFalse(restart_allowed("8.8.8.8", {}, allow_remote=True))

    def test_allow_false_requires_token_for_lan(self):
        self.assertFalse(restart_allowed("192.168.1.50", {}, allow_remote=False))
        with mock.patch.dict(
            os.environ, {"ISPY_ADMIN_TOKEN": "sekrit"}, clear=False
        ):
            self.assertTrue(
                restart_allowed(
                    "192.168.1.50",
                    {"X-iSpy-Admin-Token": "sekrit"},
                    allow_remote=False,
                )
            )
            self.assertFalse(
                restart_allowed(
                    "192.168.1.50",
                    {"X-iSpy-Admin-Token": "wrong"},
                    allow_remote=False,
                )
            )

    def test_forwarded_for_is_not_trusted(self):
        # A public client claiming 127.0.0.1 via the proxy header is still just
        # a public client. The helper never looks at that header.
        self.assertFalse(
            restart_allowed(
                "203.0.113.9",
                {"X-Forwarded-For": "127.0.0.1"},
                allow_remote=True,
            )
        )


class StatusHandoffTests(unittest.TestCase):
    def test_status_server_releases_and_rebinds_port(self):
        port = _free_port()
        state = {
            "state": "booting",
            "web_enabled": True,
            "attempt": 0,
            "max_attempts": 5,
            "log_tail": [],
        }
        server = StatusServer(
            host="127.0.0.1", port=port, snapshot_func=lambda: state
        )
        child_sock = None
        try:
            self.assertTrue(server.start(retry_s=2))
            status, body = _get_json(
                f"http://127.0.0.1:{port}/api/supervisor/state"
            )
            self.assertEqual(status, 200)
            self.assertEqual(body["state"], "booting")
            self.assertTrue(server.serving)

            server.release()
            self.assertFalse(server.serving)
            with self.assertRaises(urllib.error.URLError):
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/supervisor/state", timeout=1
                )

            # The child owns the port in this window.
            child_sock = bind_with_retry("127.0.0.1", port, timeout=2)
            child_sock.close()
            child_sock = None

            # The child went away -> supervisor rebinds the same exact port.
            self.assertTrue(server.start(retry_s=2))
            status, body = _get_json(
                f"http://127.0.0.1:{port}/api/supervisor/state"
            )
            self.assertEqual(status, 200)
            self.assertEqual(body["state"], "booting")
        finally:
            if child_sock is not None:
                child_sock.close()
            server.release()

    def test_loopback_post_reaches_the_action_handler(self):
        port = _free_port()
        called = []
        server = StatusServer(
            host="127.0.0.1",
            port=port,
            action_handler=lambda action: (
                called.append(action) or {"ok": True, "action": action}
            ),
            snapshot_func=lambda: {"state": "idle"},
        )
        try:
            with _TmpCwd():
                self.assertTrue(server.start(retry_s=2))
                status, body = _post_json(
                    f"http://127.0.0.1:{port}/api/supervisor/restart"
                )
                normal_status, normal_body = _post_json(
                    f"http://127.0.0.1:{port}/api/supervisor/restart-normal"
                )
            self.assertEqual(status, 200)
            self.assertTrue(body["ok"])
            self.assertEqual(normal_status, 200)
            self.assertTrue(normal_body["ok"])
            self.assertEqual(called, ["restart", "restart-normal"])
        finally:
            server.release()


class PageRenderTests(unittest.TestCase):
    def test_web_off_page_offers_normal_settings_restart(self):
        page = render_page(
            {
                "state": "running",
                "web_enabled": False,
                "web_off_reason": "performance level 0 (Max FPS) has the web UI off",
                "attempt_label": "0/5",
                "log_tail": [],
            }
        ).decode("utf-8")
        self.assertIn("web UI is off for this run", page)
        self.assertIn("Restart with normal settings", page)
        self.assertIn("/api/supervisor/restart-normal", page)
        self.assertIn("performance_level", page)

    def test_booting_page_shows_the_child_phase_and_polls(self):
        page = render_page(
            {
                "state": "booting",
                "phase": "loading vision pipelines",
                "web_enabled": True,
                "attempt_label": "0/5",
                "log_tail": [],
            }
        ).decode("utf-8")
        self.assertIn("loading vision pipelines", page)
        self.assertIn("/api/supervisor/state", page)
        self.assertIn("served_by", page)

    def test_crash_page_shows_attempt_and_log_tail(self):
        page = render_page(
            {
                "state": "crashed",
                "phase": "crashed",
                "web_enabled": True,
                "attempt_label": "2/5",
                "last_error": "vision process exited with code 1",
                "log_tail": ["Traceback ...", "ImportError: boom"],
            }
        ).decode("utf-8")
        self.assertIn("Restart attempt 2/5", page)
        self.assertIn("vision process exited with code 1", page)
        self.assertIn("ImportError: boom", page)


class StateMachineTests(unittest.TestCase):
    def test_fast_crashing_child_stops_after_the_limit(self):
        with _TmpCwd() as root:
            script = root / "crash_child.py"
            script.write_text(
                "import sys, time\n"
                "time.sleep(0.05)\n"
                "sys.exit(1)\n",
                encoding="utf-8",
            )
            supervisor = Supervisor(
                child_command=[sys.executable, str(script)],
                host="127.0.0.1",
                port=_free_port(),
                max_restarts=2,
                restart_delay=0.05,
            )
            try:
                result = supervisor.request_start()
                self.assertTrue(result["ok"])
                deadline = time.time() + 12
                final = None
                while time.time() < deadline:
                    final = snapshot()
                    if final["state"] == "crash_loop_stopped":
                        break
                    time.sleep(0.05)
                self.assertIsNotNone(final)
                self.assertEqual(final["state"], "crash_loop_stopped")
                self.assertEqual(final["attempt"], 2)
                self.assertEqual(final["attempt_label"], "2/2")
                self.assertIn("exited with code 1", final["last_error"])
            finally:
                supervisor.shutdown()

    def test_stop_returns_to_idle(self):
        with _TmpCwd() as root:
            script = root / "sleep_child.py"
            script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
            supervisor = Supervisor(
                child_command=[sys.executable, str(script)],
                host="127.0.0.1",
                port=_free_port(),
                max_restarts=2,
                restart_delay=0.05,
                stop_timeout=0.5,
            )
            try:
                supervisor.request_start()
                deadline = time.time() + 5
                while time.time() < deadline:
                    if supervisor.proc is not None and supervisor.proc.poll() is None:
                        break
                    time.sleep(0.05)
                self.assertIsNotNone(supervisor.proc)
                supervisor.request_stop()
                deadline = time.time() + 8
                while time.time() < deadline:
                    if snapshot()["state"] == "idle":
                        break
                    time.sleep(0.05)
                self.assertEqual(snapshot()["state"], "idle")
            finally:
                supervisor.shutdown()


class FlaskSupervisorEndpointTests(unittest.TestCase):
    def _client(self, *, allow_remote: bool = True):
        from iSpy.config.iSpyConfig import iSpyConfig
        from iSpy.web.Backend.WebApp import create_app

        config = iSpyConfig()
        config.config["allow_remote_restart"] = allow_remote
        return create_app(cameras=[], config=config).flask_app.test_client()

    def test_lan_restart_is_allowed_by_default(self):
        client = self._client()
        with mock.patch(
            "iSpy.web.Backend.WebApp.send_supervisor_command",
            return_value=(True, "ok"),
        ) as send:
            response = client.post(
                "/api/supervisor/restart",
                environ_base={"REMOTE_ADDR": "192.168.1.50"},
            )
        self.assertEqual(response.status_code, 200)
        send.assert_called_once_with("RESTART")

    def test_public_client_is_denied_even_with_forwarded_for(self):
        client = self._client()
        with mock.patch(
            "iSpy.web.Backend.WebApp.send_supervisor_command",
            return_value=(True, "ok"),
        ) as send:
            response = client.post(
                "/api/supervisor/restart",
                environ_base={"REMOTE_ADDR": "203.0.113.9"},
                headers={"X-Forwarded-For": "127.0.0.1"},
            )
        self.assertEqual(response.status_code, 403)
        send.assert_not_called()

    def test_allow_false_still_permits_loopback_and_token(self):
        client = self._client(allow_remote=False)
        with mock.patch(
            "iSpy.web.Backend.WebApp.send_supervisor_command",
            return_value=(True, "ok"),
        ) as send:
            lan = client.post(
                "/api/supervisor/restart",
                environ_base={"REMOTE_ADDR": "10.0.0.9"},
            )
            loopback = client.post(
                "/api/supervisor/restart",
                environ_base={"REMOTE_ADDR": "127.0.0.1"},
            )
            with mock.patch.dict(
                os.environ, {"ISPY_ADMIN_TOKEN": "sekrit"}, clear=False
            ):
                token = client.post(
                    "/api/supervisor/restart",
                    environ_base={"REMOTE_ADDR": "10.0.0.9"},
                    headers={"X-iSpy-Admin-Token": "sekrit"},
                )
        self.assertEqual(lan.status_code, 403)
        self.assertEqual(loopback.status_code, 200)
        self.assertEqual(token.status_code, 200)
        self.assertEqual(send.call_count, 2)


class RestartNormalActionTests(unittest.TestCase):
    def test_web_off_restart_writes_level_four_and_restarts(self):
        with _TmpCwd() as root:
            cfg_dir = root / "Config"
            cfg_dir.mkdir()
            (cfg_dir / "config.json").write_text(
                json.dumps(
                    {
                        "performance_level": 0,
                        "app_mode": False,
                        "game_mode_web_off_next_run": True,
                    }
                ),
                encoding="utf-8",
            )
            supervisor = Supervisor(port=_free_port())
            with mock.patch.object(
                supervisor, "request_restart", return_value={"ok": True}
            ) as restart:
                result = supervisor.handle_action("restart-normal")
            written = json.loads((cfg_dir / "config.json").read_text())
        self.assertTrue(result["ok"])
        self.assertEqual(written["performance_level"], 4)
        self.assertTrue(written["app_mode"])
        self.assertFalse(written["game_mode_web_off_next_run"])
        restart.assert_called_once()


class SupervisorDisabledTests(unittest.TestCase):
    def test_env_and_ci_disable_the_supervisor(self):
        with mock.patch.dict(os.environ, {"ISPY_NO_SUPERVISOR": "1"}, clear=False):
            self.assertTrue(supervisor_config.supervisor_disabled())
        env = dict(os.environ)
        env.pop("ISPY_NO_SUPERVISOR", None)
        env["CI"] = "true"
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(supervisor_config.supervisor_disabled())


if __name__ == "__main__":
    unittest.main()
