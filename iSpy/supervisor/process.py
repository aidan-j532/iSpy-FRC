"""Process supervisor: owns :5000 and a single vision child process."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time

from iSpy.core.control_channel import (
    ControlServer,
    clear_child_state,
    read_child_state,
    read_service_state,
    send_command,
    update_supervisor_state,
)
from iSpy.supervisor.config import write_normal_web_settings
from iSpy.supervisor.status_server import StatusServer

logger = logging.getLogger("iSpy.supervisor")

DEFAULT_MAX_RESTARTS = 5
DEFAULT_RESTART_DELAY_S = 5.0
DEFAULT_STOP_TIMEOUT_S = 10.0


class Supervisor:
    def __init__(
        self,
        child_args=None,
        *,
        host: str = "0.0.0.0",
        port: int = 5000,
        python: str | None = None,
        max_restarts: int = DEFAULT_MAX_RESTARTS,
        restart_delay: float = DEFAULT_RESTART_DELAY_S,
        stop_timeout: float = DEFAULT_STOP_TIMEOUT_S,
        autostart: bool = False,
        status_server: StatusServer | None = None,
        child_command: list[str] | None = None,
    ):
        self.child_args = list(child_args or [])
        self.host = host
        self.port = port
        self.python = python
        self.child_command = list(child_command) if child_command else None
        self.max_restarts = int(max_restarts)
        self.restart_delay = float(restart_delay)
        self.stop_timeout = float(stop_timeout)
        self.autostart = bool(autostart)

        self._lock = threading.RLock()
        self.proc: subprocess.Popen | None = None
        self.state = "idle"
        self.attempt = 0
        self.last_error = ""
        self.message = ""
        self._since = time.time()
        self._stop_requested = False
        self._restart_requested = False
        self._stopping = threading.Event()

        self.status_server = status_server or StatusServer(
            host=host,
            port=port,
            action_handler=self.handle_action,
        )
        self.control_server = ControlServer(
            {
                "RELEASE_PORT": self.release_port,
                "START": self.request_start,
                "STOP": self.request_stop,
                "RESTART": self.request_restart,
            },
            state_updater=update_supervisor_state,
        )
        self._watchers: set[threading.Thread] = set()

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def _pid_locked(self):
        proc = self.proc
        if proc is not None and proc.poll() is None:
            return proc.pid
        return None

    def _record_locked(self):
        update_supervisor_state(
            state=self.state,
            pid=self._pid_locked(),
            supervisor_pid=os.getpid(),
            attempt=self.attempt,
            max_attempts=self.max_restarts,
            last_error=self.last_error,
            message=self.message,
            # These are child-owned while a child exists. Reset them on every
            # supervisor transition so a previous fresh-boot/web-off value
            # cannot leak into the next boot's page.
            phase="",
            web_enabled=True,
            web_off_reason="",
            since=self._since,
            updated=time.time(),
        )

    def _set_state(self, state: str, error: str | None = None):
        with self._lock:
            if state != self.state:
                self._since = time.time()
            self.state = state
            if error is not None:
                self.last_error = error
            self._record_locked()

    def snapshot(self) -> dict:
        # Imported lazily so state.py remains the one place that assembles the
        # merged public shape; this also avoids an import cycle.
        from iSpy.supervisor.state import snapshot

        return snapshot()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def run(self) -> int:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )
        try:
            self.control_server.start()
        except OSError:
            logger.exception("Supervisor control channel failed to start")
            return 1
        try:
            self.status_server.start()
        except OSError as exc:
            logger.error("Could not bind the status page to port %d: %s", self.port, exc)
            self.control_server.stop()
            return 1

        self._set_state("idle")
        self._install_signal_handlers()
        if self.autostart:
            self._start_threaded(reset_attempts=True)
        try:
            while not self._stopping.wait(0.5):
                pass
        except KeyboardInterrupt:
            self._stopping.set()
        finally:
            self.shutdown()
        return 0

    def _install_signal_handlers(self):
        def _handler(_signum, _frame):
            self._stopping.set()

        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                # not the main thread / unsupported platform
                pass

    def shutdown(self):
        self._stopping.set()
        with self._lock:
            has_child = self.proc is not None and self.proc.poll() is None
        if has_child:
            try:
                self._stop_worker()
            except Exception:
                logger.exception("Error stopping the vision child")
        try:
            self.control_server.stop()
        except Exception:
            pass
        try:
            self.status_server.stop()
        except Exception:
            pass
        with self._lock:
            self.state = "idle"
            self.proc = None
            self.last_error = ""
            self.message = ""
            self._record_locked()
        clear_child_state()
        logger.info("Supervisor stopped")

    # ------------------------------------------------------------------
    # actions (called by the status API / control channel)
    # ------------------------------------------------------------------
    def handle_action(self, action: str) -> dict:
        if action == "start":
            return self.request_start()
        if action == "stop":
            return self.request_stop()
        if action == "restart":
            return self.request_restart()
        if action == "restart-normal":
            ok, detail = write_normal_web_settings()
            if not ok:
                return {"ok": False, "error": detail}
            return self.request_restart()
        return {"ok": False, "error": f"unknown action {action!r}"}

    def request_start(self) -> dict:
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return {"ok": True, "already_running": True}
        self._start_threaded(reset_attempts=True)
        return {"ok": True, "starting": True}

    def request_stop(self) -> dict:
        thread = threading.Thread(
            target=self._stop_worker, name="ispy-supervisor-stop", daemon=True
        )
        thread.start()
        return {"ok": True, "stopping": True}

    def request_restart(self) -> dict:
        thread = threading.Thread(
            target=self._restart_worker, name="ispy-supervisor-restart", daemon=True
        )
        thread.start()
        return {"ok": True, "restarting": True}

    # ------------------------------------------------------------------
    # workers
    # ------------------------------------------------------------------
    def _start_threaded(self, reset_attempts: bool = False):
        thread = threading.Thread(
            target=self._start_worker,
            kwargs={"reset_attempts": reset_attempts},
            name="ispy-supervisor-start",
            daemon=True,
        )
        thread.start()

    def _start_worker(self, reset_attempts: bool = False):
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return
            self._stop_requested = False
            self._restart_requested = False
            if reset_attempts:
                self.attempt = 0
            self.last_error = ""
            self.message = ""
        self._ensure_status_server()
        with self._lock:
            try:
                self._spawn_locked()
            except Exception as exc:
                logger.exception("Failed to spawn the vision child")
                self.state = "crashed"
                self.last_error = str(exc)
                self._since = time.time()
                self._record_locked()
                self._schedule_auto_restart_locked()

    def _restart_worker(self):
        with self._lock:
            self._restart_requested = True
            self._stop_requested = False
            self.state = "restarting"
            self.message = ""
            self._since = time.time()
            self._record_locked()
        try:
            self._stop_child()
        except Exception:
            logger.exception("Error while stopping the child for restart")
        with self._lock:
            self.proc = None
            self._restart_requested = False
            self.attempt = 0
            self.last_error = ""
            self.state = "restarting"
            self._since = time.time()
            self._record_locked()
        clear_child_state()
        self._ensure_status_server()
        with self._lock:
            try:
                self._spawn_locked()
            except Exception as exc:
                logger.exception("Failed to spawn the vision child during restart")
                self.state = "crashed"
                self.last_error = str(exc)
                self._since = time.time()
                self._record_locked()
                self._schedule_auto_restart_locked()

    def _stop_worker(self):
        with self._lock:
            self._stop_requested = True
            self._restart_requested = False
            self.state = "stopping"
            self.message = ""
            self._since = time.time()
            self._record_locked()
        try:
            self._stop_child()
        finally:
            with self._lock:
                self.proc = None
                self._stop_requested = False
                self.attempt = 0
                self.last_error = ""
                self.state = "idle"
                self._since = time.time()
                self._record_locked()
            clear_child_state()
            self._ensure_status_server()

    def _spawn_locked(self):
        if self._stopping.is_set():
            return
        if self.proc is not None and self.proc.poll() is None:
            return
        if not self.status_server.serving:
            # _ensure_status_server() normally did this; keep a last attempt here
            # so a spawn never leaves the port unanswered after a hand-off.
            self.status_server.start()
        clear_child_state()
        env = os.environ.copy()
        env["ISPY_SUPERVISOR"] = "1"
        env["ISPY_SUPERVISOR_PID"] = str(os.getpid())
        if self.control_server.port:
            env["ISPY_SUPERVISOR_CONTROL_PORT"] = str(self.control_server.port)
        python = self.python or _resolve_python()
        cmd = self.child_command or [
            python,
            "-m",
            "iSpy.cli",
            "_vision",
            *self.child_args,
        ]
        self.state = "booting"
        self._since = time.time()
        self.message = ""
        popen_kwargs = {}
        if os.name == "nt":
            # `ispy start` is often launched from a console on Windows; keep
            # the child from flashing a second console window.
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NO_WINDOW", 0x08000000
            )
        self.proc = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),
            env=env,
            **popen_kwargs,
        )
        logger.info(
            "Launched vision child pid=%s: %s", self.proc.pid, " ".join(cmd)
        )
        self._record_locked()
        watcher = threading.Thread(
            target=self._watch_child,
            args=(self.proc,),
            name="ispy-supervisor-watch",
            daemon=True,
        )
        self._watchers.add(watcher)
        watcher.start()

    def _watch_child(self, proc: subprocess.Popen):
        rc = proc.wait()
        with self._lock:
            self._watchers.discard(threading.current_thread())
            if self.proc is not proc:
                return
            self.proc = None
            if self._restart_requested or self._stop_requested:
                # The stop/restart worker owns the next state transition.
                return
            clear_child_state()
            if rc == 0:
                self.state = "idle"
                self.last_error = ""
                self.message = ""
                self._since = time.time()
                self._record_locked()
                schedule = False
            else:
                self.attempt += 1
                self.last_error = f"vision process exited with code {rc}"
                self._since = time.time()
                if self.attempt >= self.max_restarts:
                    self.state = "crash_loop_stopped"
                    schedule = False
                else:
                    self.state = "crashed"
                    schedule = True
                self._record_locked()
        self._ensure_status_server()
        if schedule:
            with self._lock:
                self._schedule_auto_restart_locked()

    def _schedule_auto_restart_locked(self):
        # Safe to call with the lock held; the timer re-checks everything.
        if self.attempt >= self.max_restarts:
            return
        timer = threading.Timer(self.restart_delay, self._auto_restart)
        timer.daemon = True
        timer.start()

    def _auto_restart(self):
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return
            if self.state != "crashed":
                return
            self.state = "booting"
            self._since = time.time()
            self.message = ""
            self._record_locked()
        self._ensure_status_server()
        with self._lock:
            if self.state == "booting" and self.proc is None:
                try:
                    self._spawn_locked()
                except Exception as exc:
                    logger.exception("Automatic restart failed to spawn the child")
                    self.state = "crashed"
                    self.last_error = str(exc)
                    self._since = time.time()
                    self._record_locked()
                    self._schedule_auto_restart_locked()

    # ------------------------------------------------------------------
    # child control / port hand-off
    # ------------------------------------------------------------------
    def release_port(self):
        logger.info("Vision child requested the web port; releasing :%d", self.port)
        self.status_server.release()

    def _ensure_status_server(self):
        if self.status_server.serving:
            return
        try:
            self.status_server.start()
        except OSError:
            logger.exception("Could not rebind the status page to port %d", self.port)

    def _control_port_for_child(self) -> int | None:
        # Prefer the child-state copy: the Outputs file can be stale after a
        # previous process or be removed by `boot -f`.
        child = read_child_state()
        port = child.get("control_port")
        if isinstance(port, int):
            return port
        state = read_service_state()
        port = state.get("control_port")
        return port if isinstance(port, int) else None

    def _stop_child(self):
        with self._lock:
            proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        port = self._control_port_for_child()
        if port is not None:
            try:
                send_command("SHUTDOWN", port, timeout=3.0)
            except Exception:
                logger.debug("Graceful SHUTDOWN request failed", exc_info=True)
        try:
            proc.wait(timeout=self.stop_timeout)
            return
        except subprocess.TimeoutExpired:
            logger.warning("Vision child did not exit in %.0fs; terminating", self.stop_timeout)
        try:
            proc.terminate()
            proc.wait(timeout=5.0)
            return
        except (subprocess.TimeoutExpired, OSError):
            pass
        logger.warning("Vision child ignored terminate; killing")
        try:
            proc.kill()
            proc.wait(timeout=5.0)
        except (subprocess.TimeoutExpired, OSError):
            logger.error("Vision child could not be killed")

    # kept for the older service_daemon-style callers
    def start(self):
        return self.request_start()

    def stop(self):
        return self.request_stop()

    def restart(self):
        return self.request_restart()


def _resolve_python() -> str:
    try:
        from iSpy.boot._venv import resolve_launch_python

        return resolve_launch_python(fallback=sys.executable)
    except Exception:
        return sys.executable


def run_supervisor(child_args=None, *, autostart: bool = True) -> int:
    return Supervisor(child_args=child_args, autostart=autostart).run()
