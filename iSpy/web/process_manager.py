import subprocess
import sys
import time
import logging
import threading
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class VisionState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class VisionProcessManager:
    """Manages the iSpy vision loop as a subprocess.

    The web service owns this manager. It can start/stop/restart the vision
    process independently. When the vision process is not running, the web
    UI shows a control page instead of disappearing.
    """

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._state = VisionState.STOPPED
        self._start_time: float | None = None
        self._stop_time: float | None = None
        self._error_msg: str = ""
        self._lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None

    @property
    def state(self) -> VisionState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == VisionState.RUNNING

    @property
    def uptime(self) -> float | None:
        if self._start_time and self._state == VisionState.RUNNING:
            return time.time() - self._start_time
        return None

    @property
    def error(self) -> str:
        return self._error_msg

    def status(self) -> dict:
        with self._lock:
            return {
                "state": self._state.value,
                "uptime": self.uptime,
                "error": self._error_msg,
                "pid": self._process.pid if self._process else None,
            }

    def start(self) -> dict:
        with self._lock:
            if self._state in (VisionState.RUNNING, VisionState.STARTING):
                return {"ok": False, "error": f"Vision is already {self._state.value}"}

            self._state = VisionState.STARTING
            self._error_msg = ""

        try:
            game_loop_path = Path(__file__).resolve().parent.parent / "core" / "game_loop.py"
            cmd = [sys.executable, str(game_loop_path)]
            env = {**__import__("os").environ, "ISPY_MANAGED": "1"}

            self._process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

            self._start_time = time.time()
            with self._lock:
                self._state = VisionState.RUNNING

            self._monitor_thread = threading.Thread(
                target=self._monitor, daemon=True, name="vision-monitor"
            )
            self._monitor_thread.start()

            logger.info("Vision process started (PID %d)", self._process.pid)
            return {"ok": True, "pid": self._process.pid}

        except Exception as e:
            with self._lock:
                self._state = VisionState.ERROR
                self._error_msg = str(e)
            logger.exception("Failed to start vision process")
            return {"ok": False, "error": str(e)}

    def stop(self) -> dict:
        with self._lock:
            if self._state not in (VisionState.RUNNING, VisionState.STARTING):
                return {"ok": False, "error": f"Vision is not running (state={self._state.value})"}
            self._state = VisionState.STOPPING

        if self._process is None:
            with self._lock:
                self._state = VisionState.STOPPED
            return {"ok": True}

        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)

            logger.info("Vision process stopped")
            with self._lock:
                self._state = VisionState.STOPPED
                self._stop_time = time.time()
            return {"ok": True}

        except Exception as e:
            with self._lock:
                self._state = VisionState.ERROR
                self._error_msg = str(e)
            logger.exception("Failed to stop vision process")
            return {"ok": False, "error": str(e)}

    def restart(self) -> dict:
        stop_result = self.stop()
        if not stop_result["ok"] and "not running" not in stop_result.get("error", ""):
            return stop_result

        time.sleep(0.5)
        return self.start()

    def _monitor(self):
        """Watch the subprocess and update state if it exits unexpectedly."""
        if self._process is None:
            return

        returncode = self._process.wait()

        with self._lock:
            if self._state == VisionState.STOPPING:
                self._state = VisionState.STOPPED
            elif returncode == 0:
                self._state = VisionState.STOPPED
            else:
                self._state = VisionState.ERROR
                self._error_msg = f"Process exited with code {returncode}"
                logger.error("Vision process exited with code %d", returncode)

        self._stop_time = time.time()
