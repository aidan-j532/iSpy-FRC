import subprocess
import sys
import threading
import time
import json
import logging
from pathlib import Path
from flask import Flask, jsonify, request

from iSpy.boot._venv import resolve_launch_python
from iSpy.core.control_channel import SupervisorControlClient, state_file_path

logger = logging.getLogger(__name__)


def _setup_log_file() -> None:
    """Append daemon logs to the same Outputs/log.txt the boot process uses.

    Best-effort: cwd is the project root under the systemd unit; fall back to
    the repo root derived from this module's location for other launch modes.
    """
    try:
        log_file = Path.cwd() / "Outputs" / "log.txt"
        if not log_file.parent.is_dir():
            log_file = Path(__file__).resolve().parents[2] / "Outputs" / "log.txt"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(handler)
    except Exception:
        pass


class VisionSupervisor:
    def __init__(self, entry_point: str):
        self.entry_point = entry_point
        self.proc: subprocess.Popen | None = None
        self.status = "stopped"   # stopped | running | paused | error
        self.last_error = None
        self.lock = threading.RLock()
        self._watch_thread: threading.Thread | None = None
        # talks to the control channel the vision process publishes in the
        # state file (control_port); replaces the old stdin pipe
        self._control = SupervisorControlClient()

    def start(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return {"ok": False, "error": "already running"}
            # single-source the interpreter from the install-time marker so the
            # vision child runs under the same venv the rest of iSpy booted with
            python = resolve_launch_python()
            logger.info(
                "launching vision process %r with python=%r (this daemon: %r, prefix=%r)",
                self.entry_point, python, sys.executable, sys.prefix,
            )
            self.proc = subprocess.Popen(
                [python, self.entry_point],
                cwd=str(Path.cwd()),
            )
            self._control.invalidate()
            self.status = "running"
            self.last_error = None
            self._save_state()
            self._watch_thread = threading.Thread(target=self._watch, daemon=True)
            self._watch_thread.start()
            return {"ok": True, "pid": self.proc.pid}

    def _watch(self):
        proc = self.proc
        proc.wait()
        with self.lock:
            if self.proc is proc:
                if self.status not in ("stopping",):
                    self.status = "error" if proc.returncode not in (0, None) else "stopped"
                    self.last_error = f"exited with code {proc.returncode}" if proc.returncode else None
                else:
                    self.status = "stopped"
                self._save_state()

    def _send(self, cmd: str):
        with self.lock:
            if not self.proc or self.proc.poll() is not None:
                return {"ok": False, "error": "not running"}
        ok, detail = self._control.send(cmd)
        return {"ok": ok, "error": None if ok else detail}

    def pause(self):
        r = self._send("PAUSE")
        if r["ok"]:
            with self.lock:
                self.status = "paused"
                self._save_state()
        return r

    def resume(self):
        r = self._send("RESUME")
        if r["ok"]:
            with self.lock:
                self.status = "running"
                self._save_state()
        return r

    def stop(self, timeout=10):
        with self.lock:
            if not self.proc or self.proc.poll() is not None:
                self.status = "stopped"
                self._save_state()
                return {"ok": True}
            self.status = "stopping"
            # best-effort graceful request; if the control channel isn't up
            # yet (child still booting) we fall through to terminate/kill
            self._send("SHUTDOWN")
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        watcher = self._watch_thread
        if watcher is not None and watcher is not threading.current_thread():
            # let the exit-status write land before we overwrite with "stopped"
            watcher.join(timeout=2)
        with self.lock:
            self.status = "stopped"
            self._save_state()
        return {"ok": True}

    def restart(self):
        self.stop()
        time.sleep(0.5)
        return self.start()

    def _save_state(self):
        state_file = state_file_path()
        state_file.parent.mkdir(parents=True, exist_ok=True)
        pid = self.proc.pid if (self.proc and self.proc.poll() is None) else None
        # merge instead of overwrite - the vision process owns control_port
        # in this same file and must survive our status updates
        try:
            state = json.loads(state_file.read_text())
            if not isinstance(state, dict):
                state = {}
        except Exception:
            state = {}
        state.update({
            "status": self.status, "pid": pid, "last_error": self.last_error,
            "updated": time.time(),
        })
        try:
            tmp = state_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state))
            tmp.replace(state_file)
        except OSError:
            logger.exception("Failed to write service state file")

    def get_status(self):
        with self.lock:
            return {"status": self.status, "pid": self.proc.pid if self.proc and self.proc.poll() is None else None,
                    "last_error": self.last_error}


def create_service_app(entry_point: str) -> Flask:
    app = Flask(__name__)
    # service control endpoints take no request body - a tight ceiling rejects
    # junk payloads outright
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    sup = VisionSupervisor(entry_point)

    # start/stop/restart/pause/resume can kill the vision pipeline or the whole
    # boot: only allow them from localhost, or from a remote client presenting
    # ISPY_ADMIN_TOKEN (same trust model as /api/plugins/* admin endpoints).
    from iSpy.web.Backend.PluginStatus import require_local_or_token

    @app.route("/service/status")
    def status():
        return jsonify(sup.get_status())

    @app.route("/service/start", methods=["POST"])
    @require_local_or_token
    def start():
        return jsonify(sup.start())

    @app.route("/service/stop", methods=["POST"])
    @require_local_or_token
    def stop():
        return jsonify(sup.stop())

    @app.route("/service/restart", methods=["POST"])
    @require_local_or_token
    def restart():
        return jsonify(sup.restart())

    @app.route("/service/pause", methods=["POST"])
    @require_local_or_token
    def pause():
        return jsonify(sup.pause())

    @app.route("/service/resume", methods=["POST"])
    @require_local_or_token
    def resume():
        return jsonify(sup.resume())

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _setup_log_file()
    logger.info(
        "ispy service daemon: python=%r prefix=%r", sys.executable, sys.prefix
    )
    app = create_service_app("iSpy/core/game_loop.py")
    app.run(host="0.0.0.0", port=5050)