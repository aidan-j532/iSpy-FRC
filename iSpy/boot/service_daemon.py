import subprocess
import sys
import threading
import time
import json
import logging
from pathlib import Path
from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)

_STATE_FILE = Path.cwd() / "Outputs" / "service_state.json"


class VisionSupervisor:
    def __init__(self, entry_point: str):
        self.entry_point = entry_point
        self.proc: subprocess.Popen | None = None
        self.status = "stopped"   # stopped | running | paused | error
        self.last_error = None
        self.lock = threading.RLock()

    def start(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return {"ok": False, "error": "already running"}
            env = {"ISPY_MANAGED": "1"}
            import os
            full_env = {**os.environ, **env}
            self.proc = subprocess.Popen(
                [sys.executable, self.entry_point],
                stdin=subprocess.PIPE, text=True, env=full_env,
                cwd=str(Path.cwd()),
            )
            self.status = "running"
            self.last_error = None
            self._save_state()
            threading.Thread(target=self._watch, daemon=True).start()
            return {"ok": True}

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
            try:
                self.proc.stdin.write(cmd + "\n")
                self.proc.stdin.flush()
                return {"ok": True}
            except Exception as e:
                return {"ok": False, "error": str(e)}

    def pause(self):
        r = self._send("PAUSE")
        if r["ok"]:
            self.status = "paused"
            self._save_state()
        return r

    def resume(self):
        r = self._send("RESUME")
        if r["ok"]:
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
            self._send("SHUTDOWN")
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        with self.lock:
            self.status = "stopped"
            self._save_state()
        return {"ok": True}

    def restart(self):
        self.stop()
        time.sleep(0.5)
        return self.start()

    def _save_state(self):
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        pid = self.proc.pid if (self.proc and self.proc.poll() is None) else None
        _STATE_FILE.write_text(json.dumps({
            "status": self.status, "pid": pid, "last_error": self.last_error,
            "updated": time.time(),
        }))

    def get_status(self):
        with self.lock:
            return {"status": self.status, "pid": self.proc.pid if self.proc and self.proc.poll() is None else None,
                    "last_error": self.last_error}


def create_service_app(entry_point: str) -> Flask:
    app = Flask(__name__)
    sup = VisionSupervisor(entry_point)

    @app.route("/service/status")
    def status():
        return jsonify(sup.get_status())

    @app.route("/service/start", methods=["POST"])
    def start():
        return jsonify(sup.start())

    @app.route("/service/stop", methods=["POST"])
    def stop():
        return jsonify(sup.stop())

    @app.route("/service/restart", methods=["POST"])
    def restart():
        return jsonify(sup.restart())

    @app.route("/service/pause", methods=["POST"])
    def pause():
        return jsonify(sup.pause())

    @app.route("/service/resume", methods=["POST"])
    def resume():
        return jsonify(sup.resume())

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_service_app("iSpy/core/game_loop.py")
    app.run(host="0.0.0.0", port=5050)