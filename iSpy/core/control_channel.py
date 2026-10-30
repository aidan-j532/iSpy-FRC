import json
import logging
import os
import socket
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

COMMANDS = (
    "PAUSE",
    "RESUME",
    "SHUTDOWN",
    # child -> supervisor hand-off / lifecycle commands
    "RELEASE_PORT",
    "RESTART",
    "STOP",
    "START",
)

_STATE_RELATIVE = Path("Outputs") / "service_state.json"
# This directory is deliberately outside Config/, Outputs/ and YoloModels/:
# `boot -f` wipes those, but the process supervisor has to keep its own state
# across a fresh boot. It is gitignored.
_RUN_DIR = Path(".ispy-run")
_SUPERVISOR_STATE = _RUN_DIR / "supervisor_state.json"
_CHILD_STATE = _RUN_DIR / "child_state.json"


def state_file_path() -> Path:
    return Path.cwd() / _STATE_RELATIVE


def supervisor_state_file_path() -> Path:
    return Path.cwd() / _SUPERVISOR_STATE


def child_state_file_path() -> Path:
    return Path.cwd() / _CHILD_STATE


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, state: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        logger.debug("Failed to write state file %s", path, exc_info=True)


def read_service_state() -> dict:
    return _read_json(state_file_path())


def read_supervisor_state() -> dict:
    return _read_json(supervisor_state_file_path())


def read_child_state() -> dict:
    return _read_json(child_state_file_path())


def update_supervisor_state(**fields) -> None:
    path = supervisor_state_file_path()
    state = _read_json(path)
    state.update(fields)
    _write_json(path, state)


def clear_child_state() -> None:
    try:
        child_state_file_path().unlink()
    except OSError:
        pass


def update_service_state(**fields) -> None:
    # The legacy Outputs/service_state.json is what SupervisorControlClient
    # reads. The same update also goes to .ispy-run/child_state.json, which is
    # what the status page reads for the current child phase. Keeping the two
    # writes independent means `boot -f` deleting Outputs/ cannot erase the
    # supervisor's view of the running child.
    fields = dict(fields)
    path = state_file_path()
    state = _read_json(path)
    child_path = child_state_file_path()
    child = _read_json(child_path)
    if "pid" not in fields and not state.get("pid") and not child.get("pid"):
        fields["pid"] = os.getpid()
    state.update(fields)
    _write_json(path, state)
    child.update(fields)
    _write_json(child_path, child)


class ControlServer:
    def __init__(self, handlers: dict[str, callable], state_updater=None):
        self._handlers = handlers
        self._state_updater = state_updater or update_service_state
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self.port: int | None = None

    def start(self) -> int:
        if self._server is not None:
            return self.port
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))  # ephemeral port - no clashes
        srv.listen(2)
        srv.settimeout(0.5)
        self._server = srv
        self.port = srv.getsockname()[1]
        self._thread = threading.Thread(
            target=self._serve, daemon=True, name="control-channel"
        )
        self._thread.start()
        self._state_updater(control_port=self.port)
        logger.info("Control channel listening on 127.0.0.1:%d", self.port)
        return self.port

    def stop(self) -> None:
        srv, self._server = self._server, None
        if srv is not None:
            try:
                srv.close()
            except OSError:
                pass
        self._state_updater(control_port=None)
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)

    def _serve(self):
        while self._server is not None:
            try:
                conn, addr = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break  # server closed
            if addr[0] != "127.0.0.1":
                try:
                    conn.close()
                except OSError:
                    pass
                continue
            with conn:
                conn.settimeout(5)
                self._handle_connection(conn)

    def _handle_connection(self, conn: socket.socket):
        try:
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(1024)
                if not chunk:
                    return
                data += chunk
                if len(data) > 4096:
                    conn.sendall(b"ERR command too long\n")
                    return
            cmd = data.decode("utf-8", "replace").strip().upper()
            handler = self._handlers.get(cmd)
            if handler is None:
                conn.sendall(f"ERR unknown command {cmd!r}\n".encode())
                return
            handler()
            conn.sendall(b"OK\n")
        except Exception:
            logger.debug("Control channel error", exc_info=True)
            try:
                conn.sendall(b"ERR internal error\n")
            except OSError:
                pass


def send_command(cmd: str, port: int, timeout: float = 3.0) -> tuple[bool, str]:
    cmd = cmd.strip().upper()
    if cmd not in COMMANDS:
        return False, f"unknown command {cmd!r}"
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as conn:
            conn.settimeout(timeout)
            conn.sendall((cmd + "\n").encode())
            reply = b""
            while not reply.endswith(b"\n"):
                chunk = conn.recv(1024)
                if not chunk:
                    break
                reply += chunk
    except (OSError, ValueError) as e:
        return False, str(e)
    reply_text = reply.decode("utf-8", "replace").strip()
    if reply_text == "OK":
        return True, ""
    return False, reply_text or "no response"


class SupervisorControlClient:
    def __init__(self, stale_after_s: float = 30.0):
        self._stale_after_s = stale_after_s
        self._cached_port: int | None = None
        self._cache_time = 0.0

    def _port(self) -> int | None:
        # short cache so stop()/pause()/resume() bursts don't re-read the file,
        # but still fresh enough to notice the child publishing its port
        if (
            self._cached_port is None
            or time.monotonic() - self._cache_time > self._stale_after_s
        ):
            port = read_service_state().get("control_port")
            self._cached_port = int(port) if isinstance(port, int) else None
            self._cache_time = time.monotonic()
        return self._cached_port

    def invalidate(self):
        self._cached_port = None

    def send(self, cmd: str) -> tuple[bool, str]:
        port = self._port()
        if port is None:
            return False, "not running"
        ok, detail = send_command(cmd, port)
        if not ok:
            # connection refused / reset -> the published port is stale;
            # forget it so the next attempt re-reads the state file
            self.invalidate()
        return ok, detail


def send_supervisor_command(cmd: str, timeout: float = 3.0) -> tuple[bool, str]:
    """Send a lifecycle command to the process supervisor, if one is running."""
    port = read_supervisor_state().get("control_port")
    if not isinstance(port, int):
        return False, "supervisor not running"
    return send_command(cmd, port, timeout=timeout)
