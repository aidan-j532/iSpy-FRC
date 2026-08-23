"""Local control channel between the service daemon and the vision process.

The vision subprocess opens a loopback TCP server on start and publishes its
port in Outputs/service_state.json. VisionSupervisor connects to send
PAUSE/RESUME/SHUTDOWN instead of writing to the subprocess's stdin pipe.

Protocol: newline-delimited command per connection; the server replies
"OK\\n" or "ERR <reason>\\n" and closes. Only loopback connections are
accepted.
"""

import json
import logging
import socket
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

COMMANDS = ("PAUSE", "RESUME", "SHUTDOWN")

_STATE_RELATIVE = Path("Outputs") / "service_state.json"


def state_file_path() -> Path:
    return Path.cwd() / _STATE_RELATIVE


def read_service_state() -> dict:
    try:
        return json.loads(state_file_path().read_text())
    except Exception:
        return {}


def update_service_state(**fields) -> None:
    """Merge-write fields into the shared state file.

    Both processes write to this file (the daemon owns status/pid/last_error,
    the vision process owns control_port), so each side merges into whatever
    is currently on disk instead of overwriting the whole document.
    """
    path = state_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = read_service_state()
        state.update(fields)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(path)
    except Exception:
        logger.debug("Failed to update service state file", exc_info=True)


class ControlServer:
    """In-process TCP server executing control commands via callbacks."""

    def __init__(self, handlers: dict[str, callable]):
        self._handlers = handlers
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
        update_service_state(control_port=self.port)
        logger.info("Control channel listening on 127.0.0.1:%d", self.port)
        return self.port

    def stop(self) -> None:
        srv, self._server = self._server, None
        if srv is not None:
            try:
                srv.close()
            except OSError:
                pass
        update_service_state(control_port=None)
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
    """Send a single command to a ControlServer. Returns (ok, detail)."""
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
    """Daemon-side handle to the vision process's control channel."""

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
