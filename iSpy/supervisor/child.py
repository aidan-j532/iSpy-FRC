"""Child-side port hand-off helper (stdlib only).

The vision process binds :5000 only after the supervisor has closed its status
listener.  This module deliberately has no Flask imports so it stays testable
without the web stack.
"""

from __future__ import annotations

import errno
import os
import socket
import time

from iSpy.core.control_channel import send_supervisor_command


def supervisor_enabled() -> bool:
    return os.environ.get("ISPY_SUPERVISOR", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def request_port_release(timeout: float = 10.0) -> tuple[bool, str]:
    if not supervisor_enabled():
        return False, "not running under the supervisor"
    return send_supervisor_command("RELEASE_PORT", timeout=timeout)


def bind_with_retry(
    host: str,
    port: int,
    *,
    timeout: float = 10.0,
) -> socket.socket:
    """Bind/listen with the EADDRINUSE retry window the hand-off needs."""
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            sock.listen(128)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
            # 10048 = WSAEADDRINUSE on Windows
            if exc.errno not in (errno.EADDRINUSE, 10048):
                raise
            if time.monotonic() >= deadline:
                raise last_error
            time.sleep(0.2)


def serve_flask(app, host: str = "0.0.0.0", port: int = 5000) -> None:
    """Run the Flask dev server, asking the supervisor for the port first."""
    if not supervisor_enabled():
        app.run(host=host, port=port, threaded=True)
        return

    ok, detail = request_port_release()
    if not ok:
        # Stay useful even if the supervisor went away: retry the ordinary
        # bind path rather than failing the whole vision process.
        print(f"iSpy: supervisor port release failed ({detail}); binding directly.")
    sock = bind_with_retry(host, port)
    try:
        # Flask's app.run() no longer forwards `fd` through run_simple in
        # current Werkzeug, so construct the WSGI server ourselves. It uses
        # the socket we already bound, making the retry above the only bind.
        from werkzeug.serving import make_server

        server = make_server(host, port, app, threaded=True, fd=sock.fileno())
        try:
            server.serve_forever()
        finally:
            server.server_close()
    finally:
        try:
            sock.close()
        except OSError:
            pass


__all__ = [
    "bind_with_retry",
    "request_port_release",
    "serve_flask",
    "supervisor_enabled",
]
