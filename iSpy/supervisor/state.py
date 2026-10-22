"""Shared supervisor/child state snapshot (stdlib only).

The supervisor owns ``.ispy-run/supervisor_state.json``.  The vision child
publishes its phase into ``.ispy-run/child_state.json`` through
``iSpy.core.control_channel.update_service_state``.  Both files live outside
the directories wiped by ``boot -f``.
"""

from __future__ import annotations

import os
from pathlib import Path

from iSpy.core.control_channel import (
    clear_child_state,
    read_child_state,
    read_supervisor_state,
    supervisor_state_file_path,
    update_supervisor_state,
)

STATES = (
    "idle",
    "fresh_boot",
    "booting",
    "running",
    "restarting",
    "stopping",
    "crashed",
    "crash_loop_stopped",
)

DEFAULT_MAX_ATTEMPTS = 5


def log_path() -> Path:
    return Path.cwd() / "Outputs" / "log.txt"


def read_log_tail(max_lines: int = 20, path: Path | None = None) -> list[str]:
    """Read the last N lines, tolerating a missing/rotating/truncated log."""
    path = path or log_path()
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= max_lines:
                read_size = min(block, size)
                size -= read_size
                fh.seek(size)
                data = fh.read(read_size) + data
                block = min(block * 2, 65536)
        text = data.decode("utf-8", "replace")
        return text.splitlines()[-max_lines:]
    except Exception:
        return []


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def snapshot(include_log: bool = True, supervisor: bool = True) -> dict:
    """Merge supervisor + child state into the public JSON shape."""
    sup = read_supervisor_state()
    child = read_child_state()

    state = sup.get("state") or "idle"
    supervisor_pid = sup.get("pid")
    child_pid = child.get("pid")

    # Ignore a child_state left behind by an older child.  The supervisor
    # clears it on spawn/exit, but a crash (especially kill -9) can leave it.
    child_is_current = bool(child) and (
        supervisor_pid is None
        or child_pid is None
        or str(child_pid) == str(supervisor_pid)
    )
    if not child_is_current:
        child = {}

    # The child knows when the vision loop actually starts.  Until then the
    # supervisor's booting/restarting page is the correct one.
    child_status = child.get("status")
    if state in ("booting", "restarting") and child_status == "running":
        state = "running"

    attempt = _as_int(sup.get("attempt"), 0)
    max_attempts = _as_int(sup.get("max_attempts"), DEFAULT_MAX_ATTEMPTS)
    web_enabled = child.get("web_enabled", sup.get("web_enabled", True))
    phase = child.get("phase") or sup.get("phase") or ""
    web_off_reason = child.get("web_off_reason") or sup.get("web_off_reason") or ""

    payload = {
        "state": state,
        "phase": phase,
        "web_enabled": bool(web_enabled),
        "web_off_reason": web_off_reason,
        "pid": sup.get("pid"),
        "supervisor_pid": sup.get("supervisor_pid"),
        "attempt": attempt,
        "max_attempts": max_attempts,
        "attempt_label": f"{attempt}/{max_attempts}",
        "last_error": sup.get("last_error") or "",
        "message": sup.get("message") or "",
        "since": sup.get("since"),
        "updated": sup.get("updated") or child.get("updated"),
        "managed": bool(supervisor),
        # Which HTTP server answered this snapshot. The supervisor's pages use
        # this to reload when the child has taken over the port - even in the
        # brief window where the child still reports its previous phase.
        "served_by": "supervisor",
    }
    if include_log:
        payload["log_tail"] = read_log_tail()
    return payload


def child_snapshot(include_log: bool = True) -> dict:
    """State payload for the child Flask app.

    A child started outside the supervisor (``ispy-run``, direct
    ``python -m iSpy.boot.boot``, tests) is itself the running service.  Do not
    let a stale file from a previous supervisor run make its UI show the crash
    overlay.
    """
    if os.environ.get("ISPY_SUPERVISOR", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return {
            "state": "running",
            "phase": "running",
            "web_enabled": True,
            "web_off_reason": "",
            "pid": os.getpid(),
            "supervisor_pid": None,
            "attempt": 0,
            "max_attempts": DEFAULT_MAX_ATTEMPTS,
            "attempt_label": f"0/{DEFAULT_MAX_ATTEMPTS}",
            "last_error": "",
            "message": "",
            "since": None,
            "updated": None,
            "managed": False,
            "served_by": "child",
            "log_tail": read_log_tail() if include_log else [],
        }
    payload = snapshot(include_log=include_log, supervisor=True)
    payload["served_by"] = "child"
    return payload


def mark_supervisor_exit(final_state: str = "idle") -> None:
    update_supervisor_state(
        state=final_state,
        pid=None,
        control_port=None,
        attempt=0,
        last_error="",
        message="",
        phase="",
        web_enabled=True,
        web_off_reason="",
    )
    clear_child_state()


def current_state_file() -> Path:
    return supervisor_state_file_path()
