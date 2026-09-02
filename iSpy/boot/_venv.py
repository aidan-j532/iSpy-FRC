"""Single source of truth for the iSpy Python interpreter.

BUG 3 fix: setup_service.py baked ``sys.executable`` (the install-time
interpreter) into the systemd / schtasks / launchd units, while watchdog.py
and service_daemon.py independently resolved ``sys.executable`` again at
their own runtime. Whenever a different entrypoint launched them (a bare
``python3`` on PATH, a cron line, a hand-edited unit), the venv silently
diverged and ispy-run (game_loop) crashed on missing deps while the
ispy-boot unit kept working.

Every launcher in the boot/watchdog/service chain now reads ONE canonical
interpreter path from a marker file written at install time
(``<repo root>/.ispy-python``), so children are always spawned with the
same interpreter iSpy was installed into. Before a marker exists (plain
dev/manual runs) the invoking interpreter itself is used.

Stdlib-only on purpose: this must import safely even when watchdog.py is
launched with a bare system python that has no iSpy dependencies.
"""

import os
import sys
from pathlib import Path

MARKER_NAME = ".ispy-python"

# <repo root>/iSpy/boot/_venv.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return _REPO_ROOT


def marker_path() -> Path:
    return _REPO_ROOT / MARKER_NAME


def _is_executable(python: str | None) -> bool:
    if not python:
        return False
    if os.name == "nt":
        return os.path.isfile(python)
    return os.path.isfile(python) and os.access(python, os.X_OK)


def read_marked_python() -> str | None:
    """Interpreter path recorded at install time, or None if absent/unusable."""
    try:
        recorded = marker_path().read_text(encoding="utf-8").strip().splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    if not _is_executable(recorded):
        return None
    return recorded


def record_python(python: str) -> Path:
    """Persist the canonical interpreter this install was made with."""
    marker = marker_path()
    marker.write_text(python.strip() + "\n", encoding="utf-8")
    return marker


def resolve_launch_python(fallback: str | None = None) -> str:
    """Canonical interpreter for spawning child processes.

    The install-time marker wins so the whole chain stays on one venv, even
    when the outer launcher was a bare ``python3``. Falls back to the
    invoking interpreter for dev/manual runs before any install wrote a
    marker.
    """
    marked = read_marked_python()
    if marked:
        return marked
    return fallback if fallback else sys.executable