
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
    try:
        recorded = marker_path().read_text(encoding="utf-8").strip().splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    if not _is_executable(recorded):
        return None
    return recorded


def record_python(python: str) -> Path:
    marker = marker_path()
    marker.write_text(python.strip() + "\n", encoding="utf-8")
    return marker


def resolve_launch_python(fallback: str | None = None) -> str:
    marked = read_marked_python()
    if marked:
        return marked
    return fallback if fallback else sys.executable