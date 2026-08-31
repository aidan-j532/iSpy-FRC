import json
import logging
import threading
from pathlib import Path

_SAVE_DIR = Path.cwd() / "Save"
_lock = threading.Lock()
logger = logging.getLogger(__name__)

def _path(key: str) -> Path:
    _SAVE_DIR.mkdir(parents=True, exist_ok=True)
    return _SAVE_DIR / f"{key}.json"

def read(key: str, default=None):
    p = _path(key)
    if not p.exists():
        return default
    try:
        with _lock:
            return json.loads(p.read_text())
    except Exception as e:
        logger.error(f"Failed to read {key}: {e}")
        return default

def write(key: str, data) -> None:
    with _lock:
        _path(key).write_text(json.dumps(data, indent=2))