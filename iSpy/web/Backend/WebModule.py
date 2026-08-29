import shutil
from abc import ABC
from pathlib import Path

# Uploads that would leave the filesystem with less than this much free space
# are rejected up-front instead of corrupting/failing mid-write.
_MIN_FREE_BYTES = 100 * 1024 * 1024


def ensure_disk_space(dest: Path, required_bytes: int = 0) -> str | None:
    """Reject writes that would leave < _MIN_FREE_BYTES free on *dest*.

    Returns an error string when the write should be refused, or None when
    there is (apparently) enough room. Checked before an upload handler writes
    a payload so a full disk can never truncate a model/dataset/plugin mid-save.
    """
    try:
        target = Path(dest)
        if not target.exists():
            target = target.parent
        usage = shutil.disk_usage(str(target))
    except OSError as e:
        return f"Could not check free disk space: {e}"
    needed = max(int(required_bytes or 0) + _MIN_FREE_BYTES, _MIN_FREE_BYTES)
    if usage.free < needed:
        free_mb = usage.free / (1024 * 1024)
        margin_mb = _MIN_FREE_BYTES // (1024 * 1024)
        return (
            f"Not enough free disk space for that upload: only {free_mb:.0f} MB "
            f"free (a {margin_mb} MB safety margin is required)."
        )
    return None


class WebModule(ABC):
    plugin_name = "base_web_module"

    def __init__(self, context: dict):
        self.context = context  # {"config": ..., "cameras": ..., "flask_app": ...}

    def register_routes(self, flask_app):
        pass

    def update(self, frame_data: dict):
        pass

    def stop(self):
        pass