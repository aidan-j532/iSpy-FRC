import subprocess
from pathlib import Path
import platform

def _resolve_device_id(path: str) -> str | None:
    if platform.system() != "Linux":
        return path  # best effort elsewhere - at least index-stable within a session
    try:
        by_id_dir = Path("/dev/v4l/by-id")
        if by_id_dir.exists():
            target = Path(path).resolve()
            for link in by_id_dir.iterdir():
                if link.resolve() == target:
                    return link.name  # e.g. usb-Microsoft_Microsoft_LifeCam_HD-3000-video-index0
        # fall back to udevadm vendor:product:serial
        out = subprocess.run(
            ["udevadm", "info", "--query=property", "--name", path],
            capture_output=True, text=True, timeout=3,
        )
        props = dict(l.split("=", 1) for l in out.stdout.splitlines() if "=" in l)
        vid, pid, serial = props.get("ID_VENDOR_ID"), props.get("ID_MODEL_ID"), props.get("ID_SERIAL_SHORT")
        if vid and pid:
            return f"{vid}:{pid}:{serial or ''}"
    except Exception:
        pass
    return None