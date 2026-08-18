import re
import shutil
import site
import subprocess
import sys
import sysconfig
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def cv2_has_gstreamer() -> bool:
    try:
        import cv2
        return bool(re.search(r"GStreamer:\s*YES", cv2.getBuildInformation()))
    except Exception:
        return False


def _system_python() -> str:
    if sys.platform == "win32":
        return str(Path(sys.base_prefix) / "python.exe")
    return str(Path(sys.base_prefix) / "bin" / "python3")


def _apt_install_python3_opencv() -> bool:
    env = {"DEBIAN_FRONTEND": "noninteractive"}
    cmd = ["sudo", "apt-get", "install", "-y", "python3-opencv"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        if result.returncode == 0:
            return True
        logger.warning("python3-opencv install failed, retrying after apt-get update")
        subprocess.run(["sudo", "apt-get", "update"], capture_output=True, text=True, timeout=300, env=env)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        return result.returncode == 0
    except Exception as e:
        logger.error("apt install of python3-opencv failed: %s", e)
        return False


def _find_system_cv2_path() -> Path | None:
    try:
        result = subprocess.run(
            [_system_python(), "-c",
             "import cv2, os; print(os.path.dirname(os.path.abspath(cv2.__file__)))"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        path = Path(result.stdout.strip())
        if path.name == "cv2" and (path / "__init__.py").exists():
            return path  # package-style install
        candidates = list(path.glob("cv2*.so")) + list(path.glob("cv2*.pyd")) # Added .pyd for Windows safety
        return candidates[0] if candidates else None
    except Exception:
        return None


def _current_cv2_targets() -> tuple[Path, list[Path]]:
    purelib = Path(sysconfig.get_paths()["purelib"])
    search_dirs = {purelib}
    if hasattr(site, "getsitepackages"):
        search_dirs.update(map(Path, site.getsitepackages()))
    if hasattr(site, "getusersitepackages"):
        search_dirs.add(Path(site.getusersitepackages()))

    existing = []
    for d in search_dirs:
        if not d.exists():
            continue
        if (d / "cv2").exists():
            existing.append(d / "cv2")
        existing.extend(d.glob("cv2*.so"))
        existing.extend(d.glob("cv2*.pyd")) # Added .pyd for Windows safety
        existing.extend(d.glob("opencv_python*"))
    return purelib, existing


def ensure_csi_capable_opencv(auto_fix: bool = True) -> bool:
    if sys.platform != "linux":
        return False

    if cv2_has_gstreamer():
        return False

    logger.warning("Current OpenCV has no GStreamer support - CSI cameras won't work.")
    if not auto_fix:
        return False

    if not _apt_install_python3_opencv():
        logger.error("Could not apt-install python3-opencv. CSI capture unavailable.")
        return False

    system_cv2 = _find_system_cv2_path()
    if system_cv2 is None:
        logger.error("apt install succeeded but couldn't locate the resulting cv2 module.")
        return False

    check = subprocess.run(
        [_system_python(), "-c", "import cv2; print(cv2.getBuildInformation())"],
        capture_output=True, text=True, timeout=30,
    )
    if not re.search(r"GStreamer:\s*YES", check.stdout):
        logger.error("apt's python3-opencv build lacks GStreamer too - can't auto-fix.")
        return False

    target_dir, existing = _current_cv2_targets()
    for old in existing:
        try:
            shutil.rmtree(old) if old.is_dir() else old.unlink()
        except Exception as e:
            logger.warning("Could not remove old cv2 artifact %s: %s", old, e)

    if system_cv2.is_dir():
        shutil.copytree(system_cv2, target_dir / "cv2")
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(system_cv2, target_dir / system_cv2.name)

    logger.info("Vendored GStreamer-enabled cv2 (from apt) into %s", target_dir)
    return True