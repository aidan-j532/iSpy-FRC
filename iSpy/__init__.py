import os

# kill opencv log spam before cv2 is imported. The v4l2 backend only reports
# availability failures (like QUERYCAP on a non-camera /dev/video* node, or the
# obsensor enumeration spam) through its logger - and the level it uses varies
# by OpenCV build, so 'ERROR' is NOT enough (ioctl(VIDIOC_QUERYCAP) messages
# leak through it). Errors still surface via isOpened()/return codes at the
# callers, so hard-disable the whole OpenCV logger on embedded boards.
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_LOG_LEVEL"] = "SILENT"


def _detect_version() -> str:
    # pyproject.toml is the single source of truth. Prefer the installed
    # distribution metadata; fall back to parsing pyproject directly for source
    # checkouts run straight from a clone without `pip install`.
    try:
        from importlib.metadata import version

        return version("ispy-frc")
    except Exception:
        pass
    try:
        import re

        pyproject = os.path.join(os.path.dirname(__file__), os.pardir, "pyproject.toml")
        with open(pyproject, "r", encoding="utf-8") as fh:
            match = re.search(
                r"""^\s*version\s*=\s*["']([^"']+)["']""", fh.read(), re.MULTILINE
            )
        if match:
            return match.group(1)
    except Exception:
        pass
    return "0.0.0+unknown"


__version__ = _detect_version()
