import os

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_LOG_LEVEL"] = "SILENT"

def _detect_version() -> str:
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
