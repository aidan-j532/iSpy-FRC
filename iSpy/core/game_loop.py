from pathlib import Path
import logging
import sys
import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


def _configure_quiet_logging() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(logging.INFO)
    root.propagate = False

    class _iSpyLogFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            name = record.name or ""
            return name == "root" or name == "__main__" or name.startswith("iSpy")

    formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(_iSpyLogFilter())
    root.addHandler(stream_handler)

    log_path = Path.cwd() / "Outputs" / "log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_iSpyLogFilter())
    root.addHandler(file_handler)

    logging.getLogger("iSpy").setLevel(logging.INFO)
    for name in list(logging.Logger.manager.loggerDict):
        if name != "__main__" and not name.startswith("iSpy"):
            logging.getLogger(name).setLevel(logging.WARNING)


_configure_quiet_logging()

from iSpy.iSpy import iSpy
from iSpy.config.iSpyConfig import iSpyConfig

logger = logging.getLogger("iSpy.core.game_loop")


def main():
    config_path = Path.cwd() / "Config" / "config.json"
    logger.info(f"Using config file: {config_path}")

    config = iSpyConfig(str(config_path))
    vision = iSpy(config)
    vision.run()


if __name__ == "__main__":
    main()
