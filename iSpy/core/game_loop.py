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


def _init_offline_mode(config) -> None:
    # both launch paths (ispy start, and the service daemon running this module
    # directly) end up here - the daemon never calls boot.on_boot
    from iSpy.core import network_policy

    if network_policy.is_offline():
        # boot already armed it (Game Mode) or ISPY_OFFLINE came from a parent
        return
    try:
        enabled = bool(config.get("offline_mode", False))
    except Exception:
        enabled = False
    if enabled:
        network_policy.set_offline(True, "config")


def _init_performance_level(config) -> None:
    # same reason as _init_offline_mode: the daemon path never calls on_boot,
    # so the level has to be resolved here too
    try:
        from iSpy.core import performance

        performance.arm_from_config(config)
    except Exception:
        logger.exception(
            "Could not resolve the performance profile - running at Full Features."
        )


def main(config=None):
    # a replay passes the config boot already built in memory; anything else
    # loads its own from disk
    if config is None:
        config_path = Path.cwd() / "Config" / "config.json"
        logger.info(f"Using config file: {config_path}")

        config = iSpyConfig(str(config_path))
    _init_offline_mode(config)
    _init_performance_level(config)
    vision = iSpy(config)
    vision.run()


if __name__ == "__main__":
    main()
