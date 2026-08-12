from pathlib import Path
import logging
import sys
import os
import threading

# Must run before any cv2 import so OpenCV's own MSMF/DSHOW warnings are
# suppressed (see iSpy/__init__.py).
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
            return name == "root" or name.startswith("iSpy")

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
        if not name.startswith("iSpy"):
            logging.getLogger(name).setLevel(logging.WARNING)


_configure_quiet_logging()

from iSpy.iSpy import iSpy
from iSpy.config.iSpyConfig import iSpyConfig
from iSpy.validations.ez import unit_tests
from iSpy.validations.model_validator import enforce_model_organization
from iSpy.vision.pipelines import get_pipeline_classes

logger = logging.getLogger(__name__)

_pause_event = threading.Event()
_shutdown_event = threading.Event()


def _stdin_reader(pause_event: threading.Event, shutdown_event: threading.Event):
    try:
        for line in sys.stdin:
            cmd = line.strip().upper()
            if cmd == "PAUSE":
                pause_event.set()
                logger.info("Vision paused by service")
            elif cmd == "RESUME":
                pause_event.clear()
                logger.info("Vision resumed by service")
            elif cmd == "SHUTDOWN":
                shutdown_event.set()
                logger.info("Shutdown command received from service")
                break
    except Exception:
        pass

def main():
    repo_root = Path.cwd()

    config_path = repo_root / "Config" / "config.json"
    logger.info(f"Using config file: {config_path}")

    config = iSpyConfig(str(config_path))

    # Boot the web dashboard FIRST - it is the top priority and must be
    # reachable while the vision stack is still initializing (camera open
    # and model loading can take minutes). Until the vision loop below
    # starts producing ticks, the dashboard shows a red
    # "Vision is not running" banner.
    prebuilt_web = None
    if config.config.get("app_mode", False):
        from iSpy.web.Backend.WebApp import create_app
        prebuilt_web = create_app(cameras=[], config=config)
        threading.Thread(
            target=prebuilt_web.run, daemon=True, name="web-app"
        ).start()
        logger.info("Web dashboard started - vision is still booting.")

    is_valid, corrected_model_path = enforce_model_organization(repo_root, config.config)
    if corrected_model_path:
        from iSpy.config.iSpyConfig import get_pipeline_settings
        for cam_cfg in config.config["camera_configs"].values():
            if not isinstance(cam_cfg, dict):
                continue
            vm = get_pipeline_settings(cam_cfg).get("vision_model")
            if isinstance(vm, dict):
                vm["file_path"] = corrected_model_path

    pipeline_classes = get_pipeline_classes()
    cameras = []
    for cam_name in config.camera_configs:
        cam_config = config.camera_config(cam_name)
        pipeline = cam_config.pipeline_name()
        cls = pipeline_classes.get(pipeline)
        if cls is None:
            logger.warning("Unknown pipeline '%s' for camera '%s'", pipeline, cam_name)
            continue
        cameras.append(cls(cam_config, config))

    if not cameras:
        logger.error("No cameras configured or detected. Cannot run iSpy.")
        sys.exit(1)

    vision = iSpy(cameras, config, web_app=prebuilt_web)

    reader_thread = threading.Thread(
        target=_stdin_reader,
        args=(vision.pause_event, vision.shutdown_event),
        daemon=True,
        name="stdin-reader",
    )
    reader_thread.start()

    vision.run()

if __name__ == "__main__":
    if not unit_tests():
        raise SystemExit("Unit tests failed")
    main()
