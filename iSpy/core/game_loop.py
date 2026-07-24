from pathlib import Path
import logging
import sys


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
from iSpy.plugins._loader import load_plugins
from iSpy.plugins.bases import VisionBase
import iSpy.plugins as _plugins_pkg
from iSpy.vision.ObjectDetectionCamera import ObjectDetectionCamera

logger = logging.getLogger(__name__)

def main():
    repo_root = Path.cwd()
    plugin_root = Path(_plugins_pkg.__file__).resolve().parent

    vision_classes = load_plugins(plugin_root / "vision", VisionBase)

    config_path = repo_root / "Config" / "config.json"
    logger.info(f"Using config file: {config_path}")

    config = iSpyConfig(str(config_path))

    is_valid, corrected_model_path = enforce_model_organization(repo_root, config.config)
    if corrected_model_path:
        config.config["vision_model"]["file_path"] = corrected_model_path

    cameras = []
    for cam_name in config.camera_configs:
        cam_config = config.camera_config(cam_name)
        pipeline = cam_config.get("pipeline", "object_detection")

        if pipeline == "object_detection":
            cameras.append(ObjectDetectionCamera(cam_config, config))
        elif pipeline in vision_classes:
            cameras.append(vision_classes[pipeline](cam_config, config))
        else:
            logger.warning("Unknown pipeline '%s' for camera '%s'", pipeline, cam_name)

    if not cameras:
        logger.error("No cameras configured or detected. Cannot run iSpy.")
        sys.exit(1)

    vision = iSpy(cameras, config)
    vision.run()

if __name__ == "__main__":
    if not unit_tests():
        raise SystemExit("Unit tests failed")
    main()
