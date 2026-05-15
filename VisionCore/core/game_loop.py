from pathlib import Path
import logging
from VisionCore.VisionCore import VisionCore
from VisionCore.config.VisionCoreConfig import VisionCoreConfig
from VisionCore.validations.ez import unit_tests
from VisionCore.validations.model_validator import enforce_model_organization
from VisionCore.plugins._loader import load_plugins
from VisionCore.plugins.bases import VisionBase
import VisionCore.plugins as _plugins_pkg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    plugin_root = Path(_plugins_pkg.__file__).resolve().parent

    vision_classes = load_plugins(plugin_root / "vision", VisionBase)

    config_path = repo_root / "Config" / "config.json"
    logger.info(f"Using config file: {config_path}")

    config = VisionCoreConfig(str(config_path))

    logger.info("Validating YOLO model organization...")
    is_valid, corrected_model_path = enforce_model_organization(
        repo_root, config.config
    )

    if corrected_model_path:
        config.config["vision_model"]["file_path"] = corrected_model_path

    cameras = []
    for cam_name in config.camera_configs:
        cam_config = config.camera_config(cam_name)
        pipeline = cam_config.get("pipeline", "object_detection")

        if pipeline in vision_classes:
            cameras.append(vision_classes[pipeline](cam_config, config))

    vision = VisionCore(cameras, config)
    vision.run()

if __name__ == "__main__":
    if not unit_tests():
        raise SystemExit("Unit tests failed")
    main()
