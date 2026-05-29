from pathlib import Path
import logging
from iSpy.iSpy import iSpy
from iSpy.config.iSpyConfig import iSpyConfig
from iSpy.validations.ez import unit_tests
from iSpy.validations.model_validator import enforce_model_organization
from iSpy.plugins._loader import load_plugins
from iSpy.plugins.bases import VisionBase
import sys

for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def main():
    repo_root = Path.cwd()

    plugin_root = repo_root / "iSpy" / "plugins"

    vision_classes = load_plugins(plugin_root / "vision", VisionBase)

    config_path = repo_root / "Config" / "config.json"
    logger.info(f"Using config file: {config_path}")

    config = iSpyConfig(str(config_path))

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

    vision = iSpy(cameras, config)
    vision.run()

if __name__ == "__main__":
    if not unit_tests():
        raise SystemExit("Unit tests failed")
    main()