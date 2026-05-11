from pathlib import Path
import importlib.metadata
from VisionCore.VisionCore import VisionCore
from VisionCore.config.VisionCoreConfig import VisionCoreConfig
from VisionCore.validations.ez import unit_tests
from VisionCore.validations.model_validator import enforce_model_organization
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    repo_root = Path.cwd()
    
    config_path = (
        os.environ.get("VISIONCORE_CONFIG")
        or repo_root / "config.json"
    )
    logger.info(f"Using config file: {config_path}")
    config = VisionCoreConfig(str(config_path))

    # Validate YOLO model organization
    logger.info("Validating YOLO model organization...")
    is_valid, corrected_model_path = enforce_model_organization(repo_root, config.config)
    
    if not is_valid:
        logger.error("YOLO model validation failed")
        # raise RuntimeError(
        #     "YOLO model organization validation failed. "
        #     "Ensure models are in YoloModels/[format]/[size]/ structure."
        # )
    
    if corrected_model_path:
        config.config["vision_model"]["file_path"] = corrected_model_path
        # logger.info("Using model from filesystem: %s", corrected_model_path)

    # Load vision modules dynamically
    vision_entries = importlib.metadata.entry_points(group='visioncore_vision')
    vision_classes = {ep.name: ep.load() for ep in vision_entries}

    cameras = []
    for cam_name in config.camera_configs:
        cam_config = config.camera_config(cam_name)
        pipeline = cam_config.get('pipeline', 'object_detection')
        if pipeline in vision_classes:
            vision_class = vision_classes[pipeline]
            camera = vision_class(cam_config, config)
            cameras.append(camera)
        else:
            logger.warning(f"Unknown vision pipeline: {pipeline}")

    vision = VisionCore(cameras, config)
    vision.run()

if __name__ == "__main__":
    if not unit_tests():
        raise SystemExit("Unit tests failed")
    main()