import os
import re
import sys
import logging
from pathlib import Path
from VisionCore.dataset.dataset import validate_quantization_dataset

# for name in logging.root.manager.loggerDict:
#     logging.getLogger(name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_MODEL_PATTERN = re.compile(
r"^YoloModels/" # starts with YoloModels/
r"(?:(?:pytorch|onnx|tflite|rknn|openvino|coreml)/)?" # optional format folder
r"[a-zA-Z0-9_\-]+.*\.(pt|onnx|tflite|rknn|bin|xml|yaml)$")

def is_valid_model_path(path: str) -> bool:
    return bool(_MODEL_PATTERN.match(path.replace("\\", "/")))

def validate_model_files() -> None:
    model_dir = Path("YoloModels")
    if not model_dir.exists():
        logger.warning("YoloModels directory not found - skipping model path validation.")
        return

    for root, _, files in os.walk(model_dir):
        for file in files:
            full_path = os.path.join(root, file)
            if not is_valid_model_path(full_path):
                raise ValueError(f"Invalid model file path: {full_path}")

    logger.info("All model file paths are valid.")

def validate_config_files() -> None:
    config_dir = Path("Config")
    if not config_dir.exists():
        logger.warning("Config directory not found - skipping Config file validation.")
        return

    for root, _, files in os.walk(config_dir):
        for file in files:
            if not file.endswith(".json"):
                raise ValueError(f"Invalid config file: {file}. Only .json files are allowed.")

    logger.info("All config files are valid.")

def run_unit_tests() -> None:
    logger.info("Running unit tests...")

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(Path(__file__).parent),
        pattern="unit_tests.py",
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Unit tests failed.")

    logger.info("All unit tests passed.")

def validate_config_required_fields(config_path: str = "Config/config.json") -> None:
    import json
    from pathlib import Path

    # Create if doesnt exist
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}")
        return

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file) as f:
        data = json.load(f)

    config = data.get("config", data)

    required_fields = ["unit", "vision_model", "camera_configs"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    valid_units = {"meter", "meters", "inch", "inches", "foot", "feet", "centimeter", "centimeters"}
    if config.get("unit", "").lower() not in valid_units:
        raise ValueError(f"Invalid unit: {config.get('unit')}. Must be one of: {valid_units}")

    vision_model = config.get("vision_model", {})
    if not vision_model:
        raise ValueError("vision_model config is empty")
    if "file_path" not in vision_model:
        raise ValueError("vision_model must have 'file_path'")
    if "input_size" not in vision_model:
        raise ValueError("vision_model must have 'input_size'")

    camera_configs = config.get("camera_configs", {})
    if not camera_configs:
        raise ValueError("camera_configs cannot be empty")

    for cam_name, cam_config in camera_configs.items():
        cam_required = ["name", "source", "subsystem"]
        for field in cam_required:
            if field not in cam_config:
                raise ValueError(f"Camera '{cam_name}' missing required field: {field}")

        calib = cam_config.get("calibration", {})
        if calib:
            calib_required = ["size", "distance", "game_piece_size", "fov"]
            for field in calib_required:
                if field not in calib:
                    raise ValueError(f"Camera '{cam_name}' calibration missing: {field}")

    ip = config.get("network_tables_ip")
    if ip:
        ip_parts = ip.split(".")
        if len(ip_parts) != 4:
            raise ValueError(f"Invalid network_tables_ip format: {ip}")

    logger.info("Config validation passed.")

def get_recommendations(config_path: str = "VisionCore/example_config.json") -> str:
    import json
    from pathlib import Path

    config_file = Path(config_path)
    if not config_file.exists():
        return "Config file not found. Cannot generate recommendations."

    with open(config_file) as f:
        data = json.load(f)

    config = data.get("config", data)
    recommendations = []

    dbscan = config.get("dbscan", {})
    epsilon = dbscan.get("elipson", 0)
    min_samples = dbscan.get("min_samples", 0)

    if epsilon == 0:
        recommendations.append(
            "DBSCAN epsilon is 0 - clustering is disabled. "
            "Set 'dbscan.elipson' to a positive value (e.g., 0.3 for meters) to enable clustering."
        )
    elif epsilon < 0.1:
        recommendations.append(
            f"DBSCAN epsilon is very small ({epsilon}). "
            "This may not cluster nearby detections. Consider increasing (e.g., 0.3-0.5)."
        )
    elif epsilon > 2.0:
        recommendations.append(
            f"DBSCAN epsilon is large ({epsilon}). "
            "Distant detections may be grouped together. Consider reducing."
        )

    if min_samples == 0:
        recommendations.append(
            "DBSCAN min_samples is 0 - noise filtering disabled. "
            "Set to at least 2 to filter out single-point clusters."
        )
    elif min_samples > 3:
        recommendations.append(
            f"DBSCAN min_samples is high ({min_samples}). "
            "Only dense clusters will be kept. May miss sparse detections."
        )

    dist_threshold = config.get("distance_threshold", -1)
    if dist_threshold is None or dist_threshold < 0:
        recommendations.append(
            "distance_threshold is negative/unset - using default 0.5m. "
            "Verify this merge distance works for your game pieces."
        )
    elif dist_threshold < 0.1:
        recommendations.append(
            f"distance_threshold is very small ({dist_threshold}m). "
            "Detections may not merge properly. Consider 0.3-0.5m."
        )
    elif dist_threshold > 1.5:
        recommendations.append(
            f"distance_threshold is large ({dist_threshold}m). "
            "Different game pieces may incorrectly merge. Consider 0.3-0.5m."
        )

    camera_configs = config.get("camera_configs", {})
    for cam_name, cam_cfg in camera_configs.items():
        calib = cam_cfg.get("calibration", {})

        if calib.get("size", 0) == 0 and calib.get("distance", 0) == 0:
            recommendations.append(
                f"Camera '{cam_name}' calibration is zero - distance estimates will be inaccurate. "
                "Run camera calibration and set calibration.size, calibration.distance, "
                "calibration.game_piece_size, and calibration.fov."
            )

        if cam_cfg.get("x", 0) == 0 and cam_cfg.get("y", 0) == 0:
            recommendations.append(
                f"Camera '{cam_name}' position is (0,0) - is this intentional? "
                "Set camera x, y for accurate field-relative positioning."
            )

        if cam_cfg.get("height", 0) == 0:
            recommendations.append(
                f"Camera '{cam_name}' height is 0 - distance calculations may be wrong. "
                "Set camera height for accurate distance estimation."
            )

        fps_cap = cam_cfg.get("fps_cap", -1)
        if fps_cap == -1:
            recommendations.append(
                f"Camera '{cam_name}' has no FPS cap (unlimited). "
                "Consider setting fps_cap (e.g., 30) to reduce CPU load."
            )

    vision_model = config.get("vision_model", {})
    model_path = vision_model.get("file_path", "model.pt")
    if not Path(model_path).exists():
        recommendations.append(
            f"Model file not found: {model_path}. "
            "Verify the path exists or update vision_model.file_path."
        )

    input_size = vision_model.get("input_size", [640, 640])
    if input_size[0] != input_size[1]:
        recommendations.append(
            f"Vision model input_size is non-square {input_size}. "
            "Most models expect square input. This may cause issues."
        )

    ip = config.get("network_tables_ip", "")
    if ip == "10.22.7.2":
        recommendations.append(
            "NetworkTables IP is default (10.22.7.2). "
            "Verify this matches your robot's IP address."
        )

    stale = config.get("stale_threshold", 1.0)
    if stale > 3.0:
        recommendations.append(
            f"stale_threshold is high ({stale}s). "
            "Old detections may persist too long. Consider 1.0-2.0s."
        )
    elif stale < 0.5:
        recommendations.append(
            f"stale_threshold is low ({stale}s). "
            "Detections may disappear too quickly. Consider 1.0-2.0s."
        )

    if not recommendations:
        return "All config parameters look good! No critical issues found."

    output = "=" * 60 + "\n"
    output += "PRE-DEPLOYMENT RECOMMENDATIONS\n"
    output += "=" * 60 + "\n\n"
    output += "Review these items before deploying:\n\n"
    for i, rec in enumerate(recommendations, 1):
        output += f"{i}. {rec}\n\n"
    output += "=" * 60 + "\n"
    output += "Run validate_system() for full validation.\n"

    return output

def validate_quantization_dataset_wrapper(dataset_path: str = "dataset") -> bool:
    result = validate_quantization_dataset(dataset_path)
    if result["valid"]:
        logger.info(
            "Quantization dataset valid: %d images, rknn=%s, ultralytics=%s",
            result["image_count"],
            result["rknn_ready"],
            result["ultralytics_ready"],
        )
    else:
        logger.warning("Quantization dataset issues (%s):", dataset_path)
        for issue in result["issues"]:
            logger.warning("  - %s", issue)
    return result["valid"]


def validate_system(first_boot: bool = False) -> bool:
    try:
        if not first_boot:
            validate_model_files()
            validate_config_files()

        validate_quantization_dataset_wrapper()

        run_unit_tests()
        logger.info("System validation successful.")
        return True

    except Exception as e:
        logger.error("System validation failed: %s", e)
        return False