import os
import re
import sys
import logging
from pathlib import Path
from iSpy.dataset.dataset import validate_quantization_dataset

# for name in logging.root.manager.loggerDict:
#     logging.getLogger(name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_MODEL_PATTERN = re.compile(
    r"^YoloModels/"
    r"(?:(?:pytorch|onnx|tflite|rknn|openvino|coreml|engine)/)?"
    r"[a-zA-Z0-9_\-]+.*\.(pt|onnx|tflite|rknn|bin|xml|yaml|engine)$")

def is_valid_model_path(path: str) -> bool:
    return bool(_MODEL_PATTERN.match(path.replace("\\", "/")))

def validate_model_files() -> None:
    model_dir = Path("YoloModels")
    if not model_dir.exists():
        logger.warning("YoloModels directory not found - skipping model path validation.")
        return

    for root, dirs, files in os.walk(model_dir):
        dirs[:] = [d for d in dirs if not (d.endswith(".mlpackage") or d.endswith("_openvino_model"))]
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

def run_unit_tests() -> bool:
    logger.info("Running unit tests...")

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import unittest
    import logging
    # Suppress logging during unit tests to avoid spam in boot output.
    # Setting the root level is not enough: iSpy.* loggers get an explicit
    # INFO level in boot's quiet-logging setup, so their records ignore the
    # root threshold. logging.disable() is a global tripwire that every
    # logger respects regardless of its own level.
    logging.disable(logging.ERROR)
    try:
        loader = unittest.TestLoader()
        suite = loader.discover(
            start_dir=str(Path(__file__).parent),
            pattern="unit_tests.py",
        )
        with open(os.devnull, "w") as devnull:
            runner = unittest.TextTestRunner(verbosity=2, stream=devnull)
            result = runner.run(suite)
        return result.wasSuccessful()
    finally:
        logging.disable(logging.NOTSET)

def get_addon_setting(config: dict, addon_type: str, addon_name: str,
                      key: str, default=None):
    try:
        entry = config["plugins"][addon_type][addon_name]
    except (KeyError, TypeError):
        return default
    if not isinstance(entry, dict):
        return default
    return entry.get(key, default)


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

    required_fields = ["unit", "camera_configs"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    valid_units = {"meter", "meters", "inch", "inches", "foot", "feet", "centimeter", "centimeters", "frc"}
    if config.get("unit", "").lower() not in valid_units:
        raise ValueError(f"Invalid unit: {config.get('unit')}. Must be one of: {valid_units}")

    camera_configs = config.get("camera_configs", {})
    if not camera_configs:
        raise ValueError("camera_configs cannot be empty")

    model_cameras = 0
    for cam_name, cam_config in camera_configs.items():
        cam_required = ["name", "source", "subsystem", "pipeline"]
        for field in cam_required:
            if field not in cam_config:
                raise ValueError(f"Camera '{cam_name}' missing required field: {field}")

        calib = cam_config.get("calibration", {})
        if calib:
            calib_required = ["size", "distance", "game_piece_size", "fov"]
            for field in calib_required:
                if field not in calib:
                    raise ValueError(f"Camera '{cam_name}' calibration missing: {field}")

        # models live in the camera's pipeline settings (nested layout) - legacy flat entries are folded in lazily
        pipeline = cam_config.get("pipeline")
        if isinstance(pipeline, dict):
            pipeline_name = pipeline.get("name")
            settings = pipeline.get("settings") or {}
        else:
            pipeline_name = pipeline
            settings = {k: v for k, v in cam_config.items()
                        if k not in ("name", "source", "subsystem", "pipeline", "calibration")}
        if not pipeline_name:
            raise ValueError(f"Camera '{cam_name}' pipeline must have a name")

        vision_model = settings.get("vision_model")
        if isinstance(vision_model, dict):
            model_cameras += 1
            if "file_path" not in vision_model:
                raise ValueError(f"Camera '{cam_name}' vision_model must have 'file_path'")
            if "input_size" not in vision_model:
                raise ValueError(f"Camera '{cam_name}' vision_model must have 'input_size'")

    if model_cameras == 0:
        raise ValueError("No camera has a vision_model configured")

    # network_tables_ip is an add-on setting now (network_table_handler utility); only validated when the add-on is enabled
    ip = get_addon_setting(config, "utilities", "network_table_handler",
                           "network_tables_ip")
    if ip:
        ip_parts = ip.split(".")
        if len(ip_parts) != 4:
            raise ValueError(f"Invalid network_tables_ip format: {ip}")

    logger.info("Config validation passed.")

def get_recommendations(config_path: str = "iSpy/example_config.json") -> str:
    import json
    from pathlib import Path

    config_file = Path(config_path)
    if not config_file.exists():
        return "Config file not found. Cannot generate recommendations."

    with open(config_file) as f:
        data = json.load(f)

    config = data.get("config", data)
    recommendations = []

    # Add-on settings live in plugins.<type>.<name>; absent add-on = disabled.
    dbscan_epsilon = get_addon_setting(config, "trackers", "path_planner",
                                       "epsilon")
    dbscan_min_samples = get_addon_setting(config, "trackers", "path_planner",
                                           "min_samples")
    epsilon = dbscan_epsilon if dbscan_epsilon is not None else 0
    min_samples = dbscan_min_samples if dbscan_min_samples is not None else 0

    if dbscan_epsilon is None:
        recommendations.append(
            "PathPlanner tracker is not enabled - no clustering configured."
        )
    elif epsilon == 0:
        recommendations.append(
            "DBSCAN epsilon is 0 - clustering is disabled. "
            "Set 'dbscan.epsilon' to a positive value (e.g., 0.3 for meters) to enable clustering."
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

    dist_threshold = get_addon_setting(config, "trackers", "object_tracker",
                                       "distance_threshold")
    if dist_threshold is None:
        recommendations.append(
            "object_tracker tracker is not enabled - no object merging configured. "
            "Enable it and verify distance_threshold (default 0.5m) for your game pieces."
        )
    elif dist_threshold < 0:
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

    # Models are configured per camera under their pipeline settings.
    seen_paths = set()
    for cam_name, cam_cfg in camera_configs.items():
        pipeline = cam_cfg.get("pipeline")
        if isinstance(pipeline, dict):
            settings = pipeline.get("settings") or {}
        else:
            settings = {k: v for k, v in cam_cfg.items()
                        if k not in ("name", "source", "subsystem", "pipeline", "calibration")}
        vision_model = settings.get("vision_model")
        if not isinstance(vision_model, dict):
            continue
        model_path = vision_model.get("file_path", "model.pt")
        if model_path in seen_paths:
            continue
        seen_paths.add(model_path)
        if not Path(model_path).exists():
            recommendations.append(
                f"Model file not found: {model_path} (camera '{cam_name}'). "
                "Verify the path exists or update the camera's pipeline vision_model."
            )

        input_size = vision_model.get("input_size", [640, 640])
        if input_size[0] != input_size[1]:
            recommendations.append(
                f"Vision model input_size is non-square {input_size} (camera '{cam_name}'). "
                "Most models expect square input. This may cause issues."
            )

    nt_ip = get_addon_setting(config, "utilities", "network_table_handler",
                              "network_tables_ip")
    if nt_ip is None:
        recommendations.append(
            "NetworkTables utility not enabled - vision data is not published to the robot."
        )
    elif nt_ip == "10.22.7.2":
        recommendations.append(
            "NetworkTables IP is default (10.22.7.2). "
            "Verify this matches your robot's IP address."
        )

    stale = config.get("health_stale_threshold")
    if stale is None:
        stale = get_addon_setting(config, "trackers",
                                  "object_tracker",
                                  "stale_threshold", 1.0)
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

def validate_quantization_dataset_wrapper(dataset_path: str = "QuantizeDataset") -> bool:
    root = Path(dataset_path)

    from iSpy.dataset.dataset import _find_images

    if not root.exists() or not _find_images(root):
        logger.warning(
            "No quantization dataset found at %s - skipping (only needed for "
            "RKNN conversion, which runs on demand, not at boot).", dataset_path,
        )
        return True

    # named datasets live at QuantizeDataset/<name>/ - reusable calibration picked per camera via its 'quantization_dataset' setting, never derived from a model filename. validate each one since that's what conversions actually use now
    per_model_dirs = [
        d for d in root.iterdir()
        if d.is_dir() and d.name not in ("valid",) and (d / "dataset.txt").exists()
    ] if root.exists() else []

    if not per_model_dirs:
        result = validate_quantization_dataset(dataset_path)
        if result["valid"]:
            logger.info(
                "Quantization dataset valid: %d images, rknn=%s, yolo_data=%s",
                result["image_count"], result["rknn_ready"], result["yolo_data_ready"],
            )
        else:
            logger.warning("Quantization dataset issues (%s):", dataset_path)
            for issue in result["issues"]:
                logger.warning("  - %s", issue)
        return result["valid"]

    all_valid = True
    for model_dir in per_model_dirs:
        result = validate_quantization_dataset(str(model_dir))
        if result["valid"]:
            logger.info(
                "Quantization dataset valid for %s: %d images, rknn=%s, yolo_data=%s",
                model_dir.name, result["image_count"], result["rknn_ready"], result["yolo_data_ready"],
            )
        else:
            logger.warning("Quantization dataset issues (%s):", model_dir)
            for issue in result["issues"]:
                logger.warning("  - %s", issue)
            all_valid = False

    return all_valid

def validate_system() -> bool:
    try:
        validate_model_files()
        validate_config_files()

        if not validate_quantization_dataset_wrapper():
            raise RuntimeError("Quantization dataset validation failed.")

        if not run_unit_tests():
            raise RuntimeError("Unit tests failed during boot validation.")
        logger.info("System validation successful.")
        return True

    except Exception as e:
        logger.error("System validation failed: %s", e)
        return False