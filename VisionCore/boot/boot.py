import sys
import os
from pathlib import Path

_BOOT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BOOT_DIR.parents[1] 

if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))
import subprocess
from VisionCore.config.AutoOpt import recommend_format
from VisionCore.validations.validate_system import validate_system
from VisionCore.validations.model_validator import enforce_model_organization
from VisionCore.config.VisionCoreConfig import VisionCoreConfig
import logging
import ultralytics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FORMAT_EXTENSIONS = {
    "onnx":      ".onnx",
    "openvino":  ".xml",
    "rknn":      ".rknn",
    "tflite":    ".tflite",
    "coreml":    ".mlpackage",
}

def search_for_config() -> str:
    config_dir = _REPO_ROOT / "config"
    if not config_dir.exists():
        raise FileNotFoundError(f"config directory not found at {config_dir}")

    config_files = list(config_dir.rglob("*.json"))
    if not config_files:
        return None
        # raise FileNotFoundError("No .json config files found in config/")

    chosen = str(config_files[0])
    logger.info("Found config files: %s -> using %s", len(config_files), chosen)
    return chosen

def convert_model(model_file, target_format, input_size):
    # Check if file exists
    if not os.path.exists(model_file):
        logger.warning(f"Model file {model_file} is missing or empty. Skipping conversion.")
        return model_file

    if Path(model_file).suffix.lower() != '.pt':
        logger.warning(f"Model file {model_file} is not a valid .pt file. Skipping conversion.")
        return model_file

    stem = Path(model_file).stem
    parent = Path(model_file).parent

    ext_map = {
        "rknn": f"{stem}.rknn",
        "onnx": f"{stem}.onnx",
        "tflite": f"{stem}_saved_model/{stem}_full_integer_quant.tflite",
        "openvino": f"{stem}_openvino_model",
        "coreml": f"{stem}.mlpackage",
    }

    if target_format not in ext_map:
        return model_file

    out_path = parent / ext_map[target_format]

    if out_path.exists():
        logger.info(f"Cached {target_format} model found: {out_path}")
        return str(out_path)

    logger.info(f"Converting {model_file} -> {target_format}")

    try:
        model = ultralytics.YOLO(model_file)

        if target_format == "rknn":
            model.export(format="rknn", imgsz=input_size)
        elif target_format == "onnx":
            model.export(format="onnx", imgsz=input_size, simplify=True, opset=12)
        elif target_format == "tflite":
            model.export(format="tflite", imgsz=input_size, int8=True)
        elif target_format == "openvino":
            model.export(format="openvino", imgsz=input_size, half=True)
        elif target_format == "coreml":
            model.export(format="coreml", imgsz=input_size, nms=True)
    except Exception as e:
        logger.error(f"Conversion to {target_format} raised an exception: {e}", exc_info=True)
        return model_file

    if out_path.exists():
        return str(out_path)

    logger.warning(f"Conversion to {target_format} failed, falling back to .pt")
    return model_file

def setup_files():
    # This download a ultralytics model and sets up the YoloModels thingie
    # Also setups Config/config.json with VisionCore defualt config

    # Ensure YoloModels directory exists
    yolo_dir = _REPO_ROOT / "YoloModels"
    yolo_dir.mkdir(exist_ok=True)

    # Create all sub dirs for formats and sizes
    formats = ["pytorch", "onnx", "tflite", "rknn", "openvino", "coreml"]
    sizes = ["nano", "small", "medium", "large", "xlarge", "2xlarge"]
    for fmt in formats:
        for size in sizes:
            (yolo_dir / fmt / size).mkdir(parents=True, exist_ok=True)

    # Download nano model if not already present
    nano_pt = yolo_dir / "pytorch" / "nano" / "dummy.pt"
    if not nano_pt.exists():
        logger.info("Downloading default nano.pt model...")
        try:
            model = ultralytics.YOLO("yolov8n.pt")
            model.save(str(nano_pt))
            logger.info(f"Saved default model to {nano_pt}")
        except Exception as e:
            logger.error(f"Failed to download/save default model: {e}", exc_info=True)
    
    # Ensure config directory and default config exist
    config_dir = _REPO_ROOT / "Config"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.json"
    if not config_file.exists():
        default_config = VisionCoreConfig()
        default_config.save()
        logger.info(f"Created default config at {config_file}")

def on_boot():
    logger.info("Starting VisionCore boot sequence...")
    setup_files()
    # 1. Validate system
    if not validate_system():
        raise RuntimeError("System validation failed. Aborting boot.")

    # 2. Theirs gonna be no config file, its boot
    config = VisionCoreConfig()

    # 3. Enforce YOLO model organization, actually not doing this because this is boot
    # is_valid, corrected_model_path = enforce_model_organization(_REPO_ROOT, config.config)
    
    # if not is_valid:
    #     raise RuntimeError(
    #         "YOLO model organization validation failed. "
    #         "Ensure models are in YoloModels/[format]/[size]/ structure."
    #     )
    
    # if corrected_model_path:
    #     config.config["vision_model"]["file_path"] = corrected_model_path
    #     logger.info("Using model from filesystem: %s", corrected_model_path)

    # 4. Auto-optimization (if enabled)
    if config.get("auto_opt"):
        best_format = recommend_format()
        logger.info("Auto-opt enabled. Recommended format: %s", best_format)

        extension = FORMAT_EXTENSIONS.get(best_format)
        if not extension:
            raise ValueError(f"No extension mapping for format: {best_format}")

        model_dir = _REPO_ROOT / "YoloModels"
        optimized = list(model_dir.rglob(f"*{extension}"))

        if optimized:
            chosen = str(optimized[0])
            # logger.info("Found optimised model(s): %s  ->  using %s",
                        # [str(m) for m in optimized], chosen)
            logger.info("Found optimised model.")
            config.set("vision_model", "file_path", chosen)
        else:
            logger.info("No %s models found in YoloModels/. Attempting conversion...", best_format)
            pt_path = config.get("vision_model", {}).get("file_path")
            if pt_path:
                pt_full = str(_REPO_ROOT / pt_path) if not os.path.isabs(pt_path) else pt_path
                input_size = config.get("input_size") or [640, 640]
                converted = convert_model(pt_full, best_format, input_size)
                if converted != pt_full:
                    logger.info("Conversion successful: %s", converted)
                    config.set("vision_model", "file_path", converted)
                else:
                    logger.warning("Conversion to %s failed or was skipped. Using .pt model.", best_format)
            else:
                logger.warning("No source .pt model found to convert. Using configured model.")
    else:
        logger.info("Auto-opt disabled.")

    # 5. Final model validation
    model_path = config.get("vision_model", {}).get("file_path")
    if not model_path:
        raise FileNotFoundError("No model path specified in config or found by auto-opt")
    
    model_full_path = Path(model_path)
    if not model_full_path.is_absolute():
        model_full_path = _REPO_ROOT / model_full_path
    
    if not model_full_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_full_path}")

    logger.info("Boot sequence complete. Final model path: %s", config.get("vision_model", {}).get("file_path"))

    config.save()

    # 6. Install service using the same Python interpreter that launched boot.py
    install_script = str(_BOOT_DIR / "install.py")
    try:
        subprocess.run([sys.executable, install_script], check=True, cwd=str(_REPO_ROOT))
    except subprocess.CalledProcessError as e:
        logger.error("Failed to run install.py: %s", e)
        raise RuntimeError("Boot failed during service installation.")

if __name__ == "__main__":
    on_boot()