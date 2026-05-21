import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # Crucial: overrides any setups hidden inside imports
)

logger = logging.getLogger(__name__)
import os
os.environ["YOLO_VERBOSE"] = "False"
import os
import shutil
import subprocess
import ultralytics

from VisionCore.vision.ModelInspector import fill_missing_config
from VisionCore.config.AutoOpt import recommend_format
from VisionCore.validations.validate_system import validate_system
from VisionCore.validations.model_validator import enforce_model_organization
from VisionCore.config.VisionCoreConfig import VisionCoreConfig

# Ensure root log levels didn't get overridden by imports
logging.getLogger().setLevel(logging.INFO)

_BOOT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path.cwd().resolve()
_PACKAGE_ROOT = Path(__file__).resolve().parent
_ASSETS_DIR = _PACKAGE_ROOT.parent / "assets"

FORMAT_MATCHERS = {
    "onnx": lambda p: p.suffix == ".onnx",
    "rknn": lambda p: p.suffix == ".rknn",
    "tflite": lambda p: p.suffix == ".tflite",
    "coreml": lambda p: p.suffix == ".mlpackage",
    "openvino": lambda p: p.is_dir() and p.name.endswith("_openvino_model"),
    "engine": lambda p: p.suffix == ".engine",
}

def reset_workspace():
    targets = [
        _PROJECT_ROOT / "YoloModels",
        _PROJECT_ROOT / "Outputs",
        _PROJECT_ROOT / "Config",
    ]
    for target in targets:
        try:
            if target.exists():
                logger.warning("Resetting: %s", target)
                shutil.rmtree(target, ignore_errors=False)
        except Exception as e:
            logger.warning("Failed to fully remove %s: %s", target, e)

    # recreate clean structure immediately
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
    logger.info("Workspace fully reset and reinitialized.")


def search_for_config():
    config_dir = _PROJECT_ROOT / "Config"
    if not config_dir.exists():
        return None
    config_files = sorted(config_dir.rglob("*.json"))
    if not config_files:
        return None
    non_default = [f for f in config_files if f.name != "config.json"]
    chosen = non_default[0] if non_default else config_files[0]
    logger.info("Found config: %s -> using %s", len(config_files), chosen)
    return str(chosen)


def convert_model(model_file, target_format, input_size):
    if not os.path.exists(model_file):
        logger.warning(
            f"Model file {model_file} is missing or empty. Skipping conversion."
        )
        return model_file
    if Path(model_file).suffix.lower() != ".pt":
        logger.warning(
            f"Model file {model_file} is not a valid .pt file. Skipping conversion."
        )
        return model_file

    model_path = Path(model_file)
    stem = model_path.stem
    parent = model_path.parent
    expected_outputs = {
        "rknn": parent / f"{stem}.rknn",
        "onnx": parent / f"{stem}.onnx",
        "openvino": parent / f"{stem}_openvino_model",
        "coreml": parent / f"{stem}.mlpackage",
        "engine": parent / f"{stem}.engine",
    }

    if target_format == "tflite":
        saved_model_dir = parent / f"{stem}_saved_model"
        if saved_model_dir.exists():
            tflites = list(saved_model_dir.rglob("*.tflite"))
            if tflites:
                logger.info(f"Cached tflite model found: {tflites[0]}")
                return str(tflites[0])
    else:
        out_path = expected_outputs.get(target_format)
        if out_path and out_path.exists():
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
        elif target_format == "engine":
            model.export(format="engine", imgsz=input_size, half=True, device=0)
    except Exception as e:
        logger.error(
            f"Conversion to {target_format} raised an exception: {e}", exc_info=True
        )
        return model_file

    if target_format == "tflite":
        saved_model_dir = parent / f"{stem}_saved_model"
        if saved_model_dir.exists():
            tflites = list(saved_model_dir.rglob("*.tflite"))
            if tflites:
                logger.info(f"TFLite export successful: {tflites[0]}")
                return str(tflites[0])
    else:
        out_path = expected_outputs.get(target_format)
        if out_path and out_path.exists():
            logger.info(f"{target_format} export successful: {out_path}")
            return str(out_path)

    logger.warning(f"Conversion to {target_format} failed, falling back to .pt")
    return model_file


def setup_files():
    yolo_dir = _PROJECT_ROOT / "YoloModels"
    config_dir = _PROJECT_ROOT / "Config"
    outputs_dir = _PROJECT_ROOT / "Outputs"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["pytorch", "onnx", "tflite", "rknn", "openvino", "coreml"]:
        (yolo_dir / fmt).mkdir(parents=True, exist_ok=True)
    nano_pt = yolo_dir / "pytorch" / "_default_pose.pt"
    if not nano_pt.exists():
        bundled = _ASSETS_DIR / "_default_pose.pt"
        if bundled.exists():
            shutil.copy(bundled, nano_pt)


def on_boot(install_service: bool = False, first_boot: bool = False):
    if first_boot:
        logger.info("First boot mode enabled. Resetting workspace...")
        reset_workspace()
    setup_files()
    config_path = search_for_config()
    config = None
    if not config_path:
        logger.info("No config found. Creating default config...")
        config_path = _PROJECT_ROOT / "Config" / "config.json"
        config = VisionCoreConfig(str(config_path), create=True)
    else:
        logger.info(f"Using existing config: {config_path}")

    if not validate_system():
        raise RuntimeError("System validation failed. Aborting boot.")

    if config is None:
        config = VisionCoreConfig(str(config_path))

    if config.get("auto_opt"):
        best_format = recommend_format()
        logger.info("Auto-opt enabled. Recommended format: %s", best_format)
        matcher = FORMAT_MATCHERS.get(best_format)
        if not matcher:
            raise ValueError(f"No matcher for format: {best_format}")
        model_dir = _PROJECT_ROOT / "YoloModels"
        optimized = [p for p in model_dir.rglob("*") if matcher(p)]
        if optimized:
            logger.info("Found cached optimised model: %s", optimized[0])
            config.set("vision_model", "file_path", str(optimized[0]))
        else:
            logger.info(
                f"No cached {best_format} model found. Attempting conversion..."
            )
            pt_path = config.get("vision_model", {}).get("source_pt")
            if pt_path:
                pt_full = Path(pt_path)
                if not pt_full.is_absolute():
                    pt_full = _PROJECT_ROOT / pt_path
                if not pt_full.exists():
                    logger.warning(
                        "Model missing in workspace, copying bundled _default_pose.pt"
                    )
                    bundled = _ASSETS_DIR / "_default_pose.pt"
                    target = _PROJECT_ROOT / "YoloModels/pytorch/_default_pose.pt"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(bundled, target)
                    pt_full = target
                input_size = config.get("input_size") or [640, 640]
                converted = Path(convert_model(str(pt_full), best_format, input_size))
                if converted != pt_full:
                    logger.info("Conversion successful: %s", converted)
                    config.set("vision_model", "file_path", str(converted))
                else:
                    logger.warning(
                        "Conversion to %s failed or was skipped. Using .pt model.",
                        best_format,
                    )
            else:
                logger.warning(
                    "No source .pt model found to convert. Using configured model."
                )
    else:
        logger.info("Auto-opt disabled.")

    model_path = config.get("vision_model", {}).get("file_path")
    if not model_path:
        raise FileNotFoundError(
            "No model path specified in config or found by auto-opt"
        )
    model_full_path = Path(model_path)
    if not model_full_path.is_absolute():
        model_full_path = _PROJECT_ROOT / model_full_path
    if not model_full_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_full_path}")

    vision_cfg = config.get("vision_model", {})
    filled = fill_missing_config(vision_cfg)
    config.set("vision_model", filled)
    logger.info("Model config auto-filled and saved.")
    logger.info(
        "Boot sequence complete. Final model path: %s",
        config.get("vision_model", {}).get("file_path"),
    )
    config.save(quiet=True)

    if install_service:
        install_script = str(_BOOT_DIR / "install.py")
        try:
            subprocess.run(
                [sys.executable, install_script], check=True, cwd=str(_PROJECT_ROOT)
            )
        except subprocess.CalledProcessError as e:
            logger.error("Failed to run install.py: %s", e)
            raise RuntimeError("Boot failed during service installation.")
    else:
        logger.info("Skipping service installation. Run with -s to install.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="VisionCore boot sequence")
    parser.add_argument(
        "-s",
        "--service",
        action="store_true",
        help="Install and start the watchdog service",
    )
    parser.add_argument(
        "-f",
        "--first-boot",
        action="store_true",
        help="Delete Config, Outputs, and YoloModels before booting",
    )
    args = parser.parse_args()
    on_boot(install_service=args.service, first_boot=args.first_boot)


if __name__ == "__main__":
    main()
