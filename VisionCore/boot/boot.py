import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger(__name__)
import os

os.environ["YOLO_VERBOSE"] = "False"
import shutil
import subprocess
import platform
import importlib.util
import ultralytics
from VisionCore.vision.ModelInspector import fill_missing_config
from VisionCore.config.AutoOpt import recommend_format
from VisionCore.validations.validate_system import validate_system
from VisionCore.validations.model_validator import enforce_model_organization
from VisionCore.config.VisionCoreConfig import VisionCoreConfig

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

_ARCH = platform.machine().lower()
_IS_AARCH64 = "aarch64" in _ARCH or "arm64" in _ARCH
_PY_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"

_RKNN_WHEELS_BASE = os.environ.get(
    "VISIONCORE_RKNN_WHEELS_URL",
    "https://raw.githubusercontent.com/aidan-j532/VisionCore-Deploy/main/RknnWheels",
).rstrip("/")

_KNOWN_RKNN_WHEELS: dict[tuple[str, str], str] = {
    ("aarch64", "cp311"): "rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    ("aarch64", "cp312"): "rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    ("x86_64", "cp310"): "rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    ("x86_64", "cp312"): "rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
}


def _rknn_wheel_url() -> str | None:
    key = ("aarch64" if _IS_AARCH64 else "x86_64", _PY_TAG)
    filename = _KNOWN_RKNN_WHEELS.get(key)
    if not filename:
        supported = sorted(
            f"{a} {v}" for (a, v) in _KNOWN_RKNN_WHEELS if a == key[0]
        )
        logger.error(
            "No RKNN wheel for %s (Python %s). Supported: %s",
            key[0], _PY_TAG, ", ".join(supported),
        )
        return None
    return f"{_RKNN_WHEELS_BASE}/{filename}"


def _backend_dependencies() -> dict[str, list[tuple[str, str]]]:
    deps: dict[str, list[tuple[str, str]]] = {
        "onnx": [("onnxruntime", "onnxruntime")],
        "engine": [("tensorrt", "tensorrt")],
        "openvino": [("openvino", "openvino")],
        "coreml": [("coremltools", "coremltools")],
        "tflite": [("tflite_runtime", "tflite-runtime")],
    }
    rknn_url = _rknn_wheel_url()
    if rknn_url:
        mod = "rknnlite" if _IS_AARCH64 else "rknn"
        deps["rknn"] = [(mod, rknn_url)]
    return deps


BACKEND_DEPENDENCIES = _backend_dependencies()


def _in_virtualenv() -> bool:
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.environ.get("VIRTUAL_ENV") is not None
        or os.environ.get("CONDA_DEFAULT_ENV") is not None
    )


def _is_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _pip_install(install_target: str) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", install_target]
    if not _in_virtualenv():
        cmd.append("--break-system-packages")
    try:
        logger.info("Installing: %s", install_target)
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        logger.warning("pip install failed for: %s", install_target)
        return False


def install_special_dependencies(auto_install: bool = False):
    backend = recommend_format()
    logger.info("Recommended backend: %s", backend)

    deps = BACKEND_DEPENDENCIES.get(backend)
    if not deps:
        logger.info("No extra dependencies required for %s", backend)
        return

    missing = [(mod, target) for mod, target in deps if not _is_installed(mod)]

    if not missing:
        logger.info("All dependencies already installed for %s", backend)
        return

    logger.warning(
        "Missing dependencies for %s: %s",
        backend,
        [target for _, target in missing],
    )

    if not auto_install:
        logger.info("auto_install=False — skipping installation")
        return

    if backend == "rknn":
        arch = platform.machine()
        if not (_IS_AARCH64 or "x86_64" in arch or "amd64" in arch):
            logger.error(
                "RKNN wheels are only available for aarch64 and x86_64. "
                "Your architecture (%s) is not supported.",
                arch,
            )
            return

    if backend in {"rknn", "engine"}:
        logger.warning(
            "%s is a hardware/vendor backend — installation may require "
            "system-level setup and can take a few minutes.",
            backend,
        )

    for mod, target in missing:
        if _pip_install(target):
            logger.info("Installed %s successfully.", target)
        else:
            logger.error(
                "Failed to install %s. You may need to install it manually.",
                target,
            )

    logger.info("Dependency installation complete for %s", backend)


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
        logger.warning("Model file %s is missing. Skipping conversion.", model_file)
        return model_file
    if Path(model_file).suffix.lower() != ".pt":
        logger.warning(
            "Model file %s is not a .pt file. Skipping conversion.", model_file
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
                logger.info("Cached tflite model found: %s", tflites[0])
                return str(tflites[0])
    else:
        out_path = expected_outputs.get(target_format)
        if out_path and out_path.exists():
            logger.info("Cached %s model found: %s", target_format, out_path)
            return str(out_path)

    logger.info(
        "Converting %s -> %s. "
        "Note: RKNN, TensorRT, and CoreML may take several minutes.",
        model_file,
        target_format,
    )
    try:
        model = ultralytics.YOLO(model_file)
        export_kwargs = {
            "rknn": dict(format="rknn", imgsz=input_size),
            "onnx": dict(format="onnx", imgsz=input_size, simplify=True, opset=12),
            "tflite": dict(format="tflite", imgsz=input_size, int8=True),
            "openvino": dict(format="openvino", imgsz=input_size, half=True),
            "coreml": dict(format="coreml", imgsz=input_size, nms=True),
            "engine": dict(format="engine", imgsz=input_size, half=True, device=0),
        }
        kwargs = export_kwargs.get(target_format)
        if not kwargs:
            logger.warning("Unknown format: %s. Skipping conversion.", target_format)
            return model_file
        result = model.export(**kwargs)
    except Exception as e:
        logger.error("Conversion to %s failed: %s", target_format, e, exc_info=True)
        return model_file

    if result is not None:
        result_path = Path(result)
        if result_path.exists():
            logger.info("%s export successful: %s", target_format, result_path)
            return str(result_path)

    if target_format == "tflite":
        saved_model_dir = parent / f"{stem}_saved_model"
        if saved_model_dir.exists():
            tflites = list(saved_model_dir.rglob("*.tflite"))
            if tflites:
                logger.info("TFLite export successful: %s", tflites[0])
                return str(tflites[0])
    else:
        out_path = expected_outputs.get(target_format)
        if out_path and out_path.exists():
            logger.info("%s export successful: %s", target_format, out_path)
            return str(out_path)

    logger.warning("Conversion to %s failed, falling back to .pt", target_format)
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
        logger.info("Using existing config: %s", config_path)

    if not validate_system():
        raise RuntimeError("System validation failed. Aborting boot.")

    if config is None:
        config = VisionCoreConfig(str(config_path))

    if config.get("auto_opt"):
        install_special_dependencies(auto_install=True)

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
                "No cached %s model found. Attempting conversion...", best_format
            )
            pt_path = config.get("vision_model", {}).get("source_pt")
            if pt_path:
                pt_full = Path(pt_path)
                if not pt_full.is_absolute():
                    pt_full = _PROJECT_ROOT / pt_path
                if not pt_full.exists():
                    logger.warning("Model missing, copying bundled _default_pose.pt")
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
