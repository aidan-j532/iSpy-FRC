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
import importlib.metadata
import ultralytics
from VisionCore.vision.ModelInspector import fill_missing_config
from VisionCore.config.AutoOpt import recommend_format
from VisionCore.validations.validate_system import validate_system
from VisionCore.validations.model_validator import enforce_model_organization
from VisionCore.config.VisionCoreConfig import VisionCoreConfig
from VisionCore.dataset.dataset import prepare_quantization_dataset

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
    (
        "aarch64",
        "cp311",
    ): "rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    (
        "aarch64",
        "cp312",
    ): "rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    (
        "x86_64",
        "cp310",
    ): "rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    (
        "x86_64",
        "cp312",
    ): "rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
}


def _rknn_wheel_url() -> str | None:
    key = ("aarch64" if _IS_AARCH64 else "x86_64", _PY_TAG)
    filename = _KNOWN_RKNN_WHEELS.get(key)
    if not filename:
        supported = sorted(f"{a} {v}" for (a, v) in _KNOWN_RKNN_WHEELS if a == key[0])
        logger.error(
            "No RKNN wheel for %s (Python %s). Supported: %s",
            key[0],
            _PY_TAG,
            ", ".join(supported),
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


def _get_installed_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return None


def _check_version_constraint(package_name: str, constraint: str) -> bool:
    if not constraint:
        return True
    installed = _get_installed_version(package_name)
    if installed is None:
        return False
    try:
        from packaging.version import Version
        inst = Version(installed)
        bound = constraint.lstrip("<>=!~")
        if constraint.startswith("<="):
            return inst <= Version(bound)
        if constraint.startswith(">="):
            return inst >= Version(bound)
        if constraint.startswith("!="):
            return inst != Version(bound)
        if constraint.startswith("=="):
            return inst == Version(bound)
        if constraint.startswith("<"):
            return inst < Version(bound)
        if constraint.startswith(">"):
            return inst > Version(bound)
        if constraint.startswith("~="):
            return (
                inst.release[: len(Version(bound).release)]
                == Version(bound).release
                and inst >= Version(bound)
            )
    except Exception:
        pass
    return True


def _pip_install(install_target: str, force_reinstall: bool = False) -> bool:
    cmd = [sys.executable, "-m", "pip", "install"]
    # Only use --no-deps for wheel URLs (RKNN) to avoid pulling in
    # conflicting transitive deps; regular pip packages need their deps
    if install_target.startswith("http") or install_target.endswith(".whl"):
        cmd.append("--no-deps")
    if force_reinstall:
        cmd.append("--force-reinstall")
    cmd.append(install_target)
    if not _in_virtualenv():
        cmd.append("--break-system-packages")
    try:
        logger.info("Installing: %s", install_target)
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        logger.warning("pip install failed for: %s", install_target)
        return False


def _parse_pip_target(pip_target: str) -> tuple[str, str]:
    parts = pip_target.split("[")
    base = parts[0]
    for sep in ("<=", ">=", "!=", "==", "<", ">", "~="):
        if sep in base:
            name, ver = base.split(sep, 1)
            return name, sep + ver
    return base, ""


def install_special_dependencies(auto_install: bool = False):
    backend = recommend_format(ignore_dependencies=True)
    logger.info("Recommended backend: %s", backend)

    deps = BACKEND_DEPENDENCIES.get(backend)
    if not deps:
        logger.info("No extra dependencies required for %s", backend)
        return

    missing = []
    for mod, target in deps:
        pkg_name, constraint = _parse_pip_target(target)
        if not _is_installed(mod):
            missing.append((mod, target, False))
        elif constraint and not _check_version_constraint(pkg_name, constraint):
            logger.warning(
                "Installed %s (%s) does not satisfy %s. Will reinstall.",
                pkg_name,
                _get_installed_version(pkg_name),
                target,
            )
            missing.append((mod, target, True))
        else:
            logger.debug("Dependency %s satisfied: %s", mod, target)

    if not missing:
        logger.info("All dependencies already installed for %s", backend)
        return

    logger.warning(
        "Missing dependencies for %s: %s",
        backend,
        [target for _, target, _ in missing],
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

    for mod, target, force in missing:
        if _pip_install(target, force_reinstall=force):
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


def _patch_tensorrt_for_ultralytics():
    try:
        import tensorrt as trt
    except ImportError:
        return
    if not hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
        logger.info(
            "Monkey-patched missing trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH "
            "for TensorRT %s compatibility", trt.__version__
        )


def _export_ultralytics(model_file, target_format, input_size, data_yaml=None):
    if target_format == "engine":
        _patch_tensorrt_for_ultralytics()

    model = ultralytics.YOLO(model_file)

    native_kwargs = {
        "onnx": dict(format="onnx", imgsz=input_size, simplify=True, opset=12),
        "tflite": dict(format="tflite", imgsz=input_size, int8=True),
        "openvino": dict(format="openvino", imgsz=input_size, half=True),
        "coreml": dict(format="coreml", imgsz=input_size, nms=True),
        "engine": dict(format="engine", imgsz=input_size, half=True, device=0),
    }

    kwargs = native_kwargs.get(target_format)
    if not kwargs:
        raise ValueError(f"Unsupported native format: {target_format}")

    if data_yaml and target_format in ("tflite", "openvino"):
        kwargs = dict(format=target_format, imgsz=input_size, int8=True, data=data_yaml)
        logger.info(
            "Dataset-aware %s quantization enabled (data=%s)", target_format, data_yaml
        )

    logger.info("Exporting %s -> %s with kwargs: %s", model_file, target_format, kwargs)
    return model.export(**kwargs)


def _export_rknn_metadata(pt_file: str, rknn_output: Path) -> None:
    try:
        import ultralytics
        from ruamel.yaml import YAML

        model = ultralytics.YOLO(pt_file, verbose=False)
        meta = {}
        model_task = getattr(model, "task", "detect") or "detect"
        meta["task"] = model_task
        try:
            nc = int(model.model.model[-1].nc)
            meta["nc"] = nc
            names = getattr(model, "names", None)
            if names and isinstance(names, dict):
                meta["names"] = names
        except Exception:
            pass
        if model_task == "pose":
            try:
                kpt_shape = model.model.model[-1].kpt_shape
                if kpt_shape and len(kpt_shape) == 2:
                    meta["kpt_shape"] = [int(kpt_shape[0]), int(kpt_shape[1])]
            except Exception:
                pass

        # Output format — known because WE did the conversion
        # ONNX output is (feat, anchors), RKNN compiler preserves this layout
        meta["output_format"] = "raw"
        meta["output_layout"] = "features_first"
        meta["box_format"] = "cxcywh"
        meta["quantization"] = "int8"
        meta["quant_scale"] = 255.0

        meta_path = rknn_output.parent / "metadata.yaml"
        yaml = YAML()
        yaml.default_flow_style = False
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)
        logger.info("Exported RKNN metadata: %s", meta_path)
    except Exception as e:
        logger.warning("Failed to export RKNN metadata: %s", e)

def _export_onnx_metadata(pt_file: str, onnx_output: Path) -> None:
    try:
        import ultralytics
        from ruamel.yaml import YAML

        model = ultralytics.YOLO(pt_file, verbose=False)
        meta = {}
        model_task = getattr(model, "task", "detect") or "detect"
        meta["task"] = model_task
        try:
            nc = int(model.model.model[-1].nc)
            meta["nc"] = nc
            names = getattr(model, "names", None)
            if names and isinstance(names, dict):
                meta["names"] = names
        except Exception:
            pass
        if model_task == "pose":
            try:
                kpt_shape = model.model.model[-1].kpt_shape
                if kpt_shape and len(kpt_shape) == 2:
                    meta["kpt_shape"] = [int(kpt_shape[0]), int(kpt_shape[1])]
            except Exception:
                pass

        # Standard Ultralytics ONNX export output format
        meta["output_format"] = "raw"
        meta["output_layout"] = "features_first"
        meta["box_format"] = "cxcywh"
        meta["quantization"] = "none"

        meta_path = onnx_output.parent / "metadata.yaml"
        yaml = YAML()
        yaml.default_flow_style = False
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)
        logger.info("Exported ONNX metadata: %s", meta_path)
    except Exception as e:
        logger.warning("Failed to export ONNX metadata: %s", e)

def _convert_rknn(pt_file, input_size, dataset_path, task="detect"):
    pt_path = Path(pt_file)
    stem = pt_path.stem
    parent = pt_path.parent

    onnx_path = Path(_export_ultralytics(str(pt_path), "onnx", input_size))
    if not onnx_path.exists():
        raise RuntimeError(f"Intermediate ONNX export failed: {onnx_path}")

    try:
        from rknn.api import RKNN
    except ImportError:
        raise ImportError(
            "RKNN Toolkit not found. Install it to convert to RKNN format."
        )

    prepare_quantization_dataset(dataset_path, boot=True)
    dataset_txt = Path(dataset_path) / "dataset.txt"
    if not dataset_txt.exists() or not dataset_txt.read_text().strip():
        raise FileNotFoundError(
            f"RKNN calibration dataset could not be prepared at: {dataset_txt}"
        )

    rknn_output = parent / f"{stem}.rknn"
    logger.info("Converting ONNX -> RKNN with dataset=%s", dataset_txt)

    rknn = RKNN()
    try:
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform="rk3588",
            quantized_algorithm="kl_divergence",
            quantized_dtype="w8a8",
            quantized_hybrid_level=3,
        )
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"RKNN load_onnx failed with code {ret}")
        ret = rknn.build(do_quantization=True, dataset=str(dataset_txt))
        if ret != 0:
            raise RuntimeError(f"RKNN build failed with code {ret}")
        ret = rknn.export_rknn(str(rknn_output))
        if ret != 0:
            raise RuntimeError(f"RKNN export failed with code {ret}")
    finally:
        rknn.release()

    logger.info("RKNN conversion successful: %s", rknn_output)

    _export_rknn_metadata(pt_file, rknn_output)

    return str(rknn_output)


def convert_model(model_file, target_format, input_size, quantize=False):
    if not os.path.exists(model_file):
        logger.warning("Model file %s is missing. Skipping conversion.", model_file)
        return model_file
    if Path(model_file).suffix.lower() != ".pt":
        logger.warning(
            "Model file %s is not a .pt file. Skipping conversion.", model_file
        )
        return model_file

    pt_path = Path(model_file)
    stem = pt_path.stem
    parent = pt_path.parent

    if target_format == "rknn":
        return _convert_rknn(
            pt_file=model_file,
            input_size=input_size,
            dataset_path=str(_PROJECT_ROOT / "QuantizeDataset"),
        )

    expected = {
        "onnx": parent / f"{stem}.onnx",
        "openvino": parent / f"{stem}_openvino_model",
        "coreml": parent / f"{stem}.mlpackage",
        "engine": parent / f"{stem}.engine",
    }
    if target_format == "tflite":
        saved = parent / f"{stem}_saved_model"
        if saved.exists():
            tflites = list(saved.rglob("*.tflite"))
            if tflites:
                logger.info("Cached tflite model found: %s", tflites[0])
                return str(tflites[0])
    else:
        out = expected.get(target_format)
        if out and out.exists():
            logger.info("Cached %s model found: %s", target_format, out)
            return str(out)

    dataset_root = str(_PROJECT_ROOT / "QuantizeDataset")
    data_yaml = None
    if quantize:
        prepare_quantization_dataset(dataset_root, boot=True)
        data_yaml = str(Path(dataset_root) / "data.yaml")
    else:
        Path(dataset_root).mkdir(parents=True, exist_ok=True)

    try:
        result = _export_ultralytics(model_file, target_format, input_size, data_yaml)
    except AttributeError as e:
        if "EXPLICIT_BATCH" in str(e):
            logger.error(
                "TensorRT 10+ is incompatible with the ultralytics export path. "
                "Skipping engine conversion. Falling back to .pt."
            )
        else:
            logger.error("Conversion to %s failed: %s", target_format, e, exc_info=True)
        return model_file
    except Exception as e:
        logger.error("Conversion to %s failed: %s", target_format, e, exc_info=True)
        return model_file

    if result is not None:
        result_path = Path(result)
        if result_path.exists():
            logger.info("%s export successful: %s", target_format, result_path)
            if target_format == "onnx":
                _export_onnx_metadata(model_file, result_path)
            return str(result_path)

    if target_format == "tflite":
        saved = parent / f"{stem}_saved_model"
        if saved.exists():
            tflites = list(saved.rglob("*.tflite"))
            if tflites:
                logger.info("TFLite export successful: %s", tflites[0])
                return str(tflites[0])
    else:
        out = expected.get(target_format)
        if out and out.exists():
            logger.info("%s export successful: %s", target_format, out)
            return str(out)

    logger.warning("Conversion to %s failed, falling back to .pt", target_format)
    return model_file


def setup_files():
    yolo_dir = _PROJECT_ROOT / "YoloModels"
    config_dir = _PROJECT_ROOT / "Config"
    outputs_dir = _PROJECT_ROOT / "Outputs"
    dataset_dir = _PROJECT_ROOT / "QuantizeDataset"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["pytorch", "onnx", "tflite", "rknn", "openvino", "coreml"]:
        (yolo_dir / fmt).mkdir(parents=True, exist_ok=True)
    prepare_quantization_dataset(str(dataset_dir), boot=True)
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

        best_format = recommend_format(ignore_dependencies=True)
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
                converted = Path(
                    convert_model(
                        str(pt_full),
                        best_format,
                        input_size,
                        quantize=config.get("quantize", False),
                    )
                )
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
