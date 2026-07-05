import sys
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger(__name__)
os.environ["YOLO_VERBOSE"] = "False"
import json
import shutil
import subprocess
import platform
import contextlib
import importlib.util
import importlib.metadata
import ultralytics
from iSpy.vision.ModelInspector import fill_missing_config
from iSpy.vision.metadata import (
    metadata_path_for,
    metadata_from_pt,
    write_metadata,
    read_metadata,
    derive_format_metadata,
)
from iSpy.config.AutoOpt import recommend_format
from iSpy.validations.validate_system import validate_system
from iSpy.config.iSpyConfig import iSpyConfig
from iSpy.dataset.dataset import prepare_quantization_dataset
from iSpy.config.AutoOpt import recommend_format, has_jetson
import argparse
from iSpy.config.AutoOpt import has_jetson
from iSpy.boot.opencv_fix import ensure_csi_capable_opencv
from iSpy.validations.tests.test_optimized_model import compare_models

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
    "tpu": lambda p: p.suffix == ".pt",
}

_BUNDLED_DEFAULT_MODELS = {"_default_pose.pt", "_default_box.pt", "_default_v26_detect_for_fuel.pt"}

_JETSON_SYSTEM_MANAGED = {"tensorrt", "onnxruntime"}

_ARCH = platform.machine().lower()
_IS_AARCH64 = "aarch64" in _ARCH or "arm64" in _ARCH
_PY_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"

keywords = ["frc game piece", "frc 2025 REBUILT", "frc 2025 fuel"]

_RKNN_QUANTIZE = True

_BOOT_MANAGED_VISION_FIELDS = {
    "file_path", "task", "num_classes", "input_size",
    "output", "input", "frame_batches", "device",
}

# Fields that must come from metadata, never stored in config.
_METADATA_ONLY_FIELDS = {
    "task", "num_classes", "input_size", "output", "input", "frame_batches",
}

_MANUAL_POSTPROCESS_FORMATS = {"onnx", "tflite"}


def _model_supports_end2end(model: "ultralytics.YOLO") -> bool:
    # Set by Ultralytics on every DetectionModel/SegmentationModel/PoseModel/
    # OBBModel: self.end2end = getattr(self.model[-1], "end2end", False).
    # Reflects the actual trained head - not a version guess.
    return bool(getattr(model.model, "end2end", False))


def _strip_boot_managed_fields(cfg: dict) -> dict:
    cfg = json.loads(json.dumps(cfg))  # deep copy, don't mutate caller's dict
    vm = cfg.get("vision_model")
    if isinstance(vm, dict):
        for key in _BOOT_MANAGED_VISION_FIELDS:
            vm.pop(key, None)
    return cfg


def _default_config_dict() -> dict:
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    try:
        return iSpyConfig().default_config
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)

def _find_lite_wheel_dir() -> Path:
    local = _PACKAGE_ROOT.parent / "rknn_wheels"
    if local.exists():
        return local
    try:
        spec = importlib.util.find_spec("iSpy")
        if spec and spec.origin:
            pkg = Path(spec.origin).parent / "rknn_wheels"
            if pkg.exists():
                return pkg
    except Exception:
        pass
    return local


_RKNN_LITE_DIR = _find_lite_wheel_dir()

_RKNN_FULL_BASE = os.environ.get(
    "iSpy_RKNN_WHEELS_URL",
    "https://github.com/aidan-j532/iSpy-FRC/releases/download/v1.0.2",
).rstrip("/")

_RKNN_FULL_WHEELS: dict[tuple[str, str], str] = {
    (
        "aarch64",
        "cp310",
    ): "rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    (
        "aarch64",
        "cp311",
    ): "rknn_toolkit2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    (
        "aarch64",
        "cp312",
    ): "rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    (
        "x86_64",
        "cp310",
    ): "rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    (
        "x86_64",
        "cp311",
    ): "rknn_toolkit2-2.3.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    (
        "x86_64",
        "cp312",
    ): "rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
}

_RKNN_LITE_FILENAMES: dict[tuple[str, str], str] = {
    (
        "aarch64",
        "cp310",
    ): "rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    (
        "aarch64",
        "cp311",
    ): "rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    (
        "aarch64",
        "cp312",
    ): "rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
}


def _rknn_wheel_targets() -> list[tuple[str, str]]:
    key = ("aarch64" if _IS_AARCH64 else "x86_64", _PY_TAG)
    targets: list[tuple[str, str]] = []

    full_fn = _RKNN_FULL_WHEELS.get(key)
    if full_fn:
        targets.append(("rknn", f"{_RKNN_FULL_BASE}/{full_fn}"))

    lite_fn = _RKNN_LITE_FILENAMES.get(key)
    if lite_fn:
        local_lite = _RKNN_LITE_DIR / lite_fn
        if local_lite.exists():
            targets.append(("rknnlite", str(local_lite)))
        else:
            logger.warning("Lite wheel not found in package: %s", local_lite)

    if not targets:
        supported = sorted(f"{a} {v}" for (a, v) in _RKNN_FULL_WHEELS if a == key[0])
        logger.error(
            "No RKNN wheel for %s (Python %s). Supported: %s",
            key[0],
            _PY_TAG,
            ", ".join(supported),
        )
    return targets


def _backend_dependencies() -> dict[str, list[tuple[str, str]]]:
    from iSpy.config.AutoOpt import has_nvidia, has_amd_gpu

    if has_nvidia():
        onnx_dep = ("onnxruntime", "onnxruntime-gpu")
    elif platform.system() == "Windows":
        onnx_dep = ("onnxruntime", "onnxruntime-directml")  # covers AMD/Intel/Nvidia on Windows
    else:
        onnx_dep = ("onnxruntime", "onnxruntime")
        if has_amd_gpu():
            logger.warning(
                "AMD GPU on Linux: the standard 'onnxruntime' wheel is CPU-only. "
                "ROCm onnxruntime builds aren't reliably on PyPI for every ROCm "
                "version - install one manually matching your ROCm version."
            )

    deps: dict[str, list[tuple[str, str]]] = {
        "onnx": [onnx_dep],
        "engine": [("tensorrt", "tensorrt==10.16.1.11")],
        "openvino": [("openvino", "openvino")],
        "coreml": [("coremltools", "coremltools")],
        "tflite": [("tflite_runtime", "tflite-runtime")],
        "tpu": [("torch_xla", "torch_xla[tpu]", ["-f", "https://storage.googleapis.com/libtpu-releases/index.html"])],
    }
    rknn_targets = _rknn_wheel_targets()
    if rknn_targets:
        deps["rknn"] = rknn_targets + [("onnx", "onnx<1.17"), ("google.protobuf", "protobuf<4.0")]
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
            return inst.release[: len(Version(bound).release)] == Version(
                bound
            ).release and inst >= Version(bound)
    except Exception:
        pass
    return True

def _run_optimized_model_comparison(pt_file: str, converted_result: str) -> None:
    converted_path = Path(converted_result)
    if converted_path == Path(pt_file) or converted_path.suffix.lower() == ".pt":
        # Conversion failed / fell back to the source .pt - nothing to compare.
        return

    valid_dir = _PROJECT_ROOT / "QuantizeDataset" / "valid"
    if not valid_dir.exists():
        logger.info(
            "Skipping optimized-model comparison for %s - no QuantizeDataset/valid/ found.",
            converted_path.name,
        )
        return

    logger.info("Running optimized-model comparison for %s...", converted_path.name)
    try:
        results = compare_models(
            base_path=str(pt_file),
            optimized_path=str(converted_path),
            images_dir=str(valid_dir),
            quiet=False,
        )
    except (Exception, SystemExit) as e:
        logger.warning("Optimized-model comparison failed for %s: %s", converted_path.name, e)
        return

    if results is None:
        return  # compare_models already logged why it skipped

    if results.overall_verdict == "NOT READY":
        logger.warning(
            "Optimized model %s scored NOT READY vs its base .pt: %s. "
            "Conversion is still being used - review Outputs/optimized_model_report.json.",
            converted_path.name,
            "; ".join(results.verdict_reasons),
        )
    elif results.overall_verdict == "REVIEW RECOMMENDED":
        logger.info(
            "Optimized model %s: REVIEW RECOMMENDED (%s). See Outputs/optimized_model_report.json.",
            converted_path.name,
            "; ".join(results.verdict_reasons),
        )
    else:
        logger.info("Optimized model %s: READY.", converted_path.name)

def _pip_install(
    install_target: str,
    force_reinstall: bool = False,
    extra_args: list[str] | None = None,
) -> bool:
    cmd = [sys.executable, "-m", "pip", "install"]
    if install_target.startswith("http") or install_target.endswith(".whl"):
        cmd.append("--no-deps")
    if force_reinstall:
        cmd.append("--force-reinstall")
    if extra_args:
        cmd.extend(extra_args)
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

    on_jetson = has_jetson()
    missing = []
    for entry in deps:
        mod, target = entry[0], entry[1]
        extra_args = list(entry[2]) if len(entry) > 2 else None
        pkg_name, constraint = _parse_pip_target(target)

        if on_jetson and mod in _JETSON_SYSTEM_MANAGED:
            if _is_installed(mod):
                logger.debug("Dependency %s satisfied (JetPack system package).", mod)
            else:
                logger.error(
                    "%s is not importable. On Jetson this must come from JetPack "
                    "(apt), not pip — pip has no matching GPU-enabled build for "
                    "this board. Verify it with the L4T-provided python3, and "
                    "make sure your venv was created with --system-site-packages "
                    "so it can see it.",
                    mod,
                )
            continue

        if not _is_installed(mod):
            missing.append((mod, target, False, extra_args))
        elif constraint and not _check_version_constraint(pkg_name, constraint):
            logger.warning(
                "Installed %s (%s) does not satisfy %s. Will reinstall.",
                pkg_name, _get_installed_version(pkg_name), target,
            )
            missing.append((mod, target, True, extra_args))
        else:
            logger.debug("Dependency %s satisfied: %s", mod, target)

    if not missing:
        logger.info("All pip-installable dependencies already satisfied for %s", backend)
        return

    logger.warning("Missing dependencies for %s: %s", backend, [t for _, t, _, _ in missing])

    if not auto_install:
        logger.info("auto_install=False - skipping installation")
        return

    if backend == "rknn":
        arch = platform.machine()
        if not (_IS_AARCH64 or "x86_64" in arch or "amd64" in arch):
            logger.error(
                "RKNN wheels are only available for aarch64 and x86_64. "
                "Your architecture (%s) is not supported.", arch,
            )
            return

    if backend in {"rknn", "engine"}:
        logger.warning(
            "%s is a hardware/vendor backend - installation may require "
            "system-level setup and can take a few minutes.", backend,
        )

    for mod, target, force, extra_args in missing:
        if _pip_install(target, force_reinstall=force, extra_args=extra_args):
            logger.info("Installed %s successfully.", target)
        else:
            logger.error("Failed to install %s. You may need to install it manually.", target)

    logger.info("Dependency installation complete for %s", backend)

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


def _export_ultralytics(model_file, target_format, input_size, data_yaml=None, device=0):
    model = ultralytics.YOLO(model_file)
    has_e2e = _model_supports_end2end(model)

    native_kwargs = {
        "onnx": dict(format="onnx", imgsz=input_size, simplify=True, opset=17, dynamic=False),
        "tflite": dict(format="tflite", imgsz=input_size, int8=True),
        "openvino": dict(format="openvino", imgsz=input_size, half=True),
        "coreml": dict(format="coreml", imgsz=input_size, nms=True),
        "engine": dict(format="engine", imgsz=input_size, half=True, device=device),
    }

    kwargs = native_kwargs.get(target_format)
    if not kwargs:
        raise ValueError(f"Unsupported native format: {target_format}")

    if has_e2e and target_format in _MANUAL_POSTPROCESS_FORMATS:
        kwargs["end2end"] = False
        logger.info(
            "%s has an end-to-end (dual-head) architecture - forcing "
            "end2end=False for %s export so the raw-tensor parser gets "
            "the traditional (1, nc+4, N) output instead of (1, 300, 6).",
            Path(model_file).name, target_format,
        )
    elif has_e2e:
        logger.info(
            "%s has an end-to-end architecture; leaving the default head for "
            "%s export (Ultralytics decodes it natively at runtime - no "
            "speed reason to disable it here).",
            Path(model_file).name, target_format,
        )

    if data_yaml and target_format in ("tflite", "openvino", "engine"):
        kwargs = dict(format=target_format, imgsz=input_size, int8=True, data=data_yaml)
        if has_e2e and target_format in _MANUAL_POSTPROCESS_FORMATS:
            kwargs["end2end"] = False
        if target_format == "engine":
            kwargs["device"] = device
            kwargs["half"] = True

        logger.info(
            "Dataset-aware %s quantization enabled (data=%s)", target_format, data_yaml
        )

    logger.info("Exporting %s -> %s with kwargs: %s", model_file, target_format, kwargs)
    return model.export(**kwargs)

@contextlib.contextmanager
def _silent_fd():
    devnull = "nul" if os.name == "nt" else "/dev/null"
    fd = os.open(devnull, os.O_WRONLY)
    old_out = os.dup(1)
    old_err = os.dup(2)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    try:
        yield
    finally:
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out)
        os.close(old_err)


def _normalize_box_coords_for_quantization(onnx_path: str, output_path: str, input_size) -> float:
    """Insert a Div node before the final box/confidence Concat to keep box coords
    in the same order of magnitude as confidence before RKNN quantizes the tensor.
    """
    import numpy as np
    import onnx
    import onnx.numpy_helper
    import onnx.shape_inference

    model = onnx.load(onnx_path)
    inferred = onnx.shape_inference.infer_shapes(model)
    inferred_graph = inferred.graph

    output_names = {output.name for output in inferred_graph.output}
    concat_node = None
    for node in inferred_graph.node:
        if node.op_type == "Concat" and any(out_name in output_names for out_name in node.output):
            concat_node = node
            break
    if concat_node is None:
        raise RuntimeError(
            f"No Concat node feeding a graph output found in {onnx_path}."
        )

    value_info = {
        vi.name: vi
        for vi in list(inferred_graph.value_info) + list(inferred_graph.input) + list(inferred_graph.output)
    }

    def _has_dim(name: str, target: int):
        value = value_info.get(name)
        if value is None:
            return False
        dims = value.type.tensor_type.shape.dim
        return any(int(dim.dim_value) == target for dim in dims if dim.dim_value > 0)

    box_input_name = None
    for input_name in concat_node.input:
        if _has_dim(input_name, 4):
            box_input_name = input_name
            break
    if box_input_name is None:
        raise RuntimeError(
            f"Could not identify the 4-channel box input for Concat '{concat_node.name}'."
        )

    if hasattr(input_size, "__iter__"):
        divisor = float(input_size[0] if len(input_size) > 0 else 640)
    else:
        divisor = float(input_size)

    scale_name = "iSpy_box_coord_scale"
    if not any(init.name == scale_name for init in model.graph.initializer):
        scale_tensor = onnx.numpy_helper.from_array(np.array(divisor, dtype=np.float32), name=scale_name)
        model.graph.initializer.append(scale_tensor)

    normalized_name = f"{box_input_name}_iSpy_normalized"
    div_node = onnx.helper.make_node(
        "Div",
        inputs=[box_input_name, scale_name],
        outputs=[normalized_name],
        name="iSpy_Div_box_normalize",
    )

    nodes = list(model.graph.node)
    target_idx = next(
        idx
        for idx, node in enumerate(nodes)
        if node.op_type == "Concat" and list(node.output) == list(concat_node.output)
    )
    nodes.insert(target_idx, div_node)
    target_concat = nodes[target_idx + 1]
    for idx, input_name in enumerate(target_concat.input):
        if input_name == box_input_name:
            target_concat.input[idx] = normalized_name
            break

    del model.graph.node[:]
    model.graph.node.extend(nodes)

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    logger.info(
        "Box-coordinate normalization applied: divided '%s' by %.1f before Concat '%s'",
        box_input_name,
        divisor,
        concat_node.name,
    )
    return divisor


def _export_rknn_metadata(
    pt_file: str,
    rknn_output,
    input_size=None,
    *,
    output_format=None,
    output_layout=None,
    box_format=None,
    quantize=None,
    box_coord_scale=None,
) -> None:
    try:
        pt_path = Path(pt_file)
        rknn_path = Path(rknn_output)
        pt_meta = read_metadata(pt_path) or metadata_from_pt(pt_path)
        meta = derive_format_metadata(pt_meta, "rknn")

        # Override detected fields from RKNN inference probe
        if output_format is not None:
            meta["output_format"] = output_format
        if output_layout is not None:
            meta["output_layout"] = output_layout
        if box_format is not None:
            meta["box_format"] = box_format
        if quantize is not None:
            meta["quantization"] = "int8" if quantize else "none"
            meta["quant_scale"] = 255.0 if quantize else 1.0
            meta["input_dtype"] = "uint8" if quantize else "float32"
            meta["quantize"] = quantize
        if box_coord_scale is not None:
            meta["box_coord_scale"] = float(box_coord_scale)
        if input_size is not None:
            if hasattr(input_size, "__iter__"):
                meta["input_size"] = [int(x) for x in input_size]
            else:
                meta["input_size"] = [int(input_size), int(input_size)]

        meta_path = metadata_path_for(rknn_path)
        write_metadata(meta_path, meta)
        logger.info("Exported RKNN metadata: %s", meta_path)
    except Exception as e:
        logger.warning("Failed to export RKNN metadata: %s", e)
 
 
def _yolo_models_dir() -> Path:
    return _PROJECT_ROOT / "YoloModels"
 
 
def _format_output_dir(target_format: str) -> Path:
    if target_format == "pytorch":
        return _yolo_models_dir() / "pytorch"
    return _yolo_models_dir() / target_format
 
 
def _desired_output_path(pt_path: Path, target_format: str) -> Path:
    stem = pt_path.stem
    out_dir = _format_output_dir(target_format)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    if target_format == "rknn":
        return out_dir / f"{stem}.rknn"
    if target_format == "onnx":
        return out_dir / f"{stem}.onnx"
    if target_format == "openvino":
        return out_dir / f"{stem}_openvino_model"
    if target_format == "coreml":
        return out_dir / f"{stem}.mlpackage"
    if target_format == "engine":
        return out_dir / f"{stem}.engine"
    if target_format == "tflite":
        return out_dir / f"{stem}.tflite"
    return out_dir / f"{stem}.{target_format}"
 
 
def _organize_exported_output(result_path: Path, desired_path: Path) -> Path:
    if not result_path.exists():
        return result_path
    if result_path == desired_path:
        return result_path
 
    desired_path.parent.mkdir(parents=True, exist_ok=True)
    if desired_path.exists():
        if desired_path.is_dir():
            shutil.rmtree(desired_path)
        else:
            desired_path.unlink()
 
    if result_path.is_dir():
        shutil.move(str(result_path), str(desired_path))
        return desired_path
    shutil.move(str(result_path), str(desired_path))
    return desired_path
 
 
def _find_tflite_artifact(saved_path: Path) -> Path | None:
    if saved_path.is_file() and saved_path.suffix == ".tflite":
        return saved_path
    if saved_path.is_dir():
        candidates = list(saved_path.rglob("*.tflite"))
        return candidates[0] if candidates else None
    return None
 
 



def _convert_rknn(pt_file, input_size, dataset_path, task="detect", quantize=None, kw=None):
    if quantize is None:
        quantize = _RKNN_QUANTIZE
    if kw is None:
        kw = keywords
    pt_path = Path(pt_file)
    raw_onnx = Path(_export_ultralytics(str(pt_path), "onnx", input_size))
    if not raw_onnx.exists():
        raise RuntimeError(f"Intermediate ONNX export failed: {raw_onnx}")

    # Route intermediate ONNX to the onnx folder with its own sidecar
    try:
        onnx_path = _desired_output_path(pt_path, "onnx")
        if raw_onnx != onnx_path:
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            if onnx_path.exists():
                if onnx_path.is_dir():
                    shutil.rmtree(onnx_path)
                else:
                    onnx_path.unlink()
            shutil.move(str(raw_onnx), str(onnx_path))
        pt_meta = read_metadata(pt_path) or metadata_from_pt(pt_path)
        format_meta = derive_format_metadata(pt_meta, "onnx")
        format_meta["input_size"] = (
            list(input_size) if hasattr(input_size, "__iter__") else [int(input_size), int(input_size)]
        )
        write_metadata(metadata_path_for(onnx_path), format_meta)
        logger.info("Intermediate ONNX routed to %s with sidecar", onnx_path)
    except Exception as e:
        logger.warning("Could not route intermediate ONNX: %s", e)
        onnx_path = raw_onnx

    try:
        from rknn.api import RKNN
    except ImportError:
        raise ImportError(
            "RKNN Toolkit not found. Install it to convert to RKNN format."
        )

    prepare_quantization_dataset(dataset_path, boot=True, keywords=kw)
    dataset_txt = Path(dataset_path) / "dataset.txt"
    if not dataset_txt.exists() or not dataset_txt.read_text().strip():
        raise FileNotFoundError(
            f"RKNN calibration dataset could not be prepared at: {dataset_txt}"
        )

    rknn_output = _desired_output_path(pt_path, "rknn")
    logger.info("Converting ONNX -> RKNN with dataset=%s", dataset_txt)

    box_coord_scale = None
    onnx_path_for_build = onnx_path
    surgery_path = onnx_path.parent / f"{onnx_path.stem}_rknn_ready.onnx"
    try:
        box_coord_scale = _normalize_box_coords_for_quantization(
            str(onnx_path), str(surgery_path), input_size
        )
        onnx_path_for_build = surgery_path
    except Exception as exc:
        logger.warning(
            "Box-coordinate normalization for RKNN quantization failed: %s. "
            "Proceeding without it; confidence may collapse to zero.",
            exc,
        )

    rknn = RKNN(verbose=False)
    detected_format = None
    detected_layout = None
    detected_box_format = None
    try:
        config_kwargs = dict(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform="rk3588",
            disable_rules=["fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa"],
            quantized_hybrid_level=3
        )
        if quantize:
            config_kwargs["quantized_dtype"] = "asymmetric_quantized-8"
            config_kwargs["quantized_algorithm"] = "kl_divergence"
            
        rknn.config(**config_kwargs)
        ret = rknn.load_onnx(model=str(onnx_path_for_build))
        if ret != 0:
            raise RuntimeError(f"RKNN load_onnx failed with code {ret}")
        ret = rknn.build(do_quantization=quantize, dataset=str(dataset_txt))
        if ret != 0:
            raise RuntimeError(f"RKNN build failed with code {ret}")
        ret = rknn.export_rknn(str(rknn_output))
        if ret != 0:
            raise RuntimeError(f"RKNN export failed with code {ret}")

        # Detect actual output format by running a quick inference
        try:
            import numpy as np
            rknn.init_runtime()
            if isinstance(input_size, int):
                h = w = input_size
            elif isinstance(input_size, (list, tuple)):
                h, w = int(input_size[0]), int(input_size[1]) if len(input_size) > 1 else int(input_size[0])
            else:
                h = w = 640
            dummy = np.zeros((1, h, w, 3), dtype=np.uint8)
            outputs = rknn.inference(inputs=[dummy])
            if outputs and len(outputs) > 0:
                tensor = outputs[0]
                t = tensor[0] if tensor.ndim == 3 else tensor
                smaller = min(t.shape[0], t.shape[-1])
                larger = max(t.shape[0], t.shape[-1])
                is_nms = (smaller == 6 and larger < 1000)
                detected_format = "hardware_nms" if is_nms else "raw"
                detected_layout = "anchors_first" if is_nms else "features_first"
                detected_box_format = "xyxy" if is_nms else "cxcywh"
                logger.info(
                    "RKNN output shape %s -> detected format: %s",
                    tensor.shape, detected_format,
                )
        except Exception as e:
            logger.debug("RKNN inference for format detection failed: %s", e)
    finally:
        rknn.release()

    logger.info("RKNN conversion successful: %s", rknn_output)
    _export_rknn_metadata(
        pt_file, rknn_output, input_size=input_size,
        output_format=detected_format,
        output_layout=detected_layout,
        box_format=detected_box_format,
        quantize=quantize,
        box_coord_scale=box_coord_scale,
    )
    return str(rknn_output)


def convert_model(model_file, target_format, input_size, quantize=None, force=False, kw=None):
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

    if target_format == "rknn":
        if quantize is None:
            quantize = _RKNN_QUANTIZE
        rknn_path = _desired_output_path(pt_path, "rknn")
        if rknn_path.exists() and not force:
            meta_path = metadata_path_for(rknn_path)
            if not meta_path.exists():
                _export_rknn_metadata(model_file, rknn_path, quantize=quantize)
            else:
                old_meta = read_metadata(rknn_path) or {}
                stored_quantize = old_meta.get("quantize")
                if stored_quantize is None:
                    q = old_meta.get("quantization")
                    stored_quantize = q != "none" if q is not None else None
                if stored_quantize is not None and stored_quantize != quantize:
                    logger.info(
                        "Cached rknn model %s has quantize=%s but config says %s. Re-converting.",
                        rknn_path.name, stored_quantize, quantize,
                    )
                    rknn_path.unlink()
                    meta_path.unlink()
                    rknn_result = _convert_rknn(
                        pt_file=model_file,
                        input_size=input_size,
                        dataset_path=str(_PROJECT_ROOT / "QuantizeDataset"),
                        quantize=quantize,
                        kw=kw,
                    )
                    _run_optimized_model_comparison(model_file, rknn_result)
                    return rknn_result
            logger.info("Cached rknn model found: %s", rknn_path)
            return str(rknn_path)
        if force and rknn_path.exists():
            logger.info("Fresh conversion forced for %s", rknn_path.name)
            rknn_path.unlink()
            meta = metadata_path_for(rknn_path)
            if meta.exists():
                meta.unlink()
        return _convert_rknn(
            pt_file=model_file,
            input_size=input_size,
            dataset_path=str(_PROJECT_ROOT / "QuantizeDataset"),
            quantize=quantize,
            kw=kw,
        )

    desired = _desired_output_path(pt_path, target_format)
    if not force:
        if target_format == "tflite":
            if desired.exists():
                logger.info("Cached tflite model found: %s", desired)
                return str(desired)
        else:
            if desired.exists():
                logger.info("Cached %s model found: %s", target_format, desired)
                return str(desired)

    dataset_root = str(_PROJECT_ROOT / "QuantizeDataset")
    data_yaml = None
    if quantize:
        prepare_quantization_dataset(dataset_root, boot=True, keywords=kw if kw is not None else keywords)
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
        desired_path = _desired_output_path(pt_path, target_format)
        if result_path.exists():
            if target_format == "tflite":
                if result_path.is_dir():
                    tflite_artifact = _find_tflite_artifact(result_path)
                    if tflite_artifact:
                        result_path = tflite_artifact
                if not result_path.exists() or result_path.suffix != ".tflite":
                    logger.warning(
                        "TFLite export did not produce a .tflite artifact at %s",
                        result,
                    )
                else:
                    if result_path.parent != desired_path.parent:
                        desired_path = desired_path
                        desired_path.parent.mkdir(parents=True, exist_ok=True)
                        if desired_path.exists():
                            desired_path.unlink()
                        shutil.move(str(result_path), str(desired_path))
                        result_path = desired_path
            else:
                if result_path != desired_path:
                    if desired_path.exists():
                        if desired_path.is_dir():
                            shutil.rmtree(desired_path)
                        else:
                            desired_path.unlink()
                    shutil.move(str(result_path), str(desired_path))
                    result_path = desired_path
 
            logger.info("%s export successful: %s", target_format, result_path)
            # Write derived metadata for converted file based on source .pt
            try:
                pt_meta = read_metadata(pt_path) or metadata_from_pt(pt_path)
                format_meta = derive_format_metadata(pt_meta, target_format)
                format_meta["input_size"] = list(input_size) if hasattr(input_size, "__iter__") else [int(input_size), int(input_size)]
                write_metadata(metadata_path_for(result_path), format_meta)
                logger.info("Wrote metadata for converted %s", result_path.name)
            except Exception as e:
                logger.warning("Could not write metadata for converted %s: %s", result_path.name, e)

            _run_optimized_model_comparison(model_file, str(result_path))
            return str(result_path)

    logger.warning("Conversion to %s failed, falling back to .pt", target_format)
    return model_file

def setup_files(first_boot: bool = False):
    yolo_dir = _PROJECT_ROOT / "YoloModels"
    config_dir = _PROJECT_ROOT / "Config"
    outputs_dir = _PROJECT_ROOT / "Outputs"
    dataset_dir = _PROJECT_ROOT / "QuantizeDataset"

    saved_config = None
    if first_boot:
        config_path = config_dir / "config.json"
        if config_path.exists():
            try:
                with open(str(config_path)) as f:
                    existing_config = json.load(f)

                default_cfg = _default_config_dict()
                comparable_existing = _strip_boot_managed_fields(existing_config)
                comparable_default = _strip_boot_managed_fields(default_cfg)

                if comparable_existing != comparable_default:
                    # Real user customization (camera geometry, NT settings,
                    # thresholds, plugins, min_conf, etc.) - keep it, but
                    # reset the boot-derived vision_model fields to fresh
                    # defaults since YoloModels (and the artifacts file_path/
                    # task/output/input describe) is about to be deleted.
                    saved_config = comparable_existing
                    saved_config["vision_model"] = {
                        **default_cfg.get("vision_model", {}),
                        **comparable_existing.get("vision_model", {}),
                    }
                    logger.info("Preserving user config (differs from default)")
                else:
                    logger.info("Existing config matches defaults - nothing to preserve")
            except Exception as e:
                logger.warning("Could not read existing config: %s", e)

        for d in [yolo_dir, config_dir, outputs_dir, dataset_dir]:
            if d.exists():
                shutil.rmtree(d)
                logger.info("Deleted %s", d)

    yolo_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["pytorch", "onnx", "tflite", "rknn", "openvino", "coreml", "engine"]:
        (yolo_dir / fmt).mkdir(parents=True, exist_ok=True)
    prepare_quantization_dataset(str(dataset_dir), boot=True, keywords=keywords)

    if saved_config is not None:
        config_path = config_dir / "config.json"
        with open(str(config_path), "w") as f:
            json.dump(saved_config, f, indent=4)
        logger.info("Restored user config (vision_model artifact paths reset for fresh conversion)")

    pytorch_dir = yolo_dir / "pytorch"
    _SKIP_DIRS = {
        ".venv",
        "__pycache__",
        ".git",
        ".pytest_cache",
        "env",
        "runs",
        "dist",
    }
    seen = set()
    for pt_file in _ASSETS_DIR.rglob("*.pt"):
        target = pytorch_dir / pt_file.name
        if first_boot or not target.exists():
            shutil.copy2(pt_file, target)
            seen.add(pt_file.name)
            logger.info("Copied model %s -> %s", pt_file.name, target)
    if first_boot:
        for pt_file in _PROJECT_ROOT.rglob("*.pt"):
            if pt_file.name in seen:
                continue
            if any(
                part in _SKIP_DIRS for part in pt_file.relative_to(_PROJECT_ROOT).parts
            ):
                continue
            target = pytorch_dir / pt_file.name
            try:
                if target.exists() and os.path.samefile(pt_file, target):
                    continue
            except (OSError, ValueError):
                pass
            shutil.copy2(pt_file, target)
            seen.add(pt_file.name)
            logger.info("Copied model %s -> %s", pt_file.name, target)

    # Ensure every .pt in the pytorch directory has a metadata sidecar
    try:
        for pt_file in pytorch_dir.glob("*.pt"):
            meta_path = metadata_path_for(pt_file)
            if not meta_path.exists():
                try:
                    logger.info("Generating metadata for %s", pt_file.name)
                    meta = metadata_from_pt(pt_file)
                    write_metadata(meta_path, meta)
                    logger.info("Wrote metadata %s", meta_path.name)
                except Exception as e:
                    logger.warning("Could not generate metadata for %s: %s", pt_file.name, e)
    except Exception:
        pass

def on_boot(install_service: bool = False, first_boot: bool = False):
    if first_boot:
        logger.info("First boot mode - ensuring fresh conversion for selected model")
    setup_files(first_boot=first_boot)
    config_path = search_for_config()
    config = None
    if not config_path:
        logger.info("No config found. Creating default config...")
        config_path = _PROJECT_ROOT / "Config" / "config.json"
        config = iSpyConfig(str(config_path), create=True)
    else:
        logger.info("Using existing config: %s", config_path)

    if not validate_system():
        raise RuntimeError("System validation failed. Aborting boot.")

    if config is None:
        config = iSpyConfig(str(config_path))

    best_format = None
    if config.get("auto_opt"):
        install_special_dependencies(auto_install=True)

        best_format = recommend_format(ignore_dependencies=True)
        logger.info("Auto-opt enabled. Recommended format: %s", best_format)

        def _cached_output(pt_path: Path) -> Path | None:
            if best_format == "tpu":
                return pt_path if pt_path.exists() else None
            desired = _desired_output_path(pt_path, best_format)
            if desired.exists():
                if best_format == "rknn":
                    old_meta = read_metadata(desired) or {}
                    stored_q = old_meta.get("quantize")
                    if stored_q is None:
                        stored_q = old_meta.get("quantization")
                        stored_q = stored_q != "none" if stored_q is not None else None
                    if stored_q is not None and stored_q != _RKNN_QUANTIZE:
                        logger.info(
                            "Cached rknn model %s has quantize=%s but boot says %s. Re-converting.",
                            desired.name, stored_q, _RKNN_QUANTIZE,
                        )
                        return None
                if best_format == "tflite":
                    if desired.is_file():
                        return desired
                    tflite_artifact = _find_tflite_artifact(desired)
                    if tflite_artifact:
                        return tflite_artifact
                else:
                    return desired
            return None

        pytorch_dir = _PROJECT_ROOT / "YoloModels/pytorch"
        pt_path = config.get("vision_model", {}).get("source_pt")
        pt_full = None
        if pt_path:
            pt_full = Path(pt_path)
            if not pt_full.is_absolute():
                pt_full = _PROJECT_ROOT / pt_full

        if pt_full and pt_full.exists():
            cached = _cached_output(pt_full)
            if cached:
                logger.info("Found cached %s model: %s", best_format, cached)
                config.set("vision_model", "file_path", str(cached))
                need_conversion = False
            else:
                logger.info(
                    "No cached %s model for %s. Converting...", best_format, pt_full.name
                )
                need_conversion = True
        else:
            if pt_full and not pt_full.exists():
                logger.warning("Configured source_pt %s not found, scanning...", pt_full)
            candidates = sorted(pytorch_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            user_models = [p for p in candidates if p.name not in _BUNDLED_DEFAULT_MODELS]
            if user_models:
                pt_full = user_models[0]
                logger.info("Auto-detected user model: %s", pt_full.name)
            elif candidates:
                pt_full = candidates[0]
                logger.info("Using default model: %s", pt_full.name)
            else:
                logger.warning("No .pt found, copying bundled _default_pose.pt")
                bundled = _ASSETS_DIR / "_default_pose.pt"
                pt_full = pytorch_dir / "_default_pose.pt"
                pt_full.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(bundled, pt_full)
            config.set("vision_model", "source_pt", str(pt_full))
            need_conversion = True

        if need_conversion:
            input_size = config.get("input_size") or [640, 640]
            converted = Path(
                convert_model(
                    str(pt_full),
                    best_format,
                    input_size,
                    quantize=_RKNN_QUANTIZE,
                    force=first_boot,
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

        # Convert all other .pt models too (ensures correct metadata for bench/run)
        if pytorch_dir.exists():
            other_input_size = config.get("input_size") or [640, 640]
            for other in pytorch_dir.glob("*.pt"):
                if pt_full and other.resolve() == pt_full.resolve():
                    continue
                try:
                    convert_model(
                        str(other),
                        best_format,
                        other_input_size,
                        quantize=_RKNN_QUANTIZE,
                        force=first_boot,
                    )
                except Exception as e:
                    logger.warning("Could not convert %s: %s", other.name, e)
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

    if best_format == "tpu":
        config.set("vision_model", "device", "tpu")
        logger.info("TPU backend configured - device set to 'tpu'")

    vision_cfg = config.get("vision_model", {})
    filled = fill_missing_config(vision_cfg)
    
    try:
        full_config_path = _PROJECT_ROOT / "Outputs" / "full_config.json"
        full_config_path.parent.mkdir(parents=True, exist_ok=True)
        full_resolved = json.loads(json.dumps(config.config))  # deep copy
        full_resolved["vision_model"] = filled
        with open(full_config_path, "w") as f:
            json.dump(full_resolved, f, indent=4)
        logger.info("Full resolved config (debug only) written to %s", full_config_path)
    except Exception as e:
        logger.warning("Could not write full_config.json: %s", e)

    # Save only user-facing fields to config; model architecture (task, input_size,
    # output.*, input.*, num_classes) lives exclusively in metadata sidecars.
    minimal = {
        "file_path": filled.get("file_path", vision_cfg.get("file_path", "")),
        "min_conf": filled.get("min_conf", vision_cfg.get("min_conf", 0.25)),
        "margin": filled.get("margin", vision_cfg.get("margin", 0)),
    }
    for key in ("device", "quantized"):
        val = filled.get(key, vision_cfg.get(key))
        if val is not None:
            minimal[key] = val
    # Preserve any unknown user-defined fields from original config
    for key in vision_cfg:
        if key not in _METADATA_ONLY_FIELDS and key not in minimal:
            minimal[key] = vision_cfg[key]
    config.set("vision_model", minimal)
    config.save(quiet=True)
    logger.info("Minimal model config saved to config (architecture in metadata only).")
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


def _any_camera_uses_csi() -> bool:
    config_path = search_for_config()
    if not config_path:
        return False
    try:
        with open(config_path) as f:
            data = json.load(f)
        cams = data.get("config", data).get("camera_configs", {})
        return any(c.get("csi", False) for c in cams.values())
    except Exception:
        return False


def main():
    if has_jetson() and _any_camera_uses_csi():
        if ensure_csi_capable_opencv(auto_fix=True):
            logger.info("OpenCV fixed - re-executing boot.py to pick it up...")
            os.execv(sys.executable, [sys.executable] + sys.argv)

    parser = argparse.ArgumentParser(description="iSpy boot sequence")
    parser.add_argument("-s", "--service", action="store_true",
                         help="Install and start the watchdog service")
    parser.add_argument("-f", "--first-boot", action="store_true",
                         help="Delete Config, Outputs, and YoloModels before booting")
    args = parser.parse_args()
    on_boot(install_service=args.service, first_boot=args.first_boot)

if __name__ == "__main__":
    main()
