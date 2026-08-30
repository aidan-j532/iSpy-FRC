import contextlib
import importlib.metadata
import importlib.util
import io
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time as _time
import warnings
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path.cwd().resolve()
_PACKAGE_ROOT = Path(__file__).resolve().parent

_REAL_STDOUT_FD = os.dup(1)
_REAL_STDERR_FD = os.dup(2)
_REAL_STDOUT = os.fdopen(_REAL_STDOUT_FD, "w", buffering=1, closefd=False)
_REAL_STDERR = os.fdopen(_REAL_STDERR_FD, "w", buffering=1, closefd=False)


class _NullWriter(io.TextIOBase):
    def write(self, s):
        return len(s)

    def flush(self):
        pass


@contextlib.contextmanager
def _progress_spinner(label: str):
    stop = threading.Event()

    def _spin():
        frames = "|/-\\"
        i = 0
        start = _time.time()
        while not stop.is_set():
            elapsed = _time.time() - start
            print(f"\r  {label} {frames[i % 4]} ({elapsed:.0f}s)", end="", file=_REAL_STDOUT, flush=True)
            i += 1
            stop.wait(0.2)
        print(f"\r  {label} done ({_time.time() - start:.0f}s)" + " " * 10, file=_REAL_STDOUT)

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()


@contextlib.contextmanager
def _silence_third_party():
    devnull = "nul" if os.name == "nt" else "/dev/null"
    fd = os.open(devnull, os.O_WRONLY)
    old_out_fd = os.dup(1)
    old_err_fd = os.dup(2)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _NullWriter()
    sys.stderr = _NullWriter()

    _muted_loggers = ("ultralytics", "nncf")
    _saved_levels, _saved_disabled = {}, {}
    for name in _muted_loggers:
        lg = logging.getLogger(name)
        _saved_levels[name] = lg.level
        _saved_disabled[name] = lg.disabled
        lg.setLevel(logging.CRITICAL + 1)
        lg.disabled = True

    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        os.dup2(old_out_fd, 1)
        os.dup2(old_err_fd, 2)
        os.close(old_out_fd)
        os.close(old_err_fd)
        for name in _muted_loggers:
            lg = logging.getLogger(name)
            lg.setLevel(_saved_levels[name])
            lg.disabled = _saved_disabled[name]


keywords = ["frc game piece", "frc 2025 REBUILT", "frc 2025 fuel"]

_RKNN_QUANTIZE = True
_RKNN_KNOWN_CHIPS = (
    "rk3588", "rk3576", "rk3399", "rk3568", "rk3566",
    "rk3562", "rk3528", "rv1103", "rv1106",
)

_MANUAL_POSTPROCESS_FORMATS = {"onnx", "tflite"}

_ARCH = platform.machine().lower()
_IS_AARCH64 = "aarch64" in _ARCH or "arm64" in _ARCH
_PY_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"

_JETSON_SYSTEM_MANAGED = {"tensorrt", "onnxruntime"}


def _model_supports_end2end(model: "ultralytics.YOLO") -> bool:
    return bool(getattr(model.model, "end2end", False))


def _find_lite_wheel_dir() -> Path:
    pkg_dir = _PACKAGE_ROOT.parent / "rknn_wheels"
    if pkg_dir.exists():
        return pkg_dir
    try:
        spec = importlib.util.find_spec("iSpy")
        if spec and spec.origin:
            pkg = Path(spec.origin).parent / "rknn_wheels"
            if pkg.exists():
                return pkg
    except Exception:
        pass
    # fall back to a user-dropped copy at the project root
    return _PACKAGE_ROOT.parent.parent / "rknn_wheels"


_RKNN_LITE_DIR = _find_lite_wheel_dir()

_RKNN_FULL_BASE = (
    "https://github.com/aidan-j532/iSpy-FRC/releases/download/RKNN_Wheels"
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


def _detect_rknn_target_platform() -> str | None:
    """Return the detected Rockchip SoC, or None when it cannot be determined.

    An explicit override via ISPY_RKNN_TARGET_PLATFORM always wins (users
    who know the board do not have to rely on device-tree heuristics).
    Returning None instead of a silent 'rk3588' default is deliberate: the
    caller stamps a visible warning and puts it in the metadata sidecar so a
    wrong-target artifact can never be produced unnoticed (Day 6).
    """
    override = os.environ.get("ISPY_RKNN_TARGET_PLATFORM", "").strip().lower()
    if override:
        logger.info(
            "RKNN target_platform override (ISPY_RKNN_TARGET_PLATFORM): %s",
            override,
        )
        return override

    for path in (
        "/proc/device-tree/compatible",
        "/proc/device-tree/model",
        "/sys/firmware/devicetree/base/model",
    ):
        try:
            content = open(path, "rb").read().decode(errors="ignore").lower()
            for chip in _RKNN_KNOWN_CHIPS:
                if chip in content:
                    logger.info("Detected RKNN target_platform: %s (from %s)", chip, path)
                    return chip
        except Exception:
            continue

    try:
        cpuinfo = open("/proc/cpuinfo").read().lower()
        for chip in _RKNN_KNOWN_CHIPS:
            if chip in cpuinfo:
                logger.info("Detected RKNN target_platform: %s (from /proc/cpuinfo)", chip)
                return chip
    except Exception:
        pass

    logger.warning(
        "Could not detect Rockchip SoC from device-tree or /proc/cpuinfo - "
        "RKNN target_platform is unknown on this host."
    )
    return None


def _resolve_rknn_target_platform() -> tuple[str, bool]:
    """(target_platform, detected) for RKNN builds.

    When the SoC cannot be detected the build falls back to 'rk3588' but the
    failure is made unmissable: a hard print (surfaces in the conversion log
    the UI streams), a logger warning, and a metadata sidecar stamp via
    _export_rknn_metadata. Set ISPY_RKNN_TARGET_PLATFORM to opt out."""
    target = _detect_rknn_target_platform()
    if target:
        return target, True

    message = (
        "\n"
        "WARNING: could not detect the Rockchip SoC for RKNN conversion.\n"
        "Defaulting target_platform to 'rk3588' - this is a GUESS.\n"
        "Wrong target = rknn-toolkit2 build errors or a silently broken "
        "artifact.\n"
        "Fix: set ISPY_RKNN_TARGET_PLATFORM=<chip> (rk3588/rk3566/rk3576/...)\n"
        "     or run boot.py directly on the target board.\n"
    )
    print(message, file=_REAL_STDOUT)
    logger.warning(message.strip())
    return "rk3588", False


_CALIB_DOWNSCALE_SIZE = 320


def _downscale_calib_images(dataset_path: Path, calib_size: int = _CALIB_DOWNSCALE_SIZE) -> Path:
    import cv2

    orig_txt = dataset_path / "dataset.txt"
    if not orig_txt.exists():
        return dataset_path

    lines = [l.strip() for l in orig_txt.read_text().splitlines() if l.strip()]
    if not lines:
        return dataset_path

    tmp_dir = Path(tempfile.mkdtemp(prefix="ispy_rknn_calib_"))
    resized_dir = tmp_dir / "images"
    resized_dir.mkdir(parents=True, exist_ok=True)

    resized_lines = []
    resized_count = 0
    for line in lines:
        img_path = dataset_path / line if not Path(line).is_absolute() else Path(line)
        if not img_path.exists():
            continue
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            if h == calib_size and w == calib_size:
                dest = resized_dir / img_path.name
                shutil.copy2(str(img_path), str(dest))
            else:
                resized = cv2.resize(img, (calib_size, calib_size), interpolation=cv2.INTER_LINEAR)
                dest = resized_dir / f"{img_path.stem}_{calib_size}.jpg"
                cv2.imwrite(str(dest), resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            resized_lines.append(f"images/{dest.name}")
            resized_count += 1
        except Exception:
            continue

    if resized_count == 0:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        return dataset_path

    (tmp_dir / "dataset.txt").write_text("\n".join(resized_lines) + "\n")
    logger.info(
        "Downscaled %d calibration images from original to %dx%d (saves ~4x RAM during RKNN build)",
        resized_count, calib_size, calib_size,
    )
    return tmp_dir


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
        logger.info(
            "Skipping optimized-model comparison for %s - conversion fell back "
            "to the source .pt (no optimized artifact was produced).",
            Path(pt_file).name,
        )
        return

    valid_dir = default_quantization_dataset_dir() / "valid"
    if not valid_dir.exists():
        logger.info(
            "Skipping optimized-model comparison for %s - no %s found.",
            converted_path.name,
            valid_dir,
        )
        return

    logger.info("Running optimized-model comparison for %s...", converted_path.name)

    try:
        from iSpy.validations.tests.compare_models import compare_models  # lazy: heavy import chain

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
    from iSpy.config.AutoOpt import recommend_format, has_jetson

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


def _export_ultralytics(model_file, target_format, input_size, data_yaml=None, device=0):
    """Export a .pt model to a compiled/optimized format.

    This is optional *development tooling*, NOT part of the runtime inference
    pipeline. It shells out to Ultralytics (AGPL-3.0) purely to produce build
    artifacts; the resulting model file is then consumed at runtime by the
    on-device loader, which never imports Ultralytics. Because the exporter is
    a separate, one-time build step, Ultralytics is intentionally NOT a runtime
    dependency of iSpy - install it on demand via the ``[optimizer]`` extra.

    Raises:
        ImportError: If Ultralytics is not installed (install the ``optimizer``
            extra or ``pip install ultralytics`` in a build environment).
    """
    try:
        import ultralytics  # optional dev-only dependency (AGPL) - build-time only
    except ImportError:
        raise ImportError(
            "Model export requires Ultralytics (AGPL) which is an optional, "
            "build-time tool. Install the 'optimizer' extra (pip install "
            "ispy-frc[optimizer]) in a build environment, or pre-export the "
            "model elsewhere. Ultralytics is never needed for runtime inference."
        ) from None

    try:
        model = ultralytics.YOLO(model_file, weights_only=True)
    except TypeError:
        model = ultralytics.YOLO(model_file)
    has_e2e = _model_supports_end2end(model)
    task = getattr(model, "task", None) or "detect"

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
        effective_data_yaml = data_yaml
        if task == "pose":
            try:
                kpt_shape = model.model.model[-1].kpt_shape
            except Exception:
                kpt_shape = [17, 3]
            pose_yaml = Path(data_yaml).parent / "data_pose.yaml"
            try:
                base = Path(data_yaml).read_text()
            except OSError:
                base = "train: images\nval: valid/images\nnc: 1\nnames: ['object']\n"
            if not base.strip():
                base = "train: images\nval: valid/images\nnc: 1\nnames: ['object']\n"
            pose_yaml.write_text(base.rstrip() + f"\nkpt_shape: {list(kpt_shape)}\n")
            effective_data_yaml = str(pose_yaml)
            logger.info("Pose task detected - using kpt_shape-aware data.yaml: %s", pose_yaml)

        kwargs = dict(format=target_format, imgsz=input_size, int8=True, data=effective_data_yaml)
        if has_e2e and target_format in _MANUAL_POSTPROCESS_FORMATS:
            kwargs["end2end"] = False
        if target_format == "engine":
            kwargs["device"] = device
            kwargs["half"] = True

        logger.info(
            "Dataset-aware %s quantization enabled (data=%s)", target_format, effective_data_yaml
        )

    logger.info("Exporting %s -> %s with kwargs: %s", model_file, target_format, kwargs)
    with _silence_third_party():
        return model.export(**kwargs)


def default_quantization_dataset_dir() -> Path:
    return _PROJECT_ROOT / "QuantizeDataset" / "default"


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
    kpt_coord_scale=None,
    target_platform=None,
    target_platform_detected=None,
) -> None:
    from iSpy.vision.metadata import (
        read_metadata,
        metadata_from_pt,
        derive_format_metadata,
        metadata_path_for,
        write_metadata,
    )

    try:
        pt_path = Path(pt_file)
        rknn_path = Path(rknn_output)
        pt_meta = read_metadata(pt_path) or metadata_from_pt(pt_path)
        meta = derive_format_metadata(pt_meta, "rknn")

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
        if kpt_coord_scale is not None:
            meta["kpt_coord_scale"] = float(kpt_coord_scale)
        if input_size is not None:
            if hasattr(input_size, "__iter__"):
                meta["input_size"] = [int(x) for x in input_size]
            else:
                meta["input_size"] = [int(input_size), int(input_size)]

        if target_platform:
            meta["target_platform"] = target_platform
        meta["target_platform_detected"] = bool(target_platform_detected)
        if not target_platform_detected:
            meta["warning"] = (
                "RKNN target_platform was NOT detected on the converting "
                f"host - artifact built for '{target_platform or 'rk3588'}' "
                "by default. Set ISPY_RKNN_TARGET_PLATFORM=<chip> or convert "
                "on the target board if this is wrong."
            )

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


def _artifact_name(pt_path: Path, target_format: str) -> str:
    stem = pt_path.stem
    if target_format == "rknn":
        return f"{stem}.rknn"
    if target_format == "onnx":
        return f"{stem}.onnx"
    if target_format == "openvino":
        return f"{stem}_openvino_model"
    if target_format == "coreml":
        return f"{stem}.mlpackage"
    if target_format == "engine":
        return f"{stem}.engine"
    if target_format == "tflite":
        return f"{stem}.tflite"
    return f"{stem}.{target_format}"


def _desired_output_path(pt_path: Path, target_format: str) -> Path:
    out_dir = _format_output_dir(target_format)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / _artifact_name(pt_path, target_format)


# optimized artifacts can be built for any of these - a camera runs whichever
# backend is active, so "already built" means any of them
_ARTIFACT_FORMATS = ("onnx", "rknn", "tflite", "openvino", "engine", "coreml")


def existing_artifact_for(
    source_pt: Path | str | None, target_format: str | None = None
) -> str | None:
    if not source_pt:
        return None
    source_pt = Path(source_pt)
    if source_pt.suffix.lower() != ".pt":
        return None
    requested = str(target_format or "").strip().lower()
    formats = [requested] if requested and requested != "auto" else []
    formats += [f for f in _ARTIFACT_FORMATS if f not in formats]
    for fmt in formats:
        out = _format_output_dir(fmt) / _artifact_name(source_pt, fmt)
        if out.exists():
            try:
                return out.resolve().relative_to(_PROJECT_ROOT).as_posix()
            except ValueError:
                return str(out)
    return None


import shutil
from pathlib import Path

def _remove_path_for_cleanup(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _find_tflite_artifact(saved_path: Path) -> Path | None:
    if saved_path.is_file() and saved_path.suffix == ".tflite":
        return saved_path
    if saved_path.is_dir():
        candidates = list(saved_path.rglob("*.tflite"))
        return candidates[0] if candidates else None
    return None


def _merge_rknn_build_outputs(outputs: list) -> "np.ndarray":
    import numpy as np

    if not outputs:
        return np.empty((0,), dtype=np.float32)
    if len(outputs) == 1:
        return outputs[0]

    parts: list[np.ndarray] = []
    for raw in outputs:
        t = raw[0] if raw.ndim == 3 else raw
        if t.ndim != 2:
            return outputs[0]
        if t.shape[0] > t.shape[1]:
            t = t.T
        parts.append(t)

    box_parts = [p for p in parts if p.shape[0] == 4]
    other_parts = [p for p in parts if p.shape[0] != 4]
    ordered = (box_parts + other_parts) if len(box_parts) == 1 else parts
    merged = np.concatenate(ordered, axis=0)
    return merged[np.newaxis, ...]


def _fold_scale_into_constant_input(graph, node, divisor, node_by_output):
    import numpy as np
    import onnx
    import onnx.numpy_helper

    initializer_map = {init.name: init for init in graph.initializer}
    const_name = None
    const_node = None
    for inp in node.input:
        if inp in initializer_map:
            const_name = inp
            break
    if const_name is None:
        # newer ultralytics exports bake stride multipliers into Constant nodes
        # ('/model.23/Constant_15' -> output) instead of graph initializers -
        # resolve those too or quantization proceeds with unnormalized box coords
        # and confidence collapses
        for inp in node.input:
            producer = node_by_output.get(inp)
            if producer is not None and producer.op_type == "Constant":
                const_node = producer
                const_name = inp
                break
    if const_name is None:
        raise RuntimeError(
            f"Node '{node.name}' ({node.op_type}) has no constant input among "
            f"{list(node.input)} - cannot fold a scale factor into it. "
            "Ultralytics export structure may have changed - inspect manually."
        )

    if const_node is None:
        arr = onnx.numpy_helper.to_array(initializer_map[const_name])
    else:
        value_attr = next(
            (a for a in const_node.attribute if a.name == "value"), None
        )
        if value_attr is None or value_attr.t is None:
            raise RuntimeError(
                f"Constant node '{const_node.name}' has no 'value' tensor "
                "attribute - cannot fold a scale factor into it."
            )
        arr = onnx.numpy_helper.to_array(value_attr.t)

    other_consumers = [
        n for n in graph.node if n is not node and const_name in n.input
    ]
    new_name = f"{const_name}_iSpy_scaled"
    scaled = onnx.numpy_helper.from_array(
        arr.astype(np.float32) / float(divisor), name=new_name
    )
    graph.initializer.append(scaled)
    for idx, inp in enumerate(node.input):
        if inp == const_name:
            node.input[idx] = new_name
    logger.info(
        "Cloned constant '%s' -> '%s' (scaled by 1/%.1f) so only node '%s' is "
        "affected (%d other consumer(s) left untouched).",
        const_name, new_name, divisor, node.name, len(other_consumers),
    )
    return new_name


def _normalize_box_coords_for_quantization(onnx_path: str, output_path: str, input_size) -> tuple[float, float | None]:
    import numpy as np
    import onnx
    import onnx.numpy_helper

    model = onnx.load(onnx_path)
    graph = model.graph

    output_names = {output.name for output in graph.output}
    concat_node = None
    for node in graph.node:
        if node.op_type == "Concat" and any(out_name in output_names for out_name in node.output):
            concat_node = node
            break
    if concat_node is None:
        raise RuntimeError(f"No Concat node feeding a graph output found in {onnx_path}.")

    inputs = list(concat_node.input)
    if len(inputs) < 2:
        raise RuntimeError(
            f"Output Concat '{concat_node.name}' has {len(inputs)} input(s), "
            "expected at least 2 (boxes, scores[, keypoints]) - "
            "Ultralytics export structure may have changed."
        )

    node_by_output = {out: n for n in graph.node for out in n.output}

    box_input_name = inputs[0]
    box_producer = node_by_output.get(box_input_name)
    if box_producer is None or box_producer.op_type != "Mul":
        raise RuntimeError(
            f"Expected box tensor '{box_input_name}' (Concat input 0) to be "
            f"produced by a Mul node (stride scaling), got "
            f"{box_producer.op_type if box_producer else 'unknown producer'}. "
            "Ultralytics export structure may have changed - inspect the ONNX "
            "graph manually before trusting this conversion."
        )

    if hasattr(input_size, "__iter__"):
        divisor = float(input_size[0] if len(input_size) > 0 else 640)
    else:
        divisor = float(input_size)

    _fold_scale_into_constant_input(graph, box_producer, divisor, node_by_output)
    logger.info(
        "Box-coordinate normalization applied: folded /%.1f into the existing "
        "stride constant feeding '%s' (Mul producing the box tensor) - no new "
        "nodes inserted, graph topology unchanged from an unmodified export.",
        divisor, box_producer.name,
    )

    kpt_coord_scale = None
    if len(inputs) > 2:
        kpt_input_name = inputs[2]
        kpt_reshape = node_by_output.get(kpt_input_name)
        if kpt_reshape is None or kpt_reshape.op_type != "Reshape":
            raise RuntimeError(
                f"Expected keypoint tensor '{kpt_input_name}' (Concat input 2) to "
                f"be produced by a Reshape node, got "
                f"{kpt_reshape.op_type if kpt_reshape else 'unknown'}. "
                "Ultralytics export structure may have changed - inspect manually."
            )

        kpt_concat_input = kpt_reshape.input[0]
        kpt_concat = node_by_output.get(kpt_concat_input)
        if kpt_concat is None or kpt_concat.op_type != "Concat" or len(kpt_concat.input) < 2:
            raise RuntimeError(
                f"Expected '{kpt_concat_input}' (feeding the keypoint Reshape) to "
                f"be produced by a Concat node with >=2 inputs (xy, confidence), "
                f"got {kpt_concat.op_type if kpt_concat else 'unknown'}."
            )

        xy_tensor_name = kpt_concat.input[0]  # [xy, confidence] order
        xy_producer = node_by_output.get(xy_tensor_name)
        if xy_producer is None or xy_producer.op_type != "Mul":
            raise RuntimeError(
                f"Expected keypoint xy tensor '{xy_tensor_name}' (inner Concat "
                f"input 0) to be produced by a Mul node (stride scaling), got "
                f"{xy_producer.op_type if xy_producer else 'unknown'}. "
                "Ultralytics export structure may have changed - inspect manually."
            )

        _fold_scale_into_constant_input(graph, xy_producer, divisor, node_by_output)
        kpt_coord_scale = divisor
        conf_input = kpt_concat.input[1] if len(kpt_concat.input) > 1 else "?"
        logger.info(
            "Keypoint-coordinate normalization applied: folded /%.1f into the "
            "existing stride constant feeding '%s' (Mul producing keypoint x/y) "
            "- confidence branch ('%s') left untouched, no new nodes inserted.",
            divisor, xy_producer.name, conf_input,
        )
    else:
        logger.debug(
            "Output Concat '%s' has no third input - assuming detect-only model.",
            concat_node.name,
        )

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return divisor, kpt_coord_scale

def _find_pose_output_tensors(graph, concat_node):
    node_by_output = {out: n for n in graph.node for out in n.output}
    inputs = list(concat_node.input)

    box_name = inputs[0]   # always first: decoded+stride-scaled boxes
    conf_name = inputs[1]  # always second: sigmoid'd confidence/class scores
    kpt_name = inputs[2] if len(inputs) > 2 else None  # pose only

    box_producer = node_by_output.get(box_name)
    if box_producer is None or box_producer.op_type != "Mul":
        raise RuntimeError(
            f"Expected box tensor '{box_name}' to be produced by a Mul node "
            f"(stride scaling), got {box_producer.op_type if box_producer else 'unknown'}. "
            "Ultralytics export structure may have changed - verify manually."
        )
    return box_name, conf_name, kpt_name


def _prepare_user_calibration_dataset(user_ds: Path) -> Path:
    from iSpy.dataset.dataset import _find_images as _find_ds_images

    dataset_txt = user_ds / "dataset.txt"
    stale = True
    if dataset_txt.exists():
        lines = [l for l in dataset_txt.read_text().splitlines() if l.strip()]
        stale = not all(
            (Path(l) if Path(l).is_absolute() else user_ds / l).exists()
            for l in lines
        )
    if not stale:
        return dataset_txt
    imgs = _find_ds_images(user_ds)
    if not imgs:
        raise FileNotFoundError(
            f"User quantization dataset contains no images: {user_ds}"
        )
    dataset_txt.write_text("\n".join(str(p.resolve()) for p in imgs) + "\n")
    logger.info(
        "Calibration dataset from user path %s (%d images)",
        user_ds, len(imgs),
    )
    return dataset_txt


def _user_calibration_data_yaml(pt_file, user_ds: Path) -> str:
    from iSpy.vision.metadata import read_metadata

    pt_meta = read_metadata(Path(pt_file)) or {}
    try:
        nc = max(1, int(pt_meta.get("nc") or pt_meta.get("num_classes") or 1))
    except (TypeError, ValueError):
        nc = 1
    names_meta = pt_meta.get("names") or {}
    if not names_meta:
        names = ["object"] if nc == 1 else [f"class_{i}" for i in range(nc)]
    else:
        names = [
            names_meta.get(i, names_meta.get(str(i), f"class_{i}"))
            for i in range(nc)
        ]
    if not names:
        names = ["object"]
    data_yaml = user_ds / "data.yaml"
    data_yaml.write_text(
        f"train: {user_ds}\nval: {user_ds}\nnc: {nc}\n"
        f"names: {json.dumps(names)}\n"
    )
    logger.info("Calibration data.yaml written (nc=%d) at %s", nc, data_yaml)
    return str(data_yaml)


def _resolve_calibration_dataset(dataset_path, keywords_list, count) -> tuple:
    from iSpy.dataset.dataset import prepare_quantization_dataset

    if dataset_path:
        user_ds = Path(dataset_path).resolve()
        try:
            _prepare_user_calibration_dataset(user_ds)
            return user_ds, True
        except FileNotFoundError:
            logger.warning(
                "User calibration dataset %s has no images - falling back to "
                "auto-downloading calibration images", user_ds,
            )
    ds_dir = default_quantization_dataset_dir()
    prepare_quantization_dataset(str(ds_dir), boot=True, keywords=keywords_list, count=count)
    return ds_dir, False


def _convert_rknn(pt_file, input_size, dataset_path=None, task="detect", quantize=None, kw=None):
    from iSpy.vision.metadata import (
        read_metadata,
        metadata_from_pt,
        derive_format_metadata,
        metadata_path_for,
        write_metadata,
    )
    from iSpy.vision.metadata import get_calibration_keywords
    from iSpy.dataset.dataset import calib_count_for_format, prepare_quantization_dataset

    if quantize is None:
        quantize = _RKNN_QUANTIZE
    pt_path = Path(pt_file)

    raw_onnx = Path(_export_ultralytics(str(pt_path), "onnx", input_size))
    if not raw_onnx.exists():
        raise RuntimeError(f"Intermediate ONNX export failed: {raw_onnx}")

    # Route intermediate ONNX to the onnx folder with its own sidecar
    onnx_path = _desired_output_path(pt_path, "onnx")
    try:
        if raw_onnx != onnx_path:
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            if onnx_path.exists():
                _remove_path_for_cleanup(onnx_path)
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
        # fall back to whichever copy actually exists - the raw file may have
        # already been moved to onnx_path before the metadata step failed, and
        # pointing at raw_onnx then would strand the build on a dead path
        if not onnx_path.exists():
            if raw_onnx != onnx_path and raw_onnx.exists():
                shutil.move(str(raw_onnx), str(onnx_path))
            else:
                onnx_path = raw_onnx

    try:
        from rknn.api import RKNN
        warnings.filterwarnings("ignore", category=UserWarning, module="rknnlite")
    except ImportError:
        raise ImportError("RKNN Toolkit not found. Install it to convert to RKNN format.")

    effective_kw = kw if kw is not None else get_calibration_keywords(pt_path, default=keywords)
    count = calib_count_for_format("rknn")
    ds_path, _ = _resolve_calibration_dataset(dataset_path, effective_kw, count)
    dataset_txt = ds_path / "dataset.txt"
    if not dataset_txt.exists() or not dataset_txt.read_text().strip():
        raise FileNotFoundError(f"RKNN calibration dataset could not be prepared at: {dataset_txt}")

    calib_dir = _downscale_calib_images(ds_path)
    calib_txt = calib_dir / "dataset.txt" if calib_dir != ds_path else dataset_txt

    rknn_output = _desired_output_path(pt_path, "rknn")
    logger.info("Converting ONNX -> RKNN with dataset=%s", calib_txt)

    box_coord_scale = None
    kpt_coord_scale = None
    onnx_path_for_build = onnx_path
    surgery_path = onnx_path.parent / f"{onnx_path.stem}_rknn_ready.onnx"
    try:
        box_coord_scale, kpt_coord_scale = _normalize_box_coords_for_quantization(
            str(onnx_path), str(surgery_path), input_size
        )
        onnx_path_for_build = surgery_path
    except Exception as exc:
        logger.warning(
            "Box/keypoint-coordinate normalization for RKNN quantization failed: %s. "
            "Proceeding without it; confidence may collapse to zero.",
            exc,
            exc_info=True,
        )

    try:
        with _progress_spinner("RKNN build"):
            with _silence_third_party():
                rknn = RKNN(verbose=False, )
            detected_format = None
            detected_layout = None
            detected_box_format = None
            target_platform, target_platform_detected = _resolve_rknn_target_platform()
            try:
                config_kwargs = dict(
                    mean_values=[[0, 0, 0]],
                    std_values=[[255, 255, 255]],
                    target_platform=target_platform,
                    disable_rules=[
                        "fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa"
                    ],
                )

                if quantize:
                    config_kwargs["quantized_dtype"] = "asymmetric_quantized-8"
                    config_kwargs["quantized_algorithm"] = "kl_divergence"

                rknn.config(**config_kwargs)
                ret = rknn.load_onnx(model=str(onnx_path_for_build))
                if ret != 0:
                    raise RuntimeError(f"RKNN load_onnx failed with code {ret}")
                ret = rknn.build(do_quantization=quantize, dataset=str(calib_txt))
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
                        tensor = _merge_rknn_build_outputs(outputs)
                        if len(outputs) > 1:
                            logger.info(
                                "RKNN build returned %d split output(s) %s -> merged shape %s",
                                len(outputs),
                                [o.shape for o in outputs],
                                tensor.shape,
                            )
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
    finally:
        if calib_dir != Path(dataset_path):
            shutil.rmtree(str(calib_dir), ignore_errors=True)
        if surgery_path.exists():
            surgery_path.unlink()

    logger.info("RKNN conversion successful: %s", rknn_output)
    _export_rknn_metadata(
        pt_file, rknn_output, input_size=input_size,
        output_format=detected_format,
        output_layout=detected_layout,
        box_format=detected_box_format,
        quantize=quantize,
        box_coord_scale=box_coord_scale,
        kpt_coord_scale=kpt_coord_scale,
        target_platform=target_platform,
        target_platform_detected=target_platform_detected,
    )
    return str(rknn_output)


def convert_model(model_file, target_format, input_size, quantize=None, force=False, kw=None, dataset_path=None):
    from iSpy.vision.metadata import (
        read_metadata,
        metadata_from_pt,
        derive_format_metadata,
        metadata_path_for,
        write_metadata,
    )
    from iSpy.vision.metadata import get_calibration_keywords
    from iSpy.dataset.dataset import calib_count_for_format, prepare_quantization_dataset

    if not os.path.exists(model_file):
        logger.warning("Model file %s is missing. Skipping conversion.", model_file)
        return model_file
    if Path(model_file).suffix.lower() != ".pt":
        logger.warning(
            "Model file %s is not a .pt file. Skipping conversion.", model_file
        )
        return model_file

    if target_format == "tpu":
        logger.info(
            "TPU backend uses the .pt directly via torch_xla at runtime - "
            "no export/conversion step needed for %s.",
            Path(model_file).name,
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
                    _remove_path_for_cleanup(rknn_path)
                    _remove_path_for_cleanup(meta_path)
                    rknn_result = _convert_rknn(
                        pt_file=model_file,
                        input_size=input_size,
                        quantize=quantize,
                        kw=kw,
                        dataset_path=dataset_path,
                    )
                    _run_optimized_model_comparison(model_file, rknn_result)
                    return rknn_result
            logger.info("Cached rknn model found: %s", rknn_path)
            return str(rknn_path)
        if force and rknn_path.exists():
            logger.info("Fresh conversion forced for %s", rknn_path.name)
            _remove_path_for_cleanup(rknn_path)
            meta = metadata_path_for(rknn_path)
            if meta.exists():
                _remove_path_for_cleanup(meta)
        return _convert_rknn(
            pt_file=model_file,
            input_size=input_size,
            quantize=quantize,
            kw=kw,
            dataset_path=dataset_path,
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
        kw = get_calibration_keywords(pt_path, default=keywords)
        ds_dir, used_user_ds = _resolve_calibration_dataset(
            dataset_path, kw, calib_count_for_format(target_format)
        )
        if used_user_ds:
            data_yaml = _user_calibration_data_yaml(model_file, ds_dir)
            logger.info(
                "Calibration dataset from user path %s (data.yaml=%s)",
                ds_dir, data_yaml,
            )
        else:
            # data.yaml lives inside the dataset folder that was just prepared -
            # the root QuantizeDataset/data.yaml is never written, so pointing
            # ultralytics int8 calibration at it would load nothing
            data_yaml = str(ds_dir / "data.yaml")
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
                            _remove_path_for_cleanup(desired_path)
                        shutil.move(str(result_path), str(desired_path))
                        result_path = desired_path
            else:
                if result_path != desired_path:
                    if desired_path.exists():
                        _remove_path_for_cleanup(desired_path)
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


def _convert_model_subprocess(model_file, target_format, input_size, quantize=None, force=False, kw=None, dataset_path=None) -> Path:
    outputs_dir = _PROJECT_ROOT / "Outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    args = {
        "model_file": str(model_file),
        "target_format": target_format,
        "input_size": list(input_size) if hasattr(input_size, "__iter__") else [int(input_size), int(input_size)],
        "quantize": quantize,
        "force": force,
        "kw": kw,
    }
    if dataset_path:
        args["dataset_path"] = str(dataset_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=str(outputs_dir)) as f:
        args_path = f.name
        json.dump(args, f)

    result_path = args_path + ".result.json"

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "iSpy.boot._convert_worker", args_path],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
        )

        # surface the worker's own logging - it just runs in a subprocess,
        # it shouldnt run silently
        if proc.stdout:
            print(proc.stdout, file=_REAL_STDOUT, end="")
        if proc.stderr:
            print(proc.stderr, file=_REAL_STDERR, end="")

        if proc.returncode != 0 or not os.path.exists(result_path):
            logger.error(
                "Conversion subprocess for %s -> %s failed (exit code %s). Falling back to .pt.",
                Path(model_file).name, target_format, proc.returncode,
            )
            return Path(model_file)

        with open(result_path) as f:
            result_data = json.load(f)
        return Path(result_data["result"])
    finally:
        for p in (args_path, result_path):
            try:
                os.unlink(p)
            except OSError:
                pass
