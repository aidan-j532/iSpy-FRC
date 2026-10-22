"""Persistent, isolated environments for model conversion.

Conversion toolchains do not have the same dependency requirements as iSpy's
runtime.  RKNN's full toolkit pins NumPy 1.x/protobuf 4.25.x, and Ultralytics'
TFLite exporter needs TensorFlow/onnx2tf with its own NumPy/protobuf stack.
Installing either into the runtime environment can break inference imports and
mutates packages the robot depends on.

This module owns one lazily provisioned virtual environment per conversion
format.  Environments are permanent (until their identity changes or the user
removes them) and live outside the project and the main virtual environment.
Only conversion workers run inside them; inference backends stay in the main
environment.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry types
# ---------------------------------------------------------------------------

WheelResolver = Callable[[str, str], str | None]

DEFAULT_MIN_FREE_BYTES = 8 * 1024 ** 3
_LIGHT_VERIFY_INTERVAL = 24 * 60 * 60
_SPACE_ERRORS = (
    "no space left",
    "errno 28",
    "enospc",
    "disk is full",
    "disk full",
    "not enough space",
    "there is not enough space",
    "insufficient disk space",
    "0x80070070",  # windows ERROR_DISK_FULL
)
_VENV_CREATE_TIMEOUT = 300
_PIP_INSTALL_TIMEOUT = 1800
_VERIFY_TIMEOUT = 300
_LIGHT_VERIFY_TIMEOUT = 120


class ConverterEnvError(RuntimeError):
    """Raised when a converter environment cannot be provisioned or found."""


@dataclass(frozen=True)
class EnvSpec:
    """Description of one conversion-only environment.

    ``wheel_resolver`` is optional: formats whose vendor wheel is not on PyPI
    can prepend a direct URL before the normal requirements.  A spec with a
    non-empty ``manual_install_hint`` is registry-visible and ``needs_env`` is
    true, but it is never auto-provisioned with pip.
    """

    format: str
    reason: str
    toolkit_version: str = ""
    depset_version: int = 1
    requirements: tuple[str, ...] = field(default_factory=tuple)
    wheel_resolver: WheelResolver | None = None
    verify_script: str = ""
    light_verify_script: str = ""
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
    supported: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    manual_install_hint: str = ""
    light_verify_interval: int = _LIGHT_VERIFY_INTERVAL


# ---------------------------------------------------------------------------
# RKNN toolkit details
# ---------------------------------------------------------------------------

RKNN_TOOLKIT_VERSION = "2.3.2"
_RKNN_FULL_BASE = (
    "https://github.com/aidan-j532/iSpy-FRC/releases/download/RKNN_Wheels"
).rstrip("/")

RKNN_FULL_WHEELS: dict[tuple[str, str], str] = {
    ("aarch64", "cp310"): f"rknn_toolkit2-{RKNN_TOOLKIT_VERSION}-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    ("aarch64", "cp311"): f"rknn_toolkit2-{RKNN_TOOLKIT_VERSION}-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    ("aarch64", "cp312"): f"rknn_toolkit2-{RKNN_TOOLKIT_VERSION}-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    ("x86_64", "cp310"): f"rknn_toolkit2-{RKNN_TOOLKIT_VERSION}-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    ("x86_64", "cp311"): f"rknn_toolkit2-{RKNN_TOOLKIT_VERSION}-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    ("x86_64", "cp312"): f"rknn_toolkit2-{RKNN_TOOLKIT_VERSION}-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
}

_RKNN_REQUIREMENTS: tuple[str, ...] = (
    "numpy>=1.26.0,<2",
    "scipy>=1.10.0,<1.15",
    "onnx>=1.16.0,<1.17",
    "protobuf>=4.21.6,<=4.25.4",
    # setuptools 82 removed pkg_resources; the toolkit still imports it.
    "setuptools>=64,<82",
    "torch>=2.0.0",
    "ultralytics>=8.3.0",
    "onnxslim>=0.1.87",
    "opencv-python>=4.8.0,<4.12",
    "tqdm>=4.65.0,<5",
    "ruamel.yaml>=0.17.4",
)

_RKNN_VERIFY_SCRIPT = "\n".join(
    [
        "import numpy; assert numpy.__version__.split('.')[0] == '1', numpy.__version__",
        "import scipy",
        "import onnx",
        # Exercise the exact API rknn.build() uses.  onnx 1.16 reads
        # FieldDescriptor.label; protobuf builds that dropped it (>=5.x in
        # some releases) fail here instead of deep inside the toolkit.
        "onnx.helper.strip_doc_string(onnx.ModelProto())",
        "import google.protobuf",
        "assert (4, 21, 6) <= tuple(map(int, google.protobuf.__version__.split('.')[:3])) <= (4, 25, 4), google.protobuf.__version__",
        "from ruamel.yaml import YAML",
        "import setuptools; assert int(setuptools.__version__.split('.')[0]) < 82",
        "import pkg_resources",
        "import torch",
        "import ultralytics",
        "import onnxslim",
        "import cv2",
        "import tqdm",
        "from rknn.api import RKNN",
        "print('OK')",
    ]
)

_RKNN_LIGHT_VERIFY_SCRIPT = "\n".join(
    [
        "import numpy; assert numpy.__version__.split('.')[0] == '1', numpy.__version__",
        "import onnx",
        "onnx.helper.strip_doc_string(onnx.ModelProto())",
        "from rknn.api import RKNN",
        "print('OK')",
    ]
)


def _rknn_wheel_url(arch: str, python_tag: str) -> str:
    key = (arch, python_tag)
    filename = RKNN_FULL_WHEELS.get(key)
    if filename is None:
        supported = sorted(f"{a} {v}" for (a, v) in RKNN_FULL_WHEELS if a == key[0])
        raise ConverterEnvError(
            f"No rknn-toolkit2 {RKNN_TOOLKIT_VERSION} wheel for {key[0]} "
            f"(Python {key[1]}). Supported: {', '.join(supported)}. "
            "RKNN conversion is not possible on this architecture/Python ABI."
        )
    return f"{_RKNN_FULL_BASE}/{filename}"


# ---------------------------------------------------------------------------
# TFLite exporter details
# ---------------------------------------------------------------------------

# The TFLite exporter is driven by Ultralytics and ultimately by onnx2tf and
# TensorFlow.  Ultralytics 8.3/8.4 checks for these exact packages; pinning the
# pair avoids silently accepting TensorFlow 2.20+ (which Ultralytics rejects).
_TFLITE_REQUIREMENTS: tuple[str, ...] = (
    "numpy>=1.26.0,<2.2",
    "protobuf>=3.20.3,<6,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5",
    "tensorflow==2.19.0",
    "tf_keras==2.19.0",
    "onnx>=1.16.0,<1.17",
    "onnx2tf>=1.26.3,<1.29.0",
    "onnx_graphsurgeon>=0.3.26",
    "sng4onnx>=1.0.1",
    "ai-edge-litert>=1.2.0,<2.4",
    "onnxslim>=0.1.87",
    "torch>=2.0.0",
    "ultralytics>=8.3.0",
    "opencv-python>=4.8.0,<4.12",
    "tqdm>=4.65.0,<5",
    "ruamel.yaml>=0.17.4",
)

_TFLITE_VERIFY_SCRIPT = "\n".join(
    [
        "import numpy",
        "from packaging.version import Version",
        "assert Version(numpy.__version__) < Version('2.2.0'), numpy.__version__",
        "import tensorflow as tf",
        "assert tf.__version__ == '2.19.0', tf.__version__",
        "import tf_keras",
        "import onnx",
        "onnx.helper.strip_doc_string(onnx.ModelProto())",
        "import onnx2tf",
        "import onnx_graphsurgeon",
        "import sng4onnx",
        "import ai_edge_litert",
        "import onnxslim",
        "import torch",
        "import ultralytics",
        "import cv2",
        "import tqdm",
        "from ruamel.yaml import YAML",
        "print('OK')",
    ]
)

_TFLITE_LIGHT_VERIFY_SCRIPT = "\n".join(
    [
        "import numpy",
        "from packaging.version import Version",
        "assert Version(numpy.__version__) < Version('2.2.0'), numpy.__version__",
        "import tensorflow as tf",
        "assert tf.__version__ == '2.19.0', tf.__version__",
        "import onnx",
        "onnx.helper.strip_doc_string(onnx.ModelProto())",
        "import onnx2tf",
        "import ai_edge_litert",
        "print('OK')",
    ]
)

_HAILO_MANUAL_HINT = (
    "Hailo conversion needs the Hailo Dataflow Compiler (DFC), which is not "
    "published on PyPI and requires a Hailo-supported Python/OS toolchain. "
    "Install the DFC from the Hailo Developer Zone on a supported machine, "
    "pre-convert the model with it, and place the resulting .hef in "
    "YoloModels/hailo/ (or your configured model directory). iSpy can use or "
    "benchmark an existing .hef but never auto-installs vendor SDKs."
)

#: Explanatory text for every registry entry.  EnvSpec carries its own copy for
#: env-backed formats; None entries use this table so every format has a reason.
CONVERTER_ENV_REASONS: dict[str, str] = {
    "rknn": (
        "Full rknn-toolkit2 conversion pins NumPy 1.x, SciPy, protobuf 4.25.x "
        "and setuptools<82, which conflicts with the runtime's inference stack."
    ),
    "tflite": (
        "TensorFlow 2.19/onnx2tf select NumPy 2.1 and protobuf 5.x, while the "
        "runtime uses a different NumPy/protobuf set (and aarch64 pins "
        "NumPy<2/protobuf<4); the build stack is also multi-GB and build-only."
    ),
    "hailo": (
        "Hailo Dataflow Compiler is vendor-installed, not available on PyPI, "
        "and supports a different Python toolchain; iSpy cannot auto-provision it."
    ),
    "engine": (
        "TensorRT engines are version-locked to the TensorRT runtime that "
        "created them; a separate conversion env could emit an engine the "
        "runtime cannot deserialize, so conversion stays in the runtime env."
    ),
    "openvino": (
        "OpenVINO conversion uses the same OpenVINO package the runtime loads; "
        "there is no conversion-only dependency conflict to isolate."
    ),
    "onnx": (
        "ONNX export is a plain torch/ultralytics graph export, and the main "
        "environment already carries the ONNX runtime used for inference."
    ),
    "qnn": (
        "QNN conversion produces an ONNX artifact through the normal export "
        "path; the QNN runtime provider is installed separately."
    ),
    "tpu": (
        "TPU runs the .pt through torch_xla at inference time and has no "
        "separate conversion step."
    ),
    "coreml": (
        "CoreML conversion is macOS-only and uses coremltools in the main "
        "environment; .mlpackage runtime inference is not supported."
    ),
}


CONVERTER_ENVS: dict[str, EnvSpec | None] = {
    "rknn": EnvSpec(
        format="rknn",
        reason=CONVERTER_ENV_REASONS["rknn"],
        toolkit_version=RKNN_TOOLKIT_VERSION,
        depset_version=6,
        requirements=_RKNN_REQUIREMENTS,
        wheel_resolver=_rknn_wheel_url,
        verify_script=_RKNN_VERIFY_SCRIPT,
        light_verify_script=_RKNN_LIGHT_VERIFY_SCRIPT,
        min_free_bytes=DEFAULT_MIN_FREE_BYTES,
        supported=tuple(RKNN_FULL_WHEELS),
    ),
    "tflite": EnvSpec(
        format="tflite",
        reason=CONVERTER_ENV_REASONS["tflite"],
        toolkit_version="2.19.0",
        depset_version=1,
        requirements=_TFLITE_REQUIREMENTS,
        verify_script=_TFLITE_VERIFY_SCRIPT,
        light_verify_script=_TFLITE_LIGHT_VERIFY_SCRIPT,
        min_free_bytes=DEFAULT_MIN_FREE_BYTES,
        supported=(
            ("aarch64", "cp310"),
            ("aarch64", "cp311"),
            ("aarch64", "cp312"),
            ("x86_64", "cp310"),
            ("x86_64", "cp311"),
            ("x86_64", "cp312"),
        ),
    ),
    "hailo": EnvSpec(
        format="hailo",
        reason=CONVERTER_ENV_REASONS["hailo"],
        toolkit_version="vendor",
        depset_version=1,
        requirements=(),
        verify_script="print('OK')",
        light_verify_script="print('OK')",
        min_free_bytes=DEFAULT_MIN_FREE_BYTES,
        supported=(),
        manual_install_hint=_HAILO_MANUAL_HINT,
    ),
    "engine": None,
    "openvino": None,
    "onnx": None,
    "qnn": None,
    "tpu": None,
    "coreml": None,
}


def _normalize_format(fmt: str | None) -> str:
    return str(fmt or "").strip().lower()


def get_env_spec(fmt: str | None) -> EnvSpec | None:
    """Return the EnvSpec for *fmt*, or None when conversion stays in-process."""
    return CONVERTER_ENVS.get(_normalize_format(fmt))


def needs_env(fmt: str | None) -> bool:
    """True when conversion for *fmt* runs in a dedicated environment."""
    return isinstance(get_env_spec(fmt), EnvSpec)


def converter_env_reason(fmt: str | None) -> str:
    """Human-readable reason the format does (or does not) have an env."""
    name = _normalize_format(fmt)
    spec = get_env_spec(name)
    if spec is not None and spec.reason:
        return spec.reason
    return CONVERTER_ENV_REASONS.get(name, "unknown conversion format")


def _require_env_spec(fmt: str | None) -> EnvSpec:
    name = _normalize_format(fmt)
    spec = get_env_spec(name)
    if spec is None:
        if name in CONVERTER_ENVS:
            raise ConverterEnvError(
                f"{name} conversion does not use a converter environment: "
                f"{converter_env_reason(name)}"
            )
        raise ConverterEnvError(f"unknown conversion format: {name or '<empty>'}")
    return spec


# ---------------------------------------------------------------------------
# Paths / identity
# ---------------------------------------------------------------------------


def _cache_root() -> Path:
    """Return the platform-appropriate base cache directory.

    ``ISPY_CONVERTER_ENV_HOME`` is the general override.  The historical
    ``ISPY_RKNN_CONVERTER_HOME`` still works as a fallback so existing
    deployments keep their cache location.
    """
    override = os.environ.get("ISPY_CONVERTER_ENV_HOME", "").strip()
    if not override:
        override = os.environ.get("ISPY_RKNN_CONVERTER_HOME", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "iSpy-FRC" / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ispy-frc"
    xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(xdg) / "ispy-frc"


def _arch_key() -> str:
    machine = platform.machine().lower()
    if "aarch64" in machine or "arm64" in machine:
        return "aarch64"
    if "x86_64" in machine or "amd64" in machine:
        return "x86_64"
    return machine


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _identity(spec: EnvSpec) -> dict[str, str]:
    return {
        "python_tag": _python_tag(),
        "arch": _arch_key(),
        "toolkit_version": spec.toolkit_version,
        "depset_version": str(spec.depset_version),
    }


def _identity_hash(identity: dict[str, str]) -> str:
    payload = json.dumps(identity, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _env_root(spec: EnvSpec) -> Path:
    return _cache_root() / f"{spec.format}_env"


def _env_dir(spec: EnvSpec, identity: dict[str, str] | None = None) -> Path:
    identity = identity or _identity(spec)
    return _env_root(spec) / _identity_hash(identity)


def env_dir_for(fmt: str, identity: dict[str, str] | None = None) -> Path:
    """Public path helper used by the CLI and diagnostics."""
    spec = _require_env_spec(fmt)
    return _env_dir(spec, identity)


def _manifest_path(env_dir: Path) -> Path:
    return env_dir / "env-manifest.json"


def _is_py(env_dir: Path) -> Path:
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _lock_path(spec: EnvSpec) -> Path:
    # One lock per format, so RKNN and TFLite can build concurrently while two
    # callers of the same format still serialize.
    return _env_root(spec) / "build.lock"


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x0400, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong(0)
            kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            return exit_code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@contextlib.contextmanager
def _build_lock(lock_path: Path, timeout: float = 600.0, label: str = "converter"):
    """Cross-process mutual exclusion for one format's environment build."""
    deadline = time.monotonic() + timeout
    acquired = False
    while time.monotonic() < deadline:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            try:
                pid = int(lock_path.read_text().strip())
            except (ValueError, OSError):
                pid = 0
            if not _pid_alive(pid):
                try:
                    lock_path.unlink()
                except OSError:
                    pass
                continue
            time.sleep(0.5)
    if not acquired:
        raise ConverterEnvError(
            f"Timed out waiting for the {label} converter-environment build "
            f"lock ({lock_path}). Another process may still be provisioning "
            "the environment; wait for it to finish or remove the lock manually."
        )
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Manifest / validation
# ---------------------------------------------------------------------------


def _read_manifest(env_dir: Path) -> dict | None:
    try:
        return json.loads(_manifest_path(env_dir).read_text())
    except (OSError, ValueError):
        return None


def _env_is_current(spec: EnvSpec, env_dir: Path) -> bool:
    if not _is_py(env_dir).exists():
        return False
    manifest = _read_manifest(env_dir)
    if manifest is None:
        return False
    identity = _identity(spec)
    for key, value in identity.items():
        if manifest.get(key) != value:
            return False
    return True


def _cleanup_orphans(spec: EnvSpec, current_hash: str) -> None:
    root = _env_root(spec)
    if not root.is_dir():
        return
    for entry in root.iterdir():
        if not entry.is_dir() or entry.name == current_hash:
            continue
        shutil.rmtree(entry, ignore_errors=True)
        logger.info("Removed stale %s converter environment: %s", spec.format, entry)


# ---------------------------------------------------------------------------
# Provisioning helpers
# ---------------------------------------------------------------------------


def _pip_failure_message(spec: EnvSpec, returncode: int, tail: str) -> str:
    lowered = tail.lower()
    hints: list[str] = []
    if any(
        k in lowered
        for k in (
            "could not resolve",
            "timed out",
            "connection",
            "ssl",
            "tls",
            "network is unreachable",
            "no route to host",
            "getaddrinfo",
        )
    ):
        hints.append(
            "This looks like a network/connection error - check internet "
            "access, firewall/proxy settings, and that package indexes are "
            "reachable, then retry."
        )
    if "no matching distribution" in lowered or "could not find a version" in lowered:
        hints.append(
            "pip could not find a compatible version - the pinned dependency "
            "set may not be published for this Python ABI. Check the supported "
            "arch/Python table for this format."
        )
    if "no space left" in lowered or ("disk" in lowered and "cache" in lowered):
        hints.append(
            "The cache disk is full - free space or point "
            "ISPY_CONVERTER_ENV_HOME at a different location."
        )
    if "permission" in lowered:
        hints.append(
            "Permission denied - make sure the converter cache directory is writable."
        )
    message = (
        f"pip install failed for the {spec.format} converter environment "
        f"(exit {returncode}).\n{tail}"
    )
    if hints:
        message += "\n" + "\n".join(hints)
    message += (
        "\nFailed provisioning is cleaned up automatically - the next "
        "conversion attempt rebuilds the environment from scratch."
    )
    return message


def _looks_out_of_space(text: str) -> bool:
    lowered = (text or "").lower()
    return any(sig in lowered for sig in _SPACE_ERRORS)


def _proc_text(proc) -> str:
    return f"{proc.stdout or ''}\n{proc.stderr or ''}"


def _purge_pip_cache() -> bool:
    """Blow away the pip download cache so a full disk has room to retry."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "cache", "purge"],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as exc:
        logger.warning("pip cache purge did not run: %s", exc)
        return False
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
        logger.warning("pip cache purge failed (exit %s): %s", proc.returncode, tail)
        return False
    logger.info("Purged the pip cache to free disk space.")
    return True


def _pip_install(
    spec: EnvSpec,
    py: Path,
    *requirements: str,
    no_deps: bool = False,
    purge_cache: Callable[[], bool] | None = None,
) -> None:
    """Install *requirements* into *py* with one disk-full retry."""
    purge_cache = purge_cache or _purge_pip_cache
    cmd = [
        str(py),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
    ]
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(requirements)
    logger.info(
        "Installing into %s converter env: %s", spec.format, " ".join(requirements)
    )

    def _run():
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, timeout=_PIP_INSTALL_TIMEOUT
            )
        except subprocess.TimeoutExpired:
            raise ConverterEnvError(
                f"pip install timed out for the {spec.format} converter "
                f"environment (>{_PIP_INSTALL_TIMEOUT}s). Check network/proxy "
                "settings and retry; environments can also be pre-built on an "
                "identical host and reused via the same cache path."
            ) from None

    proc = _run()
    if proc.returncode != 0 and _looks_out_of_space(_proc_text(proc)):
        logger.warning(
            "pip ran out of disk space installing %s - purging the pip cache "
            "and retrying once.",
            " ".join(requirements),
        )
        if purge_cache():
            proc = _run()
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-30:])
        raise ConverterEnvError(_pip_failure_message(spec, proc.returncode, tail))


def _verify(spec: EnvSpec, py: Path) -> None:
    if not spec.verify_script:
        return
    try:
        proc = subprocess.run(
            [str(py), "-I", "-c", spec.verify_script],
            capture_output=True,
            text=True,
            timeout=_VERIFY_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise ConverterEnvError(
            f"{spec.format} converter environment verification timed out "
            f"(>{_VERIFY_TIMEOUT}s). The install is likely too slow to be "
            "usable - retry, or provision on a faster host."
        ) from None
    if proc.returncode != 0 or "OK" not in proc.stdout:
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-30:])
        raise ConverterEnvError(
            f"{spec.format} converter environment verification failed - the "
            "installed dependency set is not usable.\n"
            f"{tail}\n"
            "Use 'ispy env rebuild <format>' to rebuild it."
        )


def _light_verify(
    spec: EnvSpec, py: Path, env_dir: Path, interval: int | None = None
) -> bool:
    """Cheap daily sanity check before reusing a cached converter env."""
    stamp = env_dir / ".verified"
    interval = spec.light_verify_interval if interval is None else interval
    if stamp.exists():
        try:
            if time.time() - stamp.stat().st_mtime < interval:
                return True
        except OSError:
            pass
    if not spec.light_verify_script:
        return bool(stamp.exists())
    try:
        proc = subprocess.run(
            [str(py), "-I", "-c", spec.light_verify_script],
            capture_output=True,
            text=True,
            timeout=_LIGHT_VERIFY_TIMEOUT,
        )
        ok = proc.returncode == 0 and "OK" in proc.stdout
    except Exception:
        ok = False
    if ok:
        try:
            stamp.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        except OSError:
            pass
    else:
        logger.warning(
            "%s converter env failed light verification (%s) - will rebuild",
            spec.format,
            py,
        )
    return ok


def _supported_text(spec: EnvSpec) -> str:
    return ", ".join(f"{a} {p}" for a, p in spec.supported)


def _ensure_supported(spec: EnvSpec) -> None:
    if not spec.supported:
        return
    key = (_arch_key(), _python_tag())
    if key not in spec.supported:
        raise ConverterEnvError(
            f"No {spec.format} converter environment is supported for "
            f"{key[0]} (Python {key[1]}). Supported: "
            f"{_supported_text(spec)}. Convert on a supported machine or use "
            "the source .pt model."
        )


def _wheel_url(spec: EnvSpec, arch: str | None = None, tag: str | None = None) -> str | None:
    if spec.wheel_resolver is None:
        return None
    return spec.wheel_resolver(arch or _arch_key(), tag or _python_tag())


def _safe_wheel_url(spec: EnvSpec) -> str | None:
    try:
        return _wheel_url(spec)
    except Exception:
        return None


def _free_bytes(env_dir: Path) -> tuple[int, Path] | None:
    parent = env_dir.parent
    for base in (parent, *parent.parents):
        try:
            return shutil.disk_usage(base).free, base
        except OSError:
            continue
    return None


def _check_disk_space(
    spec: EnvSpec,
    env_dir: Path,
    purge_cache: Callable[[], bool] | None = None,
) -> None:
    purge_cache = purge_cache or _purge_pip_cache
    probed = _free_bytes(env_dir)
    if probed is None:
        return
    free, base = probed
    if free >= spec.min_free_bytes:
        return

    logger.warning(
        "Only %.1f GiB free on %s before the %s converter build - purging "
        "the pip cache and re-checking.",
        free / 1024 ** 3,
        base,
        spec.format,
    )
    if purge_cache():
        probed = _free_bytes(env_dir)
        if probed is not None and probed[0] >= spec.min_free_bytes:
            logger.info(
                "Pip cache purge freed enough room (%.1f GiB now free) - "
                "continuing the build.",
                probed[0] / 1024 ** 3,
            )
            return
        free = probed[0] if probed is not None else free

    raise ConverterEnvError(
        f"Not enough free disk space to build the {spec.format} converter "
        f"environment: {free / 1024 ** 3:.1f} GiB free on {base}, "
        f"need at least {spec.min_free_bytes / 1024 ** 3:.1f} GiB. The pip "
        "cache has already been purged. Free space or point "
        "ISPY_CONVERTER_ENV_HOME at a different location."
    )


def _provision(
    spec: EnvSpec,
    env_dir: Path,
    *,
    wheel_url: Callable[[EnvSpec, str, str], str | None] | None = None,
    pip_install: Callable[..., None] | None = None,
    verify: Callable[[Path], None] | None = None,
    check_disk_space: Callable[[Path], None] | None = None,
) -> Path:
    """Create and populate one converter environment at *env_dir*."""
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    if env_dir.exists():
        shutil.rmtree(env_dir, ignore_errors=True)
        logger.info(
            "Removed incomplete %s converter env before rebuild: %s",
            spec.format,
            env_dir,
        )

    if check_disk_space is None:
        check_disk_space = lambda path: _check_disk_space(spec, path)  # noqa: E731
    check_disk_space(env_dir)

    try:
        logger.info("Creating %s converter virtual environment ...", spec.format)
        proc = subprocess.run(
            [sys.executable, "-m", "venv", str(env_dir)],
            capture_output=True,
            text=True,
            timeout=_VENV_CREATE_TIMEOUT,
        )
        if proc.returncode != 0:
            raise ConverterEnvError(
                f"Failed to create the {spec.format} converter virtual "
                f"environment (exit {proc.returncode}).\n{proc.stderr}"
            )

        py = _is_py(env_dir)

        if pip_install is None:
            pip_install = lambda path, *reqs, no_deps=False: _pip_install(  # noqa: E731
                spec, path, *reqs, no_deps=no_deps
            )
        if verify is None:
            verify = lambda path: _verify(spec, path)  # noqa: E731

        # 1. Optional vendor wheel first, without dependencies.  The pinned
        #    dependency set follows from PyPI at versions the runtime must not
        #    receive (RKNN's full toolkit is the canonical example).
        if spec.wheel_resolver is not None:
            url = (
                wheel_url(spec, _arch_key(), _python_tag())
                if wheel_url is not None
                else _wheel_url(spec)
            )
            if url:
                pip_install(py, url, no_deps=True)

        # 2. The pinned conversion dependency set.
        if spec.requirements:
            pip_install(py, *spec.requirements)

        # 3. Verify the environment can actually import the conversion stack.
        verify(py)

        # 4. Record the manifest only after a fully successful build.
        manifest = _identity(spec)
        manifest["format"] = spec.format
        manifest["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _manifest_path(env_dir).write_text(json.dumps(manifest, indent=2))
        try:
            (env_dir / ".verified").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        except OSError:
            pass

        logger.info("%s converter environment ready at %s", spec.format, env_dir)
        return py
    except Exception:
        shutil.rmtree(env_dir, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ispy_source_root() -> Path:
    """Return the directory that contains the ``iSpy`` package."""
    return Path(__file__).resolve().parents[2]


def _offline_error(spec: EnvSpec) -> ConverterEnvError:
    return ConverterEnvError(
        f"the {spec.format} converter environment is not provisioned and "
        "offline mode is on - provisioning it needs pip downloads. Convert "
        "the model on a machine with internet (or copy a provisioned env "
        "over), or turn offline mode off and reboot."
    )


def converter_python(fmt: str = "rknn") -> Path:
    """Return the Python executable of a ready converter environment.

    The environment is provisioned on first use and reused thereafter.  Hailo
    is registry-visible but manual-only: this function raises its actionable
    install hint rather than attempting a pip install.
    """
    spec = _require_env_spec(fmt)
    if spec.manual_install_hint:
        raise ConverterEnvError(
            f"{spec.format} conversion uses a manually installed environment. "
            f"{spec.manual_install_hint}"
        )
    _ensure_supported(spec)

    identity = _identity(spec)
    env_dir = _env_dir(spec, identity)
    py = _is_py(env_dir)

    if _env_is_current(spec, env_dir):
        if _light_verify(spec, py, env_dir):
            logger.debug("Reusing %s converter environment at %s", spec.format, env_dir)
            _cleanup_orphans(spec, _identity_hash(identity))
            return py
        logger.warning(
            "%s converter environment at %s failed light verification - rebuilding it.",
            spec.format,
            env_dir,
        )

    lock_path = _lock_path(spec)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _build_lock(lock_path, label=spec.format):
        if _env_is_current(spec, env_dir) and _light_verify(spec, py, env_dir):
            return py
        _cleanup_orphans(spec, _identity_hash(identity))
        from iSpy.core import network_policy

        if network_policy.is_offline():
            raise _offline_error(spec)
        py = _provision(spec, env_dir)
        _cleanup_orphans(spec, _identity_hash(identity))
        return py


def rebuild_env(fmt: str) -> Path:
    """Force a clean rebuild of *fmt*'s current environment."""
    spec = _require_env_spec(fmt)
    if spec.manual_install_hint:
        raise ConverterEnvError(
            f"{spec.format} cannot be rebuilt automatically. {spec.manual_install_hint}"
        )
    _ensure_supported(spec)
    identity = _identity(spec)
    env_dir = _env_dir(spec, identity)
    lock_path = _lock_path(spec)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _build_lock(lock_path, label=spec.format):
        _cleanup_orphans(spec, _identity_hash(identity))
        from iSpy.core import network_policy

        if network_policy.is_offline():
            raise _offline_error(spec)
        py = _provision(spec, env_dir)
        _cleanup_orphans(spec, _identity_hash(identity))
        return py


def remove_env(fmt: str) -> bool:
    """Delete all identities and files for one format.

    Returns True when something was removed.  The per-format lock file is kept
    until after the deletion so another process cannot start provisioning in
    the middle of it.
    """
    spec = _require_env_spec(fmt)
    root = _env_root(spec)
    if not root.exists():
        return False
    lock_path = _lock_path(spec)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    removed = False
    with _build_lock(lock_path, timeout=60.0, label=spec.format):
        for entry in list(root.iterdir()):
            if entry.name == lock_path.name:
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                try:
                    entry.unlink()
                except OSError:
                    pass
            removed = True
    try:
        lock_path.unlink()
    except OSError:
        pass
    try:
        root.rmdir()
    except OSError:
        pass
    return removed


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, dirs, files in os.walk(path):
        # Do not follow directory symlinks out of the env.
        dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def _last_verified(env_dir: Path) -> str:
    stamp = env_dir / ".verified"
    try:
        return time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(stamp.stat().st_mtime)
        )
    except OSError:
        return ""


def converter_env_status(fmt: str = "rknn") -> dict:
    """Return a JSON-serialisable status dict for *fmt*."""
    name = _normalize_format(fmt)
    spec = get_env_spec(name)
    base = {
        "format": name,
        "needs_env": spec is not None,
        "reason": converter_env_reason(name),
        "env_dir": "",
        "env_python": "",
        "exists": False,
        "current": False,
        "identity": {},
        "identity_hash": "",
        "size_bytes": 0,
        "last_verified": "",
        "toolkit_version": "",
        "depset_version": 0,
        "supported": [],
        "min_free_bytes": DEFAULT_MIN_FREE_BYTES,
        "manual_install_hint": "",
        "toolkit_wheel_url": None,
    }
    if spec is None:
        return base

    identity = _identity(spec)
    env_dir = _env_dir(spec, identity)
    py = _is_py(env_dir)
    base.update(
        {
            "env_dir": str(env_dir),
            "env_python": str(py),
            "exists": env_dir.exists(),
            "current": _env_is_current(spec, env_dir),
            "identity": identity,
            "identity_hash": _identity_hash(identity),
            "size_bytes": _dir_size(env_dir),
            "last_verified": _last_verified(env_dir),
            "toolkit_version": spec.toolkit_version,
            "depset_version": spec.depset_version,
            "supported": [f"{a} {p}" for a, p in spec.supported],
            "min_free_bytes": spec.min_free_bytes,
            "manual_install_hint": spec.manual_install_hint,
            "toolkit_wheel_url": _safe_wheel_url(spec),
        }
    )
    return base


def list_converter_envs() -> list[dict]:
    """Status for every registry entry, in registry order."""
    return [converter_env_status(name) for name in CONVERTER_ENVS]
