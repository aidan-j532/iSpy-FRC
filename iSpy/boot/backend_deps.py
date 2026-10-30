"""Runtime backend registry and provisioner.

Routes install paths (boot, setup, optimize) to the right wheels for the current platform. PyTorch is never pulled from generic PyPI on CPU-only ARM64. Conversion envs stay in iSpy.boot.converter_envs.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: Free space we want to keep *after* a small runtime wheel installs.
DEFAULT_MIN_FREE_BYTES = 1 * 1024 ** 3
#: PyTorch brings its own shared libraries; leave room for the wheel + unpack.
TORCH_MIN_FREE_BYTES = 3 * 1024 ** 3
#: TensorRT + its CUDA runtime wheels are considerably larger.
TENSORRT_MIN_FREE_BYTES = 5 * 1024 ** 3

VERIFY_TIMEOUT = 300
INSTALL_TIMEOUT = 1800
UNINSTALL_TIMEOUT = 300

TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_CUDA_TAGS = {11: "cu118", 12: "cu126", 13: "cu130"}
NVIDIA_INDEX = "https://pypi.nvidia.com"
TPU_FIND_LINKS = "https://storage.googleapis.com/libtpu-releases/index.html"

_ORT_VARIANTS = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml")

_SPACE_ERRORS = (
    "no space left",
    "disk full",
    "0x80070070",
)

#: Generic PyPI torch wheels can leave nvidia/triton behind after a CPU install.
#: On ARM64/CPU-only boards these are useless and can be hundreds of MB.
_CUDA_ONLY_DISTS = {
    "triton",
    "pytorch-triton",
}

_HAILO_MANUAL_HINT = (
    "Hailo runtime inference needs HailoRT / hailo_platform, which is "
    "installed from the Hailo Developer Zone for a specific HailoRT version. "
    "iSpy does not pip-install vendor SDKs; install it with the vendor "
    "instructions, or pre-convert the model elsewhere and use a supported "
    "backend."
)

_JETSON_TORCH_HINT = (
    "Jetson PyTorch comes from NVIDIA JetPack - install the matching wheel "
    "and rerun 'ispy-boot -f'."
)

# ---------------------------------------------------------------------------
# Platform description
# ---------------------------------------------------------------------------

def _normalize_machine(machine: str) -> str:
    value = (machine or "").strip().lower()
    return {
        "amd64": "x86_64",
        "x64": "x86_64",
        "i386": "x86",
        "i686": "x86",
        "arm64": "aarch64",
        "aarch64_be": "aarch64",
        "armv7": "armv7l",
        "armv6l": "armv6l",
    }.get(value, value)

@dataclass(frozen=True)
class PlatformInfo:
    """Everything the registry is allowed to branch on."""

    system: str
    machine: str
    python_tag: str
    is_jetson: bool = False
    has_nvidia: bool = False
    has_amd: bool = False
    has_intel: bool = False
    has_rockchip: bool = False
    has_hailo: bool = False
    has_qualcomm: bool = False
    has_tpu_hardware: bool = False
    has_apple_silicon: bool = False
    has_edge_tpu: bool = False
    cuda_major: int | None = None

    @property
    def is_linux(self) -> bool:
        return self.system == "Linux"

    @property
    def is_windows(self) -> bool:
        return self.system == "Windows"

    @property
    def is_macos(self) -> bool:
        return self.system == "Darwin"

    @property
    def is_linux_or_windows(self) -> bool:
        return self.system in ("Linux", "Windows")

    @property
    def is_aarch64(self) -> bool:
        return self.machine == "aarch64"

    @property
    def is_x86_64(self) -> bool:
        return self.machine == "x86_64"

    @property
    def has_cuda(self) -> bool:
        """NVIDIA hardware *and* a driver that nvidia-smi can talk to."""

        return self.has_nvidia and self.cuda_major is not None

    def describe(self) -> str:
        bits = [
            f"os={self.system}",
            f"arch={self.machine}",
            f"python={self.python_tag}",
        ]
        if self.is_jetson:
            bits.append("jetson")
        if self.has_nvidia:
            bits.append(f"nvidia(cuda={self.cuda_major or 'driver?'})")
        if self.has_amd:
            bits.append("amd")
        if self.has_intel:
            bits.append("intel")
        if self.has_rockchip:
            bits.append("rockchip")
        if self.has_hailo:
            bits.append("hailo")
        if self.has_qualcomm:
            bits.append("qualcomm")
        if self.has_tpu_hardware:
            bits.append("tpu-hw")
        if self.has_apple_silicon:
            bits.append("apple-silicon")
        if self.has_edge_tpu:
            bits.append("edge-tpu")
        return " ".join(bits)

def _call_autoopt(name: str, default: bool = False) -> bool:
    """Read one hardware probe from AutoOpt without making it mandatory."""

    try:
        from iSpy.config import AutoOpt
    except ImportError:  # pragma: no cover - AutoOpt is a core module
        return default
    func = getattr(AutoOpt, name, None)
    if func is None:
        return default
    try:
        return bool(func())
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        logger.debug("Hardware probe %s failed: %s", name, exc)
        return default

def _detect_cuda_major() -> int | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.", proc.stdout or "")
    return int(match.group(1)) if match else None

def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"

@lru_cache(maxsize=1)
def detect_platform() -> PlatformInfo:
    """Describe the running interpreter, respecting image-build overrides."""

    target_os = os.environ.get("ISPY_TARGET_OS", "").strip()
    target_arch = os.environ.get("ISPY_TARGET_ARCH", "").strip()
    skip_hardware = os.environ.get(
        "ISPY_SKIP_HARDWARE_DETECT", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    host_os = platform.system()
    host_arch = _normalize_machine(platform.machine())

    system = target_os or host_os
    machine = _normalize_machine(target_arch) if target_arch else host_arch
    on_target = system.lower() == host_os.lower() and machine == host_arch

    if not on_target or skip_hardware:
        return PlatformInfo(
            system=system,
            machine=machine,
            python_tag=_python_tag(),
        )

    nvidia = _call_autoopt("has_nvidia")
    jetson = _call_autoopt("has_jetson")
    return PlatformInfo(
        system=system,
        machine=machine,
        python_tag=_python_tag(),
        is_jetson=jetson,
        has_nvidia=nvidia,
        has_amd=_call_autoopt("has_amd_gpu"),
        has_intel=_call_autoopt("has_intel_gpu"),
        has_rockchip=_call_autoopt("has_rockchip_npu"),
        has_hailo=_call_autoopt("has_hailo_npu"),
        has_qualcomm=_call_autoopt("has_qualcomm_npu"),
        has_tpu_hardware=_call_autoopt("has_tpu_hardware"),
        has_apple_silicon=_call_autoopt("has_apple_silicon"),
        has_edge_tpu=_call_autoopt("has_edge_tpu"),
        cuda_major=_detect_cuda_major() if nvidia else None,
    )

def clear_platform_cache() -> None:
    detect_platform.cache_clear()

# ---------------------------------------------------------------------------
# Registry types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PackageSpec:
    """One concrete pip distribution to try."""

    dist: str
    requirement: str
    module: str
    index_url: str | None = None
    extra_index_urls: tuple[str, ...] = ()
    find_links: tuple[str, ...] = ()
    no_deps: bool = False
    no_cache: bool = False
    uninstall_first: tuple[str, ...] = ()
    verify_script: str | None = None
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
    note: str = ""

@dataclass(frozen=True)
class Resolution:
    """How a backend resolves on one platform."""

    mode: str
    packages: tuple[PackageSpec, ...] = ()
    reason: str = ""

Resolver = Callable[[PlatformInfo], Resolution]

@dataclass(frozen=True)
class BackendSpec:
    name: str
    title: str
    module: str
    description: str
    resolver: Resolver
    hardware: tuple[str, ...] = ()
    fallback: str | None = None
    #: The rich registry record for this backend.  Kept as a separate field so
    #: diagnostics can print it without invoking the resolver.
    notes: str = ""

# ---------------------------------------------------------------------------
# Individual resolvers
# ---------------------------------------------------------------------------

_TORCH_NUMPY_PROBE = (
    "import warnings\n"
    "import numpy\n"
    "with warnings.catch_warnings(record=True) as caught:\n"
    "    warnings.simplefilter('always')\n"
    "    import torch\n"
    "    torch.from_numpy(numpy.zeros(1, dtype=numpy.float32))\n"
    "bad = [w for w in caught if 'NumPy' in str(w.message)]\n"
    "assert not bad, bad[0].message\n"
)
_TORCH_CPU_VERIFY = (
    _TORCH_NUMPY_PROBE
    + "print(torch.__version__)\n"
    + "assert torch.version.cuda is None, "
    "'CPU-only target received a CUDA-enabled torch build'\n"
)
_TORCH_CUDA_VERIFY = (
    _TORCH_NUMPY_PROBE
    + "print(torch.__version__, 'cuda=' + str(torch.version.cuda))\n"
    + "assert torch.cuda.is_available(), 'torch cannot access the NVIDIA GPU'\n"
)
_TORCH_JETSON_VERIFY = (
    _TORCH_NUMPY_PROBE
    + "print(torch.__version__, 'cuda=' + str(torch.version.cuda))\n"
)

def _torch_cpu_package() -> PackageSpec:
    return PackageSpec(
        dist="torch",
        requirement="torch>=2.0",
        module="torch",
        index_url=TORCH_CPU_INDEX,
        no_cache=True,
        verify_script=_TORCH_CPU_VERIFY,
        min_free_bytes=TORCH_MIN_FREE_BYTES,
        note="CPU wheel index (never the generic CUDA-enabled PyPI wheel)",
    )

def _resolve_torch(p: PlatformInfo) -> Resolution:
    if p.is_jetson:
        return Resolution(
            mode="system",
            reason=_JETSON_TORCH_HINT,
        )
    if p.system == "Windows":
        supported_machine = p.machine == "x86_64"
    elif p.system == "Darwin":
        supported_machine = p.machine == "aarch64"
    elif p.system == "Linux":
        supported_machine = p.machine in ("x86_64", "aarch64")
    else:
        return Resolution(
            mode="unsupported",
            reason=f"PyTorch wheels are not published for {p.system}.",
        )
    if not supported_machine:
        return Resolution(
            mode="unsupported",
            reason=(
                f"PyTorch has no supported wheel for {p.system} {p.machine}. "
                "The .pt baseline cannot run on this platform; use a "
                "TFLite/ONNX model or install a vendor PyTorch build manually."
            ),
        )

    if p.has_nvidia and p.is_linux_or_windows and p.cuda_major in TORCH_CUDA_TAGS:
        tag = TORCH_CUDA_TAGS[p.cuda_major]
        cuda = PackageSpec(
            dist="torch",
            requirement="torch>=2.0",
            module="torch",
            index_url=f"https://download.pytorch.org/whl/{tag}",
            no_cache=True,
            verify_script=_TORCH_CUDA_VERIFY,
            min_free_bytes=TORCH_MIN_FREE_BYTES,
            note=f"NVIDIA CUDA {tag} build",
        )
        # Keep a CPU fallback so a driver/wheel mismatch degrades instead of
        # leaving an unusable .pt baseline.
        return Resolution(mode="install", packages=(cuda, _torch_cpu_package()))
    return Resolution(mode="install", packages=(_torch_cpu_package(),))

_ORT_PROVIDER_SCRIPT = (
    "import onnxruntime as ort\n"
    "print(ort.__version__, ','.join(ort.get_available_providers()))\n"
    "assert {provider!r} in ort.get_available_providers(), "
    "'expected {provider} in available providers'\n"
)

def _ort_package(dist: str, provider: str, *, extra: str = "") -> PackageSpec:
    return PackageSpec(
        dist=dist,
        requirement=f"{dist}>=1.16.0",
        module="onnxruntime",
        # Variant wheels overwrite the same package directory, so remove all
        # known variants before switching.  This only runs when an install is
        # actually needed.
        uninstall_first=_ORT_VARIANTS,
        verify_script=_ORT_PROVIDER_SCRIPT.format(provider=provider),
        note=extra,
    )

def _resolve_onnx(p: PlatformInfo) -> Resolution:
    if p.is_jetson:
        return Resolution(
            mode="system",
            reason=(
                "Jetson onnxruntime/TensorRT come from JetPack and must match "
                "the L4T release; install them with apt/NVIDIA, never pip."
            ),
        )
    if p.system == "Darwin":
        return Resolution(
            mode="install",
            packages=(
                _ort_package("onnxruntime", "CPUExecutionProvider"),
            ),
        )
    if p.has_nvidia and p.cuda_major is not None:
        # onnxruntime-gpu is published for Windows x86_64 only; Linux also has
        # aarch64 CUDA wheels, but Windows ARM64 does not.
        if p.is_windows and p.machine != "x86_64":
            return Resolution(
                mode="unsupported",
                reason="onnxruntime-gpu has no Windows ARM64 wheel.",
            )
        return Resolution(
            mode="install",
            packages=(
                _ort_package("onnxruntime-gpu", "CUDAExecutionProvider",
                             extra="CUDA provider for the detected NVIDIA GPU"),
            ),
        )
    if p.is_windows and p.machine == "x86_64":
        return Resolution(
            mode="install",
            packages=(
                _ort_package(
                    "onnxruntime-directml",
                    "DmlExecutionProvider",
                    extra="DirectML provider (AMD/Intel/NVIDIA on Windows)",
                ),
            ),
        )
    if p.machine in ("x86_64", "aarch64"):
        return Resolution(
            mode="install",
            packages=(
                _ort_package("onnxruntime", "CPUExecutionProvider"),
            ),
        )
    return Resolution(
        mode="unsupported",
        reason=f"onnxruntime has no wheel for {p.system} {p.machine}.",
    )

def _resolve_qnn(p: PlatformInfo) -> Resolution:
    if p.system not in ("Linux", "Windows"):
        return Resolution(
            mode="unsupported",
            reason="onnxruntime-qnn is only published for Linux and Windows.",
        )
    if p.machine not in ("x86_64", "aarch64"):
        return Resolution(
            mode="unsupported",
            reason=f"onnxruntime-qnn has no wheel for {p.machine}.",
        )
    if p.python_tag not in ("cp311", "cp312", "cp313", "cp314"):
        return Resolution(
            mode="unsupported",
            reason=(
                "onnxruntime-qnn requires Python 3.11+; no wheel is "
                f"published for {p.python_tag}."
            ),
        )
    if not p.has_qualcomm:
        return Resolution(
            mode="unsupported",
            reason=(
                "No Qualcomm NPU detected. onnxruntime-qnn is only useful on "
                "QCS6490/Rubik Pi-class hardware."
            ),
        )
    return Resolution(
        mode="install",
        packages=(
            PackageSpec(
                dist="onnxruntime-qnn",
                requirement="onnxruntime-qnn>=1.18",
                module="onnxruntime_qnn",
                verify_script=(
                    "import onnxruntime_qnn as q\n"
                    "print(q.get_library_path())\n"
                ),
                note="QNN execution-provider plugin for onnxruntime",
            ),
        ),
    )

#: RKNN Lite is shipped inside the iSpy wheel; only the inference runtime is
#: ever installed in the main environment.  The full toolkit2 stays in the
#: isolated converter env.
RKNN_LITE_FILENAMES: dict[tuple[str, str], str] = {
    ("aarch64", "cp310"): (
        "rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64."
        "manylinux2014_aarch64.whl"
    ),
    ("aarch64", "cp311"): (
        "rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64."
        "manylinux2014_aarch64.whl"
    ),
    ("aarch64", "cp312"): (
        "rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_aarch64."
        "manylinux2014_aarch64.whl"
    ),
}

def rknn_lite_wheel_dir() -> Path:
    """Directory containing the bundled RKNN Lite wheels."""

    package_root = Path(__file__).resolve().parent.parent
    for candidate in (
        package_root / "rknn_wheels",
        package_root.parent / "rknn_wheels",
    ):
        if candidate.is_dir():
            return candidate
    return package_root / "rknn_wheels"

def rknn_lite_wheel(machine: str, python_tag: str) -> Path | None:
    filename = RKNN_LITE_FILENAMES.get((machine, python_tag))
    if not filename:
        return None
    wheel = rknn_lite_wheel_dir() / filename
    return wheel if wheel.is_file() else None

def _resolve_rknn(p: PlatformInfo) -> Resolution:
    if p.system != "Linux" or p.machine != "aarch64":
        return Resolution(
            mode="unsupported",
            reason=(
                "RKNN Lite inference wheels are only published for Linux "
                "aarch64 (Rockchip boards). Use the isolated RKNN converter "
                "environment to convert a model on a dev machine, then copy "
                "the .rknn to the board."
            ),
        )
    wheel = rknn_lite_wheel(p.machine, p.python_tag)
    if wheel is None:
        supported = sorted(
            tag for (arch, tag) in RKNN_LITE_FILENAMES if arch == p.machine
        )
        return Resolution(
            mode="unsupported",
            reason=(
                f"No bundled RKNN Lite wheel for {p.machine} {p.python_tag}. "
                f"Supported Python ABIs: {', '.join(supported)}."
            ),
        )
    return Resolution(
        mode="install",
        packages=(
            PackageSpec(
                dist="rknn-toolkit-lite2",
                requirement=str(wheel),
                module="rknnlite",
                no_deps=True,
                verify_script=(
                    "import rknnlite\n"
                    "print(getattr(rknnlite, '__version__', 'installed'))\n"
                ),
                note="bundled RKNN Lite inference runtime (never toolkit2)",
            ),
        ),
    )

def _resolve_tflite(p: PlatformInfo) -> Resolution:
    # tflite-runtime is the light, stable ABI through cp311.  Newer Python
    # versions are served by Google's ai-edge-litert successor.
    if (
        p.system == "Linux"
        and p.machine in ("x86_64", "aarch64", "armv7l")
        and p.python_tag in ("cp310", "cp311")
    ):
        return Resolution(
            mode="install",
            packages=(
                PackageSpec(
                    dist="tflite-runtime",
                    requirement="tflite-runtime>=2.14.0",
                    module="tflite_runtime",
                    verify_script=(
                        "from tflite_runtime.interpreter import Interpreter\n"
                        "print(Interpreter)\n"
                    ),
                    note="TFLite runtime, main environment",
                ),
            ),
        )
    litert_systems = {"Linux", "Windows", "Darwin"}
    if p.system in litert_systems and p.machine in ("x86_64", "aarch64"):
        if p.system == "Darwin" and p.machine != "aarch64":
            return Resolution(
                mode="unsupported",
                reason="ai-edge-litert has no macOS x86_64 wheel.",
            )
        if p.system == "Windows" and p.machine != "x86_64":
            return Resolution(
                mode="unsupported",
                reason="ai-edge-litert has no Windows ARM64 wheel.",
            )
        return Resolution(
            mode="install",
            packages=(
                PackageSpec(
                    dist="ai-edge-litert",
                    requirement="ai-edge-litert>=2.0.0",
                    module="ai_edge_litert",
                    verify_script=(
                        "from ai_edge_litert.interpreter import Interpreter\n"
                        "print(Interpreter)\n"
                    ),
                    note="TFLite runtime (ai-edge-litert successor), main env",
                ),
            ),
        )
    return Resolution(
        mode="unsupported",
        reason=(
            f"No TFLite runtime wheel for {p.system} {p.machine} "
            f"({p.python_tag})."
        ),
    )

def _resolve_openvino(p: PlatformInfo) -> Resolution:
    if p.system not in ("Linux", "Windows", "Darwin"):
        return Resolution(
            mode="unsupported",
            reason=f"OpenVINO wheels are not published for {p.system}.",
        )
    if p.machine not in ("x86_64", "aarch64"):
        return Resolution(
            mode="unsupported",
            reason=f"OpenVINO has no wheel for {p.machine}.",
        )
    if p.system == "Darwin" and p.machine != "aarch64":
        return Resolution(
            mode="unsupported",
            reason="OpenVINO has no current macOS x86_64 wheel.",
        )
    if p.system == "Windows" and p.machine != "x86_64":
        return Resolution(
            mode="unsupported",
            reason="OpenVINO has no Windows ARM64 wheel.",
        )
    return Resolution(
        mode="install",
        packages=(
            PackageSpec(
                dist="openvino",
                requirement="openvino>=2024.0",
                module="openvino",
                verify_script="import openvino as ov\nprint(ov.__version__)\n",
            ),
        ),
    )

def _resolve_coreml(p: PlatformInfo) -> Resolution:
    if not p.is_macos:
        return Resolution(
            mode="unsupported",
            reason="coremltools requires macOS.",
        )
    return Resolution(
        mode="install",
        packages=(
            PackageSpec(
                dist="coremltools",
                requirement="coremltools>=8.0",
                module="coremltools",
                verify_script="import coremltools as ct\nprint(ct.__version__)\n",
                note="conversion-only on macOS; .mlpackage runtime is unsupported",
            ),
        ),
    )

def _resolve_engine(p: PlatformInfo) -> Resolution:
    if p.is_jetson:
        return Resolution(
            mode="system",
            reason=(
                "TensorRT on Jetson is part of JetPack and version-locked to "
                "the L4T release; install/verify it with NVIDIA tooling."
            ),
        )
    if not p.has_nvidia:
        return Resolution(
            mode="unsupported",
            reason="TensorRT requires an NVIDIA GPU.",
        )
    if p.system not in ("Linux", "Windows") or p.machine not in (
        "x86_64",
        "aarch64",
    ):
        return Resolution(
            mode="unsupported",
            reason=(
                "TensorRT pip wheels are only available for Linux/Windows "
                "x86_64 and aarch64."
            ),
        )
    if p.system == "Windows" and p.machine != "x86_64":
        return Resolution(
            mode="unsupported",
            reason="TensorRT pip wheels are not published for Windows ARM64.",
        )
    packages: list[PackageSpec] = []
    if p.cuda_major in (12, 13):
        packages.append(
            PackageSpec(
                dist=f"tensorrt-cu{p.cuda_major}",
                requirement=f"tensorrt-cu{p.cuda_major}>=10.0",
                module="tensorrt",
                extra_index_urls=(NVIDIA_INDEX,),
                no_cache=True,
                uninstall_first=("tensorrt", "tensorrt-cu12", "tensorrt-cu13"),
                verify_script=(
                    "import tensorrt as trt\n"
                    "print(trt.__version__)\n"
                ),
                min_free_bytes=TENSORRT_MIN_FREE_BYTES,
            )
        )
    packages.append(
        PackageSpec(
            dist="tensorrt",
            requirement="tensorrt>=10.0",
            module="tensorrt",
            extra_index_urls=(NVIDIA_INDEX,),
            no_cache=True,
            uninstall_first=("tensorrt", "tensorrt-cu12", "tensorrt-cu13"),
            verify_script="import tensorrt as trt\nprint(trt.__version__)\n",
            min_free_bytes=TENSORRT_MIN_FREE_BYTES,
        )
    )
    return Resolution(mode="install", packages=tuple(packages))

def _resolve_hailo(p: PlatformInfo) -> Resolution:
    return Resolution(mode="manual", reason=_HAILO_MANUAL_HINT)

def _resolve_tpu(p: PlatformInfo) -> Resolution:
    if not (p.is_linux and p.machine == "x86_64"):
        return Resolution(
            mode="unsupported",
            reason="torch_xla PyPI wheels are only published for Linux x86_64.",
        )
    if p.python_tag not in ("cp310", "cp311", "cp312", "cp313"):
        return Resolution(
            mode="unsupported",
            reason=f"torch_xla has no wheel for {p.python_tag}.",
        )
    return Resolution(
        mode="install",
        packages=(
            # Installing CPU torch first keeps torch_xla's torch==X.Y.Z
            # dependency from resolving to the generic CUDA-enabled wheel.
            _torch_cpu_package(),
            PackageSpec(
                dist="torch-xla",
                requirement="torch_xla[tpu]",
                module="torch_xla",
                find_links=(TPU_FIND_LINKS,),
                verify_script=(
                    "import torch_xla\n"
                    "print(torch_xla.__version__)\n"
                ),
                min_free_bytes=TORCH_MIN_FREE_BYTES,
            ),
        ),
    )

# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

BACKEND_REGISTRY: dict[str, BackendSpec] = {
    "pytorch": BackendSpec(
        name="pytorch",
        title="PyTorch (.pt)",
        module="torch",
        description="Baseline .pt inference and the source format for conversion.",
        resolver=_resolve_torch,
        hardware=(),
        notes=(
            "CPU builds come from download.pytorch.org/whl/cpu. CUDA builds "
            "are only chosen when nvidia-smi reports a driver. Never install "
            "generic PyPI torch on ARM64/CPU-only hosts."
        ),
    ),
    "onnx": BackendSpec(
        name="onnx",
        title="ONNX Runtime",
        module="onnxruntime",
        description="ONNX inference; variant selected from the OS/hardware.",
        resolver=_resolve_onnx,
        hardware=("has_nvidia", "has_qualcomm"),
        fallback=None,
        notes="Jetson is JetPack-managed; Windows uses DirectML when no NVIDIA.",
    ),
    "rknn": BackendSpec(
        name="rknn",
        title="RKNN Lite",
        module="rknnlite",
        description="Rockchip NPU inference (lite runtime only).",
        resolver=_resolve_rknn,
        hardware=("has_rockchip",),
        fallback="onnx",
        notes="Full rknn-toolkit2 is conversion-only and lives in converter_envs.",
    ),
    "tflite": BackendSpec(
        name="tflite",
        title="TFLite / LiteRT",
        module="tflite_runtime",
        description="TFLite inference; ai-edge-litert on newer Pythons.",
        resolver=_resolve_tflite,
        hardware=("has_edge_tpu",),
        fallback="onnx",
        notes="Runtime only; TensorFlow/onnx2tf conversion is isolated.",
    ),
    "openvino": BackendSpec(
        name="openvino",
        title="OpenVINO",
        module="openvino",
        description="Intel CPU/GPU/VPU inference and OpenVINO IR loading.",
        resolver=_resolve_openvino,
        hardware=("has_intel",),
        fallback="onnx",
    ),
    "coreml": BackendSpec(
        name="coreml",
        title="Core ML",
        module="coremltools",
        description="CoreML conversion on macOS (runtime .mlpackage unsupported).",
        resolver=_resolve_coreml,
        hardware=("has_apple_silicon",),
        fallback=None,
    ),
    "engine": BackendSpec(
        name="engine",
        title="TensorRT",
        module="tensorrt",
        description="TensorRT engine inference on NVIDIA GPUs.",
        resolver=_resolve_engine,
        hardware=("has_nvidia",),
        fallback="onnx",
        notes="Prefer NVIDIA-index tensorrt-cuXX wheels; Jetson is system-managed.",
    ),
    "qnn": BackendSpec(
        name="qnn",
        title="Qualcomm QNN",
        module="onnxruntime_qnn",
        description="QNN execution-provider plugin for onnxruntime.",
        resolver=_resolve_qnn,
        hardware=("has_qualcomm",),
        fallback="onnx",
    ),
    "hailo": BackendSpec(
        name="hailo",
        title="HailoRT",
        module="hailo_platform",
        description="Hailo NPU inference (vendor-installed runtime).",
        resolver=_resolve_hailo,
        hardware=("has_hailo",),
        fallback=None,
        notes="Vendor SDK; never auto-installed from PyPI.",
    ),
    "tpu": BackendSpec(
        name="tpu",
        title="Google TPU (torch_xla)",
        module="torch_xla",
        description="PyTorch/XLA TPU inference.",
        resolver=_resolve_tpu,
        hardware=("has_tpu_hardware",),
        fallback="pytorch",
    ),
}

#: User-facing aliases that may come from a model filename.
BACKEND_ALIASES = {
    "pt": "pytorch",
    "pytorch": "pytorch",
    "torch": "pytorch",
    "onnx": "onnx",
    "rknn": "rknn",
    "tflite": "tflite",
    "litert": "tflite",
    "openvino": "openvino",
    "coreml": "coreml",
    "engine": "engine",
    "tensorrt": "engine",
    "qnn": "qnn",
    "hailo": "hailo",
    "tpu": "tpu",
}

_MODEL_SUFFIX_BACKENDS = {
    ".pt": "pytorch",
    ".pth": "pytorch",
    ".onnx": "onnx",
    ".rknn": "rknn",
    ".tflite": "tflite",
    ".mlpackage": "coreml",
    ".engine": "engine",
    ".hef": "hailo",
    ".xml": "openvino",
}

def normalize_backend(name: str | None) -> str:
    key = str(name or "").strip().lower()
    return BACKEND_ALIASES.get(key, key)


#: Backends that are conversion-only or never provisioned from the boot path.
#: CoreML conversion needs coremltools on macOS but the .mlpackage runtime is
#: unsupported, so it is never auto-installed from boot.
_RUNTIME_BACKEND_EXCLUDE = frozenset({"coreml"})


def all_backend_names() -> list[str]:
    """Return every backend whose inference runtime can be provisioned by boot.

    Used by ``ispy-boot -f`` so that all backends tested by
    ``ispy-bench --all-backends`` are available out of the box instead of
    requiring a manual ``pip install`` per backend.
    """
    return [name for name in BACKEND_REGISTRY if name not in _RUNTIME_BACKEND_EXCLUDE]

def get_backend_spec(name: str | None) -> BackendSpec | None:
    return BACKEND_REGISTRY.get(normalize_backend(name))

def resolve_backend(
    name: str | None, platform_info: PlatformInfo | None = None
) -> Resolution:
    spec = get_backend_spec(name)
    if spec is None:
        return Resolution(
            mode="unsupported",
            reason=(
                f"Unknown backend {name!r}. Known backends: "
                + ", ".join(sorted(BACKEND_REGISTRY))
            ),
        )
    return spec.resolver(platform_info or detect_platform())

# ---------------------------------------------------------------------------
# Model/config -> backend selection
# ---------------------------------------------------------------------------

_MODEL_PATH_KEYS = ("file_path", "path")

def backend_for_model_path(path: str | Path | None) -> str | None:
    value = str(path or "").strip()
    if not value:
        return None
    lowered = value.lower()
    if "openvino_model" in lowered or "openvino" in Path(lowered).parts:
        return "openvino"
    suffix = Path(lowered).suffix
    return _MODEL_SUFFIX_BACKENDS.get(suffix)

def backends_for_config(
    config: dict | None, platform_info: PlatformInfo | None = None
) -> list[str]:
    """Return the runtime backends the configured models actually need."""

    found: set[str] = set()

    def _visit(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _MODEL_PATH_KEYS and isinstance(value, str):
                    backend = backend_for_model_path(value)
                    if backend:
                        found.add(backend)
                else:
                    _visit(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _visit(item)

    _visit(config or {})

    # A QNN model is stored as .onnx; the plugin is an extra runtime dependency.
    p = platform_info or detect_platform()
    if "onnx" in found and p.has_qualcomm:
        found.add("qnn")
    # CoreML cannot be loaded at runtime; do not provision its conversion tools
    # from a boot path.  ispy-optimize asks for it explicitly when converting.
    found.discard("coreml")
    order = list(BACKEND_REGISTRY)
    return [name for name in order if name in found]

# ---------------------------------------------------------------------------
# Legacy mapping (kept for callers that only need module/target pairs)
# ---------------------------------------------------------------------------

def legacy_dependencies(
    platform_info: PlatformInfo | None = None,
) -> dict[str, list[tuple]]:
    """Return {backend: [(module, target, extra_args?) ...]}."""

    p = platform_info or detect_platform()
    result: dict[str, list[tuple]] = {}
    for name, spec in BACKEND_REGISTRY.items():
        resolution = spec.resolver(p)
        entries: list[tuple] = []
        if resolution.mode == "install":
            for pkg in resolution.packages:
                extra: list[str] = []
                if pkg.index_url:
                    extra += ["--index-url", pkg.index_url]
                for url in pkg.extra_index_urls:
                    extra += ["--extra-index-url", url]
                for link in pkg.find_links:
                    extra += ["-f", link]
                if extra:
                    entries.append((pkg.module, pkg.requirement, extra))
                else:
                    entries.append((pkg.module, pkg.requirement))
        result[name] = entries
    return result

def __getattr__(name: str):
    """Resolve the compatibility view lazily.

    Building it runs platform/hardware probes.  Importing a module that merely
    wants a helper should not import torch or touch the hardware.
    """
    if name == "BACKEND_DEPENDENCIES":
        return legacy_dependencies()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ---------------------------------------------------------------------------
# Pip plumbing
# ---------------------------------------------------------------------------

def _in_virtualenv() -> bool:
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )

def looks_out_of_space(text: str) -> bool:
    lowered = (text or "").lower()
    return any(sig in lowered for sig in _SPACE_ERRORS)

def purge_pip_cache() -> bool:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "cache", "purge"],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("pip cache purge did not run: %s", exc)
        return False
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
        logger.warning("pip cache purge failed (exit %s): %s", proc.returncode, tail)
        return False
    logger.info("Purged the pip cache to free disk space.")
    return True

def pip_install(
    install_target: str,
    force_reinstall: bool = False,
    extra_args: Sequence[str] | None = None,
    *,
    no_deps: bool | None = None,
    no_cache: bool = False,
    timeout: int = INSTALL_TIMEOUT,
) -> bool:
    """Install *install_target* with the running interpreter."""

    try:
        from iSpy.core import network_policy
    except ImportError:  # pragma: no cover - core module
        network_policy = None

    if network_policy is not None and network_policy.is_offline():
        is_remote = install_target.startswith(("http://", "https://"))
        is_local = not is_remote and Path(install_target).exists()
        if not is_local:
            logger.warning(
                "Offline mode: skipping pip install of %s - pip needs the "
                "internet.",
                install_target,
            )
            return False

    if no_deps is None:
        no_deps = install_target.startswith("http") or install_target.endswith(
            ".whl"
        )

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-input",
        "--disable-pip-version-check",
    ]
    if no_deps:
        cmd.append("--no-deps")
    if force_reinstall:
        cmd.append("--force-reinstall")
    if no_cache:
        cmd.append("--no-cache-dir")
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(install_target)
    if not _in_virtualenv():
        cmd.append("--break-system-packages")

    def _run():
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )

    logger.info("Installing: %s", install_target)
    try:
        proc = _run()
        if proc.returncode != 0 and looks_out_of_space(proc.stdout or ""):
            logger.warning(
                "pip ran out of disk space installing %s - purging the pip "
                "cache and retrying once.",
                install_target,
            )
            if purge_pip_cache():
                proc = _run()
        if proc.returncode == 0:
            return True
        tail = "\n".join((proc.stdout or "").strip().splitlines()[-20:])
        logger.warning("pip install failed for: %s\n%s", install_target, tail)
        return False
    except subprocess.TimeoutExpired:
        logger.warning("pip install timed out for: %s", install_target)
        return False
    except OSError as exc:
        logger.warning("pip install failed for: %s (%s)", install_target, exc)
        return False

def uninstall_packages(names: Sequence[str]) -> bool:
    """Uninstall existing distributions, ignoring ones that are not present."""

    present = [name for name in names if installed_version(name) is not None]
    if not present:
        return True
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "-y",
        *present,
    ]
    if not _in_virtualenv():
        cmd.append("--break-system-packages")
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=UNINSTALL_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("pip uninstall failed for %s: %s", present, exc)
        return False
    if proc.returncode != 0:
        tail = (proc.stdout or "").strip()[-500:]
        logger.warning("pip uninstall failed for %s: %s", present, tail)
        return False
    logger.info("Removed conflicting distributions: %s", ", ".join(present))
    return True

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def installed_version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None

def _satisfies_version(pkg: PackageSpec, version: str | None) -> bool:
    if version is None:
        return False
    try:
        requirement = Requirement(pkg.requirement)
    except (InvalidRequirement, ValueError):
        # Local wheel / direct URL: any installed distribution is acceptable;
        # the import check below is the real gate.
        return True
    if requirement.name.lower().replace("_", "-") != pkg.dist.lower().replace(
        "_", "-"
    ):
        return True
    try:
        return requirement.specifier.contains(version, prereleases=True)
    except (InvalidVersion, ValueError):
        return True

def verify_package(pkg: PackageSpec, timeout: int = VERIFY_TIMEOUT) -> tuple[bool, str]:
    """Import and exercise *pkg* in a fresh interpreter."""

    if installed_version(pkg.dist) is None and not _module_present(pkg.module):
        return False, "not installed"

    script = pkg.verify_script or (
        "import importlib\n"
        f"importlib.import_module({pkg.module!r})\n"
        "print('import ok')\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"verification timed out after {timeout}s"
    except OSError as exc:
        return False, f"verification could not start: {exc}"

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode == 0:
        detail = stdout.splitlines()[-1] if stdout else "import ok"
        return True, detail
    detail = (stderr or stdout).splitlines()
    detail = detail[-1] if detail else f"exit {proc.returncode}"
    return False, detail

def _module_present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False

# ---------------------------------------------------------------------------
# Disk awareness
# ---------------------------------------------------------------------------

def _minimum_free_bytes(path: Path) -> int | None:
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return None

def check_disk_space(
    required: int, paths: Iterable[Path | None] = (None,)
) -> tuple[bool, str]:
    if required <= 0:
        return True, ""
    roots = [p for p in paths if p is not None]
    if not roots:
        roots = [Path(sys.prefix)]
    for root in roots:
        free = _minimum_free_bytes(root)
        if free is not None and free < required:
            return (
                False,
                f"{root} has {free / 1024 ** 3:.1f} GiB free; "
                f"{required / 1024 ** 3:.1f} GiB is required for a safe install.",
            )
    return True, ""

# ---------------------------------------------------------------------------
# ARM CUDA cleanup
# ---------------------------------------------------------------------------

def _cuda_related_distributions() -> list[str]:
    names: set[str] = set()
    for dist in importlib.metadata.distributions():
        raw = (dist.metadata.get("Name") or "").strip().lower()
        if not raw:
            continue
        name = raw.replace("_", "-")
        if name.startswith("nvidia-") or name in _CUDA_ONLY_DISTS:
            names.add(name)
    return sorted(names)

def remove_forbidden_cuda_packages(
    platform_info: PlatformInfo,
    uninstaller: Callable[[Sequence[str]], bool] | None = None,
) -> list[str]:
    """Strip CUDA-only packages from CPU-only Linux ARM64 environments."""

    if (
        platform_info.system != "Linux"
        or platform_info.machine != "aarch64"
        or platform_info.is_jetson
        or platform_info.has_nvidia
    ):
        return []
    found = _cuda_related_distributions()
    if not found:
        return []
    logger.warning(
        "Removing CUDA-only packages from this CPU-only ARM64 environment: %s",
        ", ".join(found),
    )
    ok = (uninstaller or uninstall_packages)(found)
    if not ok:
        logger.error("Could not remove CUDA-only packages: %s", ", ".join(found))
        return []
    return found

# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------

Installer = Callable[[PackageSpec, bool], bool]
Verifier = Callable[[PackageSpec], tuple[bool, str]]
Uninstaller = Callable[[Sequence[str]], bool]

@dataclass
class PackageResult:
    backend: str
    dist: str
    status: str  # reused | installed | missing | failed | offline | disk
    detail: str = ""
    version: str | None = None
    fallback: bool = False

    @property
    def ok(self) -> bool:
        return self.status in ("reused", "installed")

@dataclass
class BackendResult:
    backend: str
    title: str
    status: str  # ok | manual | unsupported | missing | failed | offline | disk
    detail: str = ""
    packages: list[PackageResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "system", "reused", "installed")

@dataclass
class ProvisionReport:
    platform: PlatformInfo
    results: list[BackendResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.results)

    @property
    def actionable_failures(self) -> list[BackendResult]:
        """Failures an install/repair run can actually fix."""

        return [
            result
            for result in self.results
            if result.status in ("missing", "failed", "offline", "disk")
        ]

    @property
    def manual_requirements(self) -> list[BackendResult]:
        return [
            result
            for result in self.results
            if result.status in ("manual", "unsupported")
        ]

    def lines(self) -> list[str]:
        lines = [f"Dependencies: {self.platform.describe()}"]
        for result in self.results:
            state = result.status
            detail = f" - {result.detail}" if result.detail else ""
            lines.append(f"  [{state}] {result.backend}: {result.title}{detail}")
            for pkg in result.packages:
                version = f" ({pkg.version})" if pkg.version else ""
                pkg_detail = f" - {pkg.detail}" if pkg.detail else ""
                lines.append(
                    f"    {pkg.status}: {pkg.dist}{version}{pkg_detail}"
                )
        return lines

class DependencyProvisionError(RuntimeError):
    """An install/repair failure that the caller must not report as success."""

    def __init__(self, message: str, report: ProvisionReport):
        super().__init__(message)
        self.report = report

def _default_install(pkg: PackageSpec, force: bool) -> bool:
    if pkg.uninstall_first:
        if not uninstall_packages(pkg.uninstall_first):
            return False
    extra: list[str] = []
    if pkg.index_url:
        extra += ["--index-url", pkg.index_url]
    for url in pkg.extra_index_urls:
        extra += ["--extra-index-url", url]
    for link in pkg.find_links:
        extra += ["-f", link]
    return pip_install(
        pkg.requirement,
        force_reinstall=force,
        extra_args=extra or None,
        no_deps=pkg.no_deps,
        no_cache=pkg.no_cache,
    )

def _verify_managed_backend(
    backend: str, title: str, verifier: Verifier
) -> BackendResult | None:
    """Verify a vendor/system-managed package when it is already importable."""

    spec = BACKEND_REGISTRY.get(backend)
    module = spec.module if spec else ""
    if not module or not _module_present(module):
        return None
    pkg = PackageSpec(
        dist=module,
        requirement=module,
        module=module,
        verify_script=f"import {module}\nprint('system package ok')\n",
    )
    ok, detail = verifier(pkg)
    return BackendResult(backend, title, "system", detail) if ok else None

def _install_one(
    backend: str,
    resolution: Resolution,
    *,
    auto_install: bool,
    installer: Installer,
    verifier: Verifier,
) -> BackendResult:
    title = (
        BACKEND_REGISTRY[backend].title
        if backend in BACKEND_REGISTRY
        else backend
    )
    if resolution.mode == "manual":
        result = _verify_managed_backend(backend, title, verifier)
        if result is not None:
            return result
        return BackendResult(backend, title, "manual", resolution.reason)
    if resolution.mode == "unsupported":
        return BackendResult(backend, title, "unsupported", resolution.reason)
    if resolution.mode == "system":
        # Verify a system-managed runtime if it is present; never pip-install it.
        result = _verify_managed_backend(backend, title, verifier)
        if result is not None:
            return result
        return BackendResult(backend, title, "manual", resolution.reason)

    result = BackendResult(backend, title, "failed")
    last_detail = ""
    for index, pkg in enumerate(resolution.packages):
        version = installed_version(pkg.dist)
        satisfies = _satisfies_version(pkg, version)
        if satisfies:
            ok, detail = verifier(pkg)
            if ok:
                result.packages.append(
                    PackageResult(
                        backend,
                        pkg.dist,
                        "reused",
                        detail,
                        version=version,
                        fallback=index > 0,
                    )
                )
                result.status = "reused"
                result.detail = pkg.note or detail
                return result
            last_detail = f"installed {pkg.dist} failed verification: {detail}"
        else:
            last_detail = (
                f"installed {pkg.dist} {version or 'missing'} does not satisfy "
                f"{pkg.requirement}"
            )

        if not auto_install:
            result.packages.append(
                PackageResult(
                    backend,
                    pkg.dist,
                    "missing",
                    last_detail,
                    version=version,
                    fallback=index > 0,
                )
            )
            result.status = "missing"
            result.detail = last_detail
            # Keep walking the ordered candidates: on a CUDA machine the
            # fallback CPU torch may be the valid, already-installed build.
            continue

        free_ok, free_detail = check_disk_space(
            pkg.min_free_bytes,
            [Path(sys.prefix), Path.home() / ".cache"],
        )
        if not free_ok:
            result.packages.append(
                PackageResult(
                    backend,
                    pkg.dist,
                    "disk",
                    free_detail,
                    version=version,
                    fallback=index > 0,
                )
            )
            result.status = "disk"
            result.detail = free_detail
            logger.error("Not enough disk space for %s: %s", pkg.dist, free_detail)
            continue

        # A version mismatch can be fixed with a normal upgrade; a broken
        # import needs the files replaced, so force only in that case.
        force = version is not None and satisfies
        logger.info(
            "Provisioning %s %s for backend %s%s",
            pkg.dist,
            pkg.requirement,
            backend,
            " (repair)" if force else "",
        )
        if not installer(pkg, force):
            result.packages.append(
                PackageResult(
                    backend,
                    pkg.dist,
                    "failed",
                    f"pip install failed for {pkg.requirement}",
                    version=version,
                    fallback=index > 0,
                )
            )
            result.status = "failed"
            result.detail = (
                f"Could not install {pkg.dist}. "
                + (pkg.note or "Check the network and retry 'ispy-boot -f'.")
            )
            continue

        ok, detail = verifier(pkg)
        if ok:
            result.packages.append(
                PackageResult(
                    backend,
                    pkg.dist,
                    "installed",
                    detail,
                    version=installed_version(pkg.dist),
                    fallback=index > 0,
                )
            )
            result.status = "installed"
            result.detail = pkg.note or detail
            return result
        last_detail = f"installed {pkg.dist} but verification failed: {detail}"
        result.packages.append(
            PackageResult(
                backend,
                pkg.dist,
                "failed",
                last_detail,
                version=installed_version(pkg.dist),
                fallback=index > 0,
            )
        )
        result.status = "failed"
        result.detail = last_detail

    if result.status == "failed" and last_detail and not result.detail:
        result.detail = last_detail
    return result

def ensure_backend(
    name: str,
    *,
    platform_info: PlatformInfo | None = None,
    auto_install: bool = True,
    installer: Installer | None = None,
    verifier: Verifier | None = None,
    uninstaller: Uninstaller | None = None,
) -> BackendResult:
    p = platform_info or detect_platform()
    backend = normalize_backend(name)
    spec = get_backend_spec(backend)
    if spec is None:
        return BackendResult(
            backend,
            backend,
            "unsupported",
            f"Unknown backend {name!r}.",
        )
    resolution = spec.resolver(p)
    result = _install_one(
        backend,
        resolution,
        auto_install=auto_install,
        installer=installer or _default_install,
        verifier=verifier or verify_package,
    )
    if backend == "pytorch" and auto_install:
        remove_forbidden_cuda_packages(p, uninstaller)
    return result

def ensure_backends(
    names: Iterable[str],
    *,
    platform_info: PlatformInfo | None = None,
    auto_install: bool = True,
    installer: Installer | None = None,
    verifier: Verifier | None = None,
    uninstaller: Uninstaller | None = None,
) -> ProvisionReport:
    """Provision every requested backend, reusing valid installs."""

    p = platform_info or detect_platform()
    normalized: list[str] = []
    for name in names:
        backend = normalize_backend(name)
        if backend and backend not in normalized:
            normalized.append(backend)
    normalized.sort(key=lambda item: 0 if item == "pytorch" else 1)

    report = ProvisionReport(platform=p)
    attempted = set(normalized)

    for backend in normalized:
        result = ensure_backend(
            backend,
            platform_info=p,
            auto_install=auto_install,
            installer=installer,
            verifier=verifier,
            uninstaller=uninstaller,
        )
        report.results.append(result)
        if not result.ok:
            spec = get_backend_spec(backend)
            fallback = normalize_backend(spec.fallback) if spec and spec.fallback else ""
            if fallback and fallback not in attempted:
                attempted.add(fallback)
                fallback_result = ensure_backend(
                    fallback,
                    platform_info=p,
                    auto_install=auto_install,
                    installer=installer,
                    verifier=verifier,
                    uninstaller=uninstaller,
                )
                fallback_result.detail = (
                    f"fallback for {backend}: {fallback_result.detail}".strip()
                )
                report.results.append(fallback_result)
    return report

def raise_for_actionable_failures(report: ProvisionReport) -> None:
    failures = report.actionable_failures
    if not failures:
        return
    summary = "; ".join(
        f"{item.backend}: {item.detail or item.status}" for item in failures
    )
    raise DependencyProvisionError(
        "Runtime dependency provisioning failed: " + summary,
        report,
    )

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def registry_snapshot(platform_info: PlatformInfo | None = None) -> list[dict]:
    p = platform_info or detect_platform()
    rows: list[dict] = []
    for name, spec in BACKEND_REGISTRY.items():
        resolution = spec.resolver(p)
        rows.append(
            {
                "backend": name,
                "title": spec.title,
                "module": spec.module,
                "mode": resolution.mode,
                "packages": [
                    {
                        "dist": pkg.dist,
                        "requirement": pkg.requirement,
                        "index": pkg.index_url,
                    }
                    for pkg in resolution.packages
                ],
                "reason": resolution.reason,
                "fallback": spec.fallback,
            }
        )
    return rows

def format_registry(platform_info: PlatformInfo | None = None) -> str:
    p = platform_info or detect_platform()
    lines = [f"Platform: {p.describe()}"]
    for row in registry_snapshot(p):
        lines.append(f"{row['backend']}: {row['mode']} ({row['title']})")
        for pkg in row["packages"]:
            index = f" from {pkg['index']}" if pkg["index"] else ""
            lines.append(f"  {pkg['dist']}: {pkg['requirement']}{index}")
        if row["reason"]:
            lines.append(f"  note: {row['reason']}")
    return "\n".join(lines)
