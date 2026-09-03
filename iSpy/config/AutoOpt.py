import platform
import subprocess
import os
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"tflite", "openvino", "coreml", "onnx", "rknn", "engine", "tpu"}


def _run(cmd: str) -> str:
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            timeout=2,
        ).stdout.lower()
    except Exception:
        return ""


def _cmd_ok(cmd: str) -> bool:
    try:
        return (
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True,
                timeout=2,
            ).returncode
            == 0
        )
    except Exception:
        return False


@lru_cache()
def _lsusb_output() -> str:
    if os.name != "nt" and _cmd_ok("which lsusb"):
        return _run("lsusb")
    if os.name == "nt":
        return _run("wmic path win32_pnpentity get name")
    return ""



@lru_cache()
def has_jetson() -> bool:
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            with open(path, "rb") as f:
                model = f.read().decode(errors="ignore").lower()
            if "jetson" in model or "tegra" in model:
                return True
        except Exception:
            pass
    return False

@lru_cache()
def has_hailo_npu() -> bool:
    import glob
    if glob.glob("/dev/hailo*"):
        return True
    if "hailo" in _lsusb_output():
        return True
    return _cmd_ok("hailortcli fw-control identify")

@lru_cache()
def has_nvidia() -> bool:
    if has_jetson():
        return True
    if any(
        os.path.exists(p)
        for p in ("/dev/nvidia0", "/dev/nvidiactl", "/proc/driver/nvidia/version")
    ):
        return True
    if _cmd_ok("nvidia-smi"):
        return True
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        logger.warning("PyTorch not installed, skipping CUDA check for NVIDIA GPU.")
    if os.name == "nt" and "nvidia" in _run("wmic path win32_videocontroller get name"):
        return True
    return False


@lru_cache()
def has_tensorrt() -> bool:
    try:
        import tensorrt  # noqa: F401

        return True
    except ImportError:
        pass
    # check system python too, in case we're in a venv without it
    return _cmd_ok("python3 -c 'import tensorrt' 2>/dev/null")


@lru_cache()
def has_tpu() -> bool:
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        return True
    except Exception:
        pass
    return False


def has_amd_gpu() -> bool:
    if os.name == "nt":
        return "amd" in _run("wmic path win32_videocontroller get name")
    return "amd" in _run("lspci") or "radeon" in _run("lspci")


def has_intel_gpu() -> bool:
    if os.name == "nt":
        return "intel" in _run("wmic path win32_videocontroller get name")
    return "intel" in platform.processor().lower() or "intel" in _run("lspci")


def has_arm() -> bool:
    return "arm" in platform.machine().lower() or "aarch" in platform.machine().lower()


def has_apple_silicon() -> bool:
    return platform.system() == "Darwin" and has_arm()


def has_intel_vpu() -> bool:
    usb = _lsusb_output()
    return "movidius" in usb or "03e7:2485" in usb


def has_edge_tpu() -> bool:
    usb = _lsusb_output()
    return "18d1:9302" in usb or "1ac1:089a" in usb


@lru_cache()
def has_rockchip_npu() -> bool:
    import glob

    if glob.glob("/dev/rknpu*"):
        return True
    if _cmd_ok("lsmod 2>/dev/null | grep -q rknpu"):
        return True

    rockchip_indicators = ("rk3588", "rk3576", "rk3399", "rk3568", "rk3566", "rk3528", "rv1103", "rv1106")
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read().lower()
        if any(s in cpuinfo for s in rockchip_indicators):
            return True
    except Exception:
        pass

    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            with open(path, "rb") as f:
                model = f.read().decode(errors="ignore").lower()
            if any(s in model for s in rockchip_indicators):
                return True
            if "orange pi" in model or "rockchip" in model:
                return True
        except Exception:
            pass

    return False

def resolve_openvino_device(requested_device=None) -> str:
    if isinstance(requested_device, str) and requested_device.startswith("intel:"):
        return requested_device
    try:
        from openvino import Core
        available = Core().available_devices
    except Exception:
        return "intel:cpu"
    if "GPU" in available:
        return "intel:gpu"
    if "NPU" in available:
        return "intel:npu"
    return "intel:cpu"


def recommend_format(
    ignore_dependencies: bool = False, runtime_supported: bool = True
) -> str:
    # runtime_supported:
    #   True  - caller wants the best accelerator *artifact to build/convert*
    #           (optimizer, dependency installer). Compiler-backed formats
    #           (engine/openvino/coreml) are all valid picks.
    #   False - caller only wants a format it can *load and run inference on*
    #           right now. engine/openvino/coreml are skipped so the pick falls
    #           through to the next best thing (e.g. onnx on CPU), because
    #           GenericYolo may not expose a runtime path for every compiled
    #           format. Only coreml stays unimplemented after Bug 7, but the
    #           engine/openvino guard keeps the contract honest if a future
    #           backend regresses either.
    # 1. embedded NPUs / TPUs
    if has_rockchip_npu():
        logger.info("Rockchip NPU detected - using RKNN format for hardware acceleration.")
        return "rknn"
    if has_hailo_npu():
        logger.info("Hailo NPU detected - using HEF format for hardware acceleration.")
        return "hef"
    if has_edge_tpu():
        logger.info("Edge TPU detected - using TFLite format for hardware acceleration.")
        return "tflite"

    # 2. apple ecosystem
    if has_apple_silicon() and runtime_supported:
        logger.info("Apple Silicon detected - using Core ML format for hardware acceleration.")
        return "coreml"

    # 3. google TPU - pytorch via XLA
    if has_tpu():
        logger.info("Google TPU detected - using TPU format for hardware acceleration.")
        return "tpu"

    # 4. nvidia - tensorrt engine > onnx for max fps
    if has_nvidia():
        if (has_tensorrt() or ignore_dependencies) and runtime_supported:
            logger.info(
                "NVIDIA GPU detected - using .engine format for maximum FPS."
            )
            return "engine"
        logger.info(
            "NVIDIA GPU detected but .engine runtime unsupported - falling back "
            "to ONNX (install tensorrt for a significant FPS boost)."
        )
        return "onnx"

    # 5. desktop hardware
    if os.name != "nt" and has_intel_vpu() and runtime_supported:
        logger.info("Intel VPU detected - using OpenVINO format for hardware acceleration.")
        return "openvino"
    if has_intel_gpu() and runtime_supported:
        logger.info("Intel GPU detected - using OpenVINO format for hardware acceleration.")
        return "openvino"
    if has_amd_gpu():
        logger.info("AMD GPU detected - using ONNX format for hardware acceleration.")
        return "onnx"  # rocm / directml exec providers

    # 6. arm edge (jetson, rpi, etc.)
    if has_arm():
        logger.info("ARM edge device detected - using TFLite format for hardware acceleration.")
        return "tflite"

    logger.info("No specialised hardware detected - defaulting to ONNX (CPU).")
    return "onnx"
