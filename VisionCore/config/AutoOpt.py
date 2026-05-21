import platform
import subprocess
import os
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"tflite", "openvino", "coreml", "onnx", "rknn", "engine"}


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
def has_nvidia() -> bool:
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
    # Check the system Python in case we're in a venv that doesn't have it
    return _cmd_ok("python3 -c 'import tensorrt' 2>/dev/null")


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
        cpuinfo = open("/proc/cpuinfo").read().lower()
        if any(s in cpuinfo for s in rockchip_indicators):
            return True
    except Exception:
        pass

    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            model = open(path).read().lower()
            if any(s in model for s in rockchip_indicators):
                return True
            if "orange pi" in model or "rockchip" in model:
                return True
        except Exception:
            pass

    return False


def recommend_format() -> str:
    # 1. Dedicated embedded NPUs / TPUs
    if has_rockchip_npu():
        return "rknn"
    if has_edge_tpu():
        return "tflite"

    # 2. Apple ecosystem
    if has_apple_silicon():
        return "coreml"

    # 3. NVIDIA - prefer TensorRT engine over raw ONNX for max FPS     ─
    if has_nvidia():
        if has_tensorrt():
            logger.info(
                "NVIDIA GPU + TensorRT detected - using .engine format for maximum FPS."
            )
            return "engine"
        logger.info(
            "NVIDIA GPU detected but TensorRT not found - falling back to ONNX "
            "(install tensorrt for a significant FPS boost)."
        )
        return "onnx"

    # 4. Other desktop hardware                       ─
    if os.name != "nt" and has_intel_vpu():
        return "openvino"
    if has_intel_gpu():
        return "openvino"
    if has_amd_gpu():
        return "onnx"  # ROCm / DirectML execution providers

    # 5. ARM edge (Jetson, RPi, etc.)
    if has_arm():
        return "tflite"

    logger.info("No specialised hardware detected - defaulting to ONNX (CPU).")
    return "onnx"
