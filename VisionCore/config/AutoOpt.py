import platform
import subprocess
import os
from functools import lru_cache

SUPPORTED_FORMATS = {"tflite", "openvino", "coreml", "onnx", "rknn"}

def _run(cmd: str) -> str:
    try:
        return subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, timeout=2
        ).stdout.lower()
    except Exception:
        return ""

def _cmd_ok(cmd: str) -> bool:
    try:
        return subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True, timeout=2
        ).returncode == 0
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
    if any(os.path.exists(p) for p in ("/dev/nvidia0", "/dev/nvidiactl", "/proc/driver/nvidia/version")):
        return True
    if _cmd_ok("nvidia-smi"):
        return True
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
    if os.name == "nt" and "nvidia" in _run("wmic path win32_videocontroller get name"):
        return True
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
    return os.path.exists("/dev/rknpu") or os.path.exists("/dev/rknpu0")

def recommend_format() -> str:
    # 1. Specialized Embedded NPUs / TPUs
    if has_rockchip_npu():
        return "rknn"
    if has_edge_tpu():
        return "tflite"
        
    # 2. Apple Ecosystem
    if has_apple_silicon():
        return "coreml"
        
    # 3. Discrete & Integrated Desktop Hardware
    if has_nvidia():
        return "onnx"  # High FPS via TensorRT/CUDA execution providers
    if os.name != "nt" and has_intel_vpu():
        return "openvino"
    if has_intel_gpu():
        return "openvino"
    if has_amd_gpu():
        return "onnx"  # High FPS via ROCm/DirectML execution providers
        
    if has_arm():
        return "tflite"
        
    return "onnx"
