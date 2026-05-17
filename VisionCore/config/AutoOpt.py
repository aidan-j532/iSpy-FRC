import platform
import subprocess
import os
from functools import lru_cache

SUPPORTED_FORMATS = {"tflite", "openvino", "coreml", "onnx", "rknn"}

@lru_cache(maxsize=16)
def _run(cmd: str) -> str:
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
        ).stdout.lower()
    except Exception:
        return ""


def _cmd_ok(cmd: str) -> bool:
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True,
        ).returncode == 0
    except Exception:
        return False


@lru_cache()
def _lsusb_output() -> str:
    if _cmd_ok("lsusb"):
        return _run("lsusb")
    # Windows fallback
    return _run("wmic path win32_pnpentity get name")

@lru_cache()
def has_nvidia() -> bool:
    if os.path.exists("/dev/nvidia0") or os.path.exists("/dev/nvidiactl"):
        return True

    if os.path.exists("/proc/driver/nvidia/version"):
        return True

    if _cmd_ok("nvidia-smi"):
        return True

    for smi in ("/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi"):
        if os.path.isfile(smi) and _cmd_ok(smi):
            return True

    cuda_libs = [
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/libcuda.so.1",
    ]
    if any(os.path.exists(p) for p in cuda_libs):
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
    lspci = _run("lspci")
    return "amd" in lspci or "radeon" in lspci


def has_intel_gpu() -> bool:
    return "intel" in platform.processor().lower() or "intel" in _run("lspci")


# Legacy alias kept for external callers.
has_intel = has_intel_gpu


def has_arm() -> bool:
    m = platform.machine().lower()
    return "arm" in m or "aarch" in m


def has_apple_silicon() -> bool:
    return platform.system() == "Darwin" and "arm" in platform.machine().lower()


def has_intel_vpu() -> bool:
    usb = _lsusb_output()
    return "movidius" in usb or "03e7:2485" in usb


def has_edge_tpu() -> bool:
    usb = _lsusb_output()
    return "18d1:9302" in usb or "1ac1:089a" in usb


@lru_cache()
def has_rockchip_npu() -> bool:
    return (
        os.path.exists("/dev/rknpu")
        or os.path.exists("/dev/rknpu0")
        or "rknn" in _run("dmesg")
    )


def has_hailo_npu() -> bool:
    return os.path.exists("/dev/hailo0")

def recommend_format() -> str:
    scores = {fmt: 0 for fmt in SUPPORTED_FORMATS}
    if has_rockchip_npu():
        scores["rknn"] += 1000

    if has_edge_tpu():
        scores["tflite"] += 1000

    if has_apple_silicon():
        scores["coreml"] += 1000

    if has_intel_vpu():
        scores["openvino"] += 900

    if has_nvidia():
        scores["onnx"] += 800

    if has_intel_gpu():
        scores["openvino"] += 700

    if has_amd_gpu():
        scores["onnx"] += 650

    if has_intel_gpu() and not has_nvidia() and not has_amd_gpu():
        scores["openvino"] += 400

    if has_arm():
        scores["tflite"] += 350

    scores["onnx"] += 100

    return max(scores, key=scores.get)