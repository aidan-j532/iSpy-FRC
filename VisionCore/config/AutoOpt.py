import platform
import subprocess
import os
from functools import lru_cache

SUPPORTED_FORMATS = {"tflite", "openvino", "coreml", "onnx", "rknn"}

@lru_cache()
def run(cmd):
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=isinstance(cmd, str)
        ).stdout.lower()
    except Exception:
        return ""

def command_exists(cmd):
    return os.system(f"{cmd} >nul 2>&1" if os.name == "nt" else f"which {cmd} >/dev/null 2>&1") == 0


def has_nvidia():
    return command_exists("nvidia-smi")

def has_amd_gpu():
    if os.name == "nt":
        return "amd" in run("wmic path win32_videocontroller get name")
    return "amd" in run(["lspci"]) or "radeon" in run(["lspci"])

def has_intel_gpu():
    return "intel" in platform.processor().lower() or "intel" in run(["lspci"])

has_intel = has_intel_gpu

def has_arm():
    return "arm" in platform.machine().lower() or "aarch" in platform.machine().lower()

def has_apple_silicon():
    return platform.system() == "Darwin" and "arm" in platform.machine().lower()

@lru_cache()
def lsusb():
    return run(["lsusb"]) if command_exists("lsusb") else run("wmic path win32_pnpentity get name")

def has_intel_vpu():
    usb = lsusb()
    return "movidius" in usb or "03e7:2485" in usb

def has_edge_tpu():
    usb = lsusb()
    return "18d1:9302" in usb or "1ac1:089a" in usb

def has_rockchip_npu():
    return (
        os.path.exists("/dev/rknpu") or
        os.path.exists("/dev/rknpu0") or
        "rknn" in run("dmesg")
    )

def has_hailo_npu():
    return os.path.exists("/dev/hailo0")

def recommend_format() -> str:
    scores = {fmt: 0 for fmt in SUPPORTED_FORMATS}

    # Edge TPU (Coral) -> TFLite is the only option
    if has_edge_tpu():
        scores["tflite"] += 100

    # Rockchip NPU -> RKNN
    if has_rockchip_npu():
        scores["rknn"] += 100

    # Intel Movidius VPU -> OpenVINO
    if has_intel_vpu():
        scores["openvino"] += 90

    # Apple Silicon -> CoreML
    if has_apple_silicon():
        scores["coreml"] += 100

    # NVIDIA GPU -> ONNX (TensorRT backend or direct)
    if has_nvidia():
        scores["onnx"] += 80

    # AMD GPU -> ONNX (best cross-runtime support)
    if has_amd_gpu():
        scores["onnx"] += 70

    # Intel GPU/CPU -> OpenVINO
    if has_intel_gpu():                  # FIX: was has_intel() — NameError at runtime
        scores["openvino"] += 70

    # ARM CPU (Raspberry Pi, Jetson, etc) -> TFLite
    if has_arm():
        scores["tflite"] += 60

    # Fallback: ONNX is the most portable format
    scores["onnx"] += 10

    best = max(scores, key=scores.get)
    return best