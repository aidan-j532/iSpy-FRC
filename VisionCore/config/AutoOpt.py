import platform
import subprocess
import os
from functools import lru_cache

SUPPORTED_FORMATS = {"tflite", "openvino", "coreml", "onnx", "rknn"}
@lru_cache(maxsize=16)
def run(cmd: str) -> str:
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

def command_exists(cmd):
    return os.system(f"{cmd} >nul 2>&1" if os.name == "nt" else f"which {cmd} >/dev/null 2>&1") == 0


def has_nvidia():
    return command_exists("nvidia-smi")

def has_amd_gpu():
    if os.name == "nt":
        return "amd" in run("wmic path win32_videocontroller get name")
    lspci = run("lspci")
    return "amd" in lspci or "radeon" in lspci

def has_intel_gpu():
    return "intel" in platform.processor().lower() or "intel" in run("lspci")

has_intel = has_intel_gpu

def has_arm():
    return "arm" in platform.machine().lower() or "aarch" in platform.machine().lower()

def has_apple_silicon():
    return platform.system() == "Darwin" and "arm" in platform.machine().lower()

@lru_cache()
def _lsusb_output() -> str:
    if command_exists("lsusb"):
        return run("lsusb")
    return run("wmic path win32_pnpentity get name")

def has_intel_vpu():
    usb = _lsusb_output()
    return "movidius" in usb or "03e7:2485" in usb

def has_edge_tpu():
    usb = _lsusb_output()
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