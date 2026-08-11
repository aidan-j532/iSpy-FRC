import sys
import os
import json
import time
import logging
import argparse
import hashlib
import warnings
import contextlib
import re
import ctypes
from pathlib import Path
import ctypes
from pathlib import Path

# Load libc for fflush() - used by _quiet() to flush C stdio buffers
_libc = None
if sys.platform == "win32":
    try:
        _libc = ctypes.cdll.msvcrt
        _libc.fflush(None)
    except Exception:
        _libc = None
else:
    for _libname in ("libc.so.6", "libc.so", "libc.musl.so"):
        try:
            _libc = ctypes.CDLL(_libname, use_errno=False)
            _libc.fflush(None)  # verify the function exists
            break
        except Exception:
            _libc = None

if _libc is not None:
    _fflush = _libc.fflush
else:
    def _fflush(_) -> None:
        pass
import numpy as np
import cv2

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.resolve()
# Fallback if installed non-editably (__file__ in site-packages)
if not (_PROJECT_ROOT / "iSpy").is_dir():
    _PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(_PROJECT_ROOT))

from iSpy.config.AutoOpt import has_rockchip_npu, has_nvidia, has_tensorrt, has_tpu
from iSpy.vision.ModelInspector import fill_missing_config
from iSpy.vision.optimizer import convert_model

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
logging.getLogger("iSpy.vision.ModelInspector").setLevel(logging.INFO)
for name in list(logging.root.manager.loggerDict):
    if not name.startswith("iSpy") and name != "ispy-test":
        logging.getLogger(name).setLevel(logging.CRITICAL + 1)
logger = logging.getLogger("ispy-test")
warnings.filterwarnings("ignore")


def _file_fingerprint(path: Path) -> tuple[int, str]:
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(4096)
    h = hashlib.sha256(head).hexdigest()
    return (size, h)


def find_pt_files():
    raw: list[Path] = []

    dirs = [
        _PROJECT_ROOT / "YoloModels" / "pytorch",
        _PROJECT_ROOT / "iSpy" / "assets",
    ]
    for d in dirs:
        if d.exists():
            for f in d.glob("*.pt"):
                raw.append(f.resolve())

    config_path = _PROJECT_ROOT / "Config" / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            from iSpy.config.iSpyConfig import get_pipeline_settings
            models: list[Path] = []
            for cam in (cfg.get("camera_configs") or {}).values():
                if not isinstance(cam, dict):
                    continue
                settings = get_pipeline_settings(cam) or {}
                settings_vm = settings.get("vision_model")
                if isinstance(settings_vm, dict):
                    for key in ("file_path", "source_pt"):
                        p = settings_vm.get(key)
                        if p:
                            p = Path(p)
                            if not p.is_absolute():
                                p = _PROJECT_ROOT / p
                            if p.suffix == ".pt" and p.exists():
                                models.append(p.resolve())
            # Legacy top-level layout, tolerated for old configs.
            if not models:
                vm = cfg.get("vision_model", {})
                for key in ("file_path", "source_pt"):
                    p = vm.get(key)
                    if p:
                        p = Path(p)
                        if not p.is_absolute():
                            p = _PROJECT_ROOT / p
                        if p.suffix == ".pt" and p.exists():
                            models.append(p.resolve())
            raw.extend(models)
        except Exception:
            pass

    try:
        for f in (Path.home() / "YoloModels" / "pytorch").glob("*.pt"):
            raw.append(f.resolve())
    except Exception:
        pass

    seen: set[tuple[int, str]] = set()
    unique = []
    for p in raw:
        fp = _file_fingerprint(p)
        if fp not in seen:
            seen.add(fp)
            unique.append(p)
        else:
            logger.debug("Skipping duplicate model: %s", p)
    return sorted(unique, key=lambda p: p.name)


def make_placeholder_frame(h=480, w=640):
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _npu_masks():
    import glob
    n = len(glob.glob("/dev/rknpu*")) or 3
    masks = [(1, "NPU-core0"), (2, "NPU-core1")]
    if n >= 3:
        masks += [(4, "NPU-core2"), (3, "NPU-cores0+1"), (7, "NPU-all3")]
    return masks


def _cuda_devices() -> list[tuple[int, str]]:
    try:
        import torch
        n = torch.cuda.device_count()
        if n > 0:
            return [(i, f"CUDA-{i}") for i in range(n)]
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            timeout=10,
        ).decode().strip()
        indices = [int(line) for line in out.splitlines() if line.strip().isdigit()]
        if indices:
            return [(i, f"CUDA-{i}") for i in indices]
    except Exception:
        pass
    return [(0, "CUDA-0")]


def detect_test_plan() -> dict:
    plan: dict[str, tuple] = {}
    if has_rockchip_npu():
        plan["rknn"] = ("rknn", 0, _npu_masks())
    if has_tpu():
        plan["tpu"] = ("tpu", "tpu", [(None, "TPU")])
    if has_nvidia():
        cuda_devs = _cuda_devices()
        for dev, label in cuda_devs:
            plan[f"pt_{label.lower()}"] = ("pt", dev, [(None, label)])
            plan[f"onnx_{label.lower()}"] = ("onnx", dev, [(None, f"ONNX-{label}")])
        if has_tensorrt():
            dev, label = cuda_devs[0]
            plan[f"engine_{label.lower()}"] = ("engine", dev, [(None, f"TensorRT")])
    if not has_rockchip_npu():
        plan["pt_cpu"] = ("pt", "cpu", [(None, "CPU")])
        plan["onnx_cpu"] = ("onnx", "cpu", [(None, "ONNX-CPU")])
    return plan


_KERNEL_PRINTK: list[int] | None = None


def _mute_kernel_rknn():
    global _KERNEL_PRINTK
    for _mod_param in ("/sys/module/rknn/parameters/debug_level",
                       "/sys/module/rknn/parameters/debug"):
        try:
            with open(_mod_param) as _f:
                _cur = int(_f.read().strip())
            if _cur > 0:
                with open(_mod_param, "w") as _f:
                    _f.write("0")
            break
        except Exception:
            continue
    try:
        with open("/proc/sys/kernel/printk") as f:
            vals = re.findall(r"\d+", f.read())
        if vals:
            _KERNEL_PRINTK = [int(v) for v in vals[:4]]
            if _KERNEL_PRINTK[0] <= 4:
                return  # already muted
            with open("/proc/sys/kernel/printk", "w") as f:
                f.write("4 4 1 7")
            return
    except PermissionError:
        pass
    except (FileNotFoundError, OSError):
        pass
    # Fallback: try dmesg -n 4
    try:
        import subprocess as _sp
        _sp.run(["dmesg", "-n", "4"], capture_output=True, timeout=5)
    except Exception:
        pass


def _restore_kernel_log():
    global _KERNEL_PRINTK
    if _KERNEL_PRINTK is not None:
        try:
            with open("/proc/sys/kernel/printk", "w") as f:
                f.write(" ".join(str(v) for v in _KERNEL_PRINTK))
        except (PermissionError, FileNotFoundError, OSError):
            try:
                import subprocess as _sp
                _sp.run(["dmesg", "-n", str(_KERNEL_PRINTK[0])], capture_output=True, timeout=5)
            except Exception:
                pass
        _KERNEL_PRINTK = None


@contextlib.contextmanager
def _quiet():
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
        _fflush(None)
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out)
        os.close(old_err)


def get_or_convert(pt_path, fmt, input_size=(640, 640)):
    if fmt in ("tpu", "pt"):
        return pt_path
    with _quiet():
        result = Path(convert_model(str(pt_path), fmt, input_size))
    if not result.exists() or result == pt_path:
        return None
    return result

def make_base_config(pt_path, model_path, device):
    return {
        "file_path": str(model_path),
        "source_pt": str(pt_path),
        "task": "detect",
        "num_classes": 1,
        "input_size": [640, 640],
        "min_conf": 0.5,
        "device": device,
        # "output": {
        #     "format": "raw",
        #     "layout": "features_first",
        #     "box_format": "cxcywh",
        #     "score_mode": "objectness",
        #     "scores_are_logits": False,
        #     "apply_software_nms": False,
        #     "nms_iou": 0.45,
        #     "quantization": "none",
        # },
        # "input": {
        #     "layout": "nhwc",
        #     "dtype": "uint8",
        #     "letterbox": True,
        #     "pad_value": 114,
        #     "normalize": False,
        # },
    }


def benchmark(model_config, core_mask, duration=5.0):
    from iSpy.vision.pipelines.object_detection import ObjectDetectionCamera
    from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig

    config = iSpyConfig()
    # Model selection lives in the camera's pipeline settings, never at the
    # config root - the object detection camera only reads it from there.
    cam_entry = {
        "name": "bench",
        "source": 99,  # won't open -> placeholder
        "fps_cap": 1000,
        "yaw": 0, "pitch": 0, "height": 1.0,
        "x": 0, "y": 0,
        "grayscale": False,
        "subsystem": "bench",
        "calibration": {"distance": 1.0, "game_piece_size": 1.0, "size": 100, "fov": 90},
        "pipeline": {"name": "object_detection", "settings": {"vision_model": model_config}},
    }
    config.set("camera_configs", {"bench": cam_entry})

    cam_cfg = iSpyCameraConfig(cam_entry)

    with _quiet():
        camera = ObjectDetectionCamera(cam_cfg, config, core_mask=core_mask)

    # warm up
    for _ in range(5):
        camera.run()

    count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        camera.run()
        count += 1

    elapsed = time.perf_counter() - start
    camera.destroy()

    fps = count / elapsed
    inference_ms = elapsed / count * 1000
    return fps, inference_ms, count, elapsed

def _fmt_result(r: dict) -> str:
    if r.get("fps"):
        return f"{r['backend']:20s}  {r['fps']:6.1f} FPS  {r.get('inference_ms', 0):6.1f}ms"
    return f"{r['backend']:20s}  ERROR"

def main():
    _mute_kernel_rknn()
    try:
        return _main_body()
    finally:
        _restore_kernel_log()


def _main_body():
    parser = argparse.ArgumentParser(description="Find the fastest inference backend")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    plan = detect_test_plan()
    active = {k: v for k, v in plan.items() if v is not None}
    print(f"Testing {len(active)} backend type(s): {', '.join(active.keys())}...")
    print()

    pt_files = []
    if args.model:
        p = Path(args.model)
        pt_files = [p.resolve() if not p.is_absolute() else p]
    else:
        pt_files = find_pt_files()

    if not pt_files:
        print("No .pt models found.")
        return 1

    best: dict[str, dict] = {}
    all_results: list[dict] = []

    for pt_path in pt_files:
        name = pt_path.stem
        print(f"  {name}")

        for _, (fmt, device, masks) in active.items():
            model_path = get_or_convert(pt_path, fmt)
            if model_path is None:
                continue
            cfg = make_base_config(pt_path, model_path, device)
            cfg = fill_missing_config(cfg)

            for core_mask, label in masks:
                try:
                    fps, inference_ms, count, elapsed = benchmark(cfg, core_mask, args.duration)
                    r = {"model": name, "backend": label, "format": fmt,
                         "device": str(device), "core_mask": core_mask,
                         "fps": round(fps, 1), "inference_ms": round(inference_ms, 1), "frames": count, "elapsed": round(elapsed, 3)}
                except Exception as e:
                    print(f"X {label}: {e}")
                    r = {"model": name, "backend": label, "format": fmt,
                         "device": str(device), "core_mask": core_mask,
                         "fps": None, "inference_ms": None, "error": str(e)}
                all_results.append(r)
                cur = best.get(name)
                if cur is None or (r["fps"] or 0) > (cur["fps"] or 0):
                    best[name] = r

        w = best.get(name)
        if w:
            print(f"SUCCESS: {_fmt_result(w)}")

    print()
    print("=" * 55)
    print("  BEST BACKEND PER MODEL")
    print("=" * 55)
    for name, r in best.items():
        print(f"  {name:40s}  {_fmt_result(r)}")

    if best:
        overall = max(best.values(), key=lambda x: x["fps"] or 0)
        print()
        print(f"  OVERALL WINNER: {overall['backend']} @ {overall['fps']} FPS ({overall['model']})")

    output_path = args.output or str(Path.cwd() / "Outputs" / "benchmark_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"best": best, "all": all_results}, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    sys.exit(main())
