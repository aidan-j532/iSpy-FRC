import sys
import os
import json
import time
import logging
import argparse
import hashlib
import warnings
import contextlib
from pathlib import Path

import numpy as np
import cv2

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

os.environ["RKNN_LOG_LEVEL"] = "3"
os.environ["ORT_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["YOLO_VERBOSE"] = "False"

from iSpy.config.AutoOpt import has_rockchip_npu, has_nvidia, has_tensorrt, has_tpu
from iSpy.vision.ModelInspector import fill_missing_config
from iSpy.boot.boot import convert_model

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
            vm = cfg.get("vision_model", {})
            for key in ("file_path", "source_pt"):
                p = vm.get(key)
                if p:
                    p = Path(p)
                    if not p.is_absolute():
                        p = _PROJECT_ROOT / p
                    if p.suffix == ".pt" and p.exists():
                        raw.append(p.resolve())
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


def detect_test_plan():
    return {
        "rknn": ("rknn", 0, _npu_masks()) if has_rockchip_npu() else None,
        "tpu": ("tpu", "tpu", [(None, "TPU")]) if has_tpu() else None,
        "engine": ("engine", 0, [(None, "TensorRT")]) if has_nvidia() and has_tensorrt() else None,
        "pt_cuda": ("pt", 0, [(None, "CUDA")]) if has_nvidia() else None,
        "onnx_cuda": ("onnx", 0, [(None, "ONNX-CUDA")]) if has_nvidia() else None,
        "pt_cpu": ("pt", "cpu", [(None, "CPU")]) if not has_rockchip_npu() else None,
        "onnx_cpu": ("onnx", "cpu", [(None, "ONNX-CPU")]) if not has_rockchip_npu() else None,
    }


@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as null:
        with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
            yield


def get_or_convert(pt_path, fmt, input_size=(640, 640)):
    if fmt == "tpu" or fmt == "pt":
        return pt_path
    with _quiet():
        result = Path(convert_model(str(pt_path), fmt, input_size))
    return result if result.exists() and result != pt_path else pt_path


def make_base_config(pt_path, model_path, device):
    return {
        "file_path": str(model_path),
        "source_pt": str(pt_path),
        "task": "detect",
        "num_classes": 1,
        "input_size": [640, 640],
        "min_conf": 0.5,
        "device": device,
        "output": {
            "format": "raw",
            "layout": "features_first",
            "box_format": "cxcywh",
            "score_mode": "objectness",
            "scores_are_logits": False,
            "apply_software_nms": False,
            "nms_iou": 0.45,
            "quantization": "none",
        },
        "input": {
            "layout": "nhwc",
            "dtype": "uint8",
            "letterbox": True,
            "pad_value": 114,
            "normalize": False,
        },
    }


def benchmark(model_config, core_mask, duration=5.0):
    from iSpy.vision.genericYolo import GenericYolo

    with _quiet():
        model = GenericYolo(model_config, core_mask=core_mask)
    target_h, target_w = model.input_size[1], model.input_size[0]
    frame = make_placeholder_frame(target_h, target_w)

    if model.model_type == "rknn":
        buf = np.empty((1, target_h, target_w, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model._letterbox_into(rgb, buf[0], model.input_size)
        infer = lambda: model.predict_preprocessed(buf, frame.shape)
    else:
        infer = lambda: model.predict(frame, orig_shape=frame.shape)

    for _ in range(5):
        infer()
    start = time.perf_counter()
    count = 0
    while time.perf_counter() - start < duration:
        infer()
        count += 1
    elapsed = time.perf_counter() - start
    fps = count / elapsed
    model.release()
    return fps, count, elapsed


def _fmt_result(r: dict) -> str:
    if r.get("fps"):
        return f"{r['backend']:20s}  {r['fps']:7.1f} FPS"
    return f"{r['backend']:20s}  ERROR"


def main():
    parser = argparse.ArgumentParser(description="Find the fastest inference backend")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    plan = detect_test_plan()
    active = {k: v for k, v in plan.items() if v is not None}
    print(f"Testing {len(active)} backend type(s)...")
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

    for pt_path in pt_files:
        name = pt_path.stem
        print(f"  {name}")

        for _, (fmt, device, masks) in active.items():
            model_path = get_or_convert(pt_path, fmt)
            if not model_path.exists():
                continue
            cfg = make_base_config(pt_path, model_path, device)
            cfg = fill_missing_config(cfg)

            for core_mask, label in masks:
                try:
                    fps, count, elapsed = benchmark(cfg, core_mask, args.duration)
                    r = {"model": name, "backend": label, "format": fmt,
                         "device": str(device), "core_mask": core_mask,
                         "fps": round(fps, 1), "frames": count, "elapsed": round(elapsed, 3)}
                except Exception as e:
                    print(f"X {label}: {e}")
                    r = {"model": name, "backend": label, "format": fmt,
                         "device": str(device), "core_mask": core_mask,
                         "fps": None, "error": str(e)}
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

    output_path = args.output or str(_PROJECT_ROOT / "Outputs" / "benchmark_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    sys.exit(main())
