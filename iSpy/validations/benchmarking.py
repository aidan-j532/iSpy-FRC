import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import cv2

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

from iSpy.config.AutoOpt import has_rockchip_npu, has_nvidia, has_tensorrt, has_tpu
from iSpy.vision.ModelInspector import fill_missing_config
from iSpy.boot.boot import convert_model

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("iSpy.vision.ModelInspector").setLevel(logging.INFO)
logger = logging.getLogger("ispy-test")


def find_pt_files():
    candidates = set()

    dirs = [
        _PROJECT_ROOT / "YoloModels" / "pytorch",
        _PROJECT_ROOT / "iSpy" / "assets",
    ]
    for d in dirs:
        if d.exists():
            for f in d.glob("*.pt"):
                candidates.add(f)

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
                        candidates.add(p.resolve())
        except Exception:
            pass

    try:
        for f in (Path.home() / "YoloModels" / "pytorch").glob("*.pt"):
            candidates.add(f.resolve())
    except Exception:
        pass

    return sorted(candidates, key=lambda p: p.name)


def make_placeholder_frame(h=480, w=640):
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def detect_test_plan():
    plans = []

    if has_rockchip_npu():
        import glob
        npu_count = len(glob.glob("/dev/rknpu*")) or 3
        masks = []
        if npu_count >= 1:
            masks.append((1, "NPU-core0"))
        if npu_count >= 2:
            masks.append((2, "NPU-core1"))
        if npu_count >= 3:
            masks.append((4, "NPU-core2"))
            masks.append((3, "NPU-cores0+1"))
            masks.append((7, "NPU-all3"))
        for mask, label in masks:
            plans.append(("rknn", 0, mask, label))

    if has_tpu():
        plans.append(("tpu", "tpu", None, "TPU"))

    if has_nvidia():
        if has_tensorrt():
            plans.append(("engine", 0, None, "TensorRT"))
        plans.append(("pt", 0, None, "CUDA"))
        plans.append(("onnx", 0, None, "ONNX-CUDA"))

    plans.append(("pt", "cpu", None, "CPU"))
    plans.append(("onnx", "cpu", None, "ONNX-CPU"))

    return plans


def get_or_convert(pt_path, fmt, input_size=(640, 640)):
    if fmt == "tpu" or fmt == "pt":
        return pt_path
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


def main():
    parser = argparse.ArgumentParser(description="iSpy multi-backend benchmark")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    test_plan = detect_test_plan()
    print(f"Detected {len(test_plan)} backend(s):")
    for _, _, _, label in test_plan:
        print(f"  - {label}")
    print()

    pt_files = []
    if args.model:
        p = Path(args.model)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        pt_files = [p]
    else:
        pt_files = find_pt_files()

    if not pt_files:
        print("No .pt files found.")
        print(f"Project root: {_PROJECT_ROOT}")
        return 1

    print(f"Found {len(pt_files)} model(s):")
    for p in pt_files:
        print(f"  - {p}")
    print()

    results = []

    for pt_path in pt_files:
        name = pt_path.stem
        print(f"{'='*60}")
        print(f"  Model: {name}")
        print(f"{'='*60}")

        for fmt, device, core_mask, label in test_plan:
            model_path = get_or_convert(pt_path, fmt)
            if not model_path.exists():
                print(f"  {label:20s}  SKIP (conversion failed)")
                continue

            cfg = make_base_config(pt_path, model_path, device)
            cfg = fill_missing_config(cfg)

            try:
                fps, count, elapsed = benchmark(cfg, core_mask, args.duration)
                print(f"  {label:20s}  {fps:7.1f} FPS  ({count} frames)")
                results.append({
                    "model": name,
                    "backend": label,
                    "format": fmt,
                    "device": str(device),
                    "core_mask": core_mask,
                    "fps": round(fps, 1),
                    "frames": count,
                    "elapsed": round(elapsed, 3),
                })
            except Exception as e:
                print(f"  {label:20s}  ERROR: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "model": name,
                    "backend": label,
                    "format": fmt,
                    "device": str(device),
                    "core_mask": core_mask,
                    "fps": None,
                    "error": str(e),
                })

        print()

    print(f"{'='*60}")
    print("  SUMMARY  (sorted best \u2192 worst per model)")
    print(f"{'='*60}")
    rows = {}
    for r in results:
        rows.setdefault(r["model"], []).append(r)
    for model, res in rows.items():
        print(f"\n  {model}:")
        for r in sorted(res, key=lambda x: x["fps"] or 0, reverse=True):
            fps = f"{r['fps']:7.1f}" if r["fps"] else "  ERROR"
            print(f"    {r['backend']:20s}  {fps} FPS")

    output_path = args.output or str(_PROJECT_ROOT / "Outputs" / "benchmark_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")
    print(f"\n  Tip: run with --model path/to/model.pt to test a single file")


if __name__ == "__main__":
    sys.exit(main())
