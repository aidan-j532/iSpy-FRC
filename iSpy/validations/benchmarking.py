import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import cv2

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from iSpy.config.AutoOpt import recommend_format
from iSpy.vision.ModelInspector import fill_missing_config, inspect_model
from iSpy.boot.boot import convert_model

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("iSpy.vision.ModelInspector").setLevel(logging.INFO)
logger = logging.getLogger("ispy-test")


def find_pt_files():
    dirs = [
        _PROJECT_ROOT / "YoloModels" / "pytorch",
        _PROJECT_ROOT / "iSpy" / "assets",
    ]
    files = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.pt")))
    return files


def make_placeholder_frame(h=480, w=640):
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def benchmark_model(model_config, duration=5.0):
    from iSpy.vision.genericYolo import GenericYolo

    model = GenericYolo(model_config)

    target_h, target_w = model.input_size[1], model.input_size[0]
    frame = make_placeholder_frame(target_h, target_w)

    if model.model_type == "rknn":
        preproc_buf = np.empty((1, target_h, target_w, 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model._letterbox_into(img_rgb, preproc_buf[0], model.input_size)
        infer = lambda: model.predict_preprocessed(preproc_buf, frame.shape)
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
    parser = argparse.ArgumentParser(description="iSpy model benchmark")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None, help="Benchmark a single .pt file")
    args = parser.parse_args()

    target_format = recommend_format(ignore_dependencies=True)
    print(f"Target format: {target_format}")
    print()

    if args.model:
        pt_files = [Path(args.model)]
    else:
        pt_files = find_pt_files()

    if not pt_files:
        print("No .pt files found.")
        return 1

    results = []

    for pt_path in pt_files:
        name = pt_path.stem
        print(f"{'='*60}")
        print(f"  Model: {name}")
        print(f"{'='*60}")

        if target_format == "tpu":
            model_path = pt_path
            device = "tpu"
        else:
            model_path = Path(convert_model(str(pt_path), target_format, [640, 640]))
            device = 0

        if not model_path.exists():
            print(f"  SKIP — {model_path} not found")
            continue

        print(f"  Using: {model_path}")

        base_cfg = {
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

        model_config = fill_missing_config(base_cfg)
        actual_task = model_config.get("task", "detect")
        print(f"  Task: {actual_task}")

        try:
            fps, count, elapsed = benchmark_model(model_config, duration=args.duration)
            print(f"  FPS: {fps:.1f}  ({count} frames in {elapsed:.2f}s)")
            results.append({
                "model": name,
                "task": actual_task,
                "pt_path": str(pt_path),
                "optimized_path": str(model_path),
                "format": target_format,
                "fps": round(fps, 1),
                "frames": count,
                "elapsed": round(elapsed, 3),
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "model": name,
                "task": actual_task,
                "pt_path": str(pt_path),
                "optimized_path": str(model_path),
                "format": target_format,
                "fps": None,
                "error": str(e),
            })

        print()

    print(f"{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        fps_str = f"{r['fps']:.1f}" if r['fps'] is not None else "ERROR"
        print(f"  {r['model']:40s} {fps_str:>8s} FPS")

    output_path = args.output or str(_PROJECT_ROOT / "Outputs" / "benchmark_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
