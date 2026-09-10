"""Benchmark available inference backends against your .pt models."""

import argparse
import contextlib
import hashlib
import io
import json
import logging
import sys
import time
import warnings
from pathlib import Path

from iSpy.config.AutoOpt import has_rockchip_npu, has_nvidia, has_tensorrt, has_tpu
from iSpy.vision.ModelInspector import fill_missing_config
from iSpy.vision.optimizer import _convert_model_subprocess

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (_PROJECT_ROOT / "iSpy").is_dir():
    _PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


@contextlib.contextmanager
def _quiet():
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        yield


def _file_fingerprint(path: Path) -> tuple[int, str]:
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(4096)
    return (size, hashlib.sha256(head).hexdigest())


def find_pt_files() -> list[Path]:
    raw: list[Path] = []

    for d in (
        _PROJECT_ROOT / "YoloModels" / "pytorch",
        _PROJECT_ROOT / "iSpy" / "assets",
    ):
        if d.exists():
            raw.extend(f.resolve() for f in d.glob("*.pt"))

    config_path = _PROJECT_ROOT / "Config" / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            from iSpy.config.iSpyConfig import get_pipeline_settings

            for cam in (cfg.get("camera_configs") or {}).values():
                if not isinstance(cam, dict):
                    continue
                settings = get_pipeline_settings(cam) or {}
                if isinstance(settings.get("vision_model"), dict):
                    _add_model_paths_from_config(settings["vision_model"], raw)
            if not raw:
                _add_model_paths_from_config(cfg.get("vision_model") or {}, raw)
        except Exception:
            pass

    if Path.home().joinpath("YoloModels", "pytorch").exists():
        try:
            raw.extend(
                f.resolve()
                for f in Path.home().joinpath("YoloModels", "pytorch").glob("*.pt")
            )
        except Exception:
            pass

    seen: set[tuple[int, str]] = set()
    unique: list[Path] = []
    for p in raw:
        fp = _file_fingerprint(p)
        if fp in seen:
            logger.debug("Skipping duplicate model: %s", p)
            continue
        seen.add(fp)
        unique.append(p)
    return sorted(unique, key=lambda p: p.name)


def _add_model_paths_from_config(vm: dict, out: list[Path]):
    for key in ("file_path", "source_pt"):
        p = vm.get(key)
        if not p:
            continue
        p = Path(p)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        if p.suffix == ".pt" and p.exists():
            out.append(p.resolve())


def _npu_masks() -> list[tuple[int, str]]:
    import glob

    n = len(glob.glob("/dev/rknpu*")) or 3
    masks: list[tuple[int, str]] = [(1, "NPU-core0"), (2, "NPU-core1")]
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

        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                timeout=10,
            )
            .decode()
            .strip()
        )
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
            plan[f"engine_{label.lower()}"] = ("engine", dev, [(None, "TensorRT")])
    if not has_rockchip_npu():
        plan["pt_cpu"] = ("pt", "cpu", [(None, "CPU")])
        plan["onnx_cpu"] = ("onnx", "cpu", [(None, "ONNX-CPU")])
    return plan


def get_or_convert(pt_path, fmt, input_size=(640, 640)):
    if fmt in ("tpu", "pt"):
        return pt_path
    with _quiet():
        result = _convert_model_subprocess(str(pt_path), fmt, input_size)
    if not result.exists() or result == pt_path:
        return None
    return result


def make_base_config(pt_path, model_path, device) -> dict:
    return {
        "file_path": str(model_path),
        "source_pt": str(pt_path),
        "task": "detect",
        "num_classes": 1,
        "input_size": [640, 640],
        "min_conf": 0.5,
        "device": device,
    }


def benchmark(model_config, core_mask, duration=5.0):
    from iSpy.config.iSpyConfig import iSpyCameraConfig, iSpyConfig
    from iSpy.vision.pipelines.object_detection import ObjectDetectionPipeline

    config = iSpyConfig()
    cam_entry = {
        "name": "bench",
        "source": 99,  # invalid source -> ObjectDetectionPipeline feeds placeholder frames
        "fps_cap": 1000,
        "yaw": 0,
        "pitch": 0,
        "height": 1.0,
        "x": 0,
        "y": 0,
        "grayscale": False,
        "subsystem": "bench",
        "calibration": {
            "distance": 1.0,
            "game_piece_size": 1.0,
            "size": 100,
            "fov": 90,
        },
        "pipeline": {
            "name": "object_detection",
            "settings": {"vision_model": model_config},
        },
    }
    config.set("camera_configs", {"bench": cam_entry})
    cam_cfg = iSpyCameraConfig(cam_entry)

    with _quiet():
        camera = ObjectDetectionPipeline(cam_cfg, config, core_mask=core_mask)

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
        return (
            f"{r['backend']:20s} {r['fps']:6.1f} FPS {r.get('inference_ms', 0):6.1f} ms"
        )
    return f"{r['backend']:20s} ERROR: {r.get('error', 'unknown')}"


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference backends")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds per run")
    parser.add_argument("--output", default=None, help="output json path")
    parser.add_argument("--model", default=None, help="benchmark a specific .pt file")
    args = parser.parse_args()

    active = {k: v for k, v in detect_test_plan().items() if v is not None}
    if not active:
        print("No supported backend detected on this machine.")
        return 1

    if args.model:
        p = Path(args.model)
        pt_files = [p.resolve()] if not p.is_absolute() else [p]
    else:
        pt_files = find_pt_files()

    if not pt_files:
        print("No .pt models found.")
        return 1

    print(f"Testing {len(active)} backend(s): {', '.join(active)}")
    print()

    best: dict[str, dict] = {}
    all_results: list[dict] = []

    for pt_path in pt_files:
        name = pt_path.stem
        print(f"- {name}")

        for fmt, device, masks in active.values():
            model_path = get_or_convert(pt_path, fmt)
            if model_path is None:
                continue
            cfg = fill_missing_config(make_base_config(pt_path, model_path, device))

            for core_mask, label in masks:
                try:
                    fps, inference_ms, count, elapsed = benchmark(
                        cfg, core_mask, args.duration
                    )
                    r = {
                        "model": name,
                        "backend": label,
                        "format": fmt,
                        "device": str(device),
                        "core_mask": core_mask,
                        "fps": round(fps, 1),
                        "inference_ms": round(inference_ms, 1),
                        "frames": count,
                        "elapsed": round(elapsed, 3),
                    }
                    print(f"    {_fmt_result(r)}")
                except Exception as e:
                    print(f"    {label:20s} ERROR: {e}")
                    r = {
                        "model": name,
                        "backend": label,
                        "format": fmt,
                        "device": str(device),
                        "core_mask": core_mask,
                        "fps": None,
                        "inference_ms": None,
                        "frames": None,
                        "elapsed": None,
                        "error": str(e),
                    }
                all_results.append(r)
                if best.get(name) is None or (r["fps"] or 0) > (
                    best[name].get("fps") or 0
                ):
                    best[name] = r
        print()

    print("Best backend per model:")
    for name, r in best.items():
        print(f"  {name:40s} {_fmt_result(r)}")

    output_path = (
        Path(args.output)
        if args.output
        else Path.cwd() / "Outputs" / "benchmark_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best": best,
        "all": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
