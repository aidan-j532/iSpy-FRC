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

from iSpy.config.AutoOpt import has_nvidia, has_rockchip_npu, has_tensorrt, has_tpu
from iSpy.vision.ModelInspector import fill_missing_config
from iSpy.vision.optimizer import _convert_model_subprocess

_REPO_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
if (
    (_REPO_CHECKOUT_ROOT / "iSpy" / "__init__.py").exists()
    and str(_REPO_CHECKOUT_ROOT) not in sys.path
):
    sys.path.insert(0, str(_REPO_CHECKOUT_ROOT))
_PROJECT_ROOT = Path.cwd()

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


# --------------------------------------------------------------------------
# Zero-config fallback: if nothing was found anywhere and the user didn't
# pass --model, grab a tiny stock YOLOv8n checkpoint so `ispy-bench` still
# has *something* to test in a brand-new environment (a fresh Colab runtime
# with no YoloModels/ or Config/ around yet). Same checkpoint/license terms
# as the "_default_detect.pt" stock model boot/default_models.py downloads
# (Ultralytics, AGPL-3.0) - it is NOT bundled with ispy-frc, only fetched
# on demand and only when nothing else is available.
# --------------------------------------------------------------------------
_DEFAULT_BENCH_MODEL_NAME = "_default_detect.pt"
_DEFAULT_BENCH_MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
)


def _download_default_bench_model() -> Path | None:
    target_dir = _PROJECT_ROOT / "YoloModels" / "pytorch"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / _DEFAULT_BENCH_MODEL_NAME
    if target.exists() and target.stat().st_size > 1024:
        return target

    print(
        f"No .pt models found under {_PROJECT_ROOT} - downloading a stock "
        f"YOLOv8n checkpoint (Ultralytics, AGPL-3.0) to {target} ..."
    )
    try:
        import requests

        tmp = target.with_suffix(".part")
        with requests.get(_DEFAULT_BENCH_MODEL_URL, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
        tmp.replace(target)
    except Exception as exc:
        print(f"Automatic download failed: {exc}")
        return None

    if target.stat().st_size < 1024:
        target.unlink(missing_ok=True)
        print("Downloaded file looked truncated - discarding it.")
        return None

    return target


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


# Formats that expect quantization (rknn/tflite are int8-only) or support it
# (engine/openvino get a real int8 build from the calibration dataset) get a
# quantized artifact whenever we can build one, because ispy-bench wants the
# fastest possible FPS for the device - never a float32 fallback.
_QUANTIZABLE_FORMATS = {"rknn", "tflite", "openvino", "engine"}


def _recommended_backend_plan() -> dict[str, tuple]:
    """Single-backend plan: the exact backend normal iSpy would pick for this
    machine (AutoOpt.recommend_format) with default settings - no prospecting,
    no benchmark-only knobs. Formats GenericYolo can't live-run fall back to
    onnx, which is what normal iSpy would use anyway."""
    from iSpy.config.AutoOpt import recommend_format, resolve_openvino_device

    fmt = recommend_format(ignore_dependencies=True)
    if fmt == "engine" and not has_tensorrt():
        # normal iSpy needs TensorRT installed to build *and* run an engine -
        # without it recommend_format is wrong, fall back to onnx like the
        # optimizer's own engine failure path does
        fmt = "onnx"
    if fmt in ("coreml", "hailo", "qnn"):
        fmt = "onnx"

    if fmt == "onnx":
        if has_nvidia():
            dev = _cuda_devices()[0][0]
            label = "Auto/ONNX-CUDA"
        else:
            dev = "cpu"
            label = "Auto/ONNX-CPU"
    elif fmt == "engine":
        dev = _cuda_devices()[0][0]
        label = "Auto/TensorRT"
    elif fmt == "openvino":
        dev = resolve_openvino_device()
        label = "Auto/OpenVINO"
    elif fmt == "rknn":
        dev = 0
        label = "Auto/RKNN"
    elif fmt == "tflite":
        dev = "cpu"
        label = "Auto/TFLite"
    else:  # tpu
        dev = "tpu"
        label = "Auto/TPU"
    return {fmt: (fmt, dev, [(None, label)])}


def _ensure_calibration_dataset(fmt) -> Path:
    from iSpy.dataset.dataset import calib_count_for_format, prepare_quantization_dataset
    from iSpy.vision.optimizer import default_quantization_dataset_dir

    ds = default_quantization_dataset_dir()
    prepare_quantization_dataset(
        str(ds),
        boot=False,
        keywords=[],
        count=calib_count_for_format(fmt),
    )
    return ds


def get_or_convert(pt_path, fmt, input_size=(640, 640)):
    if fmt in ("tpu", "pt"):
        return pt_path
    quantize = fmt in _QUANTIZABLE_FORMATS
    dataset_path = str(_ensure_calibration_dataset(fmt)) if quantize else None
    with _quiet():
        result = _convert_model_subprocess(
            str(pt_path),
            fmt,
            input_size,
            quantize=quantize,
            dataset_path=dataset_path,
        )
    if not result.exists() or result == pt_path:
        return None
    return result


def make_base_config(pt_path, model_path, device) -> dict:
    return {
        "file_path": str(model_path),
        "source_pt": str(pt_path),
        "task": "detect",
        "input_size": [640, 640],
        "min_conf": 0.5,
        "device": device,
    }


_BENCH_IMAGE_PATH: Path | None = None


def _bench_source_image() -> Path:
    """Return (and once create) a synthetic image the benchmark camera uses.

    is_image=True sources make the detection pipeline's preprocess worker
    feed the queue continuously, so real per-frame inference is timed.
    """
    global _BENCH_IMAGE_PATH
    if _BENCH_IMAGE_PATH is not None:
        return _BENCH_IMAGE_PATH

    import cv2
    import numpy as np

    out = _PROJECT_ROOT / "Outputs" / "bench_source.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (35, 35, 35)
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 80, (480, 640, 3), dtype=np.uint8)
        img = cv2.addWeighted(img, 0.4, noise, 0.6, 0)
        cv2.circle(img, (320, 240), 110, (60, 120, 220), -1)
        cv2.rectangle(img, (80, 70), (250, 260), (60, 200, 90), -1)
        cv2.putText(
            img,
            "bench",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])

    _BENCH_IMAGE_PATH = out
    return out


def benchmark(model_config, core_mask, duration=5.0):
    from iSpy.config.iSpyConfig import iSpyCameraConfig, iSpyConfig
    from iSpy.vision.pipelines.object_detection import ObjectDetectionPipeline

    config = iSpyConfig()
    cam_entry = {
        "name": "bench",
        # static image source (is_image=True) -> the pipeline's preprocess
        # worker keeps feeding frames, so real inference gets timed. An
        # invalid device source never connects, the preproc queue stays
        # empty, and every run() just burns the 0.1s queue timeout -> the
        # benchmark reads exactly 10.0 FPS for every backend.
        "source": str(_bench_source_image()),
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
    parser.add_argument(
        "--all-backends",
        action="store_true",
        help="prospect every reachable backend (pt/onnx per CUDA device, rknn "
        "NPU core masks, etc.) instead of the single recommend_format() pick",
    )
    args = parser.parse_args()

    try:
        import torch  # noqa: F401

        torch_note = "found"
    except ImportError:
        torch_note = (
            "NOT FOUND - install it (`pip install torch`) or every backend "
            "below will fail to load"
        )
    print(f"Working directory : {_PROJECT_ROOT}")
    print(f"torch              : {torch_note}")

    plan = detect_test_plan() if args.all_backends else _recommended_backend_plan()
    active = {k: v for k, v in plan.items() if v is not None}
    if not active:
        print("No supported backend detected on this machine.")
        return 1

    if args.model:
        p = Path(args.model)
        pt_files = [p.resolve()] if not p.is_absolute() else [p]
    else:
        pt_files = find_pt_files()

    if not pt_files:
        downloaded = _download_default_bench_model()
        pt_files = [downloaded] if downloaded else []

    if not pt_files:
        print(
            "No .pt models found, and the automatic download failed "
            "(probably no network access).\n"
            "Point ispy-bench at a model explicitly instead:\n"
            "  ispy-bench --model /path/to/your_model.pt"
        )
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
            fallback_fmt = None
            if model_path is None and not args.all_backends and fmt != "onnx":
                # recommended backend couldn't build (e.g. tensorrt absent) -
                # normal iSpy would fall back to onnx too, so still report a
                # number instead of silently skipping the model
                fallback_fmt, model_path = "onnx", get_or_convert(pt_path, "onnx")
            if model_path is None:
                continue
            if fallback_fmt:
                device = _cuda_devices()[0][0] if has_nvidia() else "cpu"
                fmt = fallback_fmt
                masks = [
                    (mask, "Auto/ONNX-CUDA (fallback)" if has_nvidia() else "Auto/ONNX-CPU (fallback)")
                    for mask, _ in masks
                ]

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
        Path(args.output) if args.output else _PROJECT_ROOT / "Outputs" / "benchmark_results.json"
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