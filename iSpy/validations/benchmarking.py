import argparse
import contextlib
from functools import partial
import hashlib
import io
import json
import logging
import multiprocessing
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from iSpy.config.AutoOpt import has_nvidia, has_rockchip_npu, has_tensorrt, has_tpu
from iSpy.validations.bench_to_matplotlib import collect_system_info, render_report
from iSpy.vision.ModelInspector import fill_missing_config
from iSpy.vision.optimizer import _convert_model_subprocess

_REPO_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
if (_REPO_CHECKOUT_ROOT / "iSpy" / "__init__.py").exists() and str(
    _REPO_CHECKOUT_ROOT
) not in sys.path:
    sys.path.insert(0, str(_REPO_CHECKOUT_ROOT))
_PROJECT_ROOT = Path.cwd()


def _repo_root() -> Path | None:
    # cwd is unreliable here - colab runs it from /content with the repo nested
    # below. walk up from this file instead, same as tests/compare_models.py.
    for base in (Path(__file__).resolve().parent, _PROJECT_ROOT):
        for d in (base, *base.parents):
            if (d / "iSpy").is_dir() and (d / "pyproject.toml").is_file():
                return d
    return None


_REPO_ROOT = _repo_root()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
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


def _install_plan_dependencies(active: dict) -> None:
    from iSpy.vision.optimizer import BACKEND_DEPENDENCIES, _is_installed, _pip_install

    fmts = {fmt for fmt, _dev, _masks in active.values()}
    for fmt in sorted(fmts):
        deps = BACKEND_DEPENDENCIES.get(fmt)
        if not deps:
            continue
        for entry in deps:
            mod, target = entry[0], entry[1]
            extra_args = list(entry[2]) if len(entry) > 2 else None
            if _is_installed(mod):
                continue
            print(f"Installing {target} for backend '{fmt}'...")
            if not _pip_install(target, extra_args=extra_args):
                print(f"  failed to install {target} - '{fmt}' will likely fail below")


def find_pt_files() -> list[Path]:
    raw: list[Path] = []

    roots = [_PROJECT_ROOT]
    if _REPO_ROOT and _REPO_ROOT != _PROJECT_ROOT:
        roots.append(_REPO_ROOT)

    for root in roots:
        for d in (
            root / "YoloModels" / "pytorch",
            root / "iSpy" / "assets",
        ):
            if d.exists():
                raw.extend(f.resolve() for f in d.glob("*.pt"))

    for root in roots:
        config_path = root / "Config" / "config.json"
        if not config_path.exists():
            continue
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
        if raw:
            break

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


# nothing to test with and no --model, so grab stock yolov8n - same
# checkpoint/license as the _default_detect.pt boot downloads (ultralytics,
# AGPL-3.0). not bundled, only fetched when there is literally nothing else
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

    searched = [_PROJECT_ROOT]
    if _REPO_ROOT and _REPO_ROOT != _PROJECT_ROOT:
        searched.append(_REPO_ROOT)
    print(
        f"No .pt models found under {', '.join(str(p) for p in searched)} - "
        f"downloading a stock YOLOv8n checkpoint (Ultralytics, AGPL-3.0) "
        f"to {target} ..."
    )
    try:
        import requests

        tmp = target.with_suffix(".part")
        with requests.get(_DEFAULT_BENCH_MODEL_URL, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                fh.writelines(resp.iter_content(chunk_size=1 << 16))
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


# rknn/tflite are int8-only, engine/openvino get a real int8 build from the
# calibration dataset. ispy-bench wants max fps, so never a float32 fallback
_QUANTIZABLE_FORMATS = {"rknn", "tflite", "openvino", "engine"}


def _recommended_backend_plan() -> dict[str, tuple]:
    # the exact backend normal iSpy would pick for this machine (AutoOpt.recommend_format)
    # at default settings. formats GenericYolo cant live-run fall back to onnx
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
    from iSpy.dataset.dataset import (
        calib_count_for_format,
        prepare_quantization_dataset,
    )
    from iSpy.vision.optimizer import default_quantization_dataset_dir

    ds = default_quantization_dataset_dir()
    prepare_quantization_dataset(
        str(ds),
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


def make_base_config(pt_path, model_path, device, batch_size=1) -> dict:
    return {
        "file_path": str(model_path),
        "source_pt": str(pt_path),
        "task": "detect",
        "input_size": [640, 640],
        "min_conf": 0.5,
        "device": device,
        "batch_size": batch_size,
    }


_BENCH_IMAGE_PATH: Path | None = None


def _bench_source_image() -> Path:
    # is_image=True sources make the preprocess worker feed the queue continuously,
    # so what we time is real per-frame inference and not a one-shot drain
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


def _benchmark_frames() -> list:
    import cv2

    suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
    roots = [_PROJECT_ROOT / "QuantizeDataset"]
    paths = sorted(
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    frames = [cv2.imread(str(path)) for path in paths[:32]]
    frames = [frame for frame in frames if frame is not None]
    if frames:
        return frames
    fallback = cv2.imread(str(_bench_source_image()))
    if fallback is None:
        raise RuntimeError("could not read benchmark input image")
    return [fallback]


def _percentile(values: list[float], percent: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percent / 100)
    return ordered[index]


def _sync_camera(camera) -> None:
    if getattr(camera.model, "model_type", None) == "tpu":
        camera.model._tpu_sync()


def _drain_pipeline(camera) -> None:
    if not camera._use_pipeline:
        _sync_camera(camera)
        return
    camera._benchmark_pause = True
    deadline = time.perf_counter() + 5.0
    last_count = camera.inference_count
    stable_since = time.perf_counter()
    while time.perf_counter() < deadline:
        camera.run()
        count = camera.inference_count
        queues_empty = all(
            target.empty()
            for target in (
                camera._preproc_q,
                camera._tpu_device_q,
                camera._tpu_result_q,
            )
        )
        if count != last_count:
            last_count = count
            stable_since = time.perf_counter()
        elif queues_empty and time.perf_counter() - stable_since >= 0.05:
            break
    _sync_camera(camera)


def _require_benchmark_model(camera) -> None:
    if camera.model is None:
        raise RuntimeError("model failed to load")


def _require_benchmark_count(count: int) -> None:
    if count < 1:
        raise RuntimeError("no inference completed during benchmark")


def _benchmark_model_forward(model, frames: list, batch_size: int) -> dict | None:
    """Time backend model execution without iSpy preprocessing or postprocessing."""
    import numpy as np

    model_type = getattr(model, "model_type", None)
    if model_type not in ("yolo", "onnx", "engine", "tpu"):
        return None

    if model_type == "tpu":
        import torch

        prepared = [
            model._preprocess_tpu_frame(frames[index % len(frames)])
            for index in range(batch_size)
        ]
        tensor = torch.from_numpy(np.stack(prepared)).to(model._tpu_device)
        tensor = tensor.permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)
        if model.tpu_dtype == "bf16":
            tensor = tensor.to(dtype=torch.bfloat16)
        network = model.model

        def infer():
            with torch.inference_mode(), model._tpu_autocast(torch):
                return network(tensor)

        synchronize = model._tpu_sync
    else:
        prepared = np.concatenate(
            [
                np.array(
                    model._preprocess_frame(frames[index % len(frames)]), copy=True
                )
                for index in range(batch_size)
            ],
            axis=0,
        )
        synchronize = lambda: None

        if model_type == "yolo":
            import torch

            device = model.device
            if isinstance(device, int):
                device = f"cuda:{device}"
            tensor = torch.from_numpy(prepared)
            if tensor.ndim == 4 and tensor.shape[-1] in (1, 3, 4):
                tensor = tensor.permute(0, 3, 1, 2)
            tensor = tensor.contiguous().to(device=device)
            if tensor.dtype == torch.uint8:
                tensor = tensor.to(dtype=torch.float32).div_(255.0)
            network = model.model.model

            def infer():
                with torch.inference_mode():
                    return network(tensor)

            if str(device).startswith("cuda"):
                synchronize = partial(torch.cuda.synchronize, device)
        elif model_type == "onnx":
            providers = model.model.get_providers()
            if "CUDAExecutionProvider" in providers:
                import torch

                device = model.device if isinstance(model.device, int) else 0
                tensor = torch.from_numpy(prepared).to(f"cuda:{device}")
                binding = model.model.io_binding()
                binding.bind_input(
                    model._onnx_inp_name,
                    device_type="cuda",
                    device_id=device,
                    element_type=prepared.dtype,
                    shape=prepared.shape,
                    buffer_ptr=tensor.data_ptr(),
                )
                for name in model._onnx_out_names:
                    binding.bind_output(name, device_type="cuda", device_id=device)
                infer = partial(model.model.run_with_iobinding, binding)
                synchronize = partial(torch.cuda.synchronize, device)
            else:
                infer = lambda: model.model.run(
                    model._onnx_out_names, {model._onnx_inp_name: prepared}
                )
        else:
            import tensorrt as trt
            import torch

            engine = model.model
            context = engine.create_execution_context()
            device = f"cuda:{model.device}"
            buffers = []
            bindings = []
            with torch.cuda.device(model.device):
                for index in range(engine.num_io_tensors):
                    name = engine.get_tensor_name(index)
                    dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
                    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        context.set_input_shape(name, prepared.shape)
                        buffer = torch.from_numpy(
                            np.ascontiguousarray(prepared, dtype=dtype)
                        ).to(device=device)
                    else:
                        shape = tuple(context.get_tensor_shape(name))
                        if any(size < 0 for size in shape):
                            raise RuntimeError(
                                f"TensorRT output {name!r} has unresolved shape "
                                f"{shape}"
                            )
                        torch_dtype = torch.from_numpy(np.empty((), dtype=dtype)).dtype
                        buffer = torch.empty(shape, dtype=torch_dtype, device=device)
                    buffers.append(buffer)
                    bindings.append(buffer.data_ptr())

            def infer():
                if not context.execute_v2(bindings):
                    raise RuntimeError("TensorRT execute_v2 returned false")

            synchronize = partial(torch.cuda.synchronize, device)

    for _ in range(3):
        infer()
    synchronize()

    durations_ms = []
    for _ in range(15):
        synchronize()
        started = time.perf_counter()
        infer()
        synchronize()
        durations_ms.append((time.perf_counter() - started) * 1000)

    batch_ms = sum(durations_ms) / len(durations_ms)
    frame_ms = batch_ms / batch_size
    return {
        "model_forward_ms": frame_ms,
        "model_forward_fps": 1000.0 / frame_ms,
    }


def _benchmark_stream(
    model_config: dict,
    core_mask,
    duration: float,
    batch_size: int,
    mode: str,
    frames: list,
) -> dict:
    from iSpy.config.iSpyConfig import iSpyCameraConfig, iSpyConfig
    from iSpy.vision.pipelines.object_detection import ObjectDetectionPipeline

    config = iSpyConfig()
    cam_entry = {
        "name": "bench",
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
        camera = ObjectDetectionPipeline(
            cam_cfg,
            config,
            core_mask=core_mask,
            use_tpu_pipeline=(mode == "pipelined"),
            benchmark_frames=frames,
        )

    try:
        _require_benchmark_model(camera)

        model = camera.model
        provider = getattr(model, "provider", None) or getattr(
            model, "model_type", str(getattr(model, "device", "unknown"))
        )
        frame_index = 0

        def next_batch():
            nonlocal frame_index
            batch = [frames[(frame_index + i) % len(frames)] for i in range(batch_size)]
            frame_index = (frame_index + batch_size) % len(frames)
            return batch

        if mode == "serial":
            for _ in range(5):
                batch = next_batch()
                outputs = model.predict(
                    batch, orig_shape=[frame.shape for frame in batch]
                )
                if not isinstance(outputs, list):
                    outputs = [outputs]
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        f"model returned {len(outputs)} warmup results for "
                        f"{len(batch)} frames"
                    )
            _sync_camera(camera)
        else:
            warmup_deadline = time.perf_counter() + 60
            warmup_count = camera.inference_count
            while time.perf_counter() < warmup_deadline:
                camera.run()
                completed = camera.inference_count - warmup_count
                if completed >= 5:
                    break
            if camera.inference_count == warmup_count:
                raise RuntimeError("no inference completed during pipeline warmup")
            _sync_camera(camera)

        start_count = camera.inference_count
        start_detections = camera.detections_seen
        with camera._metrics_lock:
            sample_start = camera._timing_samples_seen
        latencies = []
        start = time.perf_counter()
        if mode == "serial":
            while time.perf_counter() - start < duration:
                batch = next_batch()
                batch_start = time.perf_counter()
                outputs = model.predict(
                    batch, orig_shape=[frame.shape for frame in batch]
                )
                batch_ms = (time.perf_counter() - batch_start) * 1000
                if not isinstance(outputs, list):
                    outputs = [outputs]
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        f"model returned {len(outputs)} results for {len(batch)} frames"
                    )
                stages = getattr(model, "last_tpu_stage_ms", None) or {
                    "preprocess": 0.0,
                    "device": batch_ms / len(outputs),
                    "postprocess": 0.0,
                }
                frame_latency_ms = batch_ms / len(outputs)
                for result in outputs:
                    camera._record_inference(result, batch_start, stages)
                    latencies.append(frame_latency_ms)
        else:
            while time.perf_counter() - start < duration:
                camera.run()
            _drain_pipeline(camera)

        _sync_camera(camera)
        elapsed = time.perf_counter() - start
        count = camera.inference_count - start_count
        _require_benchmark_count(count)

        metrics = camera.benchmark_metrics(sample_start)
        samples = metrics.get("samples", [])
        if mode == "pipelined":
            latencies = [sample["latency"] for sample in samples]
        inference_ms = sum(latencies) / len(latencies) if latencies else 0.0
        if inference_ms < 0.5:
            raise RuntimeError(
                f"implausible inference time ({inference_ms:.3f} ms/frame)"
            )
        stage_ms = metrics.get("stage_ms") or {
            "preprocess": 0.0,
            "device": inference_ms,
            "postprocess": 0.0,
        }
        predict_fps = count / elapsed
        result = {
            "ok": True,
            "fps": predict_fps,
            "predict_fps": predict_fps,
            "predict_ms": inference_ms,
            "inference_ms": inference_ms,
            "frames": count,
            "elapsed": elapsed,
            "detections": camera.detections_seen - start_detections,
            "latencies": latencies,
            "stage_ms": stage_ms,
            "provider": provider,
        }
        try:
            model_forward = _benchmark_model_forward(model, frames, batch_size)
        except Exception as exc:
            logger.warning("Model-forward-only benchmark failed: %s", exc)
            result["model_forward_error"] = str(exc)
        else:
            if model_forward is not None:
                result.update(model_forward)
        return result
    finally:
        camera.destroy()


def benchmark(
    model_config,
    core_mask,
    duration=5.0,
    *,
    batch_size=1,
    mode="serial",
    streams=1,
    frames=None,
):
    model_config = dict(model_config)
    model_config["batch_size"] = batch_size
    frames = frames or _benchmark_frames()
    if streams < 1:
        raise ValueError("streams must be at least 1")

    if streams == 1:
        runs = [
            _benchmark_stream(
                model_config, core_mask, duration, batch_size, mode, frames
            )
        ]
    else:
        with ThreadPoolExecutor(max_workers=streams) as executor:
            futures = [
                executor.submit(
                    _benchmark_stream,
                    model_config,
                    core_mask,
                    duration,
                    batch_size,
                    mode,
                    frames,
                )
                for _ in range(streams)
            ]
            runs = [future.result() for future in futures]

    return _combine_benchmark_runs(runs)


def _combine_benchmark_runs(runs: list[dict]) -> dict:
    count = sum(run["frames"] for run in runs)
    elapsed = max(run["elapsed"] for run in runs)
    latencies = [latency for run in runs for latency in run.get("latencies", [])]
    stage_count = sum(run["frames"] for run in runs)
    stage_ms = {
        stage: sum(run["stage_ms"].get(stage, 0.0) * run["frames"] for run in runs)
        / max(stage_count, 1)
        for stage in ("preprocess", "device", "postprocess")
    }
    inference_ms = sum(latencies) / len(latencies) if latencies else 0.0
    if not count or inference_ms < 0.5:
        raise RuntimeError("benchmark produced no valid inference measurements")
    predict_fps = count / elapsed
    result = {
        "ok": True,
        "fps": predict_fps,
        "predict_fps": predict_fps,
        "predict_ms": inference_ms,
        "inference_ms": inference_ms,
        "frames": count,
        "elapsed": elapsed,
        "detections": sum(run["detections"] for run in runs),
        "latencies": latencies,
        "latency_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
        },
        "stage_ms": stage_ms,
        "bottleneck": max(stage_ms, key=stage_ms.get),
        "provider": ", ".join(dict.fromkeys(run["provider"] for run in runs)),
        "stream_fps": [
            fps for run in runs for fps in run.get("stream_fps", [run["fps"]])
        ],
    }
    forward_runs = [run for run in runs if run.get("model_forward_fps")]
    if forward_runs:
        result["model_forward_fps"] = sum(
            run["model_forward_fps"] for run in forward_runs
        )
        result["model_forward_ms"] = sum(
            run["model_forward_ms"] for run in forward_runs
        ) / len(forward_runs)
    forward_errors = [
        run["model_forward_error"] for run in runs if run.get("model_forward_error")
    ]
    if forward_errors:
        result["model_forward_error"] = "; ".join(dict.fromkeys(forward_errors))
    return result


def _local_tpu_core_count() -> int:
    try:
        import torch_xla.runtime as xr

        count = getattr(xr, "local_device_count", None)
        if callable(count):
            return int(count())
    except Exception:
        pass
    try:
        import torch_xla.core.xla_model as xm

        count = getattr(xm, "xla_device_count", None)
        return int(count()) if callable(count) else 1
    except Exception:
        return 0


def _tpu_core_worker(index, task, result_queue) -> None:
    import torch_xla.core.xla_model as xm

    xm.xla_device()
    result = benchmark(
        task["config"],
        task["core_mask"],
        task["duration"],
        batch_size=task["batch_size"],
        mode=task["mode"],
        streams=task["streams"],
    )
    result_queue.put(result)


def _benchmark_tpu_cores(task: dict) -> dict | None:
    try:
        import torch_xla.distributed.xla_multiprocessing as xmp
    except Exception as exc:
        print(f"TPU core mode skipped: torch_xla multiprocessing unavailable ({exc})")
        return None

    count = task.get("tpu_cores", 0)
    if count <= 1:
        print("TPU core mode skipped: only one local TPU core was detected")
        return None
    result_queue = multiprocessing.get_context("spawn").Queue()
    xmp.spawn(
        _tpu_core_worker,
        args=(task, result_queue),
        nprocs=count,
        start_method="spawn",
    )
    runs = [result_queue.get() for _ in range(count)]
    result = _combine_benchmark_runs(runs)
    result["tpu_cores"] = count
    return result


def _fmt_result(r: dict) -> str:
    if r.get("ok") and r.get("fps"):
        config = (
            f"b{r.get('batch_size', 1)} {r.get('mode', 'serial')} "
            f"x{r.get('streams', 1)}"
        )
        model_forward = ""
        if r.get("model_forward_fps"):
            model_forward = (
                f" | vision-only {r['model_forward_fps']:.1f} FPS "
                f"{r['model_forward_ms']:.2f} ms/frame"
            )
        elif r.get("model_forward_error"):
            model_forward = f" | vision-only ERROR: {r['model_forward_error']}"
        return (
            f"{r['backend']:20s} {config:20s} "
            f"predict {r.get('predict_fps', r['fps']):6.1f} FPS "
            f"{r.get('predict_ms', r.get('inference_ms', 0)):6.1f} ms/frame"
            f"{model_forward}"
        )
    return f"{r['backend']:20s} ERROR: {r.get('error', 'unknown')}"


def _run_benchmark_task(task: dict) -> dict:
    result = {
        "model": task["model"],
        "backend": task["backend"],
        "format": task["format"],
        "device": str(task["device"]),
        "core_mask": task["core_mask"],
        "batch_size": task.get("batch_size", 1),
        "mode": task.get("mode", "serial"),
        "streams": task.get("streams", 1),
        "latency_ms": {"p50": None, "p95": None, "p99": None},
        "stage_ms": {},
        "provider": "unavailable",
        "detections": 0,
        "ok": False,
    }
    try:
        measured = None
        if task.get("tpu_cores", 0) > 1 and task["format"] == "tpu":
            measured = _benchmark_tpu_cores(task)
        if measured is None:
            measured = benchmark(
                task["config"],
                task["core_mask"],
                task["duration"],
                batch_size=result["batch_size"],
                mode=result["mode"],
                streams=result["streams"],
            )
        if isinstance(measured, tuple):
            fps, inference_ms, count, elapsed = measured
            measured = {
                "ok": bool(count),
                "fps": fps,
                "predict_fps": fps,
                "predict_ms": inference_ms,
                "inference_ms": inference_ms,
                "frames": count,
                "elapsed": elapsed,
            }
        result.update(measured)
        result.pop("latencies", None)
        if not result.get("provider"):
            result["provider"] = str(task["device"])
        result["ok"] = bool(
            result.get("ok")
            and result.get("frames", 0) > 0
            and result.get("inference_ms", 0) >= 0.5
        )
        if not result["ok"]:
            result["fps"] = None
            result["error"] = result.get("error") or "invalid benchmark measurements"
    except Exception as e:
        result.update(
            fps=None,
            inference_ms=None,
            frames=None,
            elapsed=None,
            ok=False,
            error=str(e),
        )
    return result


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(
            dict.fromkeys(_positive_int(part.strip()) for part in value.split(","))
        )
    except argparse.ArgumentTypeError as exc:
        raise argparse.ArgumentTypeError(
            "use comma-separated positive integers"
        ) from exc
    if not sizes:
        raise argparse.ArgumentTypeError("provide at least one batch size")
    return sizes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark inference backends")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds per run")
    parser.add_argument("--output", default=None, help="output json path")
    parser.add_argument("--model", default=None, help="benchmark a specific .pt file")
    parser.add_argument(
        "--batch-sizes",
        type=_parse_batch_sizes,
        default=(1,),
        metavar="N[,N...]",
        help="comma-separated batch sizes to test (default: 1)",
    )
    parser.add_argument(
        "--pipelined",
        action="store_true",
        help="run serial and pipelined measurements side by side",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="run only the serial baseline (the default)",
    )
    parser.add_argument(
        "--streams",
        type=_positive_int,
        default=1,
        metavar="N",
        help="concurrent camera pipelines on the same device (default: 1)",
    )
    parser.add_argument(
        "--tpu-cores",
        action="store_true",
        help="try one torch_xla worker per local TPU core",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help=(
            "run up to N inference benchmarks concurrently (default: 1; "
            "results may compete for accelerator resources)"
        ),
    )
    parser.add_argument(
        "--all-backends",
        action="store_true",
        help="prospect every reachable backend (pt/onnx per CUDA device, rknn "
        "NPU core masks, etc.) instead of the single recommend_format() pick",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Best-effort pip-install the runtime dependencies needed by the "
        "active backend plan (onnxruntime-gpu, tensorrt, torch_xla, ...) "
        "before benchmarking.",
    )
    parser.add_argument("--no-plot", action="store_true", help="skip the PNG report")
    parser.add_argument(
        "--plot-output",
        default=None,
        help="PNG path (default: next to the outputed JSON)",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.parallel < 1:
        parser.error("--parallel must be at least 1 bro like what you doing?")
    if args.pipelined and args.serial:
        parser.error("--pipelined and --serial cannot be used together")

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
    tpu_core_count = 0
    if args.tpu_cores:
        tpu_core_count = _local_tpu_core_count()
        if tpu_core_count == 0:
            print("TPU core mode skipped: torch_xla runtime is unavailable")
        elif tpu_core_count == 1:
            print("TPU core mode skipped: only one local TPU core was detected")
        else:
            print(f"TPU core mode: launching {tpu_core_count} local workers")
    if args.install_deps:
        _install_plan_dependencies(active)
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

    tasks = []
    modes = ["serial", "pipelined"] if args.pipelined else ["serial"]

    for pt_path in pt_files:
        name = pt_path.stem

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
                    (
                        mask,
                        "Auto/ONNX-CUDA (fallback)"
                        if has_nvidia()
                        else "Auto/ONNX-CPU (fallback)",
                    )
                    for mask, _ in masks
                ]

            for core_mask, label in masks:
                for batch_size in args.batch_sizes:
                    cfg = fill_missing_config(
                        make_base_config(pt_path, model_path, device, batch_size)
                    )
                    for mode in modes:
                        tasks.append(
                            {
                                "model": name,
                                "backend": label,
                                "format": fmt,
                                "device": device,
                                "core_mask": core_mask,
                                "config": cfg,
                                "duration": args.duration,
                                "batch_size": batch_size,
                                "mode": mode,
                                "streams": args.streams,
                                "tpu_cores": tpu_core_count if fmt == "tpu" else 0,
                            }
                        )

    if args.parallel > 1 and len(tasks) > 1 and not tpu_core_count:
        _bench_source_image()
        with ProcessPoolExecutor(
            max_workers=min(args.parallel, len(tasks)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            all_results = list(executor.map(_run_benchmark_task, tasks))
    else:
        all_results = [_run_benchmark_task(task) for task in tasks]

    best: dict[str, dict] = {}
    current_model = None
    for r in all_results:
        if r["model"] != current_model:
            if current_model is not None:
                print()
            current_model = r["model"]
            print(f"- {current_model}")
        print(f"    {_fmt_result(r)}")
        if not r.get("ok"):
            continue
        if best.get(r["model"]) is None or (r["fps"] or 0) > (
            best[r["model"]].get("fps") or 0
        ):
            best[r["model"]] = r
    if current_model is not None:
        print()

    print("Best backend per model:")
    for name, r in best.items():
        print(f"  {name:40s} {_fmt_result(r)}")

    output_path = (
        Path(args.output)
        if args.output
        else _PROJECT_ROOT / "Outputs" / "benchmark_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": collect_system_info(),
        "best": best,
        "all": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if not args.no_plot:
        png = render_report(
            payload,
            Path(args.plot_output)
            if args.plot_output
            else output_path.with_suffix(".png"),
            title=f"iSpy benchmark - {', '.join(best) or 'no results'}",
        )
        if png:
            print(f"Chart saved to {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
