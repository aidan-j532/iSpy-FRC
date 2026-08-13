import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_quantized_model(
    source_pt,
    target_format="auto",
    input_size=(640, 640),
    quantize=True,
    force=False,
    dataset_path=None,
):
    """provision a converted/quantized artifact for a YOLO .pt through iSpy's
    boot conversion framework. shared by the YOLO pipelines; reuses
    AutoOpt.recommend_format() for hardware detection and convert_model() for
    the export, so the output drops straight into GenericYolo. on failure logs
    and falls back to (source_pt, False) so the pipeline keeps running"""
    source_pt = str(source_pt)
    if not quantize or not Path(source_pt).exists():
        return source_pt, False

    from iSpy.config.AutoOpt import recommend_format
    from iSpy.vision import optimizer

    if target_format in (None, "", "auto"):
        target_format = recommend_format()

    if target_format == "tpu":
        # TPU backends consume the .pt directly via torch_xla at runtime.
        return source_pt, True

    if target_format not in {"onnx", "rknn", "tflite", "openvino", "engine", "coreml"}:
        logger.warning(
            "Unsupported quantized target format %r for %s - falling back to .pt",
            target_format, Path(source_pt).name,
        )
        return source_pt, False

    if hasattr(input_size, "__iter__"):
        input_size = [int(x) for x in input_size]
    else:
        input_size = [int(input_size), int(input_size)]

    try:
        artifact = optimizer.convert_model(
            source_pt,
            target_format,
            input_size,
            quantize=True,
            force=force,
            dataset_path=dataset_path,
        )
    except Exception as exc:
        logger.warning(
            "Quantized conversion of %s -> %s failed (%s); falling back to .pt",
            Path(source_pt).name, target_format, exc,
        )
        return source_pt, False

    if not artifact or str(artifact) == source_pt:
        logger.warning(
            "Quantized conversion of %s -> %s produced no artifact; falling back to .pt",
            Path(source_pt).name, target_format,
        )
        return source_pt, False

    logger.info("Quantized artifact ready: %s", artifact)
    return str(artifact), True


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class _CalibrationDataReader:
    """feeds images from a quantization dataset to onnxruntime's static int8
    calibrator, preprocessed the same way the depth pipeline preprocesses
    frames (ImageNet-normalized NCHW float32)"""

    def __init__(self, image_paths, input_size, mean=None, std=None):
        self._images = list(image_paths)
        self._size = int(input_size)
        self._mean = mean if mean is not None else _IMAGENET_MEAN
        self._std = std if std is not None else _IMAGENET_STD
        self._index = 0

    def _preprocess(self, path):
        import cv2

        img = cv2.imread(str(path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (
            cv2.resize(img, (self._size, self._size), interpolation=cv2.INTER_CUBIC)
            .astype(np.float32)
            / 255.0
        )
        return (
            (img.transpose(2, 0, 1) - self._mean) / self._std
        ).astype(np.float32)

    def get_next(self):
        batch = []
        while self._index < len(self._images) and len(batch) < 1:
            sample = self._preprocess(self._images[self._index])
            self._index += 1
            if sample is not None:
                batch.append(sample)
        if not batch:
            return None
        return {"pixel_values": np.concatenate(batch, axis=0)[None]}

    def rewind(self):
        self._index = 0


def _calibration_image_paths(dataset_path) -> list:
    """all usable calibration images under a QuantizeDataset folder, honoring
    dataset.txt when present so renamed/removed files dont break calibration"""
    ds = Path(dataset_path)
    if not ds.exists():
        return []
    paths = []
    dataset_txt = ds / "dataset.txt"
    if dataset_txt.exists():
        for line in dataset_txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line) if Path(line).is_absolute() else ds / line
            if p.exists():
                paths.append(p)
    if not paths:
        try:
            from iSpy.dataset.dataset import _find_images

            paths = _find_images(ds)
        except Exception:
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                paths.extend(ds.rglob(ext))
    return paths


def _quantize_static_onnx(fp32_path: Path, int8_path: Path, dataset_path, input_size):
    """static int8 quantization using the provided calibration dataset; the
    best-quality quantization the onnxruntime stack offers"""
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    images = _calibration_image_paths(dataset_path)
    if not images:
        raise FileNotFoundError(
            f"No calibration images found in quantization dataset: {dataset_path}"
        )
    logger.info(
        "Static int8 quantization of %s using %d calibration images from %s",
        fp32_path.name, len(images), dataset_path,
    )
    reader = _CalibrationDataReader(images, input_size)
    quantize_static(
        str(fp32_path),
        str(int8_path),
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
    )


def _artifact_loads(path: Path) -> bool:
    """True when the artifact exists and onnxruntime can actually load it;
    guards against copied/corrupt/partial artifacts being picked up silently"""
    if not path.exists():
        return False
    try:
        import onnxruntime as ort

        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return True
    except Exception as exc:
        logger.warning("Cached ONNX artifact %s failed to load (%s).", path.name, exc)
        return False


def ensure_onnx_model(
    build_module,
    artifact_stem,
    input_size=(518, 518),
    quantize=True,
    force=False,
    dataset_path=None,
):
    """export any torch model to a cached ONNX artifact, optionally int8-quantized.
    generic-model counterpart to ensure_quantized_model (which only understands
    ultralytics .pt files); build_module returns an eval torch.nn.Module taking a
    (N,3,H,W) tensor. artifact cached under YoloModels/onnx/ and reused; on failure
    logs and returns (None, False) so the caller can fall back to its unoptimized path.

    An existing artifact is used as-is whenever it loads - both the size-tagged
    names this function writes and the legacy `{stem}[_int8].onnx` names, so a
    model copied into YoloModels/onnx by hand works without any download or
    rebuild (useful offline). If quantization is requested but no int8 artifact
    exists (or the int8 build fails), an existing fp32 artifact is reused as a
    fallback rather than failing the camera.

    quantization honors the dataset_path the same way the YOLO path does: a
    calibration folder enables static int8 quantization (best quality); without
    one, dynamic int8 quantization is used instead"""
    import torch
    import torch.nn as nn

    if hasattr(input_size, "__iter__"):
        height, width = int(input_size[0]), int(input_size[1])
    else:
        height = width = int(input_size)

    out_dir = _PROJECT_ROOT / "YoloModels" / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    # input size baked into the artifact name so changing it forces a rebuild
    # instead of silently reusing a stale export
    size_tag = f"{height}x{width}"
    fp32_path = out_dir / f"{artifact_stem}_{size_tag}.onnx"
    int8_path = out_dir / f"{artifact_stem}_{size_tag}_int8.onnx"
    # legacy/hand-copied artifact names (no size tag)
    legacy_fp32 = out_dir / f"{artifact_stem}.onnx"
    legacy_int8 = out_dir / f"{artifact_stem}_int8.onnx"

    if not force:
        if quantize:
            for candidate in (int8_path, legacy_int8):
                if _artifact_loads(candidate):
                    logger.info("Reusing cached int8 ONNX artifact: %s", candidate)
                    return str(candidate), True
        else:
            for candidate in (fp32_path, legacy_fp32):
                if _artifact_loads(candidate):
                    logger.info("Reusing cached fp32 ONNX artifact: %s", candidate)
                    return str(candidate), True

    try:
        model = build_module()
        if not isinstance(model, nn.Module):
            raise TypeError("build_module must return a torch.nn.Module")
        model.eval()
        dummy = torch.zeros(1, 3, height, width)
        torch.onnx.export(
            model,
            dummy,
            str(fp32_path),
            input_names=["pixel_values"],
            output_names=["predicted_depth"],
            opset_version=17,
            dynamo=False,
        )
    except Exception as exc:
        logger.warning("ONNX export of %s failed (%s).", artifact_stem, exc)
        # offline fallback: keep the camera running on an existing fp32 copy
        if quantize:
            for candidate in (fp32_path, legacy_fp32):
                if _artifact_loads(candidate):
                    logger.warning(
                        "int8 build unavailable - reusing existing fp32 artifact %s",
                        candidate,
                    )
                    return str(candidate), True
        return None, False

    artifact = fp32_path
    if quantize:
        try:
            if dataset_path:
                try:
                    _quantize_static_onnx(fp32_path, int8_path, dataset_path, height)
                except Exception as exc:
                    logger.warning(
                        "static int8 quantization of %s failed (%s); "
                        "falling back to dynamic quantization.",
                        artifact_stem, exc,
                    )
                    if int8_path.exists():
                        int8_path.unlink()
            if not int8_path.exists():
                from onnxruntime.quantization import QuantType, quantize_dynamic

                quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
            if int8_path.exists():
                artifact = int8_path
        except Exception as exc:
            logger.warning(
                "int8 quantization of %s failed (%s); keeping fp32 ONNX.", artifact_stem, exc
            )
            if int8_path.exists():
                int8_path.unlink()

    if not _artifact_loads(artifact):
        return None, False

    logger.info("Optimized ONNX artifact ready: %s", artifact)
    return str(artifact), True
