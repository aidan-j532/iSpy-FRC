import logging
from pathlib import Path

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


def ensure_onnx_model(
    build_module,
    artifact_stem,
    input_size=(518, 518),
    quantize=True,
    force=False,
):
    """export any torch model to a cached ONNX artifact, optionally int8-quantized.
    generic-model counterpart to ensure_quantized_model (which only understands
    ultralytics .pt files); build_module returns an eval torch.nn.Module taking a
    (N,3,H,W) tensor. artifact cached under YoloModels/onnx/ and reused; on failure
    logs and returns (None, False) so the caller can fall back to its unoptimized path"""
    import torch
    import torch.nn as nn

    if hasattr(input_size, "__iter__"):
        height, width = int(input_size[0]), int(input_size[1])
    else:
        height = width = int(input_size)

    out_dir = _PROJECT_ROOT / "YoloModels" / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = out_dir / f"{artifact_stem}.onnx"
    int8_path = out_dir / f"{artifact_stem}_int8.onnx"

    if not force:
        if quantize and int8_path.exists():
            return str(int8_path), True
        if fp32_path.exists():
            return str(fp32_path), True

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
        return None, False

    artifact = fp32_path
    if quantize:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
            if int8_path.exists():
                artifact = int8_path
        except Exception as exc:
            logger.warning(
                "int8 quantization of %s failed (%s); keeping fp32 ONNX.", artifact_stem, exc
            )

    try:
        import onnxruntime as ort

        ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
    except Exception as exc:
        logger.warning("Cached %s artifact failed to load (%s).", artifact.name, exc)
        return None, False

    logger.info("Optimized ONNX artifact ready: %s", artifact)
    return str(artifact), True
