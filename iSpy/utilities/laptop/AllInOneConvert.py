import shutil
from ultralytics import YOLO
import logging
import os
from pathlib import Path
from iSpy.vision.metadata import (
    read_metadata,
    metadata_from_pt,
    derive_format_metadata,
    metadata_path_for,
    write_metadata,
)


YOLO_MODELS_DIR = Path("YoloModels")


def _desired_output_path(pt_path: Path, target_format: str) -> Path:
    stem = pt_path.stem
    out_dir = YOLO_MODELS_DIR / target_format
    out_dir.mkdir(parents=True, exist_ok=True)
    if target_format == "rknn":
        return out_dir / f"{stem}.rknn"
    if target_format == "onnx":
        return out_dir / f"{stem}.onnx"
    if target_format == "openvino":
        return out_dir / f"{stem}_openvino_model"
    if target_format == "coreml":
        return out_dir / f"{stem}.mlpackage"
    if target_format == "engine":
        return out_dir / f"{stem}.engine"
    if target_format == "tflite":
        return out_dir / f"{stem}.tflite"
    return out_dir / f"{stem}.{target_format}"


def _move_to_format_dir(source: Path, pt_path: Path, target_format: str) -> Path:
    dest = _desired_output_path(pt_path, target_format)
    if source == dest:
        return source
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    shutil.move(str(source), str(dest))
    return dest

logger = logging.getLogger(__name__)

EXPORT_CONFIGS = {
    "onnx": {
        "detect":   {"simplify": True, "opset": 17, "dynamic": False, "half": False},
        "classify": {"simplify": True, "opset": 17, "dynamic": False, "half": False},
        "segment":  {"simplify": True, "opset": 17, "dynamic": False, "half": False},
    },
    "openvino": {
        "detect":   {"half": True,  "int8": False},
        "classify": {"half": True,  "int8": False},
        "segment":  {"half": True,  "int8": False},
    },
    "coreml": {
        # half=True uses Float16 weights, nms=True embeds NMS (overlapping box thingie) in the model graph (detect only)
        "detect":   {"half": True, "nms": True},
        "classify": {"half": True, "nms": False},
        "segment":  {"half": True, "nms": False},
    },
    "tflite": {
        # int8=True requires a calibration dataset, half gives float16 without one
        "detect":   {"half": True, "int8": False},
        "classify": {"half": True, "int8": False},
        "segment":  {"half": True, "int8": False},
    },
}

ULTRALYTICS_FORMATS = set(EXPORT_CONFIGS.keys())  # onnx, openvino, coreml, tflite
ALL_FORMATS = ULTRALYTICS_FORMATS | {"rknn"}

RKNN_CONFIG = {
    "mean_values": [[0, 0, 0]],
    "std_values": [[255, 255, 255]],
    "target_platform": "rk3588",
    "disable_rules": ["fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa"],
    # "quantized_algorithm": "kl_divergence",
    # "quantized_dtype": "w8a8",
    # "quantized_hybrid_level": 3,
}

VALID_TASKS = ("detect", "classify", "segment")


def _export_ultralytics(file: str, format: str, task: str) -> str:
    cfg = EXPORT_CONFIGS[format][task]
    logger.info(f"Exporting '{file}' -> {format.upper()} (task={task}) with config: {cfg}")
    model = YOLO(file, task=task)
    export_path = model.export(format=format, **cfg)
    logger.info(f"Export complete -> {export_path}")
    return export_path

def _export_rknn(onnx_file: str, task: str, dataset_txt: str, output_path: str = "model.rknn") -> str:
    try:
        from rknn.api import RKNN
    except ImportError:
        raise ImportError("RKNN Toolkit not found. Please install it to convert to RKNN format.")

    if not os.path.isfile(onnx_file):
        raise FileNotFoundError(f"ONNX model not found: {onnx_file}")
    if not os.path.isfile(dataset_txt):
        raise FileNotFoundError(f"Calibration dataset not found: {dataset_txt}")

    rknn = RKNN()

    logger.info(f"Configuring RKNN for RK3588 (task={task})...")
    rknn.config(**RKNN_CONFIG)

    logger.info(f"Loading ONNX model: {onnx_file}")
    ret = rknn.load_onnx(model=onnx_file)
    if ret != 0:
        raise RuntimeError(f"RKNN load_onnx failed with code {ret}")

    logger.info("Building RKNN model with quantization...")
    ret = rknn.build(do_quantization=True, dataset=dataset_txt)
    if ret != 0:
        raise RuntimeError(f"RKNN build failed with code {ret}")

    logger.info(f"Exporting RKNN model -> {output_path}")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        raise RuntimeError(f"RKNN export failed with code {ret}")

    rknn.release()
    logger.info("RKNN conversion complete.")
    return output_path


def _export_rknn_metadata(pt_file: str, rknn_output_path: str) -> None:
    try:
        from ultralytics import YOLO
        from ruamel.yaml import YAML

        model = YOLO(pt_file, verbose=False)

        meta = {}

        model_task = getattr(model, "task", "detect") or "detect"
        meta["task"] = model_task

        try:
            nc = int(model.model.model[-1].nc)
            meta["nc"] = nc
        except Exception:
            pass

        try:
            names = getattr(model, "names", None)
            if names is not None and isinstance(names, dict):
                meta["names"] = names
        except Exception:
            pass

        if model_task == "pose":
            try:
                kpt_shape = model.model.model[-1].kpt_shape
                if kpt_shape is not None and len(kpt_shape) == 2:
                    meta["kpt_shape"] = [int(kpt_shape[0]), int(kpt_shape[1])]
            except Exception:
                pass

        # Output format - known because WE did the conversion
        # ONNX output is (feat, anchors), RKNN compiler preserves this layout
        meta["output_format"] = "raw"
        meta["output_layout"] = "features_first"
        meta["box_format"] = "cxcywh"
        meta["quantization"] = "int8"
        meta["quant_scale"] = 255.0

        meta_path = Path(rknn_output_path).parent / f"{Path(rknn_output_path).stem}_metadata.yaml"
        yaml = YAML()
        yaml.default_flow_style = False
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

        logger.info(f"Exported RKNN metadata -> {meta_path}")
    except Exception as e:
        logger.warning(f"Failed to export RKNN metadata: {e}")


def convert_model(
    file: str,
    format: str,
    task: str = "detect",
    rknn_dataset_txt: str = "dataset.txt",
    rknn_output_path: str = "model.rknn",
) -> str:
    if not file.endswith(".pt"):
        raise ValueError("Input file must be a .pt file")
    if not os.path.isfile(file):
        raise FileNotFoundError(f"Model file not found: {file}")
    if format not in ALL_FORMATS:
        raise ValueError(f"Unsupported format '{format}'. Choose from: {', '.join(sorted(ALL_FORMATS))}")
    if task not in VALID_TASKS:
        raise ValueError(f"Unsupported task '{task}'. Choose from: {', '.join(VALID_TASKS)}")

    pt_path = Path(file)

    if format == "rknn":
        logger.info("RKNN target -- exporting to ONNX first...")
        raw_onnx = _export_ultralytics(file, "onnx", task)
        logger.info(f"Intermediate ONNX saved -> {raw_onnx}")
        # Route intermediate ONNX to onnx folder with sidecar
        try:
            onnx_path = _move_to_format_dir(Path(raw_onnx), pt_path, "onnx")
            pt_meta = read_metadata(pt_path) or metadata_from_pt(pt_path)
            fmt_meta = derive_format_metadata(pt_meta, "onnx")
            fmt_meta["input_size"] = list(pt_meta.get("input_size", [640, 640]))
            write_metadata(metadata_path_for(onnx_path), fmt_meta)
            logger.info("Routed intermediate ONNX to %s", onnx_path)
        except Exception as e:
            logger.warning("Could not route intermediate ONNX: %s", e)
            onnx_path = Path(raw_onnx)

        rknn_path = _export_rknn(
            onnx_file=str(onnx_path),
            task=task,
            dataset_txt=rknn_dataset_txt,
            output_path=str(_desired_output_path(pt_path, "rknn")),
        )
        try:
            pt_meta = read_metadata(pt_path) or metadata_from_pt(pt_path)
            fmt_meta = derive_format_metadata(pt_meta, "rknn")
            fmt_meta["input_size"] = list(pt_meta.get("input_size", [640, 640]))
            write_metadata(metadata_path_for(Path(rknn_path)), fmt_meta)
            logger.info("Wrote metadata for converted %s", Path(rknn_path).name)
        except Exception as e:
            logger.warning("Could not write metadata for %s: %s", rknn_path, e)
        return rknn_path

    raw_out = _export_ultralytics(file, format, task)
    out_path = _move_to_format_dir(Path(raw_out), pt_path, format)
    try:
        pt_meta = read_metadata(pt_path) or metadata_from_pt(pt_path)
        fmt_meta = derive_format_metadata(pt_meta, format)
        fmt_meta["input_size"] = list(pt_meta.get("input_size", [640, 640]))
        write_metadata(metadata_path_for(out_path), fmt_meta)
        logger.info("Wrote metadata for converted %s", out_path.name)
    except Exception as e:
        logger.warning("Could not write metadata for converted output %s: %s", out_path, e)
    return str(out_path)