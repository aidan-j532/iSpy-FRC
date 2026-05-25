import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def inspect_model(model_path: str, task: str = "detect") -> dict:
    ext = Path(model_path).suffix.lower()
    suffix = Path(model_path).name

    if ext == ".onnx":
        return _inspect_onnx(model_path, task)
    elif ext == ".rknn":
        return _inspect_rknn(model_path, task)
    elif ext == ".tflite":
        return _inspect_tflite(model_path, task)
    elif ext == ".pt" or "openvino_model" in suffix or ext == ".mlpackage" or ext == ".engine":
        return _inspect_ultralytics(model_path, task)
    else:
        raise ValueError(f"Unsupported model extension: {ext}")


def print_detected_config(result: dict) -> None:
    certain = result.pop("_certain_fields", [])
    detected = result.pop("_detected_fields", [])
    manual = result.pop("_manual_fields", [])
    warnings = result.pop("_warnings", [])

    print("\n" + "=" * 65)
    print("  ModelInspector - Detected Config")
    print("=" * 65)

    _print_dict(result, indent=0, certain_fields=certain, detected_fields=detected)

    if warnings:
        print("\n Assumptions made:")
        for w in warnings:
            print(f"   - {w}")

    if manual:
        print("\n Fields to verify / set manually:")
        for f in manual:
            print(f"   - {f}")

    print("\n  Copy the block above into your config under 'vision_model'.")
    print("=" * 65 + "\n")


def _inspect_onnx(model_path: str, task: str) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required for ONNX inspection. "
            "pip install onnxruntime --break-system-packages"
        )

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp_meta = sess.get_inputs()[0]
    out_metas = sess.get_outputs()

    certain, detected, manual, warnings = [], [], [], []

    # Check for metadata.yaml saved by VisionCore during conversion
    meta_path = Path(model_path).parent / "metadata.yaml"
    meta_task = task
    meta_nc = None
    meta_kpt_shape = None
    meta_output_format = None
    meta_output_layout = None
    meta_box_format = None
    meta_quantization = None

    if meta_path.exists():
        try:
            from ruamel.yaml import YAML
            meta = YAML(typ="safe").load(meta_path)
            if isinstance(meta, dict):
                meta_task = meta.get("task", task)
                meta_nc = meta.get("nc")
                meta_kpt_shape = meta.get("kpt_shape")
                meta_output_format = meta.get("output_format")
                meta_output_layout = meta.get("output_layout")
                meta_box_format = meta.get("box_format")
                meta_quantization = meta.get("quantization")
        except Exception as e:
            warnings.append(f"Failed to read metadata.yaml: {e}")

    inp_shape = inp_meta.shape
    inp_type = inp_meta.type

    layout, h, w, c = _parse_input_shape(inp_shape)
    certain += ["input.layout", "input_size"]

    ORT_DTYPE_MAP = {
        "tensor(float)":   "float32",
        "tensor(float32)": "float32",
        "tensor(double)":  "float32",
        "tensor(uint8)":   "uint8",
        "tensor(int8)":    "uint8",
    }
    dtype = ORT_DTYPE_MAP.get(inp_type, "float32")
    certain.append("input.dtype")

    normalize = dtype == "float32"
    certain.append("input.normalize")
    if normalize:
        warnings.append("input.dtype is float32 -> normalize=true, scale=255.0 assumed.")

    out_meta = out_metas[0]
    out_shape = out_meta.shape
    out_type = out_meta.type

    quant = _ort_type_to_quantization(out_type)
    certain.append("output.quantization")

    out_layout, n_anchors, feat_width = _parse_output_shape(out_shape)
    certain.append("output.layout")

    fmt, num_classes, score_mode, box_format, box_fmt_source = _detect_output_format(
        feat_width, task, warnings
    )

    # hardware_nms detection from col count is reliable; raw format inference is not
    if fmt == "hardware_nms":
        certain.append("output.format")
    else:
        detected.append("output.format")

    detected += ["output.box_format", "output.score_mode"]

    if num_classes is not None:
        detected.append("num_classes")
    else:
        manual.append("num_classes  (could not be inferred - check your model's output width)")
        num_classes = 1

    scores_are_logits = False
    warnings.append(
        "output.scores_are_logits defaults to false. "
        "If detections are always near 0 conf, try setting it to true."
    )
    manual.append(
        "output.scores_are_logits  (verify: false if ultralytics export, "
        "true if custom/raw export without sigmoid)"
    )

    # Override tensor-inferred values with ground-truth metadata when available
    if meta_nc is not None:
        num_classes = meta_nc
        certain.append("num_classes")
    if meta_task != task:
        task = meta_task
        certain.append("task")
    if meta_kpt_shape:
        certain += [
            "output.num_keypoints",
            "output.keypoint_dims",
            "output.keypoint_scores_are_logits",
        ]
    if meta_output_format:
        fmt = meta_output_format
        certain.append("output.format")
    if meta_output_layout:
        out_layout = meta_output_layout
        certain.append("output.layout")
    if meta_box_format:
        box_format = meta_box_format
        certain.append("output.box_format")
    if meta_quantization:
        quant = meta_quantization
        certain.append("output.quantization")

    cfg = {
        "file_path": model_path,
        "task": task,
        "num_classes": num_classes,
        "input_size": [w, h],
        "min_conf": 0.5,
        "output": {
            "format": fmt,
            "layout": out_layout,
            "box_format": box_format,
            "score_mode": score_mode,
            "scores_are_logits": scores_are_logits,
            "apply_software_nms": fmt == "raw",
            "nms_iou": 0.45,
            "quantization": quant,
            **({"quant_scale": 255.0} if quant != "none" else {}),
            **({"num_keypoints": meta_kpt_shape[0],
                "keypoint_dims": meta_kpt_shape[1],
                "keypoint_scores_are_logits": False}
               if meta_kpt_shape else {}),
        },
        "input": {
            "layout": layout,
            "dtype": dtype,
            "letterbox": True,
            "pad_value": 114,
            "normalize": normalize,
            **({"scale": 255.0} if normalize else {}),
        },
        "_certain_fields": certain,
        "_detected_fields": detected,
        "_manual_fields": manual + [
            "min_conf              (default 0.5 - adjust for your use-case)",
            "output.nms_iou        (default 0.45 - standard YOLO value)",
        ],
        "_warnings": warnings,
    }

    if quant != "none" and "quant_scale" not in cfg["output"]:
        manual.append(
            "output.quant_scale  (required for int8/uint8 - check your quantization params)"
        )

    return cfg


def _inspect_rknn(model_path: str, task: str) -> dict:
    warnings = []
    manual = []

    # These are RKNN hardware guarantees, not guesses
    certain_fields = [
        "input.layout",    # RKNN runtime always expects NHWC
        "input.dtype",     # RKNN runtime always expects uint8
        "input.normalize", # always false for uint8 input
    ]

    # These are reasonable defaults but the inspector can't actually
    # read RKNN output shapes without running inference
    detected_fields = [
        "output.format",      # hardware_nms is correct for all Ultralytics exports
        "output.layout",
        "output.box_format",
        "output.score_mode",
        "output.quantization",
    ]

    result: dict[str, Any] = {
        "file_path": model_path,
        "task": task,
        "num_classes": 1,
        "input_size": [640, 640],
        "min_conf": 0.5,
        "output": {
            "format": "hardware_nms",  # Ultralytics RKNN exports are always end2end NMS
            "layout": "anchors_first",
            "box_format": "xyxy",
            "score_mode": "objectness",
            "scores_are_logits": False,
            "apply_software_nms": False,
            "nms_iou": 0.45,
            "quantization": "int8",
            "quant_scale": 255.0,
        },
        "input": {
            "layout": "nhwc",
            "dtype": "uint8",
            "letterbox": True,
            "pad_value": 114,
            "normalize": False,
        },
    }

    meta_path = Path(model_path).parent / "metadata.yaml"
    if meta_path.exists():
        try:
            from ruamel.yaml import YAML
            meta = YAML(typ="safe").load(meta_path)
            if isinstance(meta, dict):
                rknn_task = meta.get("task", "")
                if rknn_task == "pose":
                    result["task"] = "pose"
                    certain_fields.append("task")
                    kpt_shape = meta.get("kpt_shape")
                    if kpt_shape and len(kpt_shape) == 2:
                        n_kpts, kpt_dims = kpt_shape
                        out = result["output"]
                        out["num_keypoints"] = int(n_kpts)
                        out["keypoint_dims"] = int(kpt_dims)
                        out["keypoint_scores_are_logits"] = False
                        out["score_mode"] = "objectness"
                        certain_fields += [
                            "output.num_keypoints",
                            "output.keypoint_dims",
                            "output.keypoint_scores_are_logits",
                            "output.score_mode",
                        ]
                        warnings.append(
                            "Pose model detected from metadata.yaml. "
                            "Verify keypoint ordering in your config."
                        )

                names = meta.get("names")
                if isinstance(names, dict):
                    num_names = len(names)
                    result["num_classes"] = num_names
                    certain_fields.append("num_classes")
                    score_mode = "objectness" if num_names == 1 else "multi_class"
                    result["output"]["score_mode"] = score_mode
                    certain_fields.append("output.score_mode")

                meta_output_format = meta.get("output_format")
                if meta_output_format:
                    result["output"]["format"] = meta_output_format
                    certain_fields.append("output.format")

                meta_output_layout = meta.get("output_layout")
                if meta_output_layout:
                    result["output"]["layout"] = meta_output_layout
                    certain_fields.append("output.layout")

                meta_box_format = meta.get("box_format")
                if meta_box_format:
                    result["output"]["box_format"] = meta_box_format
                    certain_fields.append("output.box_format")

                meta_quant = meta.get("quantization")
                if meta_quant:
                    result["output"]["quantization"] = meta_quant
                    certain_fields.append("output.quantization")

                meta_scale = meta.get("quant_scale")
                if meta_scale is not None:
                    result["output"]["quant_scale"] = float(meta_scale)
                    certain_fields.append("output.quant_scale")

        except Exception as e:
            warnings.append(f"Failed to parse metadata.yaml: {e}")

    if not result.get("output", {}).get("num_keypoints"):
        manual += [
            "input_size           (verify against your training config)",
            "output.format        (hardware_nms if end2end export, raw otherwise)",
            "output.layout        (features_first for Ultralytics exports)",
            "output.score_mode    (objectness for 1-class, multi_class otherwise)",
            "output.quantization  (int8 default - check your rknn build config)",
        ]
        if not warnings:
            warnings.append(
                "RKNN models carry limited metadata. "
                "Input layout/size are inferred from RKNN conventions (NHWC uint8). "
                "Output fields are defaults only - your config values will be kept."
            )

    result["_certain_fields"] = certain_fields
    result["_detected_fields"] = detected_fields
    result["_manual_fields"] = manual
    result["_warnings"] = warnings
    return result


def _inspect_tflite(model_path: str, task: str) -> dict:
    certain, detected, manual, warnings = [], [], [], []

    try:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter

        interp = Interpreter(model_path=model_path)
        interp.allocate_tensors()
        inp_det = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]

        inp_shape = inp_det["shape"]  # always [1, H, W, C] for TFLite
        out_shape = out_det["shape"]

        _, h, w, _ = inp_shape
        certain += ["input.layout", "input_size"]  # TFLite is always NHWC

        import numpy as np
        dtype = "float32" if inp_det["dtype"] == np.float32 else "uint8"
        certain.append("input.dtype")

        normalize = dtype == "float32"
        certain.append("input.normalize")

        quant_params = out_det.get("quantization_parameters", {})
        has_quant = bool(quant_params.get("scales", []))
        quant = (
            "int8"  if out_det["dtype"] == np.int8  else
            "uint8" if out_det["dtype"] == np.uint8 else
            "none"
        )
        certain.append("output.quantization")
        certain.append("output.layout")  # read from actual tensor shape

        out_layout, n_anchors, feat_width = _parse_output_shape(list(out_shape))

        fmt, num_classes, score_mode, box_format, _ = _detect_output_format(
            feat_width, task, warnings
        )

        if fmt == "hardware_nms":
            certain.append("output.format")
        else:
            detected.append("output.format")

        detected += ["output.box_format", "output.score_mode"]

        if num_classes is not None:
            detected.append("num_classes")
        else:
            num_classes = 1
            manual.append("num_classes")

        quant_scale = float(quant_params["scales"][0]) if has_quant else 255.0

    except Exception as e:
        warnings.append(f"Could not fully inspect TFLite model ({e}). Using safe defaults.")
        w, h, dtype, normalize = 640, 640, "uint8", False
        out_layout, fmt, num_classes, score_mode, box_format = (
            "anchors_first", "raw", 1, "objectness", "cxcywh",
        )
        quant, quant_scale = "none", 255.0
        manual += ["input_size", "num_classes", "output.format", "output.score_mode"]

    return {
        "file_path": model_path,
        "task": task,
        "num_classes": num_classes,
        "input_size": [w, h],
        "min_conf": 0.5,
        "output": {
            "format": fmt,
            "layout": out_layout,
            "box_format": box_format,
            "score_mode": score_mode,
            "scores_are_logits": False,
            "apply_software_nms": fmt == "raw",
            "nms_iou": 0.45,
            "quantization": quant,
            **({"quant_scale": quant_scale} if quant != "none" else {}),
        },
        "input": {
            "layout": "nhwc",
            "dtype": dtype,
            "letterbox": True,
            "pad_value": 114,
            "normalize": normalize,
            **({"scale": 255.0} if normalize else {}),
        },
        "_certain_fields": certain,
        "_detected_fields": detected,
        "_manual_fields": manual + ["min_conf", "output.nms_iou", "output.scores_are_logits"],
        "_warnings": warnings,
    }


def _inspect_ultralytics(model_path: str, task: str) -> dict:
    try:
        from ultralytics import YOLO
        model = YOLO(model_path, task=task, verbose=False)
        model_task = getattr(model, "task", task) or task
        try:
            num_classes = int(model.model.model[-1].nc)
        except Exception:
            num_classes = 1

        if model_task == "pose":
            try:
                kpt_shape = model.model.model[-1].kpt_shape
                num_keypoints = int(kpt_shape[0])
                keypoint_dims = int(kpt_shape[1])
            except Exception:
                num_keypoints, keypoint_dims = 17, 3
        else:
            num_keypoints, keypoint_dims = None, None

        input_size = [640, 640]
        try:
            if hasattr(model, "model") and hasattr(model.model, "args"):
                imgsz = model.model.args.get("imgsz", 640)
                input_size = [imgsz, imgsz] if isinstance(imgsz, int) else list(imgsz[:2])
        except Exception:
            pass

    except Exception:
        model_task = task
        num_classes = 1
        num_keypoints, keypoint_dims = None, None
        input_size = [640, 640]

    # Ultralytics handles its own inference - all of these are reliable
    certain = [
        "task",
        "num_classes",
        "input_size",
        "input.layout",
        "input.dtype",
        "input.normalize",
        "output.quantization",
        "output.layout",
        "output.format",
        "output.box_format",
        "output.score_mode",
    ]

    base = {
        "file_path": model_path,
        "task": model_task,
        "num_classes": num_classes,
        "input_size": input_size,
        "min_conf": 0.5,
        "output": {
            "format": "hardware_nms",
            "layout": "anchors_first",
            "box_format": "xyxy",
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

    if model_task == "pose":
        base["output"]["num_keypoints"] = num_keypoints
        base["output"]["keypoint_dims"] = keypoint_dims
        base["output"]["keypoint_scores_are_logits"] = False
        certain += ["output.num_keypoints", "output.keypoint_dims"]

    base["_certain_fields"] = certain
    base["_detected_fields"] = []
    base["_manual_fields"] = []
    base["_warnings"] = [
        ".pt / OpenVINO / CoreML models are handled by Ultralytics directly - "
        "input/output config fields are informational only and not used at runtime."
    ]
    return base

def _parse_input_shape(shape) -> tuple[str, int, int, int]:
    def _to_int(v, fallback=640):
        try:
            return int(v)
        except (TypeError, ValueError):
            return fallback

    if len(shape) != 4:
        return "nhwc", 640, 640, 3

    d = [_to_int(x) for x in shape]

    if 1 <= d[1] <= 4 and d[2] > 4 and d[3] > 4:
        return "nchw", d[2], d[3], d[1]

    if d[1] > 4 and d[2] > 4 and 1 <= d[3] <= 4:
        return "nhwc", d[1], d[2], d[3]
    return "nchw", d[2], d[3], d[1]


def _parse_output_shape(shape) -> tuple[str, int, int]:
    def _to_int(v, fallback=1):
        try:
            return int(v)
        except (TypeError, ValueError):
            return fallback

    # Strip batch dim if present
    if len(shape) == 3:
        shape = shape[1:]  # (N, F) or (F, N) after batch removal
    if len(shape) != 2:
        return "anchors_first", 8400, 5  # YOLO default fallback

    d0, d1 = _to_int(shape[0]), _to_int(shape[1])

    if d0 >= d1:
        return "anchors_first", d0, d1
    else:
        return "features_first", d1, d0


def _detect_output_format(
    feat_width: int, task: str, warnings: list
) -> tuple[str, Any, str, str, str]:
    if feat_width == 6:
        return "hardware_nms", None, "objectness", "xyxy", "hardware_nms cols==6"

    score_cols = feat_width - 4
    if score_cols < 1:
        warnings.append(
            f"Output feature width {feat_width} is unusually small. "
            "Defaulting to raw/objectness - verify your model."
        )
        return "raw", 1, "objectness", "cxcywh", "fallback"

    # 1 score col -> objectness (single class)
    if score_cols == 1:
        num_classes = 1
        score_mode = "objectness"
    else:
        num_classes = score_cols
        score_mode = "multi_class"

    # YOLO always uses cxcywh internally (even when xyxy in Ultralytics output)
    # Raw ONNX exports preserve the internal cxcywh encoding.
    box_format = "cxcywh"

    return "raw", num_classes, score_mode, box_format, "inferred from feat_width"


def _ort_type_to_quantization(ort_type: str) -> str:
    if "int8" in ort_type:
        return "int8"
    if "uint8" in ort_type:
        return "uint8"
    return "none"


def _get_dotpath(d: dict, dotpath: str):
    val = d
    for key in dotpath.split("."):
        if not isinstance(val, dict) or key not in val:
            return None
        val = val[key]
    return val


def _set_dotpath(d: dict, dotpath: str, value) -> None:
    keys = dotpath.split(".")
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def fill_missing_config(model_config: dict) -> dict:
    model_path = model_config.get("file_path", "")
    if not model_path or not os.path.exists(model_path):
        return model_config

    task = model_config.get("task", "detect")

    try:
        detected = inspect_model(model_path, task)
    except Exception as e:
        logger.warning("ModelInspector could not inspect %s: %s", model_path, e)
        return model_config

    certain_fields  = set(detected.pop("_certain_fields", []))
    detected_fields = set(detected.pop("_detected_fields", []))
    detected.pop("_manual_fields", None)
    detected.pop("_warnings", None)

    merged = _deep_merge_missing(detected, model_config)

    for field_path in certain_fields | detected_fields:
        detected_val = _get_dotpath(detected, field_path)
        user_val     = _get_dotpath(model_config, field_path)

        if detected_val is None:
            continue

        if user_val is None:
            # Field is missing entirely — always fill it in
            _set_dotpath(merged, field_path, detected_val)
            logger.info("Auto-filled  %-35s = %r", field_path, detected_val)

        elif user_val != detected_val:
            if field_path in certain_fields:
                # Inspector is certain — overwrite and warn loudly
                _set_dotpath(merged, field_path, detected_val)
                logger.warning(
                    "Corrected    %-35s  your value=%r  model says=%r",
                    field_path, user_val, detected_val,
                )
            else:
                # Inspector is guessing — trust the user, just log at debug
                logger.debug(
                    "Keeping user %-35s = %r  (inspector guessed %r)",
                    field_path, user_val, detected_val,
                )

    return merged


def _deep_merge_missing(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge_missing(result[k], v)
        else:
            result[k] = v
    return result


def _print_dict(d: dict, indent: int, certain_fields: list, detected_fields: list, prefix: str = ""):
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        pad = "  " * indent
        if full_key in certain_fields:
            tag = " [CERTAIN]"
        elif full_key in detected_fields:
            tag = " [GUESSED]"
        else:
            tag = ""
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            _print_dict(v, indent + 1, certain_fields, detected_fields, full_key)
        else:
            print(f"{pad}{k}: {v!r}{tag}")