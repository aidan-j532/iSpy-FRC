from pathlib import Path
from typing import Any, Dict
METADATA_SCHEMA = {
    "task": str,
    "nc": int,
    "names": dict,
    "input_size": list,
    "kpt_shape": list,

    "output_format": str,
    "output_layout": str,
    "box_format": str,
    "score_mode": str,
    "scores_are_logits": bool,
    "apply_software_nms": bool,
    "nms_iou": float,
    "quantization": str,
    "quant_scale": float,
    "box_coord_scale": float,

    "input_layout": str,
    "input_dtype": str,
    "input_letterbox": bool,
    "input_pad_value": int,
    "input_normalize": bool,
    "input_scale": float,
    "calibration_keywords": list,
    "box_coord_scale": float,
    "kpt_coord_scale": float,
}


def metadata_path_for(model_path: Path) -> Path:
    return model_path.parent / f"{model_path.stem}_metadata.yaml"


def write_metadata(path: Path, meta: Dict[str, Any]) -> None:
    try:
        from ruamel.yaml import YAML
    except Exception:
        raise
    yaml = YAML()
    yaml.default_flow_style = False
    with open(path, "w") as f:
        yaml.dump(meta, f)


def read_metadata(model_path: Path) -> Dict[str, Any] | None:
    mp = metadata_path_for(model_path)
    if not mp.exists():
        return None
    try:
        from ruamel.yaml import YAML
        return YAML(typ="safe").load(mp) or {}
    except Exception:
        return None
    
def get_calibration_keywords(pt_path: Path, default: list[str] | None = None) -> list[str]:
    meta = read_metadata(pt_path)
    if meta and meta.get("calibration_keywords"):
        return list(meta["calibration_keywords"])
    return list(default) if default else []

def set_calibration_keywords(pt_path: Path, keywords: list[str]) -> None:
    meta = read_metadata(pt_path) or metadata_from_pt(pt_path)
    meta["calibration_keywords"] = list(keywords)
    write_metadata(metadata_path_for(pt_path), meta)

def metadata_from_pt(pt_path: Path) -> Dict[str, Any]:
    import torch
    from iSpy.vision.genericYolo import torch_load

    ckpt = torch_load(pt_path)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model = ckpt["model"]
    elif isinstance(ckpt, torch.nn.Module):
        model = ckpt
    else:
        model = ckpt

    if hasattr(model, "eval"):
        model.eval()
    task = getattr(model, "task", "detect") or "detect"

    try:
        nc = int(getattr(model, "nc", 1))
    except Exception:
        nc = 1

    try:
        names = dict(getattr(model, "names", {0: "object"}))
    except Exception:
        names = {0: "object"}

    try:
        imgsz = 640
        if hasattr(model, "model") and hasattr(model.model, "args"):
            imgsz = model.model.args.get("imgsz", 640)
        input_size = [imgsz, imgsz] if isinstance(imgsz, int) else list(imgsz[:2])
    except Exception:
        input_size = [640, 640]

    meta: Dict[str, Any] = {
        "task": task,
        "nc": nc,
        "names": names,
        "input_size": input_size,
        "scores_are_logits": False,
        "nms_iou": 0.45,
        "output_format": "hardware_nms",
        "output_layout": "anchors_first",
        "box_format": "xyxy",
        "score_mode": "objectness" if nc == 1 else "multi_class",
        "apply_software_nms": False,
        "quantization": "none",
        "quant_scale": 255.0,
        "input_layout": "nhwc",
        "input_dtype": "uint8",
        "input_letterbox": True,
        "input_pad_value": 114,
        "input_normalize": False,
    }

    if task == "pose":
        try:
            kpt = getattr(model, "kpt_shape", None)
            if kpt is None and hasattr(model, "model"):
                kpt = getattr(model.model, "kpt_shape", None) if hasattr(model.model, "__getitem__") else None
            if kpt is not None:
                meta["kpt_shape"] = [int(kpt[0]), int(kpt[1])]
            else:
                meta["kpt_shape"] = [17, 3]
        except Exception:
            meta["kpt_shape"] = [17, 3]

    return meta


def derive_format_metadata(pt_meta: Dict[str, Any], target_format: str) -> Dict[str, Any]:
    base = dict(pt_meta)

    FORMAT_CONTRACTS = {
        "rknn": {
            "output_format":      "raw",
            "output_layout":      "features_first",
            "box_format":         "cxcywh",
            "score_mode":         "objectness" if pt_meta.get("nc", 1) == 1 else "multi_class",
            "scores_are_logits":  False,
            "apply_software_nms": True,
            "quantization":       "int8",
            "quant_scale":        255.0,
            "input_layout":       "nhwc",
            "input_dtype":        "uint8",
            "input_letterbox":    True,
            "input_pad_value":    114,
            "input_normalize":    False,
        },
        "onnx": {
            "output_format":      "raw",
            "output_layout":      "features_first",
            "box_format":         "cxcywh",
            "score_mode":         "objectness" if pt_meta.get("nc", 1) == 1 else "multi_class",
            "scores_are_logits":  False,
            "apply_software_nms": True,
            "quantization":       "none",
            "input_layout":       "nchw",
            "input_dtype":        "float32",
            "input_letterbox":    True,
            "input_pad_value":    114,
            "input_normalize":    True,
            "input_scale":        255.0,
        },
        "tflite": {
            "output_format":      "raw",
            "output_layout":      "anchors_first",
            "box_format":         "cxcywh",
            "score_mode":         "objectness" if pt_meta.get("nc", 1) == 1 else "multi_class",
            "scores_are_logits":  False,
            "apply_software_nms": True,
            "quantization":       "int8",
            "quant_scale":        255.0,
            "input_layout":       "nhwc",
            "input_dtype":        "uint8",
            "input_letterbox":    True,
            "input_pad_value":    114,
            "input_normalize":    False,
        },
        "openvino": {
            "output_format":      "hardware_nms",
            "output_layout":      "anchors_first",
            "box_format":         "xyxy",
            "score_mode":         "objectness",
            "scores_are_logits":  False,
            "apply_software_nms": False,
            "quantization":       "none",
            "input_layout":       "nhwc",
            "input_dtype":        "uint8",
            "input_letterbox":    True,
            "input_pad_value":    114,
            "input_normalize":    False,
        },
        "coreml": {
            "output_format":      "hardware_nms",
            "output_layout":      "anchors_first",
            "box_format":         "xyxy",
            "score_mode":         "objectness",
            "scores_are_logits":  False,
            "apply_software_nms": False,
            "quantization":       "none",
            "input_layout":       "nhwc",
            "input_dtype":        "float32",
            "input_letterbox":    True,
            "input_pad_value":    114,
            "input_normalize":    True,
            "input_scale":        255.0,
        },
        "engine": {
            "output_format":      "hardware_nms",
            "output_layout":      "anchors_first",
            "box_format":         "xyxy",
            "score_mode":         "objectness",
            "scores_are_logits":  False,
            "apply_software_nms": False,
            "quantization":       "none",
            "input_layout":       "nchw",
            "input_dtype":        "float32",
            "input_letterbox":    True,
            "input_pad_value":    114,
            "input_normalize":    True,
            "input_scale":        255.0,
        },
    }

    contract = FORMAT_CONTRACTS.get(target_format, {})
    base.update(contract)
    return base
