import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REQUIRED_DIRS = ["images"]
_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
_CALIB_COUNT = 20
_IMGSZ = 640


def _generate_calibration_images(folder: Path, count: int = _CALIB_COUNT, imgsz: int = _IMGSZ, boot: bool = False):
    if not boot:
        logger.warning("Generating SYNTHETIC calibration images, expect poor quantization results! Add real images to %s to improve accuracy.", folder / "images")
    import numpy as np
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not available — cannot generate calibration images")
        return []

    images_dir = folder / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for i in range(count):
        path = images_dir / f"calib_{i:03d}.jpg"
        if path.exists():
            generated.append(path)
            continue
        rng = np.random.default_rng(seed=i)
        mode = i % 4
        if mode == 0:
            arr = rng.integers(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)
        elif mode == 1:
            ramp = np.tile(np.linspace(0, 255, imgsz, dtype=np.uint8), (imgsz, 1))
            arr = np.stack([ramp, ramp[::-1], ramp], axis=-1)
        elif mode == 2:
            block = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            arr = np.tile(block, (imgsz // 32 + 1, imgsz // 32 + 1, 1))[:imgsz, :imgsz]
        else:
            arr = rng.integers(200, 256, (imgsz, imgsz, 3), dtype=np.uint8)
            blur_k = rng.integers(0, 50, (imgsz, imgsz), dtype=np.uint8)
            arr = np.clip(arr.astype(np.int16) - blur_k[..., None], 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(path, quality=85)
        generated.append(path)

    logger.info("Generated %d calibration images in %s", len(generated), images_dir)
    return generated


def _find_images(folder: Path):
    imgs = []
    for ext in _IMAGE_EXTS:
        imgs.extend(folder.rglob(ext))
    return sorted(imgs)


def _rebuild_dataset_txt(ds: Path):
    imgs = _find_images(ds)
    if imgs:
        (ds / "dataset.txt").write_text(
            "\n".join(str(img.relative_to(ds)) for img in imgs) + "\n"
        )
        return True
    return False


def prepare_quantization_dataset(dataset_path: str = "dataset", imgsz: int = _IMGSZ, boot: bool = False) -> Path:
    ds = Path(dataset_path)
    for sub in _REQUIRED_DIRS:
        (ds / sub).mkdir(parents=True, exist_ok=True)

    data_yaml = ds / "data.yaml"
    if not data_yaml.exists():
        data_yaml.write_text(
            "train: images\n"
            "val: images\n"
            "nc: 1\n"
            "names: ['object']\n"
        )
        logger.info("Created %s", data_yaml)

    if not _find_images(ds):
        _generate_calibration_images(ds, imgsz=imgsz, boot=boot)

    _rebuild_dataset_txt(ds)

    logger.info("Quantization dataset directory ready at %s", ds.resolve())
    return ds


def validate_quantization_dataset(dataset_path: str = "dataset") -> dict:
    ds = Path(dataset_path)
    issues = []
    result = {
        "valid": True,
        "issues": [],
        "image_count": 0,
        "rknn_ready": False,
        "ultralytics_ready": False,
        "dataset_path": str(ds.resolve()),
    }

    if not ds.exists():
        result["valid"] = False
        result["issues"].append(f"Dataset folder not found: {ds.resolve()}")
        return result

    imgs = _find_images(ds)
    result["image_count"] = len(imgs)

    if not imgs:
        issues.append("No calibration images found — add images (*.jpg, *.png, etc.) to the dataset folder")

    dataset_txt = ds / "dataset.txt"
    if dataset_txt.exists():
        lines = [
            l.strip() for l in dataset_txt.read_text().splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        if not lines:
            issues.append("RKNN dataset.txt is empty or all-comment")
        else:
            missing = [l for l in lines if not (ds / l).exists()]
            if missing:
                issues.append(f"RKNN dataset.txt: {len(missing)} image(s) missing: {missing[:3]}" + ("..." if len(missing) > 3 else ""))
            else:
                result["rknn_ready"] = True
    else:
        issues.append("Missing dataset.txt (required for RKNN quantization)")

    data_yaml = ds / "data.yaml"
    if data_yaml.exists():
        try:
            import yaml
            with open(data_yaml) as f:
                cfg = yaml.safe_load(f) or {}
            train_path = cfg.get("train") or cfg.get("val")
            if train_path:
                tp = Path(train_path)
                if not tp.is_absolute():
                    tp = ds / tp
                if not tp.exists():
                    issues.append(f"data.yaml points to non-existent path: {train_path}")
                else:
                    val_imgs = list(tp.rglob("*"))
                    img_val = [v for v in val_imgs if v.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")]
                    if not img_val:
                        issues.append(f"data.yaml path '{train_path}' has no images")
                    else:
                        result["ultralytics_ready"] = True
            else:
                issues.append("data.yaml missing 'train' or 'val' key")
        except Exception as e:
            issues.append(f"data.yaml parse error: {e}")
    else:
        issues.append("Missing data.yaml (required for TFLite/OpenVINO int8 quantization)")

    if issues:
        result["valid"] = False
    result["issues"] = issues
    return result
