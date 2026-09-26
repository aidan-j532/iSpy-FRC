import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
_CALIB_COUNT = 200
_IMGSZ = 640

_FORMAT_CALIB_COUNTS = {
    "rknn": 20,  # KL-divergence wants broader coverage
    "hailo": 20,  # DFC quantization - moderate coverage like rknn
    "qnn": 0,  # fp32 onnx artifact - NPU weights quantized on-device, no calib
    "tflite": 100,  # simpler min/max calibration, converges faster
    "openvino": 300,
    "engine": 500,  # tensorrt entropy calibration wants more samples
    "coreml": 0,  # float16, no calibration needed
}


def calib_count_for_format(target_format: str, default: int = _CALIB_COUNT) -> int:
    return _FORMAT_CALIB_COUNTS.get(target_format, default)


def get_active_dataset_dir(default_root: str = "QuantizeDataset") -> Path:
    return Path.cwd() / default_root


def _generate_synthetic_images(
    folder: Path,
    count: int,
    imgsz: int = _IMGSZ,
    target_dir: str = "",
) -> list[Path]:
    import numpy as np

    try:
        import cv2
    except ImportError:
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            logger.error(
                "Cannot generate synthetic images: neither OpenCV nor Pillow available"
            )
            return []
        return _generate_synthetic_images_pil(
            folder, count, imgsz, target_dir=target_dir
        )

    generated: list[Path] = []
    images_dir = folder / target_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(images_dir.glob("*")))
    for i in range(count):
        dest = images_dir / f"img_{existing + i:03d}.jpg"

        img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

        noise = np.random.randint(0, 100, (imgsz, imgsz, 3), dtype=np.uint8)
        img = cv2.addWeighted(img, 0.3, noise, 0.7, 0)

        color = (
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255)),
        )
        center = (
            int(np.random.randint(imgsz // 4, 3 * imgsz // 4)),
            int(np.random.randint(imgsz // 4, 3 * imgsz // 4)),
        )
        radius = int(np.random.randint(imgsz // 8, imgsz // 3))
        cv2.circle(img, center, radius, color, -1)

        color2 = (
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255)),
        )
        pt1 = (
            int(np.random.randint(0, imgsz // 2)),
            int(np.random.randint(0, imgsz // 2)),
        )
        pt2 = (
            int(np.random.randint(imgsz // 2, imgsz)),
            int(np.random.randint(imgsz // 2, imgsz)),
        )
        cv2.rectangle(img, pt1, pt2, color2, -1)

        cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        generated.append(dest)

    return generated


def _generate_synthetic_images_pil(
    folder: Path,
    count: int,
    imgsz: int = _IMGSZ,
    target_dir: str = "",
) -> list[Path]:
    from PIL import Image, ImageDraw
    import random

    generated: list[Path] = []
    images_dir = folder / target_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(images_dir.glob("*")))
    for i in range(count):
        dest = images_dir / f"img_{existing + i:03d}.jpg"

        base = bytearray(random.randint(0, 127) for _ in range(imgsz * imgsz * 3))
        img = Image.frombytes("RGB", (imgsz, imgsz), bytes(base))
        draw = ImageDraw.Draw(img)

        for _ in range(random.randint(2, 5)):
            x1 = random.randint(0, imgsz - 1)
            y1 = random.randint(0, imgsz - 1)
            x2 = random.randint(x1, imgsz - 1)
            y2 = random.randint(y1, imgsz - 1)
            fill = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            shape = random.choice(["rect", "ellipse"])
            if shape == "rect":
                draw.rectangle([x1, y1, x2, y2], fill=fill)
            else:
                draw.ellipse([x1, y1, x2, y2], fill=fill)

        img.save(dest, "JPEG", quality=85)
        generated.append(dest)

    return generated


def _find_images(folder: Path):
    imgs = []
    for ext in _IMAGE_EXTS:
        imgs.extend(folder.rglob(ext))
    return sorted(imgs)


def _rebuild_dataset_txt(ds: Path, root: Path | None = None):
    search_root = root or ds
    imgs = [
        p
        for p in _find_images(search_root)
        if "valid" not in p.relative_to(search_root).parts
    ]
    if imgs:
        (ds / "dataset.txt").write_text(
            "\n".join(str(img.relative_to(ds)) for img in imgs) + "\n"
        )
        return True
    return False


def add_image_to_dataset_txt(ds_root: Path, rel_path: str):
    txt = ds_root / "dataset.txt"
    existing = txt.read_text().splitlines() if txt.exists() else []
    if rel_path not in existing:
        existing.append(rel_path)
        txt.write_text("\n".join(existing) + "\n")


def remove_image_from_dataset_txt(ds_root: Path, rel_path: str):
    txt = ds_root / "dataset.txt"
    if txt.exists():
        lines = [l for l in txt.read_text().splitlines() if l.strip() != rel_path]
        txt.write_text("\n".join(lines) + ("\n" if lines else ""))


def add_validate_images(
    dataset_path: str | Path,
    count: int = _CALIB_COUNT,
    imgsz: int = _IMGSZ,
) -> Path:
    ds = Path(dataset_path)
    validation_dir = ds / "valid" / "images"
    validation_dir.mkdir(parents=True, exist_ok=True)

    validation_count = max(1, math.ceil(count / 10))
    existing = _find_images(validation_dir)
    if len(existing) >= validation_count:
        return validation_dir

    logger.info(
        "Preparing %d validation images under %s", validation_count, validation_dir
    )
    if len(existing) < validation_count:
        logger.warning(
            "Only have %d / %d uploaded validation images. Generating %d synthetic fallback images...",
            len(existing),
            validation_count,
            validation_count - len(existing),
        )
        _generate_synthetic_images(
            ds,
            validation_count - len(existing),
            imgsz,
            target_dir="valid/images",
        )

    if not (ds / "valid" / "dataset.txt").exists():
        _rebuild_dataset_txt(ds / "valid", root=validation_dir)

    return validation_dir


def _find_train_images(ds: Path) -> list[Path]:
    return [p for p in _find_images(ds) if "valid" not in p.relative_to(ds).parts]


def prepare_quantization_dataset(
    dataset_path: str = "dataset",
    imgsz: int = _IMGSZ,
    count: int = _CALIB_COUNT,
) -> Path:
    ds = Path(dataset_path)
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "valid" / "images").mkdir(parents=True, exist_ok=True)

    data_yaml = ds / "data.yaml"
    data_yaml.write_text(
        f"train: {ds.resolve()}\n"
        f"val: {(ds / 'valid' / 'images').resolve()}\n"
        "nc: 1\n"
        "names: ['object']\n"
    )

    existing = _find_train_images(ds)
    if len(existing) >= count:
        logger.info("Dataset already has %d images", len(existing))
    else:
        logger.warning(
            "Only have %d / %d uploaded calibration images. Generating synthetic fallback...",
            len(existing),
            count,
        )
        _generate_synthetic_images(ds, count - len(existing), imgsz, target_dir="")

    _rebuild_dataset_txt(ds)

    add_validate_images(ds, count=count, imgsz=imgsz)

    final_count = len(_find_train_images(ds))
    logger.info(
        "Quantization dataset ready at %s (%d images)", ds.resolve(), final_count
    )
    return ds


def validate_quantization_dataset(dataset_path: str = "dataset") -> dict:
    ds = Path(dataset_path)
    issues = []
    result = {
        "valid": True,
        "issues": [],
        "image_count": 0,
        "rknn_ready": False,
        "yolo_data_ready": False,
        "dataset_path": str(ds.resolve()),
    }

    if not ds.exists():
        result["valid"] = False
        result["issues"].append(f"Dataset folder not found: {ds.resolve()}")
        return result

    imgs = _find_images(ds)
    result["image_count"] = len(imgs)

    if not imgs:
        issues.append("No calibration images found")

    if imgs:
        from PIL import Image

        bad = 0
        for img_path in imgs:
            try:
                img = Image.open(img_path)
                img.load()
                if img.width < 32 or img.height < 32:
                    bad += 1
            except Exception:
                bad += 1
        if bad > 0:
            issues.append(f"{bad} image(s) failed validation (corrupt or too small)")

    dataset_txt = ds / "dataset.txt"
    if dataset_txt.exists():
        lines = [
            l.strip()
            for l in dataset_txt.read_text().splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        if not lines:
            issues.append("RKNN dataset.txt is empty or all-comment")
        else:
            missing = [l for l in lines if not (ds / l).exists()]
            if missing:
                issues.append(
                    f"RKNN dataset.txt: {len(missing)} image(s) missing: {missing[:3]}"
                    + ("..." if len(missing) > 3 else "")
                )
            else:
                result["rknn_ready"] = True
    else:
        issues.append("Missing dataset.txt (required for RKNN quantization)")

    data_yaml = ds / "data.yaml"
    if data_yaml.exists():
        try:
            from ruamel.yaml import YAML

            yaml = YAML()
            with open(data_yaml) as f:
                cfg = yaml.load(f) or {}
            train_path = cfg.get("train") or cfg.get("val")
            if train_path:
                tp = Path(train_path)
                if not tp.is_absolute():
                    tp = ds / tp
                if not tp.exists():
                    issues.append(
                        f"data.yaml points to non-existent path: {train_path}"
                    )
                else:
                    val_imgs = list(tp.rglob("*"))
                    img_val = [
                        v
                        for v in val_imgs
                        if v.suffix.lower()
                        in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
                    ]
                    if not img_val:
                        issues.append(f"data.yaml path '{train_path}' has no images")
                    else:
                        result["yolo_data_ready"] = True
            else:
                issues.append("data.yaml missing 'train' or 'val' key")
        except Exception as e:
            issues.append(f"data.yaml parse error: {e}")
    else:
        issues.append(
            "Missing data.yaml (required for TFLite/OpenVINO int8 quantization)"
        )

    if issues:
        result["valid"] = False
    result["issues"] = issues
    return result
