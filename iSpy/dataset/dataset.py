import logging
import math
import os
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
_CALIB_COUNT = 200
_IMGSZ = 640
_CALIBRATION_RELEASE_URL = "https://github.com/aidan-j532/iSpy-FRC/releases/download/RKNN_Quantization/200.Robotics.Images.zip"
_VALIDATION_RELEASE_URL = "https://github.com/aidan-j532/iSpy-FRC/releases/download/Test_Images/valid.zip"
_VALIDATION_KEYWORDS = [
    "robotics validation images",
    "robotics test images",
    "machine vision calibration",
]

# Search-engine HTML scraping (DuckDuckGo/Bing/Google) is isolated behind an
# explicit opt-in so the project never depends on scraping search engines at
# build time - and never routes builds through barely-configured third-party
# HTML crawlers. Set ISPY_ALLOW_SEARCH_FALLBACK=1 to re-enable the old fallback.
_SEARCH_FALLBACK_ENV = "ISPY_ALLOW_SEARCH_FALLBACK"

_FORMAT_CALIB_COUNTS = {
    "rknn": 20,      # KL-divergence wants broader coverage
    "tflite": 100,     # simpler min/max calibration, converges faster
    "openvino": 300,
    "engine": 500,     # tensorrt entropy calibration wants more samples
    "coreml": 0,       # float16, no calibration needed
}

def calib_count_for_format(target_format: str, default: int = _CALIB_COUNT) -> int:
    return _FORMAT_CALIB_COUNTS.get(target_format, default)

def _download_release_images(
    folder: Path,
    count: int = _CALIB_COUNT,
    release_url: str | None = None,
    target_dir: str = "",
) -> list[Path]:
    import zipfile
    import io

    downloaded: list[Path] = []
    images_dir = folder / target_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    url = release_url or _CALIBRATION_RELEASE_URL
    logger.info("Trying release calibration images: %s", url)
    try:
        sess = _session()
        resp = sess.get(url, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue

                path = Path(member.filename)

                if path.suffix.lower() not in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".tiff"
                }:
                    continue

                dest = images_dir / path.name
                with zf.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                downloaded.append(dest)
                if len(downloaded) >= count:
                    break
    except Exception as e:
        logger.warning("Failed to download release calibration images: %s", e)
        return []

    logger.info("Got %d calibration images from release", len(downloaded))
    return downloaded

def _extract_release_url(keywords: list[str] | None) -> str | None:
    if not keywords:
        return None
    for kw in keywords:
        if isinstance(kw, str) and kw.startswith(("http://", "https://")):
            return kw
    return None

def _session():
    import requests as _requests
    sess = _requests.Session()
    # TLS verification stays ON (requests default) - downloads come from
    # GitHub/NuGet-style HTTPS endpoints we have no reason to distrust.
    for scheme in ("http://", "https://"):
        adapter = sess.get_adapter(scheme)
        adapter.max_retries = _requests.adapters.Retry(total=1, backoff_factor=0.5, raise_on_status=False)
    return sess


def _search_urls_ddg(sess, keyword: str, count: int, headers: dict) -> list[str]:
    import re
    import urllib.parse
    import json

    try:
        resp = sess.get(
            f"https://duckduckgo.com/?q={urllib.parse.quote(keyword)}",
            headers=headers,
            timeout=10,
        )
        html = resp.text

        vqd = None
        patterns = [
            r'vqd=([\w-]+)&',
            r'"vqd"\s*:\s*"([\w-]+)"',
            r'vqd["\']?\s*:\s*["\']([\w-]{60,})["\']',
            r'vqd=([\w-]{60,})(?:&|"|\s|$)',
        ]
        for p in patterns:
            m = re.search(p, html)
            if m:
                candidate = m.group(1)
                if len(candidate) > 60:
                    vqd = candidate
                    break
                if len(candidate) > 20:
                    vqd = candidate
                    break
        if not vqd:
            for token in re.findall(r'[\w-]{60,}', html):
                vqd = token
                break

        if not vqd:
            return []

        img_headers = {
            **headers,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": "https://duckduckgo.com/",
            "X-Requested-With": "XMLHttpRequest",
        }

        resp = sess.get(
            "https://duckduckgo.com/i.js",
            params={"q": keyword, "vqd": vqd, "o": "json", "p": "1", "v": "1"},
            headers=img_headers,
            timeout=10,
        )
        data = resp.json()
        results = data.get("results", [])

        urls: list[str] = []
        for r in results:
            image_url = r.get("image", "")
            if image_url and image_url.startswith("http"):
                urls.append(image_url)
            if len(urls) >= count:
                break
        return urls
    except (json.JSONDecodeError, KeyError, TypeError, Exception):
        return []


def _search_urls_bing(sess, keyword: str, count: int, headers: dict) -> list[str]:
    import re
    import urllib.parse

    found: list[str] = []
    seen: set[str] = set()

    try:
        query = urllib.parse.quote(keyword)
        search_url = f"https://www.bing.com/images/search?q={query}&count={count * 3}"
        resp = sess.get(search_url, headers=headers, timeout=10)
        resp.raise_for_status()
        html = resp.text

        img_urls = set()

        for attr in ("src", "data-src", "data-src2", "data-original"):
            for m in re.finditer(
                rf'{attr}="(https?://[^"]+)"', html, re.IGNORECASE,
            ):
                u = m.group(1)

                u_clean = u.replace("&amp;", "&").replace("\\/", "/")

                if "/th?id=OIP" in u_clean or "/th/id/OIP" in u_clean:
                    img_urls.add(u_clean)
                    continue

                if u_clean.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_urls.add(u_clean)

        for m in re.finditer(
            r'<a[^>]+href="(https?://[^"]+)"[^>]*>',
            html, re.IGNORECASE,
        ):
            u = m.group(1)
            u_clean = u.replace("&amp;", "&").replace("\\/", "/")
            if "/th?id=OIP" in u_clean or "/th/id/OIP" in u_clean:
                img_urls.add(u_clean)

        for u in img_urls:
            u = u.replace("&#8203;", "")
            if u not in seen:
                seen.add(u)
                found.append(u)
                if len(found) >= count:
                    break

    except Exception as e:
        logger.debug("Bing image search failed for %r: %s", keyword, e)

    return found


def _search_urls_google(sess, keyword: str, count: int, headers: dict) -> list[str]:
    import re
    import urllib.parse

    found: list[str] = []
    seen: set[str] = set()

    try:
        query = urllib.parse.quote(keyword)
        search_url = f"https://www.google.com/search?q={query}&tbm=isch&hl=en"
        resp = sess.get(search_url, headers=headers, timeout=10)
        resp.raise_for_status()
        html = resp.text

        img_urls = set()

        for m in re.finditer(r'"(https?://[^"]+\.(?:jpg|jpeg|png|bmp|gif|webp)(?:\?[^"]*)?)"', html, re.IGNORECASE):
            u = m.group(1).replace("\\/", "/").replace("\\u0026", "&").replace("\\x26", "&")
            if any(ext in u.lower() for ext in [".jpg", ".jpeg", ".png"]):
                if "google" not in u.lower() and "gstatic" not in u.lower():
                    img_urls.add(u)

        for m in re.finditer(r'src="(https?://[^"]+)"', html):
            u = m.group(1)
            if any(ext in u.lower() for ext in [".jpg", ".jpeg", ".png"]):
                if "google" not in u.lower() and "gstatic" not in u.lower():
                    img_urls.add(u)

        for m in re.finditer(r'\["(https?://[^"]+)",\d+,\d+\]', html):
            u = m.group(1).replace("\\/", "/").replace("\\u0026", "&")
            img_urls.add(u)

        for u in img_urls:
            if u not in seen:
                seen.add(u)
                found.append(u)
                if len(found) >= count:
                    break

    except Exception as e:
        logger.debug("Google image search failed for %r: %s", keyword, e)

    return found

def get_active_dataset_dir(default_root: str = "QuantizeDataset") -> Path:
    return Path.cwd() / default_root

def _is_host_reachable(host: str, timeout: int = 3) -> bool:
    try:
        sess = _session()
        sess.get(f"https://{host}", timeout=timeout)
        return True
    except Exception as e:
        logger.debug("Host %s unreachable: %s", host, e)
        return False

def _search_fallback_allowed() -> bool:
    """Search-engine HTML scraping is opt-in (ISPY_ALLOW_SEARCH_FALLBACK=1)."""
    return os.environ.get(_SEARCH_FALLBACK_ENV, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _collect_urls(keywords: list[str], count: int) -> tuple[list[str], dict, object]:
    sess = _session()

    if not _search_fallback_allowed():
        logger.warning(
            "Search-engine image scraping is disabled by default (set "
            "%s=1 to allow it). Falling back to synthetic images.",
            _SEARCH_FALLBACK_ENV,
        )
        return [], {}, sess

    search_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    dl_headers_base = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    all_urls: list[str] = []
    seen: set[str] = set()

    engines = [
        ("Bing", _search_urls_bing),
        ("Google", _search_urls_google),
        ("DuckDuckGo", _search_urls_ddg),
    ]

    for name, search_fn in engines:
        if len(all_urls) >= count * 2:
            break
        host = name.lower().strip(")") + ".com"
        if not _is_host_reachable(host):
            logger.debug("Skipping %s (unreachable)", name)
            continue
        for kw in keywords:
            try:
                urls = search_fn(sess, kw, count * 2, search_headers)
                added = 0
                for url in urls:
                    if url not in seen and len(url) < 500:
                        seen.add(url)
                        all_urls.append(url)
                        added += 1
                logger.info("%s:%s returned %d new URLs", name, kw, added)
            except Exception:
                continue

    if len(all_urls) < count:
        logger.warning("Only collected %d / %d image URLs from search engines", len(all_urls), count)

    return all_urls, dl_headers_base, sess


def _validate_image(image_path: Path) -> bool:
    try:
        from PIL import Image
        img = Image.open(image_path)
        img.load()

        if img.width < 32 or img.height < 32:
            logger.debug("Rejecting %s: too small (%dx%d)", image_path.name, img.width, img.height)
            return False

        needs_save = image_path.suffix.lower() != ".jpg"

        if img.mode in ("RGBA", "P", "L", "CMYK", "LA", "PA"):
            img = img.convert("RGB")
            needs_save = True

        if img.mode != "RGB":
            img = img.convert("RGB")
            needs_save = True

        if needs_save:
            dest = image_path.with_suffix(".jpg")
            img.save(dest, "JPEG", quality=90)
            if dest != image_path:
                image_path.unlink(missing_ok=True)
            return True

        extrema = img.getextrema()
        if all(mn == mx for mn, mx in extrema):
            logger.debug("Rejecting %s: blank image (all pixels identical)", image_path.name)
            return False

        return True
    except Exception as e:
        logger.debug("Rejecting %s: %s", image_path.name, e)
        return False


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
            logger.error("Cannot generate synthetic images: neither OpenCV nor Pillow available")
            return []
        return _generate_synthetic_images_pil(folder, count, imgsz, target_dir=target_dir)

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
            fill = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            shape = random.choice(["rect", "ellipse"])
            if shape == "rect":
                draw.rectangle([x1, y1, x2, y2], fill=fill)
            else:
                draw.ellipse([x1, y1, x2, y2], fill=fill)

        img.save(dest, "JPEG", quality=85)
        generated.append(dest)

    return generated


def _download_images(
    keywords: list[str],
    folder: Path,
    count: int = _CALIB_COUNT,
    boot: bool = False,
    start_index: int = 0,
    target_dir: str = "",
) -> list[Path]:
    images_dir = folder / target_dir
    images_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    all_urls, dl_headers_base, sess = _collect_urls(keywords, count)
    logger.info("Collected %d image URLs, attempting to download %d", len(all_urls), count)

    for url in all_urls:
        if len(downloaded) >= count:
            break
        try:
            headers = dict(dl_headers_base)
            headers["Referer"] = "https://www.bing.com/"
            resp = sess.get(url, headers=headers, timeout=15, stream=False)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").lower()
            if "image" not in content_type:
                continue
            ext = ".jpg"
            if "png" in content_type:
                ext = ".png"
            elif "bmp" in content_type:
                ext = ".bmp"
            elif "gif" in content_type:
                ext = ".gif"
            elif "webp" in content_type:
                ext = ".webp"
            elif "svg" in content_type:
                continue
            elif "jpeg" in content_type or "jpg" in content_type:
                ext = ".jpg"

            if len(resp.content) < 256:
                continue

            dest = images_dir / f"img_{start_index + len(downloaded):03d}{ext}"
            with open(dest, "wb") as f:
                f.write(resp.content)

            if _validate_image(dest):
                downloaded.append(dest)
                logger.info("Downloaded %d/%d: %s (%.0f KB)", len(downloaded), count, dest.name, len(resp.content) / 1024)
            else:
                dest.unlink(missing_ok=True)
        except Exception:
            continue

    if len(downloaded) < count:
        needed = count - len(downloaded)
        logger.warning("Only downloaded %d / %d real images. Generating %d synthetic calibration images...", len(downloaded), count, needed)
        synthetic = _generate_synthetic_images(folder, needed, _IMGSZ, target_dir=target_dir)
        downloaded.extend(synthetic)
        logger.info("Total calibration images: %d (%d real + %d synthetic)", len(downloaded), len(downloaded) - len(synthetic), len(synthetic))

    logger.info("Downloaded %d images", len(downloaded))
    return downloaded

def _find_images(folder: Path):
    imgs = []
    for ext in _IMAGE_EXTS:
        imgs.extend(folder.rglob(ext))
    return sorted(imgs)


def _rebuild_dataset_txt(ds: Path, root: Path | None = None):
    search_root = root or ds
    imgs = [
        p for p in _find_images(search_root)
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
    boot: bool = False,
    keywords: list[str] | None = None,
) -> Path:
    ds = Path(dataset_path)
    validation_dir = ds / "valid" / "images"
    validation_dir.mkdir(parents=True, exist_ok=True)

    validation_count = max(1, math.ceil(count / 10))
    existing = _find_images(validation_dir)
    if len(existing) >= validation_count:
        return validation_dir

    logger.info("Preparing %d validation images under %s", validation_count, validation_dir)
    release_url = _extract_release_url(keywords) or _VALIDATION_RELEASE_URL
    _download_release_images(
        ds,
        validation_count,
        release_url=release_url,
        target_dir="valid/images",
    )
    existing = _find_images(validation_dir)

    fallback_keywords = list(keywords or _VALIDATION_KEYWORDS)
    is_url_only = _extract_release_url(keywords) is not None
    if not is_url_only and fallback_keywords and len(existing) < validation_count:
        remaining = validation_count - len(existing)
        logger.info(
            "Have %d/%d validation images from release download; fetching %d more via keyword search (%s)",
            len(existing),
            validation_count,
            remaining,
            ", ".join(fallback_keywords),
        )
        _download_images(
            fallback_keywords,
            ds,
            remaining,
            boot=boot,
            start_index=len(existing),
            target_dir="valid/images",
        )

    existing = _find_images(validation_dir)
    if len(existing) < validation_count:
        logger.warning(
            "Only have %d / %d validation images. Generating %d synthetic fallback images...",
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
    """Calibration images only - the internal valid/ split is excluded."""
    return [
        p for p in _find_images(ds)
        if "valid" not in p.relative_to(ds).parts
    ]


def prepare_quantization_dataset(
    dataset_path: str = "dataset",
    imgsz: int = _IMGSZ,
    boot: bool = False,
    keywords: list[str] | None = None,
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
        logger.info("Dataset already has %d images, skipping download", len(existing))
        _rebuild_dataset_txt(ds)
    else:
        release_url = _extract_release_url(keywords)
        if release_url:
            logger.info("Keyword is a release URL - downloading calibration images from %s", release_url)
            _download_release_images(ds, count, release_url=release_url, target_dir="")
            existing = _find_train_images(ds)
        else:
            _download_release_images(ds, count, target_dir="")
            existing = _find_train_images(ds)

            if keywords and len(existing) < count:
                remaining = count - len(existing)
                logger.info(
                    "Have %d/%d calibration images from release download; "
                    "fetching %d more via keyword search (%s) instead of discarding them.",
                    len(existing), count, remaining, ", ".join(keywords),
                )
                _download_images(keywords, ds, remaining, boot=boot, start_index=len(existing))
                existing = _find_train_images(ds)

        if len(existing) < count:
            logger.warning("Only have %d / %d images. Generating synthetic fallback...", len(existing), count)
            _generate_synthetic_images(ds, count - len(existing), imgsz, target_dir="")

        _rebuild_dataset_txt(ds)

    add_validate_images(ds, count=count, imgsz=imgsz, boot=boot, keywords=keywords)

    final_count = len(_find_train_images(ds))
    logger.info("Quantization dataset ready at %s (%d images)", ds.resolve(), final_count)
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
                    issues.append(f"data.yaml points to non-existent path: {train_path}")
                else:
                    val_imgs = list(tp.rglob("*"))
                    img_val = [v for v in val_imgs if v.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")]
                    if not img_val:
                        issues.append(f"data.yaml path '{train_path}' has no images")
                    else:
                        result["yolo_data_ready"] = True
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
