import logging
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

_REQUIRED_DIRS = ["images"]
_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
_CALIB_COUNT = 20
_IMGSZ = 640

warnings.filterwarnings("ignore", message="Unverified HTTPS request")


def _search_urls_ddg(keyword: str, count: int, headers: dict) -> list[str]:
    import re
    import urllib.parse

    try:
        import requests as _requests
    except ImportError:
        return []

    session = _requests.Session()
    session.headers.update(headers)
    session.verify = False

    try:
        resp = session.get(
            f"https://duckduckgo.com/?q={urllib.parse.quote(keyword)}",
            timeout=15,
        )
        resp.raise_for_status()

        vqd = None
        m = re.search(r'vqd=([\d-]+)', resp.text)
        if m:
            vqd = m.group(1)
        if not vqd:
            m = re.search(r'vqd["\']:\s*["\']([\d-]+)["\']', resp.text)
            if m:
                vqd = m.group(1)
        if not vqd:
            return []

        resp = session.get(
            f"https://duckduckgo.com/i.js?q={urllib.parse.quote(keyword)}&vqd={vqd}&o=json&p=1",
            timeout=15,
        )
        resp.raise_for_status()

        data = resp.json()
        results = data.get("results", [])

        urls = []
        for r in results:
            image_url = r.get("image", "")
            if image_url and image_url.startswith("http"):
                urls.append(image_url)
            if len(urls) >= count:
                break
        return urls
    except Exception:
        return []


def _search_urls_bing(keyword: str, count: int, headers: dict) -> list[str]:
    import re
    import urllib.parse

    try:
        import requests as _requests
    except ImportError:
        return []

    found: list[str] = []
    seen: set[str] = set()

    try:
        query = urllib.parse.quote(keyword)
        search_url = f"https://www.bing.com/images/search?q={query}&count={count * 2}"
        resp = _requests.get(search_url, headers=headers, timeout=15, verify=False)
        resp.raise_for_status()
        html = resp.text

        patterns = [
            r'"mediaurl"\s*:\s*"(https?://[^"]+)"',
            r'"contentUrl"\s*:\s*"(https?://[^"]+)"',
            r'"thumbUrl"\s*:\s*"(https?://[^"]+)"',
            r'<img[^>]+src="(https?://[^"]+)"[^>]+class="mimg"',
            r'<img[^>]+class="mimg"[^>]+src="(https?://[^"]+)"',
        ]

        for pat in patterns:
            for match in re.finditer(pat, html, re.IGNORECASE):
                img_url = match.group(1).replace("\\/", "/")
                if img_url not in seen:
                    seen.add(img_url)
                    found.append(img_url)
                    if len(found) >= count:
                        break
            if len(found) >= count:
                break
    except Exception:
        pass

    return found


def _collect_urls(keywords: list[str], count: int) -> tuple[list[str], dict]:
    import requests as _requests

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

    # 1) DuckDuckGo
    for kw in keywords:
        urls = _search_urls_ddg(kw, count * 3, search_headers)
        for url in urls:
            if url not in seen:
                seen.add(url)
                all_urls.append(url)

    # 2) Bing fallback
    if len(all_urls) < count:
        for kw in keywords:
            urls = _search_urls_bing(kw, count * 3, search_headers)
            for url in urls:
                if url not in seen:
                    seen.add(url)
                    all_urls.append(url)

    # 3) picsum.photos guaranteed fallback
    if len(all_urls) < count:
        needed = count * 2 - len(all_urls)
        for i in range(needed):
            seed = f"{keywords[0]}_{i}_{len(all_urls)}"
            all_urls.append(f"https://picsum.photos/seed/{seed}/{_IMGSZ}/{_IMGSZ}")

    return all_urls, dl_headers_base


def _download_images(
    keywords: list[str],
    folder: Path,
    count: int = _CALIB_COUNT,
    boot: bool = False,
) -> list[Path]:
    images_dir = folder / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    try:
        import requests as _requests
    except ImportError:
        return []

    all_urls, dl_headers_base = _collect_urls(keywords, count)

    logger.info("Collected %d image URLs, attempting to download %d", len(all_urls), count)

    for url in all_urls:
        if len(downloaded) >= count:
            break
        try:
            dl_headers = dict(dl_headers_base)
            dl_headers["Referer"] = url
            resp = _requests.get(
                url, headers=dl_headers, timeout=30, stream=True, verify=False,
            )
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
            elif "jpeg" in content_type:
                ext = ".jpg"

            dest = images_dir / f"img_{len(downloaded):03d}{ext}"
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded.append(dest)
        except Exception:
            continue

    logger.info("Downloaded %d images", len(downloaded))
    return downloaded


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


def prepare_quantization_dataset(
    dataset_path: str = "dataset",
    imgsz: int = _IMGSZ,
    boot: bool = False,
    keywords: list[str] | None = None,
    count: int = _CALIB_COUNT,
) -> Path:
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

    existing = _find_images(ds)
    if len(existing) < count:
        if keywords:
            for f in existing:
                f.unlink()
            _download_images(keywords, ds, count, boot=boot)

        existing = _find_images(ds)
        if len(existing) < count:
            logger.warning("Only downloaded %d / %d requested images", len(existing), count)

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
        issues.append("No calibration images found - add images (*.jpg, *.png, etc.) to the dataset folder")

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
