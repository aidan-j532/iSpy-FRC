"""Boot-time downloads for iSpy's stock default models.

The three default checkpoints (detect, pose, and the v26 fuel-detect model)
used to ship inside ``iSpy/assets/`` and get staged into ``YoloModels/pytorch/``
at boot. They are third-party AGPL checkpoints, so bundling them conflicts with
shipping the rest of iSpy under PolyForm Noncommercial 1.0.0. Instead they are
downloaded from external URLs on a fresh/first boot, mirroring the calibration-
image download conventions in iSpy/dataset/dataset.py (_download_release_images:
same requests-session retry semantics, streamed to a temp file, atomic rename,
no partial files, never crash boot on failure).

A missing or None download URL is a hard stop for that one model only: it gets a
clear warning and is skipped so the rest of boot proceeds. A failed download is
the same. iSpy must always be able to boot, add a camera, and upload a user
model even with zero default models on disk.
"""

import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "YoloModels" / "pytorch"

# Same size floor as genericYolo.py::_validate_model_file.
_MIN_MODEL_BYTES = 1024

# Download URLs for the three stock default models.
# detect/pose reuse the already-verified Ultralytics release assets that
# iSpy.vision.optimizer.ensure_default_model() has always downloaded from -
# keep the two in sync. The v26 checkpoint URL is unresolved until the project
# owner supplies it: a None URL is logged and skipped, never guessed.
_DEFAULT_MODEL_URLS = {
    "_default_detect.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
    "_default_pose.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt",
    "_default_v26_detect_for_fuel.pt": None,  # TODO(aidan): fill in URL
}

# Per-model license status. Never assert a license for a model ourselves - the
# status stays UNRESOLVED until the project owner confirms it.
_MODEL_LICENSE_STATUS = {name: "UNRESOLVED - see project owner" for name in _DEFAULT_MODEL_URLS}


def _session() -> requests.Session:
    sess = requests.Session()
    # TLS verification stays ON (requests default) - models come from the same
    # GitHub/HTTPS endpoints we already trust for calibration images.
    for scheme in ("http://", "https://"):
        adapter = sess.get_adapter(scheme)
        adapter.max_retries = requests.adapters.Retry(
            total=1, backoff_factor=0.5, raise_on_status=False
        )
    return sess


def _download_one(name: str, url: str, target: Path) -> bool:
    """Stream a single model to ``target`` via an atomic .part rename.

    Returns True only on a complete download above the size floor. On any
    failure the .part file is removed and False is returned.
    """
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        tmp.unlink(missing_ok=True)
        sess = _session()
        resp = sess.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
        tmp.replace(target)
    except Exception as exc:
        logger.warning("Failed to download default model %s from %s: %s (skipping)", name, url, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False

    try:
        size = target.stat().st_size
    except OSError:
        size = 0
    if size < _MIN_MODEL_BYTES:
        logger.warning(
            "Downloaded default model %s appears truncated (%s bytes) - removing.",
            name, size,
        )
        try:
            target.unlink()
        except OSError:
            pass
        return False

    logger.info("Downloaded default model %s -> %s", name, target)
    return True


def download_default_models(fresh: bool = False) -> dict[str, Path]:
    """Download every configured default model into YoloModels/pytorch/.

    Idempotent: a file that already exists above the size floor is left alone
    (unless ``fresh`` forces a re-download). Returns {filename: Path} for every
    model that is present and valid after the pass - pre-existing or freshly
    downloaded - so callers can report what is actually available.
    """
    _DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    present: dict[str, Path] = {}

    for name, url in _DEFAULT_MODEL_URLS.items():
        target = _DEFAULT_MODEL_DIR / name

        if not fresh and target.exists() and target.stat().st_size >= _MIN_MODEL_BYTES:
            logger.info("Default model %s already present - skipping download", name)
            present[name] = target
            continue

        if not url:
            logger.warning(
                "No download URL configured for %s - set it in "
                "iSpy.boot.default_models._DEFAULT_MODEL_URLS before shipping.",
                name,
            )
            continue

        if _download_one(name, url, target):
            present[name] = target

    return present


def default_models_license_text() -> str:
    lines = [
        "Default model license notices",
        "============================",
        "",
        "iSpy's own source code is licensed under the PolyForm Noncommercial",
        "1.0.0 license (see the repo root LICENSE file). The ultralytics",
        "package is used only as an optional build-time [optimizer] dependency,",
        "under AGPL-3.0, and is never imported at runtime.",
        "",
        "The stock default checkpoint files in this folder are downloaded from",
        "external URLs, not bundled with iSpy. Their per-model license status",
        "is listed below and is UNRESOLVED until confirmed by the project owner:",
        "",
    ]
    for name in _DEFAULT_MODEL_URLS:
        lines.append(f"- {name}: {_MODEL_LICENSE_STATUS[name]}")
    lines.append("")
    return "\n".join(lines)


def write_default_models_license() -> Path:
    """Write YoloModels/LICENSE.txt with the per-model status placeholder."""
    path = _PROJECT_ROOT / "YoloModels" / "LICENSE.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(default_models_license_text(), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path