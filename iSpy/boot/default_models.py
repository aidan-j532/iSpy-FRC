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
    "_default_v26_detect_for_fuel.pt": "https://github.com/aidan-j532/iSpy-FRC/releases/download/Fuel_Detect_Model/fuel_detection_v26.pt",
}

# Per-model license status. All three default checkpoints are AGPL-3.0:
# _default_detect.pt and _default_pose.pt are stock Ultralytics pretrained
# weights (AGPL-3.0 by Ultralytics); _default_v26_detect_for_fuel.pt was
# trained by the project owner and is also released under AGPL-3.0.
_MODEL_LICENSE_STATUS = {name: "AGPL-3.0" for name in _DEFAULT_MODEL_URLS}


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
        logger.warning(
            "Failed to download default model %s from %s: %s (skipping)", name, url, exc
        )
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
            name,
            size,
        )
        try:
            target.unlink()
        except OSError:
            pass
        return False

    logger.info("Downloaded default model %s -> %s", name, target)
    return True


def download_default_models(fresh: bool = False) -> dict[str, Path]:
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
        "iSpy-authored source code is licensed under the PolyForm Noncommercial",
        "1.0.0 license (see the repo root LICENSE file). The Ultralytics package",
        "is AGPL-3.0. The v26 fuel model was trained with Ultralytics code and",
        "is an Ultralytics-derived AGPL-3.0 model, not a PolyForm asset.",
        "Ultralytics is only an optional build-time [optimizer] dependency for",
        "the runtime package and is not imported by the runtime inference path.",
        "",
        "The stock default checkpoint files in this folder are downloaded from",
        "external URLs, not bundled with iSpy. Their per-model license status",
        "is listed below (all AGPL-3.0):",
        "",
    ]
    for name in _DEFAULT_MODEL_URLS:
        lines.append(f"- {name}: {_MODEL_LICENSE_STATUS[name]}")
    lines.append("")
    return "\n".join(lines)


def write_default_models_license() -> Path:
    path = _PROJECT_ROOT / "YoloModels" / "LICENSE.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(default_models_license_text(), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path
