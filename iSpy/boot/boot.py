import sys
import os
import json
import shutil
import subprocess
import logging
import argparse
import time as _time
from pathlib import Path

from iSpy.config.iSpyConfig import iSpyConfig, get_pipeline_name
from iSpy.config.AutoOpt import has_jetson
from iSpy.validations.validate_system import validate_system
from iSpy.dataset.dataset import get_active_dataset_dir
from iSpy.vision.metadata import (
    metadata_path_for,
    metadata_from_pt,
    write_metadata,
    read_metadata,
)
from iSpy.boot.opencv_fix import ensure_csi_capable_opencv
from iSpy.vision.pipelines import get_pipeline_classes

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

_BOOT_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = Path.cwd().resolve()
_ASSETS_DIR = _PACKAGE_ROOT.parent / "assets"

_READINESS_POLL_S = 2.0
_READINESS_WAIT_TIMEOUT_S = 1200

# Bound to the real stdout/stderr before anything can swap sys.stdout, so
# silencing third-party libs later can't take iSpy's own logging down with it.
_REAL_STDOUT_FD = os.dup(1)
_REAL_STDERR_FD = os.dup(2)
_REAL_STDOUT = os.fdopen(_REAL_STDOUT_FD, "w", buffering=1, closefd=False)
_REAL_STDERR = os.fdopen(_REAL_STDERR_FD, "w", buffering=1, closefd=False)


def _remove_path_for_cleanup(path: Path) -> None:
    def onerror(func, p, exc_info):
        try:
            os.chmod(p, 0o777)
            func(p)
        except OSError:
            pass
    shutil.rmtree(str(path), onerror=onerror)


def _bootstrap_default_camera(config: iSpyConfig):
    cams = config.get("camera_configs", {})
    if cams and any(c.get("source") not in (None, "") for c in cams.values()):
        return  # already has something real
    from iSpy.web.modules.cameras import CamerasModule  # or move probing to a shared util
    devices = CamerasModule({})._probe_devices()
    if not devices:
        logger.warning("No cameras detected at first boot - leaving default placeholder config.")
        return
    dev = devices[0]
    name = "camera_1"
    default_cam = config.default_config["camera_configs"]["default_cam"]
    cam_cfg = json.loads(json.dumps(default_cam))
    cam_cfg["name"] = name
    cam_cfg["source"] = dev["path"]
    cam_cfg["device_id"] = dev.get("device_id")
    config.set("camera_configs", {name: cam_cfg})
    config.save()
    logger.info(
        "First boot: auto-configured camera '%s' -> %s (pipeline=%s)",
        name, dev["path"], get_pipeline_name(cam_cfg),
    )


def _close_logging_handlers() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        try:
            root.removeHandler(handler)
            handler.flush()
            handler.close()
        except Exception:
            pass


def _configure_quiet_logging() -> None:
    _close_logging_handlers()

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.propagate = False

    class _iSpyLogFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            name = record.name or ""
            return name == "root" or name.startswith("iSpy")

    formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    # Bind directly to the real stdout object (captured before anything ever
    # swaps sys.stdout) so silencing third-party libs later can't take
    # iSpy's own logging down with it - no fd tricks, works on every OS.
    stream_handler = logging.StreamHandler(_REAL_STDOUT)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(_iSpyLogFilter())
    root.addHandler(stream_handler)

    log_path = Path.cwd() / "Outputs" / "log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_iSpyLogFilter())
    root.addHandler(file_handler)

    logging.getLogger("iSpy").setLevel(logging.INFO)
    for name in list(logging.Logger.manager.loggerDict):
        if not name.startswith("iSpy"):
            logging.getLogger(name).setLevel(logging.WARNING)


def search_for_config():
    config_dir = _PROJECT_ROOT / "Config"
    if not config_dir.exists():
        return None
    config_files = sorted(config_dir.rglob("*.json"))
    if not config_files:
        return None
    non_default = [f for f in config_files if f.name != "config.json"]
    chosen = non_default[0] if non_default else config_files[0]
    logger.info("Found config: %s -> using %s", len(config_files), chosen)
    return str(chosen)


def setup_files(fresh: bool = False):
    """Fresh boot (`boot -f`): wipe generated application state (Config,
    Outputs, YoloModels, QuantizeDataset) and stage the bundled models.
    Normal boot: ensure runtime directories exist, stage any missing bundled
    assets, and keep model metadata sidecars up to date."""
    yolo_dir = _PROJECT_ROOT / "YoloModels"
    config_dir = _PROJECT_ROOT / "Config"
    outputs_dir = _PROJECT_ROOT / "Outputs"
    dataset_dir = get_active_dataset_dir()

    if fresh:
        for d in [yolo_dir, config_dir, outputs_dir, dataset_dir]:
            if d.exists():
                _remove_path_for_cleanup(d)
                logger.info("boot -f: deleted generated %s", d)

    keywords_path = _ASSETS_DIR / "keywords.json"

    try:
        with open(keywords_path, "r") as f:
            default_keywords = json.load(f)
    except FileNotFoundError:
        default_keywords = {}

    yolo_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["pytorch", "onnx", "tflite", "rknn", "openvino", "coreml", "engine"]:
        (yolo_dir / fmt).mkdir(parents=True, exist_ok=True)

    pytorch_dir = yolo_dir / "pytorch"
    for pt_file in _ASSETS_DIR.rglob("*.pt"):
        target = pytorch_dir / pt_file.name
        if fresh or not target.exists():
            shutil.copy2(pt_file, target)
            logger.info("Staged bundled model %s -> %s", pt_file.name, target)

    # Ensure every .pt in the pytorch directory has a metadata sidecar
    for pt_file in pytorch_dir.glob("*.pt"):
        meta_path = metadata_path_for(pt_file)
        try:
            if meta_path.exists():
                meta = read_metadata(meta_path)
            else:
                logger.info("Generating metadata for %s", pt_file.name)
                meta = metadata_from_pt(pt_file)

            # Apply bundled keyword overrides
            if pt_file.stem in default_keywords:
                meta["calibration_keywords"] = default_keywords[pt_file.stem]
            write_metadata(meta_path, meta)
            logger.info("Wrote metadata %s", meta_path.name)

        except Exception as e:
            logger.warning("Could not generate metadata for %s: %s", pt_file.name, e)


def _wait_for_pipeline_ready(
    config: iSpyConfig, pipeline_classes: dict[str, type]
) -> None:
    """Construct every configured camera pipeline (each pipeline starts its
    own background preparation in __init__) and wait until every pipeline
    reports ready. Fails fast with a clear error if a pipeline enters an
    unrecoverable error state, and raises if readiness is not reached within
    the timeout - boot never silently proceeds on an unready system."""
    from iSpy.config.iSpyConfig import iSpyCameraConfig

    cams = {k: v for k, v in config.config.get("camera_configs", {}).items() if isinstance(v, dict)}
    if not cams:
        raise RuntimeError("No cameras configured - nothing to boot.")

    default_pipeline = get_pipeline_name(
        config.default_config.get("camera_configs", {}).get("default_cam", {})
    )
    instances: dict[str, object] = {}
    for name, cam_cfg in cams.items():
        pipeline = get_pipeline_name(cam_cfg) if isinstance(cam_cfg, dict) else default_pipeline
        cls = pipeline_classes.get(pipeline)
        if cls is None:
            raise RuntimeError(
                f"Camera '{name}' uses unknown pipeline '{pipeline}' - cannot boot."
            )
        try:
            inst = cls(iSpyCameraConfig(cam_cfg), config, None)
        except Exception as e:
            raise RuntimeError(
                f"Camera '{name}' pipeline '{pipeline}' failed to construct: {e}"
            )
        instances[name] = inst

    if not instances:
        raise RuntimeError("No loadable pipelines to boot - aborting.")

    deadline = _time.monotonic() + _READINESS_WAIT_TIMEOUT_S
    logger.info(
        "Waiting for %d camera pipeline(s) to become ready (background "
        "preparation may still be running)...", len(instances),
    )
    pending = set(instances)
    last_status: dict[str, str] = {n: "" for n in instances}
    while pending and _time.monotonic() < deadline:
        for name in tuple(pending):
            inst = instances[name]
            try:
                ready, status = inst.is_ready()
            except Exception as e:
                ready, status = False, f"error: {e}"
            if status != last_status[name]:
                logger.info("  camera %-16s -> %s", name, status)
                last_status[name] = status
            if "error:" in status and not ready:
                raise RuntimeError(
                    f"Camera '{name}' pipeline entered an unrecoverable error "
                    f"state: {status}"
                )
            if ready:
                pending.discard(name)
        if pending:
            _time.sleep(_READINESS_POLL_S)

    if pending:
        detail = "; ".join(
            f"camera '{name}' -> {instances[name].is_ready()[1]}"
            for name in sorted(pending)
        )
        raise RuntimeError(
            f"Timed out waiting for pipeline readiness after "
            f"{_READINESS_WAIT_TIMEOUT_S}s: {detail}"
        )
    logger.info("All camera pipelines ready.")


def on_boot(install_service: bool = False, fresh: bool = False):
    _configure_quiet_logging()

    if fresh:
        logger.info("boot -f: forcefully fresh installation state")
        setup_files(fresh=True)
        config_path = str(_PROJECT_ROOT / "Config" / "config.json")
        config = iSpyConfig(config_path, create=True)
        _bootstrap_default_camera(config)
    else:
        setup_files(fresh=False)
        config_path = search_for_config()
        if not config_path:
            raise RuntimeError(
                "No configuration found. First-run initialization is only "
                "performed by 'boot -f' (forcefully fresh) - run that first."
            )
        logger.info("Using existing config: %s", config_path)
        config = iSpyConfig(config_path, create=False)

    if not validate_system():
        raise RuntimeError("System validation failed. Aborting boot.")

    pipeline_classes = get_pipeline_classes()
    _wait_for_pipeline_ready(config, pipeline_classes)
    config.save(quiet=True)
    logger.info("Boot sequence complete.")

    if install_service:
        install_script = str(_BOOT_DIR / "install.py")
        try:
            subprocess.run(
                [sys.executable, install_script], check=True, cwd=str(_PROJECT_ROOT)
            )
        except subprocess.CalledProcessError as e:
            logger.error("Failed to run install.py: %s", e)
            raise RuntimeError("Boot failed during service installation.")
    else:
        logger.info("Skipping service installation. Run with -s to install.")


def _any_camera_uses_csi() -> bool:
    config_path = search_for_config()
    if not config_path:
        return False
    try:
        with open(config_path) as f:
            data = json.load(f)
        cams = data.get("config", data).get("camera_configs", {})
        return any(c.get("csi", False) for c in cams.values())
    except Exception:
        return False


def main():
    if has_jetson() and _any_camera_uses_csi():
        if ensure_csi_capable_opencv(auto_fix=True):
            logger.info("OpenCV fixed - re-executing boot.py to pick it up...")
            os.execv(sys.executable, [sys.executable] + sys.argv)

    parser = argparse.ArgumentParser(description="iSpy boot sequence")
    parser.add_argument("-s", "--service", action="store_true",
                         help="Install and start the watchdog service")
    parser.add_argument("-f", "--fresh", action="store_true",
                         help="Forcefully wipe generated state (Config, Outputs, "
                              "YoloModels, QuantizeDataset) and create a fresh "
                              "default setup")
    args = parser.parse_args()
    on_boot(install_service=args.service, fresh=args.fresh)


if __name__ == "__main__":
    main()
