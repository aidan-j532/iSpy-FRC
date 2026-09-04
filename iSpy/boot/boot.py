import sys
import os
import re
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

logger = logging.getLogger("iSpy.boot.boot")
logging.getLogger().setLevel(logging.INFO)

_BOOT_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = Path.cwd().resolve()
_ASSETS_DIR = _PACKAGE_ROOT.parent / "assets"

_READINESS_POLL_S = 2.0
_READINESS_WAIT_TIMEOUT_S = 1200

# grab real stdout/stderr before anything swaps sys.stdout so silencing 3rd-party libs cant kill our logging
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


# vid/pid (+uvc interface) chunk of a windows-style device id - the trailing
# instance segment changes whenever a cam moves to another usb port, so
# presence checks match on this instead of the raw id
_USB_ID_RE = re.compile(r"VID_[0-9A-F]+&PID_[0-9A-F]+(?:&MI_\d+)?", re.IGNORECASE)
_IMAGE_SOURCE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _usb_signature(device_id) -> str | None:
    if not isinstance(device_id, str):
        return None
    m = _USB_ID_RE.search(device_id.upper())
    return m.group(0) if m else None


def _is_non_device_source(source) -> bool:
    # network streams and image files are not pluggable hardware - never
    # retire a camera just because no local capture device matches them
    if not isinstance(source, str):
        return False
    return "://" in source or source.lower().endswith(_IMAGE_SOURCE_EXTS)


def _camera_present(cam_cfg: dict, probed_ids: set, probed_paths: set,
                    probed_sigs: set) -> bool:
    device_id = cam_cfg.get("device_id")
    source = cam_cfg.get("source")
    if _is_non_device_source(source):
        return True
    if device_id:
        if str(device_id) in probed_ids:
            return True
        sig = _usb_signature(device_id)
        if sig and sig in probed_sigs:
            return True  # same physical cam, different usb port
    src_str = str(source)
    if src_str in probed_paths:
        return True
    if isinstance(source, str) and ("/" in source or "\\" in source):
        # path-like sources (/dev/video0 ...) are stable identities - trust
        # the filesystem over the probe result
        return os.path.exists(source)
    if device_id:
        # hardware identity is known and nothing on the system matched it -
        # the camera is genuinely unplugged
        return False
    # bare index without a device_id: index assignment shifts when cams unplug,
    # so there is no trustworthy signal here - keep the entry
    return True


def cleanup_missing_cameras(config: iSpyConfig) -> None:
    """Boot-time cleanup: retire configured cameras whose hardware is gone.

    A retired camera's full entry is stashed in Save/camera_profiles.json under
    its device_id - the same store the web ui reads when re-adding a device -
    so plugging it back in and re-creating it restores the previous settings.
    """
    cams = {
        k: v for k, v in (config.get("camera_configs") or {}).items()
        if isinstance(v, dict)
    }
    if len(cams) <= 1:
        return

    try:
        from iSpy.web.modules.cameras import CamerasModule  # same probing the web ui uses
        devices = CamerasModule({})._probe_devices()
    except Exception as exc:
        logger.warning("Boot camera cleanup skipped - device probe failed: %s", exc)
        return
    if not devices:
        logger.info(
            "Boot camera cleanup skipped - no capture devices detected, "
            "probe results not trustworthy."
        )
        return

    probed_ids = {str(d.get("device_id")) for d in devices if d.get("device_id")}
    probed_paths = {str(d.get("path")) for d in devices if d.get("path")}
    probed_sigs = {s for s in (_usb_signature(i) for i in probed_ids) if s}

    missing = [
        name for name, cfg in cams.items()
        if not _camera_present(cfg, probed_ids, probed_paths, probed_sigs)
    ]
    if not missing:
        return

    # never retire everything - validate_system and the vision core both need
    # at least one configured camera to boot
    if len(missing) >= len(cams):
        kept = next(iter(cams))
        missing = [name for name in missing if name != kept]
        logger.warning(
            "Every configured camera is missing from the system - keeping "
            "'%s' anyway so the boot still has something to run.", kept,
        )

    from iSpy.web.Backend.save_store import read, write
    profiles = read("camera_profiles", {}) or {}
    for name in missing:
        entry = cams.pop(name)
        device_id = entry.get("device_id")
        if device_id:
            profiles[device_id] = entry
        logger.info(
            "Boot camera cleanup: camera '%s' (%s) is gone from the system - "
            "removed from config, settings saved%s.",
            name, entry.get("source"),
            " under its device profile" if device_id else "",
        )
    config.set("camera_configs", cams)
    write("camera_profiles", profiles)
    config.save()


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
            return name == "root" or name == "__main__" or name.startswith("iSpy")

    formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    # bind to the real stdout captured before sys.stdout gets swapped, so silencing 3rd-party libs cant kill our logging
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
        if name != "__main__" and not name.startswith("iSpy"):
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


def on_boot(install_service: bool = False, fresh: bool = False, wait: bool = False):
    _configure_quiet_logging()
    logger.info(
        "ispy-boot python: executable=%r prefix=%r", sys.executable, sys.prefix
    )

    # in iSpy/boot/boot.py, inside on_boot(), right after setup_files(fresh=True)
    # and iSpyConfig construction, before cleanup_missing_cameras:

    if fresh:
        logger.info("boot -f: forcefully fresh installation state")
        setup_files(fresh=True)
        config_path = str(_PROJECT_ROOT / "Config" / "config.json")
        config = iSpyConfig(config_path, create=True)
        _bootstrap_default_camera(config)

        # fresh install with no prior deps installed - pull in whatever backend
        # this hardware needs (rknn wheel, onnxruntime-gpu, etc.) so first boot
        # doesn't fail on a missing import mid-pipeline-construction
        try:
            from iSpy.vision.optimizer import install_special_dependencies
            logger.info("First-boot dependency install starting (auto_install=True)...")
            install_special_dependencies(auto_install=True)
        except Exception:
            logger.exception(
                "First-boot dependency install failed - continuing boot anyway; "
                "the relevant pipeline will fall back or error clearly at runtime."
            )
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

    cleanup_missing_cameras(config)

    if not validate_system():
        raise RuntimeError("System validation failed. Aborting boot.")

    pipeline_classes = get_pipeline_classes()
    if wait:
        _wait_for_pipeline_ready(config, pipeline_classes)
    config.save(quiet=True)
    logger.info("Boot sequence complete.")

    # Start UDP announce beacon so tools/find_ispy.py can locate this board
    # even when mDNS and DHCP hostname resolution both fail.
    try:
        from iSpy.boot.announce import start_announcer
        start_announcer(daemon=True)
        logger.info("UDP announce beacon started.")
    except Exception as exc:
        logger.warning("Could not start UDP announcer (non-fatal): %s", exc)

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


def add_boot_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the flags shared between `python -m iSpy.boot.boot` and the
    `ispy` CLI (iSpy.cli), so both entry points stay in lockstep."""
    parser.add_argument("-s", "--service", action="store_true",
                         help="Install and start the watchdog service")
    parser.add_argument("-w", "--wait", action="store_true",
                         help="Wait for all pipelines to be ready before running vision")
    return parser


def main():
    if has_jetson() and _any_camera_uses_csi():
        if ensure_csi_capable_opencv(auto_fix=True):
            logger.info("OpenCV fixed - re-executing boot.py to pick it up...")
            os.execv(sys.executable, [sys.executable] + sys.argv)

    parser = argparse.ArgumentParser(description="iSpy boot sequence")
    parser = add_boot_arguments(parser)
    parser.add_argument("-f", "--fresh", action="store_true",
                         help="Forcefully wipe generated state (Config, Outputs, "
                              "YoloModels, QuantizeDataset) and create a fresh "
                              "default setup")
    args = parser.parse_args()
    on_boot(install_service=args.service, fresh=args.fresh, wait=args.wait)

    # RKNN/OpenCV native extensions segfault during Python interpreter
    # teardown on ARM.  Flush everything and hard-exit to avoid it.
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

if __name__ == "__main__":
    main()
