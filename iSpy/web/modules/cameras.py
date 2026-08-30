import base64
import copy
import glob
import json
import logging
import os
import platform
import re
import subprocess
import threading
from pathlib import Path
import cv2
import time
from flask import Response, jsonify, render_template, request, send_file
import numpy as np
from iSpy.web.Backend.WebModule import WebModule
from iSpy.web.Backend.save_store import read, write
from iSpy.web.Backend.PluginStatus import _build_vision_pipeline_payloads
from iSpy.config.iSpyConfig import (
    _CAMERA_CORE_KEYS,
    default_vision_model,
    is_model_backed_pipeline,
    get_pipeline_name,
    get_pipeline_settings,
    ensure_camera_entries_ready,
)
from iSpy.vision import calibration as cam_calibration
from iSpy.vision.Cameras import OpenCVCamera, TelloCamera
from iSpy.vision.Cameras.base import CameraOpenTimeout

COCO17_OBJECT_POINTS = [
    [ 0.00,  0.62,  0.00],  # 0  nose
    [ 0.02,  0.67,  0.00],  # 1  left_eye
    [-0.02,  0.67,  0.00],  # 2  right_eye
    [ 0.05,  0.65,  0.00],  # 3  left_ear
    [-0.05,  0.65,  0.00],  # 4  right_ear
    [ 0.17,  0.45,  0.00],  # 5  left_shoulder
    [-0.17,  0.45,  0.00],  # 6  right_shoulder
    [ 0.30,  0.22,  0.00],  # 7  left_elbow
    [-0.30,  0.22,  0.00],  # 8  right_elbow
    [ 0.35,  0.00,  0.00],  # 9  left_wrist
    [-0.35,  0.00,  0.00],  # 10 right_wrist
    [ 0.10, -0.05,  0.00],  # 11 left_hip
    [-0.10, -0.05,  0.00],  # 12 right_hip
    [ 0.10, -0.40,  0.00],  # 13 left_knee
    [-0.10, -0.40,  0.00],  # 14 right_knee
    [ 0.10, -0.75,  0.00],  # 15 left_ankle
    [-0.10, -0.75,  0.00],  # 16 right_ankle
]

_STALE_EVICT_S = 10.0
_FEED_TIMEOUT_S = 15.0

# auto-capture loop: the wizard's live feed stores a board frame by itself once
# it is complete enough and moved from every prior capture, and runs a rolling
# solve for a live RMS readout - the user just keeps moving the board around in
# front of the camera until they pause and calibrate intrinsics
_AUTO_CAPTURE_MIN_SOLVE = 6
_AUTO_COVERAGE_MIN = 0.6
_AUTO_DIVERSITY_PX = 12.0

# live calibration feed tuning: board detection is by far the most expensive
# step, so it runs on a downscaled work copy (corners are mapped back to the
# full-res pixel space the solve expects) and the served overlay feed is
# capped in width so JPEG encode stays cheap. Detection runs at a higher res
# than before (subpixel corner accuracy scales with detection resolution).
_DETECT_MAX_DIM = 1280
_FEED_MAX_DIM = 1280
# overlay feeds hold their opening chunk until the detector's first tick (or
# this deadline) so the first frame served is already annotated
_CALIB_FEED_WARMUP_S = 1.5
# the rolling solve is the other big periodic CPU spike - left unchecked it
# re-runs a full bundle solve on every captured frame (every ~0.25-0.5s while
# the board is being moved), which stutters the live feed. Throttle it and
# bound the number of frames the live RMS preview is computed over.
_AUTO_SOLVE_INTERVAL_S = 1.5
_AUTO_SOLVE_MAX_CAPTURES = 10
# how many captured-frame overlays to draw on each streamed frame (see
# _draw_captured_overlays - captures never stop accumulating under auto capture)
_MAX_OVERLAYS_DRAWN = 20

logger = logging.getLogger(__name__)

# legacy alias keys - dropped when the canonical key is present so old configs
# dont carry dead twins around forever
_SETTING_ALIASES = {"quantized": "quantize", "auto_opt": "optimize"}

# image tuning knobs (sliders in the camera lightbox). these live on the cam
# entry like the other capture keys; the live Camera applies them immediately
_TUNING_KEYS = (
    "brightness", "contrast",
    "saturation", "white_balance", "tint", "gamma",
    "exposure_time", "gain",
)
_TUNING_DEFAULTS = {
    "brightness": 0,
    "contrast": 0,
    "saturation": 0,
    "white_balance": 0,
    "tint": 0,
    "gamma": 1.0,
    "exposure_time": 100,
    "gain": 200,
}


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _camera_calibrated(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    calib = entry.get("calibration") or {}
    return bool(
        calib.get("camera_matrix") and calib.get("dist_coeffs") is not None
    ) or bool(calib.get("focal_length_pixels")) or _to_float(calib.get("fov")) > 0


def _decode_base64_frame(image_b64):
    if isinstance(image_b64, str) and image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[-1]
    try:
        raw = base64.b64decode(image_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _pipeline_schema_keys(pipeline_name: str) -> set:
    try:
        from iSpy.vision.pipelines import get_pipeline_classes
        cls = get_pipeline_classes().get(pipeline_name)
    except Exception:
        cls = None
    if cls is None:
        return set()
    try:
        schema = cls.config_schema() or {}
    except Exception:
        return set()
    return set(schema) | {"vision_model"}


def _prune_stale_pipeline_settings(entry: dict) -> None:
    if not isinstance(entry, dict):
        return
    pipeline_name = get_pipeline_name(entry)
    settings = get_pipeline_settings(entry)
    if not settings:
        return
    try:
        from iSpy.vision.pipelines import get_pipeline_classes
        foreign = set()
        for name, cls in get_pipeline_classes().items():
            if name == pipeline_name:
                continue
            try:
                foreign |= set((cls.config_schema() or {}).keys())
            except Exception:
                continue
    except Exception:
        foreign = set()
    allowed = _pipeline_schema_keys(pipeline_name)
    for key in [k for k in settings if k in foreign and k not in allowed]:
        del settings[key]
    for alias, canonical in _SETTING_ALIASES.items():
        if alias in settings and canonical in settings:
            del settings[alias]


def _vision_model_target_format(settings: dict) -> str:
    fmt = str(settings.get("target_format") or "auto").strip().lower()
    if fmt and fmt != "auto":
        return fmt
    try:
        from iSpy.vision.pipelines.object_detection import ObjectDetectionCamera
        return ObjectDetectionCamera.recommended_format()
    except Exception:
        return "onnx"


def _vision_model_rel_path(path) -> str:
    p = Path(str(path or ""))
    if p.is_absolute():
        try:
            return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return str(p)
    return p.as_posix()


def _resolve_vision_model_files(settings: dict) -> None:
    vm = settings.get("vision_model")
    if not isinstance(vm, dict):
        return
    src = str(vm.get("source_pt") or vm.get("file_path") or "")
    if not src.lower().endswith(".pt"):
        return
    try:
        from iSpy.vision.optimizer import existing_artifact_for
    except Exception:
        return
    pt_rel = _vision_model_rel_path(src)
    artifact = existing_artifact_for(Path(src), _vision_model_target_format(settings))
    vm["source_pt"] = pt_rel
    vm["file_path"] = artifact or pt_rel


def _windows_cameras_from_registry():
    devices = []
    try:
        import winreg
    except ImportError:
        return devices

    class_key = r"SYSTEM\CurrentControlSet\Control\DeviceClasses\{e5323777-f976-4f5b-9b55-b94699c46e44}"
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, class_key)
    except OSError:
        return devices

    try:
        subkey_pos = 0
        camera_index = 0
        while True:
            try:
                iface = winreg.EnumKey(key, subkey_pos)
            except OSError:
                break
            subkey_pos += 1
            if "#{" not in iface:
                continue
            raw_hw = iface[:iface.find("#{")]
            # USB cameras expose two UVC interfaces (MI_00 video, MI_01 metadata)
            # - strip &MI_XX so they collapse to the same physical device
            dedup_key = re.sub(r"&MI_\w+", "", raw_hw, flags=re.IGNORECASE)
            devices.append({
                "index": camera_index,
                "name": _windows_camera_name(iface) or f"Camera {camera_index}",
                "hw_id": raw_hw,
                "dedup_key": dedup_key,
            })
            camera_index += 1
    finally:
        winreg.CloseKey(key)
    return devices


def _windows_camera_name(iface):
    try:
        import winreg
    except ImportError:
        return None

    end = iface.find("#{")
    if end == -1:
        return None
    body = iface[:end]
    for prefix in ("##?#", "#?#"):
        if body.startswith(prefix):
            body = body[len(prefix):]
            break

    def _read_name(enum_path):
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, enum_path)
        except OSError:
            return None
        try:
            for value in ("FriendlyName", "DriverDesc", "DeviceDesc"):
                try:
                    name, _ = winreg.QueryValueEx(key, value)
                except OSError:
                    continue
                if not isinstance(name, str):
                    continue
                if name.startswith("@"):
                    # "inf_path,%desc%;Friendly Name" -> drop the locator, keep the friendly part
                    name = name.rsplit(";", 1)[-1] if ";" in name else ""
                name = name.strip()
                if name:
                    return name
        finally:
            winreg.CloseKey(key)
        return None

    # Each # in the interface path maps to \ in the registry Enum path:
    #   USB#VID_xxx&PID_yyy&MI_00#instance_id -> Enum\USB\VID_xxx&PID_yyy&MI_00\instance_id
    parts = body.split("#")
    if len(parts) < 3:
        return None
    enum_path = "SYSTEM\\CurrentControlSet\\Enum\\" + "\\".join(parts)
    hw_id = "\\".join(parts[:2])

    name = _read_name(enum_path)
    if name and name.lower() not in _GENERIC_WINDOWS_NAMES:
        return name

    # generic iface name -> real one usually lives on the parent usb node, not the uvc iface
    m = re.match(r"^USB\\(VID_\w+&PID_\w+)(?:&MI_\w+)?$", hw_id, re.IGNORECASE)
    if m:
        parent_hw = f"USB\\{m.group(1)}"
        parent_root = f"SYSTEM\\CurrentControlSet\\Enum\\{parent_hw}"
        try:
            parent_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, parent_root)
        except OSError:
            parent_key = None
        if parent_key is not None:
            try:
                try:
                    sub, _ = winreg.EnumKey(parent_key, 0)
                except OSError:
                    sub = None
                if sub:
                    parent_name = _read_name(f"{parent_root}\\{sub}")
                    if parent_name:
                        return parent_name
            finally:
                winreg.CloseKey(parent_key)
    return name or None


# names windows hands out when it has nothing better - treat as unknown and keep digging
_GENERIC_WINDOWS_NAMES = {
    "", "usb video device", "usb camera", "camera", "uvc camera",
    "video capture device", "usb2.0 camera",
}

# v4l2 capability bits (videodev2.h) - caps u32 lives at offset 84 in QUERYCAP
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_VIDEO_CAPTURE_MPLANE = 0x00001000
_V4L2_CAP_VIDEO_M2M = 0x00004000
_VIDIOC_QUERYCAP = 0x80685600


def _v4l2_caps(video_path):
    try:
        import fcntl
        import struct
    except ImportError:
        return 0
    for mode in (os.O_RDWR, os.O_RDONLY):
        try:
            fd = os.open(video_path, mode | getattr(os, "O_NONBLOCK", 0))
        except OSError:
            continue
        try:
            buf = fcntl.ioctl(fd, _VIDIOC_QUERYCAP, b"\0" * 104)
            return struct.unpack_from("<I", buf, 84)[0]
        except OSError:
            continue
        finally:
            os.close(fd)
    return 0


_LINUX_NON_CAMERA_KEYWORDS = {"codec", "encode", "decode", "radio", "loopback"}


def _linux_is_known_non_camera(video_path):
    name = _linux_sysfs_name(video_path)
    if name:
        lower = name.lower()
        if any(kw in lower for kw in _LINUX_NON_CAMERA_KEYWORDS):
            return True
    # e.g. /dev/video-dec0, /dev/video-enc0 — the part after "video-" starts
    # with a non-digit which real camera nodes never do
    basename = os.path.basename(video_path).lower()
    if basename.startswith("video-") and len(basename) > 6 and not basename[6].isdigit():
        return True
    return False


def _linux_is_capture_node(video_path):
    caps = _v4l2_caps(video_path)
    if caps == 0:
        return None
    if caps & _V4L2_CAP_VIDEO_M2M:
        # m2m codecs (bcm2835-codec-decode) claim the capture bit but aint cameras
        return False
    return bool(caps & (_V4L2_CAP_VIDEO_CAPTURE | _V4L2_CAP_VIDEO_CAPTURE_MPLANE))


def _linux_device_key(video_path):
    m = re.search(r"/video(?P<num>\d+)$", video_path)
    if not m:
        return video_path
    sysfs = f"/sys/class/video4linux/video{m.group('num')}/device"
    try:
        real = os.path.realpath(sysfs)
    except OSError:
        return None
    suffix = f"video4linux/video{m.group('num')}"
    if real.endswith(suffix):
        real = real[: -len(suffix)]
    return real


def _linux_sysfs_name(video_path):
    m = re.search(r"/video(\d+)$", video_path)
    if not m:
        return None
    try:
        with open(
            f"/sys/class/video4linux/video{m.group(1)}/name",
            encoding="utf-8", errors="replace",
        ) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _linux_device_groups():
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=3,
        )
    except Exception:
        result = None
    if result is not None and result.returncode == 0:
        groups = {}
        current_name = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                current_name = None
                continue
            if line.endswith(":") and not line.lower().startswith("/dev/"):
                current_name = line.rstrip(":")
                continue
            if line.startswith("/dev/video") and current_name:
                groups.setdefault(current_name, []).append(line)
        return [(name, nodes) for name, nodes in groups.items()]

    groups = {}
    for path in sorted(glob.glob("/dev/video*")):
        key = _linux_device_key(path) or path
        groups.setdefault(key, []).append(path)
    return [(None, nodes) for nodes in groups.values()]

class CamerasModule(WebModule):
    plugin_name = "cameras"

    def __init__(self, context: dict):
        super().__init__(context)
        self.lock = threading.Lock()
        self.frames: dict[str, "np.ndarray"] = {}
        self.dims: dict[str, tuple[int, int]] = {}
        self.last_seen: dict[str, float] = {}
        self.sources: dict[str, str] = {}
        self.live_cameras: dict[str, object] = {}
        self._discover_cache: list = []
        self._discover_cache_ts: float = 0.0
        self._discover_lock = threading.Lock()
        self.calib_sessions: dict[str, dict] = {}
        self.calib_lock = threading.Lock()
        self._auto_discover_stop = threading.Event()
        self._auto_discover_thread: threading.Thread | None = None
        self._sse_lock = threading.Lock()
        self._sse_clients: list = []

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/cameras", "cameras_page", lambda: render_template("cameras.html"))
        flask_app.add_url_rule("/api/cameras", "api_cameras", self._api_cameras)
        flask_app.add_url_rule("/api/cameras/discover", "api_cameras_discover", self._discover)
        flask_app.add_url_rule("/api/cameras/sources", "api_cameras_sources", self._sources)
        flask_app.add_url_rule("/video/<camera_name>", "video_feed", self._video_feed)
        flask_app.add_url_rule("/api/cameras/config", "api_cameras_config_add", self._add_camera, methods=["POST"])
        flask_app.add_url_rule("/api/cameras/config/<cam_name>", "api_cameras_config_get", self._get_camera, methods=["GET"])
        flask_app.add_url_rule("/api/cameras/config/<cam_name>", "api_cameras_config_update", self._update_camera, methods=["PUT"])
        flask_app.add_url_rule("/api/cameras/config/<cam_name>", "api_cameras_config_delete", self._remove_camera, methods=["DELETE"])
        flask_app.add_url_rule("/api/cameras/profile/<device_id>", "api_cameras_profile", self._get_profile, methods=["GET"])
        flask_app.add_url_rule("/api/cameras/tuning/<cam_name>", "api_cameras_tuning_get", self._tuning_get, methods=["GET"])
        flask_app.add_url_rule("/api/cameras/tuning/<cam_name>", "api_cameras_tuning_set", self._tuning_set, methods=["POST"])
        flask_app.add_url_rule("/api/vision_pipelines", "api_vision_pipelines", self._vision_pipelines)
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>", "api_cameras_calibration_get", self._calibration_get, methods=["GET"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>", "api_cameras_calibration_reset", self._calibration_reset, methods=["DELETE"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/focal", "api_cameras_calibration_focal", self._calibration_focal, methods=["POST"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/charuco/capture", "api_cameras_charuco_capture", self._charuco_capture, methods=["POST"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/charuco/status", "api_cameras_charuco_status", self._charuco_status)
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/charuco", "api_cameras_charuco_clear", self._charuco_clear, methods=["DELETE"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/charuco/finish", "api_cameras_charuco_finish", self._charuco_finish, methods=["POST"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/feed", "api_cameras_calibration_feed", self._calibration_feed)
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/mode", "api_cameras_calibration_mode", self._calibration_mode, methods=["POST"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/heartbeat", "api_cameras_calibration_heartbeat", self._calibration_heartbeat, methods=["POST"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/auto", "api_cameras_auto_set", self._auto_set, methods=["POST"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/auto/status", "api_cameras_auto_status", self._auto_status)
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/pnp", "api_cameras_pnp_get", self._pnp_get, methods=["GET"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/pnp", "api_cameras_pnp_save", self._pnp_save, methods=["POST"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/pnp", "api_cameras_pnp_clear", self._pnp_clear, methods=["DELETE"])
        flask_app.add_url_rule("/api/cameras/calibration/board", "api_calibration_board_pdf", self._calibration_board_pdf)
        flask_app.add_url_rule("/api/cameras/events", "api_cameras_events", self._sse_stream)

    def start(self):
        if self._auto_discover_thread is None or not self._auto_discover_thread.is_alive():
            self._auto_discover_stop.clear()
            self._auto_discover_thread = threading.Thread(
                target=self._auto_discover_loop, daemon=True, name="AutoDiscover"
            )
            self._auto_discover_thread.start()

    def stop(self):
        self._auto_discover_stop.set()
        if self._auto_discover_thread and self._auto_discover_thread.is_alive():
            self._auto_discover_thread.join(timeout=3)

    def _auto_discover_loop(self):
        last_device_ids: set = set()
        while not self._auto_discover_stop.is_set():
            self._auto_discover_stop.wait(10.0)
            if self._auto_discover_stop.is_set():
                break
            try:
                devices = self._probe_devices()
                current_ids = {d.get("device_id") or d.get("path") for d in devices}
                if current_ids != last_device_ids:
                    new_ids = current_ids - last_device_ids
                    removed_ids = last_device_ids - current_ids
                    last_device_ids = current_ids
                    self._push_sse({
                        "type": "cameras_changed",
                        "devices": devices,
                        "new": [d for d in devices if (d.get("device_id") or d.get("path")) in new_ids],
                        "removed_ids": list(removed_ids),
                    })
            except Exception:
                pass

    def _sse_stream(self):
        def generate():
            q: list = []
            with self._sse_lock:
                self._sse_clients.append(q)
            try:
                while True:
                    while q:
                        payload = q.pop(0)
                        yield f"data: {json.dumps(payload)}\n\n"
                    time.sleep(0.1)
            except GeneratorExit:
                pass
            finally:
                with self._sse_lock:
                    if q in self._sse_clients:
                        self._sse_clients.remove(q)
        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def _push_sse(self, payload: dict):
        with self._sse_lock:
            for q in self._sse_clients:
                q.append(payload)

    def _camera_display_name(self, cam, fallback: str = "camera") -> str:
        if hasattr(cam, "config") and cam.config is not None:
            name = cam.config.get("name") if hasattr(cam.config, "get") else None
            if name:
                return str(name)
        source = getattr(cam, "source", None)
        if source is not None:
            return str(source)
        return fallback

    def _pnp_model_info(self, entry):
        settings = get_pipeline_settings(entry) or {}
        vm = settings.get("vision_model")
        if not isinstance(vm, dict) or not vm.get("file_path"):
            return None, None, "Camera has no vision_model configured"
        from iSpy.vision.metadata import read_metadata
        model_path = Path(vm["file_path"])
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        meta = read_metadata(model_path) or {}
        if meta.get("task") != "pose":
            return vm, None, "Model is not a pose model - PnP needs keypoints (task=pose)"
        kpt_shape = meta.get("kpt_shape")
        if not kpt_shape:
            return vm, None, "Model metadata has no kpt_shape - re-run boot to regenerate the sidecar"
        return vm, int(kpt_shape[0]), None

    def _pnp_get(self, cam_name):
        cams, key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404
        vm, num_kpts, err = self._pnp_model_info(entry)
        calib = entry.get("calibration") or {}
        has_intrinsics = bool(calib.get("camera_matrix") and calib.get("dist_coeffs"))
        existing_pnp = vm.get("pnp") if isinstance(vm, dict) else None

        # Auto-fill COCO17 defaults for shipped pose models
        default_object_points = None
        if num_kpts == 17 and not existing_pnp:
            fp = (vm or {}).get("file_path", "") or ""
            sp = (vm or {}).get("source_pt", "") or ""
            if "_default_pose" in fp or "_default_pose" in sp:
                default_object_points = COCO17_OBJECT_POINTS

        return jsonify(
            model_error=err,
            num_keypoints=num_kpts,
            has_intrinsics=has_intrinsics,
            pnp=existing_pnp,
            default_object_points=default_object_points,
        )

    def _pnp_save(self, cam_name):
        data = request.get_json(force=True) or {}
        cams, key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404

        vm, num_kpts, err = self._pnp_model_info(entry)
        if err:
            return jsonify(error=err), 400

        calib = entry.get("calibration") or {}
        camera_matrix = calib.get("camera_matrix")
        dist_coeffs = calib.get("dist_coeffs")
        if not camera_matrix or dist_coeffs is None:
            return jsonify(error="No ChArUco intrinsics on this camera yet - run that calibration first"), 400

        object_points = data.get("object_points")
        if not isinstance(object_points, list) or len(object_points) != num_kpts:
            return jsonify(error=f"object_points must have exactly {num_kpts} [x,y,z] entries"), 400
        for pt in object_points:
            if not (isinstance(pt, list) and len(pt) == 3 and all(isinstance(v, (int, float)) for v in pt)):
                return jsonify(error="Each object_points entry must be [x, y, z] numbers"), 400

        mode = data.get("mode", "flexible")
        if mode not in ("flexible", "rigid"):
            return jsonify(error="mode must be 'flexible' or 'rigid'"), 400
        try:
            min_keypoint_conf = float(data.get("min_keypoint_conf", 0.5))
        except (TypeError, ValueError):
            return jsonify(error="min_keypoint_conf must be a number"), 400

        settings = get_pipeline_settings(entry)
        vm_entry = settings.get("vision_model")
        if not isinstance(vm_entry, dict):
            return jsonify(error="Camera has no vision_model configured"), 400

        vm_entry["pnp"] = {
            "object_points": object_points,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "min_keypoint_conf": min_keypoint_conf,
            "mode": mode,
        }

        config = self.context["config"]
        config.set("camera_configs", cams)
        config.save()
        return jsonify(success=True, pnp=vm_entry["pnp"], note="Restart vision to apply.")

    def _pnp_clear(self, cam_name):
        cams, key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404
        settings = get_pipeline_settings(entry)
        vm_entry = settings.get("vision_model")
        if isinstance(vm_entry, dict):
            vm_entry.pop("pnp", None)
        config = self.context["config"]
        config.set("camera_configs", cams)
        config.save()
        return jsonify(success=True)

    def _camera_aliases(self, cam) -> set[str]:
        aliases = {self._camera_display_name(cam, "camera")}
        source = getattr(cam, "source", None)
        if source is not None:
            aliases.add(str(source))
        device_id = getattr(cam, "device_id", None)
        if device_id:
            aliases.add(str(device_id))
        return aliases

    def update(self, frame_data: dict):
        frame = frame_data.get("frame")
        if frame is None:
            return
        cameras = frame_data.get("cameras") or []
        cam_by_name = {
            self._camera_display_name(cam, str(getattr(cam, "source", ""))): cam
            for cam in cameras
        }
        self.live_cameras = cam_by_name
        per_cam = frame_data.get("camera_frames")
        now = time.monotonic()
        with self.lock:
            if per_cam:
                for name, f in per_cam.items():
                    if f is None:
                        continue
                    matched_cams = []
                    if name in cam_by_name:
                        matched_cams.append(cam_by_name[name])
                    else:
                        for cam in cameras:
                            if name in self._camera_aliases(cam):
                                matched_cams.append(cam)
                                break
                    if not matched_cams and cameras:
                        matched_cams = [cameras[0]]

                    for cam in matched_cams:
                        display_name = self._camera_display_name(cam, str(name))
                        self.frames[display_name] = f
                        self.dims[display_name] = f.shape[1], f.shape[0]
                        self.last_seen[display_name] = now
                        self.sources[display_name] = self._device_key(cam)
            elif cameras:
                cam = cameras[0]
                name = self._camera_display_name(cam, "camera_1")
                self.frames[name] = frame
                self.dims[name] = frame.shape[1], frame.shape[0]
                self.last_seen[name] = now
                self.sources[name] = self._device_key(cam)
            else:
                self.frames["camera_1"] = frame
                self.dims["camera_1"] = frame.shape[1], frame.shape[0]
                self.last_seen["camera_1"] = now

            self._evict_stale(now)

    def _device_key(self, cam) -> str:
        dev_id = getattr(cam, "device_id", None)
        if dev_id:
            return dev_id
        return str(getattr(cam, "source", ""))

    def _get_profile(self, device_id):
        profiles = read("camera_profiles", {})
        return jsonify(profile=profiles.get(device_id))

    # ------------------------------------------------------------------
    # Camera image tuning (live sliders in the lightbox)
    # ------------------------------------------------------------------

    def _tuning_values(self, entry: dict) -> dict:
        values = dict(_TUNING_DEFAULTS)
        for key in _TUNING_KEYS:
            if key in entry:
                values[key] = entry[key]
        return values

    def _live_cam_for(self, key: str, cam_name: str):
        for lookup in (key, cam_name):
            cam = self.live_cameras.get(lookup)
            if cam is not None:
                return cam
        return None

    def _tuning_get(self, cam_name):
        cams, key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404
        values = self._tuning_values(entry)
        live = None
        cam = self._live_cam_for(key, cam_name)
        if cam is not None and hasattr(cam, "get_image_adjustments"):
            try:
                live = cam.get_image_adjustments()
            except Exception:
                live = None
        return jsonify(camera=key, values=values, live=live)

    def _tuning_set(self, cam_name):
        data = request.get_json(force=True) or {}
        cams, key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404
        cleaned = {}
        for k in _TUNING_KEYS:
            if k in data:
                cleaned[k] = data[k]
        if not cleaned:
            return jsonify(error="No tuning values provided"), 400
        for k, v in cleaned.items():
            if v is None:
                entry.pop(k, None)
            else:
                entry[k] = v
        config = self.context["config"]
        config.set("camera_configs", cams)
        config.save()
        cam = self._live_cam_for(key, cam_name)
        if cam is not None and hasattr(cam, "set_image_adjustments"):
            try:
                applied = cam.set_image_adjustments(cleaned)
            except Exception as exc:
                return jsonify(error=f"Failed to apply tuning: {exc}"), 500
        else:
            applied = self._tuning_values(entry)
            return jsonify(
                success=True,
                note="Saved to config. Restart vision to apply - camera is not live.",
                applied=applied,
            )
        return jsonify(
            success=True,
            note="Saved - applies immediately to the live feed.",
            applied=applied,
        )

    def _vision_pipelines(self):
        return jsonify(pipelines=_build_vision_pipeline_payloads())

    # ------------------------------------------------------------------
    # Camera calibration (web wizard)
    # ------------------------------------------------------------------

    def _save_calibration(self, cam_name: str, calib_dict: dict):
        config = self.context["config"]
        cams, key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return None
        merged = dict(entry.get("calibration") or {})
        merged.update(calib_dict)
        entry["calibration"] = merged
        config.set("camera_configs", cams)
        config.save()
        return merged

    def _calibration_get(self, cam_name):
        cams, key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404
        with self.calib_lock:
            session = self.calib_sessions.get(key) or {}
        return jsonify(
            camera=key,
            pipeline=get_pipeline_name(entry),
            calibration=entry.get("calibration") or {},
            charuco_captures=len(session.get("charuco_captures", [])),
            charuco_pattern=session.get("charuco_pattern", list(cam_calibration.DEFAULT_CHARUCO_PATTERN)),
            charuco_dict=session.get("charuco_dict"),
            auto_enabled=bool(session.get("auto_enabled", True)),
        )

    def _calibration_reset(self, cam_name):
        cams, key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404
        with self.calib_lock:
            self.calib_sessions.pop(key, None)
        saved = self._save_calibration(key, {
            "distance": 0, "game_piece_size": 0, "size": 0, "fov": 0,
        })
        for stale in ("camera_matrix", "dist_coeffs", "resolution", "rms",
                      "count", "focal_length_pixels"):
            saved.pop(stale, None)
        config = self.context["config"]
        entry["calibration"] = saved
        config.set("camera_configs", cams)
        config.save()
        return jsonify(success=True, calibration=saved)

    def _calibration_board_pdf(self):
        assets = Path(__file__).resolve().parent.parent.parent / "assets"
        pdf_path = assets / "calibration_board.pdf"
        if not pdf_path.exists():
            cam_calibration.generate_calibration_board_pdf(str(pdf_path))
        if not pdf_path.exists():
            return jsonify(error="Calibration board PDF not available"), 404
        return send_file(str(pdf_path), mimetype="application/pdf",
                         as_attachment=True, download_name="calibration_board.pdf")

    def _calibration_focal(self, cam_name):
        data = request.get_json(force=True) or {}
        cams, key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404
        real_size = _to_float(data.get("real_size"))
        distance = _to_float(data.get("distance"))
        pixel_height = _to_float(data.get("pixel_height"))
        frame_width = int(_to_float(data.get("frame_width")))
        if real_size <= 0 or distance <= 0 or pixel_height <= 0 or frame_width <= 0:
            return jsonify(
                error="Object size, distance, measured pixel height and frame width must all be positive"
            ), 400
        focal_px = cam_calibration.focal_from_object(real_size, distance, pixel_height)
        fov_deg = cam_calibration.fov_from_focal(focal_px, frame_width)
        saved = self._save_calibration(key, {
            "distance": round(distance, 4),
            "game_piece_size": round(real_size, 4),
            "size": round(pixel_height, 2),
            "fov": round(fov_deg, 3),
            "focal_length_pixels": round(focal_px, 2),
        })
        return jsonify(
            success=True,
            focal_length_px=round(focal_px, 2),
            fov_deg=round(fov_deg, 3),
            calibration=saved,
        )


    def _calibration_mode(self, cam_name):
        data = request.get_json(force=True) or {}
        active = bool(data.get("active", True))
        cam = self.live_cameras.get(cam_name)
        if cam is not None and hasattr(cam, "set_calibration"):
            cam.set_calibration(active)
        return jsonify(success=True)

    def _calibration_heartbeat(self, cam_name):
        cam = self.live_cameras.get(cam_name)
        if cam is not None and hasattr(cam, "calibration_heartbeat"):
            cam.calibration_heartbeat()
        return jsonify(success=True)

    # ------------------------------------------------------------------
    # Auto capture: the live feed stores diverse board frames by itself and
    # runs a rolling solve for a live RMS readout. It never stops or saves on
    # its own - it keeps going until the user pauses, then the wizard's
    # "Calibrate intrinsics" button saves over the captured frames.
    # ------------------------------------------------------------------

    def _auto_set(self, cam_name):
        data = request.get_json(force=True) or {}
        _, key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404
        key = key or cam_name
        enabled = bool(data.get("enabled", True))
        with self.calib_lock:
            session = self.calib_sessions.setdefault(key, {})
            session["auto_enabled"] = enabled
            if enabled:
                session["auto_state"] = "capturing"
                session["auto_msg"] = ""
                session["auto_rms"] = None
        return jsonify(success=True, enabled=enabled)

    def _auto_status(self, cam_name):
        _, key, _ = self._find_camera_entry(cam_name)
        key = key or cam_name
        with self.calib_lock:
            session = self.calib_sessions.get(key) or {}
            payload = {
                "enabled": bool(session.get("auto_enabled", False)),
                "state": session.get("auto_state", "idle"),
                "rms": session.get("auto_rms"),
                "message": session.get("auto_msg", ""),
                "captured": {
                    "charuco": len(session.get("charuco_captures") or []),
                },
            }
        return jsonify(**payload)

    def _auto_hint(self, cam_name, msg):
        _, key, _ = self._find_camera_entry(cam_name)
        key = key or cam_name
        with self.calib_lock:
            session = self.calib_sessions.get(key)
            if session and session.get("auto_enabled"):
                session["auto_msg"] = msg

    def _auto_consider(self, key, kind, gray, corners, ids=None, pattern=None, dict_id=None, _synchronous=True):
        need_solve = False
        now = time.monotonic()
        with self.calib_lock:
            session = self.calib_sessions.get(key)
            if not session or not session.get("auto_enabled"):
                return
            if pattern is None:
                pattern = tuple(
                    session.get("charuco_pattern", list(cam_calibration.DEFAULT_CHARUCO_PATTERN))
                )
            pattern = tuple(int(x) for x in pattern[:2])
            expected = cam_calibration.expected_charuco_corners(pattern)
            if cam_calibration.capture_coverage(corners, expected) < _AUTO_COVERAGE_MIN:
                session["auto_msg"] = (
                    "Board too small or partly out of frame - fill more of the view"
                )
                return
            captures = session.setdefault("charuco_captures", [])
            if not cam_calibration.frame_diverse(corners, captures, _AUTO_DIVERSITY_PX):
                session["auto_msg"] = (
                    f"Same pose as a captured frame - keep moving the board ({len(captures)} frames)"
                )
                return
            if ids is None or len(corners) < 4:
                return
            captures.append((gray, corners, ids))
            session.setdefault("charuco_overlays", []).append(
                (corners, ids, None, None, cam_calibration.random_overlay_color())
            )
            session["charuco_pattern"] = list(pattern)
            session["auto_rms"] = None
            session["auto_state"] = "capturing"
            session["auto_msg"] = (
                f"Auto-captured {len(captures)} frames - keep moving the board"
            )
            if len(captures) >= _AUTO_CAPTURE_MIN_SOLVE:
                if not session.get("auto_solving") and (
                    now - session.get("last_solve_at", 0.0) >= _AUTO_SOLVE_INTERVAL_S
                ):
                    session["auto_solving"] = True
                    session["last_solve_at"] = now
                    need_solve = True
        if not need_solve:
            return
        if _synchronous:
            self._auto_solve_and_finalize(key, kind, pattern, dict_id)
        else:
            threading.Thread(
                target=self._auto_solve_and_finalize,
                args=(key, kind, pattern, dict_id),
                daemon=True,
            ).start()

    def _auto_solve_and_finalize(self, key, kind, pattern, dict_id):
        try:
            result = self._auto_solve(key, kind, pattern, dict_id)
        except Exception as exc:
            logger.warning("Auto calibration solve failed: %s", exc)
            result = None
        finally:
            with self.calib_lock:
                session = self.calib_sessions.get(key)
                if session:
                    session["auto_solving"] = False
        if result is None:
            with self.calib_lock:
                session = self.calib_sessions.get(key)
                if session:
                    session["auto_state"] = "capturing"
                    session["auto_msg"] = (
                        "Captured frames solve poorly - keep moving the board through more angles"
                    )

    def _auto_solve(self, key, kind, pattern, dict_id):
        with self.calib_lock:
            session = self.calib_sessions.get(key) or {}
            captures = list(session.get("charuco_captures") or [])
            kwargs = {}
            if dict_id is not None:
                kwargs["dictionary_id"] = dict_id
        if len(captures) < 3:
            return None
        captures = captures[-_AUTO_SOLVE_MAX_CAPTURES:]
        try:
            result = cam_calibration.calibrate_charuco(captures, pattern, **kwargs)
        except cv2.error as exc:
            logger.warning("Auto calibration failed: %s", exc)
            return None
        if result is None:
            return None
        with self.calib_lock:
            session = self.calib_sessions.get(key)
            if session:
                session["auto_rms"] = result.get("rms")
        return result

    def _calibration_feed(self, cam_name):
        overlay = request.args.get("overlay", "")
        pattern_arg = request.args.get("pattern")
        dict_arg = request.args.get("dict")
        pattern = None
        if pattern_arg:
            try:
                pattern = tuple(int(p) for p in pattern_arg.split(",")[:2])
            except ValueError:
                pattern = None
        dict_id = None
        if dict_arg:
            try:
                dict_id = int(dict_arg)
            except (TypeError, ValueError):
                dict_id = None
        return Response(
            self._generate_calibration(cam_name, overlay=overlay, pattern=pattern, dict_id=dict_id),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _charuco_session_layout(self, cam_name):
        _, key, _ = self._find_camera_entry(cam_name)
        key = key or cam_name
        with self.calib_lock:
            session = self.calib_sessions.get(key) or {}
            p = session.get("charuco_pattern")
            d = session.get("charuco_dict")
        return tuple(p) if p else None, d

    def _remember_charuco_layout(self, cam_name, pattern, dictionary_id):
        if pattern is None:
            return
        _, key, _ = self._find_camera_entry(cam_name)
        key = key or cam_name
        with self.calib_lock:
            session = self.calib_sessions.setdefault(key, {})
            session["charuco_pattern"] = [int(pattern[0]), int(pattern[1])]
            if dictionary_id is not None:
                session["charuco_dict"] = int(dictionary_id)

    def _draw_captured_overlays(self, frame, cam_name, overlay):
        _, key, _ = self._find_camera_entry(cam_name)
        key = key or cam_name
        with self.calib_lock:
            session = self.calib_sessions.get(key) or {}
            layers = session.get("charuco_overlays", [])
            # captures accumulate for as long as auto capture is enabled, so
            # cap what gets drawn per streamed frame - drawing every capture
            # ever made would eventually starve the feed thread
            layers = list(layers[-_MAX_OVERLAYS_DRAWN:])
        if not layers:
            return frame
        out = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for layer in layers:
            corners, ids, mc, mi, color = layer
            cam_calibration.draw_charuco_into(out, corners, ids, mc, mi, color=color)
        return out

    def _detect_workframe(self, frame, max_dim=_DETECT_MAX_DIM):
        h, w = frame.shape[:2]
        largest = max(w, h)
        if largest <= max_dim:
            return frame, 1.0
        scale = max_dim / float(largest)
        work = cv2.resize(
            frame,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        )
        return work, scale

    def _rescale_corners(self, corners, factor):
        if corners is None:
            return None
        return (np.asarray(corners, dtype=np.float32) * factor).astype(np.float32)

    def _generate_calibration(self, cam_name, overlay="", pattern=None, dict_id=None):
        last_frame_time = time.monotonic()
        result_lock = threading.Lock()
        latest = {"detect": None}
        stop_evt = threading.Event()

        def detection_worker():
            last_detect = 0.0
            corners, ids, marker_corners, marker_ids = None, None, None, None
            while not stop_evt.is_set():
                cam = self.live_cameras.get(cam_name)
                frame = None
                if cam is not None and hasattr(cam, "get_raw_frame"):
                    frame = cam.get_raw_frame()
                if frame is None:
                    stop_evt.wait(0.05)
                    continue
                now = time.monotonic()
                result = None
                if overlay == "charuco" and now - last_detect >= 0.1:
                    last_detect = now
                    work, scale = self._detect_workframe(frame)
                    session_pattern, session_dict = self._charuco_session_layout(cam_name)
                    effective_dict = dict_id if dict_id is not None else session_dict
                    candidates = []
                    if session_pattern is not None:
                        candidates.append(tuple(session_pattern))
                    if pattern is not None and len(pattern) == 2 and tuple(pattern) not in candidates:
                        candidates.append(tuple(pattern))
                    found = False
                    matched_pattern = None
                    matched_dict = effective_dict
                    for cand in candidates:
                        kwargs = {}
                        if effective_dict is not None:
                            kwargs["dictionary_id"] = effective_dict
                        found, corners, ids, marker_corners, marker_ids, _ = (
                            cam_calibration.detect_charuco(
                                work, cand[0], cand[1], **kwargs
                            )
                        )
                        if found:
                            matched_pattern = cand
                            break
                    if not found:
                        # nothing matched the session/requested layout - sweep
                        # common layouts and dictionaries so a board printed
                        # outside the defaults is still picked up live
                        afound, acorners, aids, amc, ami, _, apat, adict = (
                            cam_calibration.detect_charuco_auto(
                                work,
                                preferred_pattern=candidates[0] if candidates else None,
                                preferred_dict=effective_dict,
                            )
                        )
                        if afound:
                            found = True
                            corners, ids = acorners, aids
                            marker_corners, marker_ids = amc, ami
                            matched_pattern = tuple(apat)
                            matched_dict = adict
                            self._remember_charuco_layout(cam_name, matched_pattern, adict)
                    if scale != 1.0:
                        corners = self._rescale_corners(corners, 1.0 / scale)
                        marker_corners = self._rescale_corners(marker_corners, 1.0 / scale)
                    if found and matched_pattern is not None:
                        gray_full = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        _, key, _ = self._find_camera_entry(cam_name)
                        key = key or cam_name
                        self._auto_consider(
                            key, "charuco", gray_full, corners, ids,
                            pattern=matched_pattern, dict_id=matched_dict,
                            _synchronous=False,
                        )
                    if not found:
                        corners, ids, marker_corners, marker_ids = None, None, None, None
                        self._auto_hint(
                            cam_name,
                            "Move the ChArUco board into view - auto capture waits for a detection",
                        )
                    result = {
                        "found": found and matched_pattern is not None,
                        "corners": corners,
                        "ids": ids,
                        "marker_corners": marker_corners,
                        "marker_ids": marker_ids,
                    }
                if result is not None:
                    result["ts"] = time.monotonic()
                    with result_lock:
                        latest["detect"] = result
                else:
                    stop_evt.wait(0.02)

        worker = None
        if overlay == "charuco":
            worker = threading.Thread(target=detection_worker, daemon=True)
            worker.start()
        try:
            warmup_deadline = (time.monotonic() + _CALIB_FEED_WARMUP_S) if overlay == "charuco" else 0
            # hold chunks until the detector's first tick so the opening frame
            # of an overlay feed is already annotated rather than raw
            while overlay == "charuco" and time.monotonic() < warmup_deadline:
                with result_lock:
                    ready = latest["detect"] is not None
                if ready:
                    break
                stop_evt.wait(0.02)
            while True:
                cam = self.live_cameras.get(cam_name)
                frame = None
                if cam is not None and hasattr(cam, "get_raw_frame"):
                    frame = cam.get_raw_frame()
                if frame is None:
                    if time.monotonic() - last_frame_time > _FEED_TIMEOUT_S:
                        break
                    time.sleep(0.05)
                    continue
                last_frame_time = time.monotonic()
                to_serve = frame
                if overlay == "charuco":
                    with result_lock:
                        res = latest.get("detect")
                    if res is not None and time.monotonic() - res.get("ts", 0.0) < 0.5:
                        if res.get("found"):
                            to_serve = cam_calibration.draw_charuco(
                                to_serve,
                                res["corners"], res["ids"],
                                res["marker_corners"], res["marker_ids"],
                            )
                    to_serve = self._draw_captured_overlays(to_serve, cam_name, "charuco")
                # overlay feeds are diagnostic - serve them capped in width so
                # JPEG encode stays cheap. The plain feed (focal measurement)
                # keeps full resolution: the UI reads the frame width from it.
                serve_frame = to_serve
                if overlay and _FEED_MAX_DIM:
                    w = to_serve.shape[1]
                    if w > _FEED_MAX_DIM:
                        f = _FEED_MAX_DIM / float(w)
                        serve_frame = cv2.resize(
                            to_serve,
                            (int(round(w * f)), int(round(to_serve.shape[0] * f))),
                        )
                ok, buf = cv2.imencode(".jpg", serve_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ok:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                time.sleep(1.0 / 30)
        except GeneratorExit:
            pass
        finally:
            stop_evt.set()

    def _charuco_status(self, cam_name):
        cam = self.live_cameras.get(cam_name)
        if cam is None or not hasattr(cam, "get_raw_frame"):
            return jsonify(found=False, message="Camera not live")
        frame = cam.get_raw_frame()
        if frame is None:
            return jsonify(found=False, message="No frame yet")
        session_pattern, session_dict = self._charuco_session_layout(cam_name)
        preferred_pattern = session_pattern or tuple(cam_calibration.DEFAULT_CHARUCO_PATTERN)
        # always sweep: a compatible-but-wrong grid can match the visible
        # corners, so let the scan rank candidates instead of trusting the
        # preferred layout blindly
        found, corners, ids, marker_corners, marker_ids, _, apat, adict = (
            cam_calibration.detect_charuco_auto(
                frame,
                preferred_pattern=preferred_pattern,
                preferred_dict=session_dict,
            )
        )
        matched_pattern = tuple(apat) if found else preferred_pattern
        matched_dict = adict if found else session_dict
        if found:
            self._remember_charuco_layout(cam_name, matched_pattern, adict)
        payload = {
            "found": bool(found),
            "corners": int(len(corners)) if found else 0,
            "markers": int(len(marker_corners)) if found else 0,
            "pattern": list(matched_pattern),
            "dictionary": matched_dict,
        }
        if not found:
            payload["message"] = "Board not visible - show the whole board in frame, even lighting helps."
        else:
            payload["message"] = (
                f"Board detected as {matched_pattern[0]}x{matched_pattern[1]} "
                f"(dictionary {matched_dict}). Move it around and capture."
            )
        return jsonify(**payload)

    def _charuco_capture(self, cam_name):
        data = request.get_json(force=True) or {}
        image_b64 = data.get("image")
        cols = int(_to_float(data.get("cols"), cam_calibration.DEFAULT_CHARUCO_PATTERN[0]))
        rows = int(_to_float(data.get("rows"), cam_calibration.DEFAULT_CHARUCO_PATTERN[1]))
        if cols < 2 or rows < 2 or cols > 30 or rows > 30:
            return jsonify(error="Invalid ChArUco board pattern"), 400
        cams, key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404
        if image_b64:
            frame = _decode_base64_frame(image_b64)
            if frame is None:
                return jsonify(error="Could not decode image"), 400
        else:
            # no image posted - detect on the live camera's raw frame so the
            # board detection runs on a clean frame instead of the overlaid
            # one shown in the wizard feed
            cam = self.live_cameras.get(cam_name)
            if cam is None or not hasattr(cam, "get_raw_frame"):
                return jsonify(error="Camera not live - start vision first"), 400
            frame = cam.get_raw_frame()
            if frame is None:
                return jsonify(error="Camera has no frame yet"), 400
        session_pattern, session_dict = self._charuco_session_layout(cam_name)
        matched_dict = session_dict
        kwargs = {}
        if matched_dict is not None:
            kwargs["dictionary_id"] = matched_dict
        found, corners, ids, marker_corners, marker_ids, gray = cam_calibration.detect_charuco(
            frame, cols, rows, **kwargs
        )
        if not found:
            return jsonify(
                success=False,
                board_found=False,
                message="ChArUco board not detected in that frame. Show the whole board, use even lighting, or try another pattern.",
            )
        color = cam_calibration.random_overlay_color()
        with self.calib_lock:
            session = self.calib_sessions.setdefault(key, {})
            session["charuco_pattern"] = [cols, rows]
            if matched_dict is not None:
                session["charuco_dict"] = matched_dict
            session.setdefault("charuco_captures", []).append((gray, corners, ids))
            session.setdefault("charuco_overlays", []).append((corners, ids, marker_corners, marker_ids, color))
            count = len(session["charuco_captures"])
        return jsonify(success=True, board_found=True, captured=count, color=list(color))

    def _charuco_clear(self, cam_name):
        cams, key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404
        with self.calib_lock:
            session = self.calib_sessions.get(key)
            if session:
                session.pop("charuco_captures", None)
                session.pop("charuco_overlays", None)
                session.pop("charuco_pattern", None)
                session.pop("charuco_dict", None)
                session.pop("auto_rms", None)
                session.pop("auto_state", None)
        return jsonify(success=True)

    def _charuco_finish(self, cam_name):
        cams, key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404
        with self.calib_lock:
            session = self.calib_sessions.get(key) or {}
            captures = list(session.get("charuco_captures", []))
            pattern = tuple(session.get("charuco_pattern", list(cam_calibration.DEFAULT_CHARUCO_PATTERN)))
            dict_id = session.get("charuco_dict")
        if len(captures) < 3:
            return jsonify(
                error=f"Need at least 3 captured ChArUco frames, have {len(captures)}"
            ), 400
        kwargs = {}
        if dict_id is not None:
            kwargs["dictionary_id"] = dict_id
        result = cam_calibration.calibrate_charuco(captures, pattern, **kwargs)
        if result is None:
            return jsonify(error="Calibration failed - try capturing more varied frames"), 500
        derived = cam_calibration.derive_fov_from_intrinsics(result)
        saved = self._save_calibration(key, {**result, **derived})
        with self.calib_lock:
            session = self.calib_sessions.get(key)
            if session:
                session.pop("charuco_captures", None)
                session.pop("charuco_pattern", None)
                session.pop("auto_rms", None)
                session.pop("auto_state", None)
        return jsonify(success=True, result=result, fov=saved.get("fov"),
                       focal_length_pixels=saved.get("focal_length_pixels"),
                       calibration=saved)

    def _add_camera(self):
        data = request.get_json(force=True) or {}
        name = data.get("name")
        device_id = data.get("device_id")
        source = data.get("source")
        camera_type = data.get("camera_type", "opencv")

        # Auto-fill source for Tello cameras
        if camera_type == "tello" and (source is None or source == ""):
            video_port = data.get("tello_video_port", 11111)
            source = f"udp://0.0.0.0:{video_port}"

        if not name or source is None:
            return jsonify(error="name and source required"), 400

        config = self.context["config"]
        cams = dict(config.get("camera_configs", {}))
        if name in cams:
            return jsonify(error=f"Camera '{name}' already exists"), 409

        # refuse to add a device already claimed by another configured cam
        if device_id:
            for existing in cams.values():
                if existing.get("device_id") == device_id:
                    return jsonify(error=f"Device already in use by camera '{existing.get('name')}'"), 409

        raw_pipeline = data.get("pipeline", "object_detection")
        if isinstance(raw_pipeline, dict):
            pipeline_name = raw_pipeline.get("name") or "object_detection"
            pipeline_settings = dict(raw_pipeline.get("settings") or {})
        else:
            pipeline_name = str(raw_pipeline or "object_detection")
            pipeline_settings = {}

        # calibration is edited through the dedicated wizard, not the add/edit
        # form anymore - when re-adding a device that was removed before,
        # carry its last saved calibration over from the profile
        calibration = data.get("calibration")
        if not calibration and device_id:
            profile = (read("camera_profiles", {}) or {}).get(device_id)
            if isinstance(profile, dict) and isinstance(profile.get("calibration"), dict):
                calibration = profile["calibration"]

        # Build cam_entry from ALL core keys provided in data
        cam_entry = {
            "name": name,
            "source": source,
            "camera_type": camera_type,
            "device_id": device_id,
            "calibration": calibration or {"distance": 0, "game_piece_size": 0, "size": 0, "fov": 0},
            "pipeline": {"name": pipeline_name, "settings": pipeline_settings},
        }
        # Copy all _CAMERA_CORE_KEYS from data if present (explicitly provided)
        for k in _CAMERA_CORE_KEYS:
            if k in data and data[k] is not None:
                cam_entry[k] = data[k]

        # model-backed pipelines crash at construction without a vision_model
        # block - accept the picker dict, a raw path, or let ensure_camera_entries_ready drop one in
        vm = data.get("vision_model")
        if isinstance(vm, dict) and vm.get("file_path"):
            pipeline_settings["vision_model"] = vm
        elif isinstance(vm, str) and vm:
            pipeline_settings["vision_model"] = {"file_path": vm, "source_pt": vm}

        # Anything not in core keys goes to pipeline settings
        handled = _CAMERA_CORE_KEYS | {"pipeline", "vision_model", "camera_type"}
        for k, v in data.items():
            if k not in handled:
                pipeline_settings[k] = v

        if is_model_backed_pipeline(pipeline_name) and isinstance(
            pipeline_settings.get("vision_model"), dict
        ):
            _resolve_vision_model_files(pipeline_settings)

        ensure_camera_entries_ready({name: cam_entry})
        _prune_stale_pipeline_settings(cam_entry)
        cams[name] = cam_entry
        config.set("camera_configs", cams)
        config.save()

        if device_id:
            profiles = read("camera_profiles", {})
            profiles[device_id] = cam_entry
            write("camera_profiles", profiles)

        return jsonify(success=True, note="Restart vision to apply.")

    def _find_camera_entry(self, cam_name):
        cams = self.context["config"].get("camera_configs", {})
        entry = cams.get(cam_name)
        if entry is not None:
            return cams, cam_name, entry
        for key, existing in cams.items():
            if existing.get("name") == cam_name or str(existing.get("source")) == cam_name:
                return cams, key, existing
        return None, None, None

    def _get_camera(self, cam_name):
        cams, entry_key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404
        return jsonify(camera=entry)

    def _update_camera(self, cam_name):
        data = request.get_json(force=True) or {}
        config = self.context["config"]
        cams, entry_key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404

        new_entry = copy.deepcopy(entry)
        model_picked = False
        pipeline_entry = new_entry.get("pipeline")
        if not isinstance(pipeline_entry, dict):
            pipeline_entry = {
                "name": get_pipeline_name(new_entry),
                "settings": dict(get_pipeline_settings(new_entry) or {}),
            }
            new_entry["pipeline"] = pipeline_entry

        for key, value in data.items():
            if value is None:
                settings = pipeline_entry.get("settings", {})
                if key in settings:
                    del settings[key]
                new_entry.pop(key, None)
                continue
            if key == "pipeline":
                if isinstance(value, str):
                    pipeline_entry["name"] = value
                elif isinstance(value, dict):
                    if value.get("name"):
                        pipeline_entry["name"] = value["name"]
                    if isinstance(value.get("settings"), dict):
                        if "vision_model" in value["settings"]:
                            model_picked = True
                        pipeline_entry.setdefault("settings", {}).update(value["settings"])
                continue
            if key == "name":
                new_entry[key] = value
                continue
            if key in _CAMERA_CORE_KEYS:
                if isinstance(value, dict) and isinstance(new_entry.get(key), dict):
                    merged = dict(new_entry[key])
                    merged.update(value)
                    new_entry[key] = merged
                else:
                    new_entry[key] = value
                continue
            # everything else is a pipeline settings-field - route it into
            # pipeline.settings so stale top-level keys cant reappear
            pipeline_entry.setdefault("settings", {})[key] = value

        new_name = str(new_entry.get("name") or "").strip()
        if not new_name:
            return jsonify(error="Camera name is required"), 400
        if new_entry.get("source") is None:
            return jsonify(error="Camera source is required"), 400
        if new_name != entry_key and new_name in cams:
            return jsonify(error=f"Camera '{new_name}' already exists"), 409

        pipeline_name = str(pipeline_entry.get("name") or "object_detection")
        try:
            from iSpy.vision.pipelines import get_pipeline_classes
            if pipeline_name not in get_pipeline_classes():
                return jsonify(error=f"Unknown pipeline '{pipeline_name}'"), 400
        except Exception:
            pass

        # switching onto a model-backed pipeline needs a vision_model block
        # or the new pipeline instance crashes at construction
        if is_model_backed_pipeline(pipeline_name) and not isinstance(
            pipeline_entry.get("settings", {}).get("vision_model"), dict
        ):
            pipeline_entry.setdefault("settings", {})["vision_model"] = default_vision_model()

        # a model pick must move both source_pt and file_path together, or the
        # camera keeps running the old model (stale artifact from a previous pick)
        if model_picked and is_model_backed_pipeline(pipeline_name):
            _resolve_vision_model_files(pipeline_entry.setdefault("settings", {}))

        ensure_camera_entries_ready({new_name: new_entry})
        _prune_stale_pipeline_settings(new_entry)

        cams.pop(entry_key)
        cams[new_name] = new_entry
        config.set("camera_configs", cams)
        config.save()

        # settings apply on the next vision start - never rebuild the running pipeline mid-run
        return jsonify(
            success=True,
            note="Settings saved - restart vision to apply.",
            camera=new_entry,
        )

    def _remove_camera(self, cam_name):
        config = self.context["config"]
        cams = dict(config.get("camera_configs", {}))
        if cam_name not in cams:
            return jsonify(error="Camera not found"), 404
        if len(cams) <= 1:
            return jsonify(error="Cannot remove the last camera - at least one is required."), 400

        removed = cams.pop(cam_name)
        config.set("camera_configs", cams)
        config.save()

        device_id = removed.get("device_id")
        if device_id:
            profiles = read("camera_profiles", {})
            profiles[device_id] = removed  # remember full params for autofill next time
            write("camera_profiles", profiles)

        return jsonify(success=True, note="Restart vision to apply.")

    def _evict_stale(self, now: float):
        stale = [n for n, ts in self.last_seen.items() if now - ts > _STALE_EVICT_S]
        for n in stale:
            self.frames.pop(n, None)
            self.dims.pop(n, None)
            self.last_seen.pop(n, None)
            self.sources.pop(n, None)

    def _api_cameras(self):
        now = time.monotonic()
        with self.lock:
            self._evict_stale(now)
            cameras = []
            for n, d in self.dims.items():
                payload = {
                    "name": n,
                    "w": d[0],
                    "h": d[1],
                    "age_ms": round((now - self.last_seen.get(n, now)) * 1000, 1),
                    "source": self.sources.get(n),
                }
                _, _, entry = self._find_camera_entry(n)
                payload["calibrated"] = _camera_calibrated(entry)
                payload["pipeline"] = (
                    get_pipeline_name(entry) if isinstance(entry, dict) else "object_detection"
                )
                inst = self.live_cameras.get(n)
                if inst is not None and hasattr(inst, "is_ready"):
                    try:
                        ready, status = inst.is_ready()
                    except Exception:
                        ready, status = False, "error: is_ready() raised"
                    payload["ready"] = bool(ready)
                    payload["status"] = str(status)
                    state = getattr(inst, "get_state", None)
                    payload["state"] = state() if callable(state) else None
                else:
                    payload["ready"] = None
                    payload["status"] = None
                    payload["state"] = None
                cameras.append(payload)
        return jsonify(cameras=cameras)

    def _discover(self):
        now = time.monotonic()
        with self._discover_lock:
            if self._discover_cache and (now - self._discover_cache_ts) < 5.0:
                devices = self._discover_cache
            else:
                devices = self._probe_devices()
                self._discover_cache = devices
                self._discover_cache_ts = now

        # a discovered device is active when its device_id (or raw source if
        # resolution failed) matches a configured cam
        config = self.context.get("config")
        configured_device_ids = set()
        configured_sources = set()
        if config:
            for cam_cfg in config.get("camera_configs", {}).values():
                dev_id = cam_cfg.get("device_id")
                if dev_id:
                    configured_device_ids.add(dev_id)
                source = cam_cfg.get("source")
                if source is not None:
                    configured_sources.add(str(source))
                if source is None and cam_cfg.get("path") is not None:
                    configured_sources.add(str(cam_cfg.get("path")))

        for dev in devices:
            dev_id = dev.get("device_id")
            source = dev.get("path")
            dev["active"] = bool(dev_id and dev_id in configured_device_ids) or str(source) in configured_sources

        return jsonify(devices=devices)

    def _sources(self):
        """Return all discoverable camera sources, grouped by camera_type."""
        claimed = set()
        config = self.context.get("config")
        if config:
            for cam_cfg in config.get("camera_configs", {}).values():
                source = cam_cfg.get("source")
                if source is not None:
                    claimed.add(str(source))
                if source is None and cam_cfg.get("path") is not None:
                    claimed.add(str(cam_cfg.get("path")))

        opencv_sources = OpenCVCamera.discover(claimed_sources=claimed)
        tello_sources = TelloCamera.discover(claimed_sources=claimed)

        return jsonify(
            sources={
                "opencv": opencv_sources,
                "tello": tello_sources,
            }
        )

    def _probe_devices(self):
        """Delegate to the OpenCVCamera discovery logic so the module-level
        test mock in test_boot_camera_cleanup.py continues to work."""
        with self.lock:
            claimed = {s for s in self.sources.values() if s}
        return OpenCVCamera.discover(claimed_sources=claimed)

    def _video_feed(self, camera_name):
        return Response(
            self._generate(camera_name),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _generate(self, camera_name):
        target_interval = 1.0 / 20
        last_frame_time = time.monotonic()
        try:
            while True:
                t0 = time.perf_counter()
                with self.lock:
                    frame = self.frames.get(camera_name)
                if frame is None:
                    if time.monotonic() - last_frame_time > _FEED_TIMEOUT_S:
                        break
                    time.sleep(0.05)
                    continue
                last_frame_time = time.monotonic()
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ok:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                elapsed = time.perf_counter() - t0
                sleep_time = target_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except GeneratorExit:
            pass