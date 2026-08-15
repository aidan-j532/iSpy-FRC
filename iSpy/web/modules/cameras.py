import base64
import copy
import glob
import os
import platform
import re
import subprocess
import threading
from pathlib import Path
import cv2
import time
from flask import Response, jsonify, render_template, request
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

_STALE_EVICT_S = 10.0
_FEED_TIMEOUT_S = 15.0

# legacy alias keys - dropped when the canonical key is present so old configs
# dont carry dead twins around forever
_SETTING_ALIASES = {"quantized": "quantize", "auto_opt": "optimize"}

# image tuning knobs (sliders in the camera lightbox). these live on the cam
# entry like the other capture keys; the live Camera applies them immediately
_TUNING_KEYS = (
    "auto_brightness", "auto_exposure", "brightness", "contrast",
    "saturation", "white_balance", "tint", "gamma",
    "exposure_time", "gain",
)
_TUNING_DEFAULTS = {
    "auto_brightness": True,
    "auto_exposure": True,
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
    """settings keys the pipeline accepts; empty when unknown so pruning never nukes data"""
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
    """drop settings from a different pipeline (they pile up when a cam
    switches pipelines) + legacy aliases; unknown keys stay - hand-written tuning knobs"""
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


def _windows_cameras_from_registry():
    """find capture sources via the registry without opening cams; key order ==
    the order CAP_MSMF assigns indices"""
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
            devices.append({
                "index": camera_index,
                "name": _windows_camera_name(iface) or f"Camera {camera_index}",
                # unique per device interface (USB\VID_...&PID_...&MI_01\<instance>) - dedupe key + stable device_id
                "hw_id": iface[:iface.find("#{")],
            })
            camera_index += 1
    finally:
        winreg.CloseKey(key)
    return devices


def _windows_camera_name(iface):
    """grab the friendly name from the registry; generic uvc names get replaced
    by walking up to the parent usb node (thats where the real name lives)"""
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
    parts = body.split("#", 1)
    if len(parts) != 2:
        return None
    hw_id, instance_id = parts

    def _read_name(enum_parent, instance):
        enum_path = f"SYSTEM\\CurrentControlSet\\Enum\\{enum_parent}\\{instance}"
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

    name = _read_name(hw_id, instance_id)
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
                    parent_name = _read_name(parent_hw, sub)
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


def _linux_is_capture_node(video_path):
    """true if a real capture device; None = caps unreadable, keep the node"""
    caps = _v4l2_caps(video_path)
    if caps == 0:
        return None
    if caps & _V4L2_CAP_VIDEO_M2M:
        # m2m codecs (bcm2835-codec-decode) claim the capture bit but aint cameras
        return False
    return bool(caps & (_V4L2_CAP_VIDEO_CAPTURE | _V4L2_CAP_VIDEO_CAPTURE_MPLANE))


def _linux_device_key(video_path):
    """physical device behind a /dev/videoN node - the dedupe key that kills
    linux "same cam twice" detection (video0+video1 of one uvc cam share it)"""
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
    """group /dev/videoN nodes by physical device - v4l2-ctl first, sysfs fallback"""
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

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/cameras", "cameras_page", lambda: render_template("cameras.html"))
        flask_app.add_url_rule("/api/cameras", "api_cameras", self._api_cameras)
        flask_app.add_url_rule("/api/cameras/discover", "api_cameras_discover", self._discover)
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
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/chessboard/capture", "api_cameras_chessboard_capture", self._chessboard_capture, methods=["POST"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/chessboard", "api_cameras_chessboard_clear", self._chessboard_clear, methods=["DELETE"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/chessboard/finish", "api_cameras_chessboard_finish", self._chessboard_finish, methods=["POST"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/charuco/capture", "api_cameras_charuco_capture", self._charuco_capture, methods=["POST"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/charuco/status", "api_cameras_charuco_status", self._charuco_status)
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/charuco", "api_cameras_charuco_clear", self._charuco_clear, methods=["DELETE"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/charuco/finish", "api_cameras_charuco_finish", self._charuco_finish, methods=["POST"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/feed", "api_cameras_calibration_feed", self._calibration_feed)
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/mode", "api_cameras_calibration_mode", self._calibration_mode, methods=["POST"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/heartbeat", "api_cameras_calibration_heartbeat", self._calibration_heartbeat, methods=["POST"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/pnp", "api_cameras_pnp_get", self._pnp_get, methods=["GET"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/pnp", "api_cameras_pnp_save", self._pnp_save, methods=["POST"])
        flask_app.add_url_rule("/api/cameras/calibration/<cam_name>/pnp", "api_cameras_pnp_clear", self._pnp_clear, methods=["DELETE"])

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
        """(vision_model dict, num_keypoints, error) - error is None if usable"""
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
        return jsonify(
            model_error=err,
            num_keypoints=num_kpts,
            has_intrinsics=has_intrinsics,
            pnp=existing_pnp,
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
            return jsonify(error="No chessboard/ChArUco intrinsics on this camera yet - run that calibration first"), 400

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
        """hardware id for a running cam; falls back to source when device_id
        resolution failed (image placeholders etc)"""
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
        """saved tuning values for a cam entry, defaults for keys not set"""
        values = dict(_TUNING_DEFAULTS)
        for key in _TUNING_KEYS:
            if key in entry:
                values[key] = entry[key]
        return values

    def _live_cam_for(self, key: str, cam_name: str):
        """the running Camera object (if web + vision share a process), by
        config key or display name"""
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
            elif k == "auto_brightness":
                entry[k] = bool(v)
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
            note="Saved - applies immediately to the live feed.",
            applied=applied,
        )

    def _vision_pipelines(self):
        return jsonify(pipelines=_build_vision_pipeline_payloads())

    # ------------------------------------------------------------------
    # Camera calibration (web wizard)
    # ------------------------------------------------------------------

    def _save_calibration(self, cam_name: str, calib_dict: dict):
        """merge calib_dict into the camera's calibration and persist; returns the merged dict"""
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
            chessboard_captures=len(session.get("captures", [])),
            chessboard_pattern=session.get("pattern", list(cam_calibration.DEFAULT_CHESSBOARD_PATTERN)),
            charuco_captures=len(session.get("charuco_captures", [])),
            charuco_pattern=session.get("charuco_pattern", list(cam_calibration.DEFAULT_CHARUCO_PATTERN)),
            charuco_dict=session.get("charuco_dict"),
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

    def _chessboard_capture(self, cam_name):
        data = request.get_json(force=True) or {}
        image_b64 = data.get("image")
        if not image_b64:
            return jsonify(error="No image provided"), 400
        cols = int(_to_float(data.get("cols"), cam_calibration.DEFAULT_CHESSBOARD_PATTERN[0]))
        rows = int(_to_float(data.get("rows"), cam_calibration.DEFAULT_CHESSBOARD_PATTERN[1]))
        if cols < 2 or rows < 2 or cols > 20 or rows > 20:
            return jsonify(error="Invalid chessboard pattern"), 400
        cams, key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404
        frame = _decode_base64_frame(image_b64)
        if frame is None:
            return jsonify(error="Could not decode image"), 400
        found, corners, gray = cam_calibration.detect_chessboard(frame, cols, rows)
        if not found:
            return jsonify(
                success=False,
                board_found=False,
                message="Chessboard not detected in that frame. Show the whole board, use even lighting, or try another pattern.",
            )
        color = cam_calibration.random_overlay_color()
        with self.calib_lock:
            session = self.calib_sessions.setdefault(key, {})
            session["pattern"] = [cols, rows]
            session.setdefault("captures", []).append((gray, corners))
            session.setdefault("chess_overlays", []).append((corners, color))
            count = len(session["captures"])
        return jsonify(success=True, board_found=True, captured=count, color=list(color))

    def _chessboard_clear(self, cam_name):
        cams, key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404
        with self.calib_lock:
            self.calib_sessions.pop(key, None)
        return jsonify(success=True)

    def _chessboard_finish(self, cam_name):
        cams, key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404
        with self.calib_lock:
            session = self.calib_sessions.get(key) or {}
            captures = list(session.get("captures", []))
            pattern = tuple(session.get("pattern", list(cam_calibration.DEFAULT_CHESSBOARD_PATTERN)))
        if len(captures) < 3:
            return jsonify(
                error=f"Need at least 3 captured chessboard frames, have {len(captures)}"
            ), 400
        result = cam_calibration.calibrate_chessboard(captures, pattern)
        if result is None:
            return jsonify(error="Calibration failed - try capturing more varied frames"), 500
        saved = self._save_calibration(key, result)
        with self.calib_lock:
            self.calib_sessions.pop(key, None)
        return jsonify(success=True, result=result, calibration=saved)

    def _calibration_mode(self, cam_name):
        """pause/resume detections while the calibration wizard is open"""
        data = request.get_json(force=True) or {}
        active = bool(data.get("active", True))
        cam = self.live_cameras.get(cam_name)
        if cam is not None and hasattr(cam, "set_calibration"):
            cam.set_calibration(active)
        return jsonify(success=True)

    def _calibration_heartbeat(self, cam_name):
        """keeps the calibration pause alive; stops arriving -> detections resume"""
        cam = self.live_cameras.get(cam_name)
        if cam is not None and hasattr(cam, "calibration_heartbeat"):
            cam.calibration_heartbeat()
        return jsonify(success=True)

    def _calibration_feed(self, cam_name):
        """MJPEG feed for the calibration wizard - frames straight from the
        capture thread (never run through the pipeline's detectors). Pass
        ?overlay=charuco or ?overlay=chessboard to draw the detected board on
        top so the user can see what the detector sees while they move it.
        Captured frames are drawn on top in their own random color."""
        overlay = request.args.get("overlay", "")
        pattern_arg = request.args.get("pattern")
        pattern = None
        if pattern_arg:
            try:
                pattern = tuple(int(p) for p in pattern_arg.split(",")[:2])
            except ValueError:
                pattern = None
        return Response(
            self._generate_calibration(cam_name, overlay=overlay, pattern=pattern),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _charuco_session_layout(self, cam_name):
        """cached (pattern, dictionary_id) the wizard auto-detected for this
        camera, so the live feed/captures stay on the board's actual layout."""
        _, key, _ = self._find_camera_entry(cam_name)
        key = key or cam_name
        with self.calib_lock:
            session = self.calib_sessions.get(key) or {}
            p = session.get("charuco_pattern")
            d = session.get("charuco_dict")
        return tuple(p) if p else None, d

    def _chessboard_session_pattern(self, cam_name):
        _, key, _ = self._find_camera_entry(cam_name)
        key = key or cam_name
        with self.calib_lock:
            session = self.calib_sessions.get(key) or {}
            p = session.get("pattern")
        return tuple(p) if p else None

    def _draw_captured_overlays(self, frame, cam_name, overlay):
        """draw every captured frame's detection on the live feed in its own
        random color, so each capture is confirmed on the live view
        (PhotonVision style) instead of a thumbnail strip under the wizard."""
        _, key, _ = self._find_camera_entry(cam_name)
        key = key or cam_name
        with self.calib_lock:
            session = self.calib_sessions.get(key) or {}
            layers = {
                "charuco": session.get("charuco_overlays", []),
                "chessboard": session.get("chess_overlays", []),
            }.get(overlay, [])
            layers = list(layers)
        out = frame
        for layer in layers:
            if overlay == "charuco":
                corners, ids, mc, mi, color = layer
                out = cam_calibration.draw_charuco(out, corners, ids, mc, mi, color=color)
            elif overlay == "chessboard":
                corners, color = layer
                out = cam_calibration.draw_corners(out, corners, color)
        return out

    def _generate_calibration(self, cam_name, overlay="", pattern=None):
        last_frame_time = time.monotonic()
        last_detect = 0.0
        last_auto_scan = 0.0
        corners, ids, marker_corners, marker_ids = None, None, None, None
        chess_pattern = None
        try:
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
                    now = time.monotonic()
                    # board detection is the expensive bit - refresh the
                    # overlay ~10x/s and reuse the last result in between
                    if now - last_detect >= 0.1:
                        last_detect = now
                        session_pattern, session_dict = self._charuco_session_layout(cam_name)
                        # layouts to try this tick: the layout the wizard
                        # auto-detected/cached first, then the one picked in
                        # the UI dropdown
                        candidates = []
                        if session_pattern is not None:
                            candidates.append(tuple(session_pattern))
                        if pattern is not None and len(pattern) == 2 and tuple(pattern) not in candidates:
                            candidates.append(tuple(pattern))
                        found = False
                        for cand in candidates:
                            kwargs = {}
                            if session_dict is not None:
                                kwargs["dictionary_id"] = session_dict
                            found, corners, ids, marker_corners, marker_ids, _ = (
                                cam_calibration.detect_charuco(
                                    frame, cand[0], cand[1], **kwargs
                                )
                            )
                            if found:
                                break
                        if not found:
                            # nothing matched a known layout - throttled scan
                            # of the common printed boards so any ChArUco board
                            # shows live. The match is cached so captures and
                            # calibration stay on the board's real layout.
                            if now - last_auto_scan >= 1.0:
                                last_auto_scan = now
                                found, corners, ids, marker_corners, marker_ids, _, fp, fd = (
                                    cam_calibration.detect_charuco_auto(
                                        frame,
                                        preferred_pattern=candidates[0] if candidates else None,
                                        preferred_dict=session_dict,
                                    )
                                )
                                if found:
                                    _, key, _ = self._find_camera_entry(cam_name)
                                    key = key or cam_name
                                    with self.calib_lock:
                                        session = self.calib_sessions.setdefault(key, {})
                                        session["charuco_pattern"] = list(fp)
                                        if fd is not None:
                                            session["charuco_dict"] = fd
                        if not found:
                            corners, ids, marker_corners, marker_ids = None, None, None, None
                    if corners is not None and ids is not None:
                        to_serve = cam_calibration.draw_charuco(
                            frame, corners, ids, marker_corners, marker_ids
                        )
                    to_serve = self._draw_captured_overlays(to_serve, cam_name, "charuco")
                elif overlay == "chessboard":
                    now = time.monotonic()
                    if now - last_detect >= 0.1:
                        last_detect = now
                        if pattern is None:
                            pattern = self._chessboard_session_pattern(cam_name)
                        if pattern is not None and len(pattern) == 2:
                            chess_pattern = tuple(pattern[:2])
                            found, corners, _ = cam_calibration.detect_chessboard(
                                frame, chess_pattern[0], chess_pattern[1]
                            )
                            if not found:
                                corners = None
                    if corners is not None and chess_pattern is not None:
                        to_serve = cam_calibration.draw_chessboard(
                            frame, corners, chess_pattern[0], chess_pattern[1]
                        )
                    to_serve = self._draw_captured_overlays(to_serve, cam_name, "chessboard")
                ok, buf = cv2.imencode(".jpg", to_serve, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ok:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                time.sleep(1.0 / 20)
        except GeneratorExit:
            pass

    def _charuco_status(self, cam_name):
        """one-shot board detection on the latest raw frame - the wizard polls
        this to show live 'board found: N corners' feedback next to the feed.
        The board layout + dictionary are auto-detected across the common
        printed sizes and cached so the feed and captures stay consistent."""
        cam = self.live_cameras.get(cam_name)
        if cam is None or not hasattr(cam, "get_raw_frame"):
            return jsonify(found=False, message="Camera not live")
        frame = cam.get_raw_frame()
        if frame is None:
            return jsonify(found=False, message="No frame yet")
        session_pattern, session_dict = self._charuco_session_layout(cam_name)
        preferred_pattern = session_pattern or tuple(cam_calibration.DEFAULT_CHARUCO_PATTERN)
        found, corners, ids, marker_corners, marker_ids, _, found_pattern, found_dict = (
            cam_calibration.detect_charuco_auto(
                frame,
                preferred_pattern=preferred_pattern,
                preferred_dict=session_dict,
            )
        )
        if found and (
            found_pattern != (tuple(session_pattern) if session_pattern else None)
            or found_dict != session_dict
        ):
            _, key, _ = self._find_camera_entry(cam_name)
            key = key or cam_name
            with self.calib_lock:
                session = self.calib_sessions.setdefault(key, {})
                session["charuco_pattern"] = list(found_pattern)
                session["charuco_dict"] = found_dict
        payload = {
            "found": bool(found),
            "corners": int(len(corners)) if found else 0,
            "markers": int(len(marker_corners)) if found else 0,
            "pattern": list(found_pattern) if found else list(preferred_pattern),
            "dictionary": found_dict if found else session_dict,
        }
        if not found:
            payload["message"] = "Board not visible - show the whole board in frame, even lighting helps."
        else:
            payload["message"] = (
                f"Board detected as {found_pattern[0]}x{found_pattern[1]} "
                f"(dictionary {found_dict}). Move it around and capture."
            )
        return jsonify(**payload)

    def _charuco_capture(self, cam_name):
        data = request.get_json(force=True) or {}
        image_b64 = data.get("image")
        if not image_b64:
            return jsonify(error="No image provided"), 400
        cols = int(_to_float(data.get("cols"), cam_calibration.DEFAULT_CHARUCO_PATTERN[0]))
        rows = int(_to_float(data.get("rows"), cam_calibration.DEFAULT_CHARUCO_PATTERN[1]))
        if cols < 2 or rows < 2 or cols > 30 or rows > 30:
            return jsonify(error="Invalid ChArUco board pattern"), 400
        cams, key, entry = self._find_camera_entry(cam_name)
        if entry is None:
            return jsonify(error="Camera not found"), 404
        frame = _decode_base64_frame(image_b64)
        if frame is None:
            return jsonify(error="Could not decode image"), 400
        session_pattern, session_dict = self._charuco_session_layout(cam_name)
        matched_dict = session_dict
        kwargs = {}
        if matched_dict is not None:
            kwargs["dictionary_id"] = matched_dict
        found, corners, ids, marker_corners, marker_ids, gray = cam_calibration.detect_charuco(
            frame, cols, rows, **kwargs
        )
        if not found:
            # fall back to a full auto-detect - the board may simply be a
            # different layout/dictionary than the one selected in the UI
            found, corners, ids, marker_corners, marker_ids, gray, found_pattern, found_dict = (
                cam_calibration.detect_charuco_auto(
                    frame,
                    preferred_pattern=(cols, rows),
                    preferred_dict=matched_dict,
                )
            )
            if found:
                cols, rows = found_pattern
                matched_dict = found_dict
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
        saved = self._save_calibration(key, result)
        with self.calib_lock:
            session = self.calib_sessions.get(key)
            if session:
                session.pop("charuco_captures", None)
                session.pop("charuco_pattern", None)
        return jsonify(success=True, result=result, calibration=saved)

    def _add_camera(self):
        data = request.get_json(force=True) or {}
        name = data.get("name")
        device_id = data.get("device_id")
        source = data.get("source")
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

        cam_entry = {
            "name": name, "source": source, "device_id": device_id,
            "yaw": data.get("yaw", 0), "pitch": data.get("pitch", 0),
            "height": data.get("height", 0), "x": data.get("x", 0),
            "y": data.get("y", 0), "z": data.get("z", 0),
            "grayscale": data.get("grayscale", False),
            "subsystem": data.get("subsystem", "field"),
            "calibration": data.get("calibration", {"distance": 0, "game_piece_size": 0, "size": 0, "fov": 0}),
            "pipeline": {"name": pipeline_name, "settings": pipeline_settings},
        }

        # model-backed pipelines crash at construction without a vision_model
        # block - accept the picker dict, a raw path, or let ensure_camera_entries_ready drop one in
        vm = data.get("vision_model")
        if isinstance(vm, dict) and vm.get("file_path"):
            pipeline_settings["vision_model"] = vm
        elif isinstance(vm, str) and vm:
            pipeline_settings["vision_model"] = {"file_path": vm, "source_pt": vm}

        handled = _CAMERA_CORE_KEYS | {"pipeline", "vision_model"}
        for k, v in data.items():
            if k not in handled:
                pipeline_settings[k] = v

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

    def _probe_devices(self):
        devices = []
        system_name = platform.system()

        if system_name == "Linux":
            # one entry per physical device, not per node: uvc cams have a
            # metadata companion node (video0+video1) and codec/radio nodes
            # pollute the glob - QUERYCAP filters + sysfs identity dedupes
            seen_keys = set()
            for label, nodes in _linux_device_groups():
                capture = [n for n in nodes if _linux_is_capture_node(n)]
                if not capture:
                    capture = [n for n in nodes if _linux_is_capture_node(n) is None]
                if not capture:
                    # all nodes confirmed non-capture (encoder/codec/radio...) - skip the group
                    continue
                node = sorted(capture)[0]
                key = _linux_device_key(node) or node
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                devices.append({
                    "path": node,
                    "name": label or _linux_sysfs_name(node) or node,
                    "device_id": key,
                })

            # catch whatever grouping missed - still drop non-capture nodes + dupes
            for path in sorted(glob.glob("/dev/video*")):
                is_capture = _linux_is_capture_node(path)
                if is_capture is False:
                    continue
                key = _linux_device_key(path) or path
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                devices.append({
                    "path": path,
                    "name": _linux_sysfs_name(path) or path,
                    "device_id": key,
                })

        elif system_name == "Windows":
            # no index probing - MSMF cant open by index, every attempt spams
            # "VIDEOIO(MSMF)..." to stderr, and discover reruns on refresh so
            # it'd spam forever. registry key order == CAP_MSMF index order
            with self.lock:
                claimed = {s for s in self.sources.values() if s}
            seen_interfaces = set()
            for cam in _windows_cameras_from_registry():
                hw_id = cam.get("hw_id")
                if hw_id in seen_interfaces:
                    continue
                seen_interfaces.add(hw_id)
                index = str(cam["index"])
                if index in claimed:
                    continue
                devices.append({
                    "path": index,
                    "name": cam["name"],
                    "device_id": hw_id,
                })

        else:
            # macos / other: best-effort /dev/video glob + index probing
            with self.lock:
                claimed = {s for s in self.sources.values() if s}

            for path in sorted(glob.glob("/dev/video*")):
                devices.append({
                    "path": path, "name": path,
                    "device_id": None,
                })

            if not devices:
                for i in range(0, 10):
                    if str(i) in claimed:
                        devices.append({"path": str(i), "name": f"Camera {i}", "device_id": None})
                        continue
                    try:
                        cap = cv2.VideoCapture(i, cv2.CAP_ANY)
                        if cap is not None and cap.isOpened():
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            devices.append({
                                "path": str(i), "name": f"Camera {i}",
                                "resolution": f"{w}x{h}" if w and h else None,
                                "device_id": None,
                            })
                            cap.release()
                    except Exception:
                        continue

        # dedupe
        seen = {}
        for dev in devices:
            key = dev.get("device_id") or dev.get("path")
            if key in seen:
                continue
            seen[key] = dev
        return list(seen.values())

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