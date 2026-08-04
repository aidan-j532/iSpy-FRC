import copy
import glob
import os
import platform
import subprocess
import threading
import cv2
import time
from flask import Response, jsonify, render_template, request
import numpy as np
from iSpy.web.Backend.WebModule import WebModule
from iSpy.web.Backend.save_store import read, write
from iSpy.web.Backend.PluginStatus import _build_vision_pipeline_payloads
from pathlib import Path

_STALE_EVICT_S = 10.0
_FEED_TIMEOUT_S = 15.0


def _windows_cameras_from_registry():
    """Discover video capture sources from the Windows registry without opening cameras.

    Reads the KSCATEGORY_VIDEO_CAMERA device-interface class, whose key order is
    the same order cv2.VideoCapture(i, cv2.CAP_MSMF) assigns indices (index 0 is
    the first camera). Returns a list of {"index": int, "name": str} dicts.
    """
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
            })
            camera_index += 1
    finally:
        winreg.CloseKey(key)
    return devices


def _windows_camera_name(iface):
    """Resolve a friendly camera name from a DeviceClasses interface key name."""
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
    instance = "SYSTEM\\CurrentControlSet\\Enum\\" + body.replace("#", "\\")
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, instance)
    except OSError:
        return None
    try:
        for value in ("FriendlyName", "DeviceDesc"):
            try:
                name, _ = winreg.QueryValueEx(key, value)
            except OSError:
                continue
            if isinstance(name, str) and name.startswith("@"):
                name = name.rsplit(";", 1)[-1]
            name = name.strip() if isinstance(name, str) else ""
            if name:
                return name
    finally:
        winreg.CloseKey(key)
    return None

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
        flask_app.add_url_rule("/api/cameras/<name>/optimize", "api_cameras_optimize", self._optimize_camera, methods=["POST"])
        flask_app.add_url_rule("/api/vision_pipelines", "api_vision_pipelines", self._vision_pipelines)

    def _camera_display_name(self, cam, fallback: str = "camera") -> str:
        if hasattr(cam, "config") and cam.config is not None:
            name = cam.config.get("name") if hasattr(cam.config, "get") else None
            if name:
                return str(name)
        source = getattr(cam, "source", None)
        if source is not None:
            return str(source)
        return fallback

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
        """Stable hardware identity for a running camera - prefers the
        resolved device_id (by-id path / vendor:product:serial), falling
        back to the raw source if resolution failed (e.g. it's an image
        placeholder, not a real device)."""
        dev_id = getattr(cam, "device_id", None)
        if dev_id:
            return dev_id
        return str(getattr(cam, "source", ""))

    def _get_profile(self, device_id):
        profiles = read("camera_profiles", {})
        return jsonify(profile=profiles.get(device_id))

    def _vision_pipelines(self):
        return jsonify(pipelines=_build_vision_pipeline_payloads())

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

        # Refuse to add a device already claimed by another configured camera
        if device_id:
            for existing in cams.values():
                if existing.get("device_id") == device_id:
                    return jsonify(error=f"Device already in use by camera '{existing.get('name')}'"), 409

        cam_entry = {
            "name": name, "source": source, "device_id": device_id,
            "pipeline": data.get("pipeline", "object_detection"),
            "yaw": data.get("yaw", 0), "pitch": data.get("pitch", 0),
            "height": data.get("height", 0), "x": data.get("x", 0),
            "y": data.get("y", 0), "z": data.get("z", 0),
            "grayscale": data.get("grayscale", False),
            "subsystem": data.get("subsystem", "field"),
            "calibration": data.get("calibration", {"distance": 0, "game_piece_size": 0, "size": 0, "fov": 0}),
        }
        handled = {"name", "source", "device_id", "pipeline", "yaw", "pitch", "height", "x", "y", "z", "grayscale", "subsystem", "calibration"}
        for k, v in data.items():
            if k not in handled:
                cam_entry[k] = v
        cams[name] = cam_entry
        config.set("camera_configs", cams)
        config.save()

        if device_id:
            profiles = read("camera_profiles", {})
            profiles[device_id] = cam_entry
            write("camera_profiles", profiles)

        return jsonify(success=True, note="Restart vision to apply.")

    def _find_camera_entry(self, cam_name):
        """Locate a camera config entry by its config key, name, or source."""
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
        for key, value in data.items():
            if value is None:
                new_entry.pop(key, None)
                continue
            if isinstance(value, dict) and isinstance(new_entry.get(key), dict):
                merged = dict(new_entry[key])
                merged.update(value)
                new_entry[key] = merged
            else:
                new_entry[key] = value

        new_name = str(new_entry.get("name") or "").strip()
        if not new_name:
            return jsonify(error="Camera name is required"), 400
        if new_entry.get("source") is None:
            return jsonify(error="Camera source is required"), 400
        if new_name != entry_key and new_name in cams:
            return jsonify(error=f"Camera '{new_name}' already exists"), 409

        cams.pop(entry_key)
        cams[new_name] = new_entry
        config.set("camera_configs", cams)
        config.save()

        return jsonify(success=True, note="Restart vision to apply.", camera=new_entry)

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
                else:
                    payload["ready"] = None
                    payload["status"] = None
                cameras.append(payload)
        return jsonify(cameras=cameras)

    def _optimize_camera(self, camera_name):
        inst = self.live_cameras.get(camera_name)
        if inst is None:
            return jsonify(error=f"Camera '{camera_name}' is not running"), 404
        if not hasattr(inst, "request_optimize"):
            pipeline = getattr(inst, "plugin_name", "?")
            return jsonify(
                error=f"Pipeline '{pipeline}' does not support on-demand optimization"
            ), 400
        status = inst.request_optimize()
        return jsonify(status=status)

    def _discover(self):
        now = time.monotonic()
        with self._discover_lock:
            if self._discover_cache and (now - self._discover_cache_ts) < 5.0:
                devices = self._discover_cache
            else:
                devices = self._probe_devices()
                self._discover_cache = devices
                self._discover_cache_ts = now

        # Cross-reference against configured cameras using the same identity
        # that the add-camera flow writes into config. A discovered device is
        # considered active if either its resolved device_id matches a
        # configured camera, or (when device_id resolution failed) its raw
        # source/path matches a configured camera source.
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

    # def _probe_devices(self):
    #     devices = []
    #     if platform.system() == "Linux":
    #         v4l = subprocess.run(
    #             ["v4l2-ctl", "--list-devices"],
    #             capture_output=True, text=True, timeout=5,
    #         )
    #         if v4l.returncode == 0:
    #             current_name = None
    #             for line in v4l.stdout.splitlines():
    #                 line = line.strip()
    #                 if not line:
    #                     continue
    #                 if line.endswith(":"):
    #                     current_name = line.rstrip(":")
    #                 elif line.startswith("/dev/video"):
    #                     devices.append({
    #                         "path": line,
    #                         "name": current_name or line,
    #                         "device_id": _resolve_device_id(line),
    #                     })

    #         for path in sorted(glob.glob("/dev/video*")):
    #             if any(existing.get("path") == path for existing in devices):
    #                 continue
    #             devices.append({
    #                 "path": path,
    #                 "name": path,
    #                 "device_id": _resolve_device_id(path),
    #             })
    #     else:
    #         with self.lock:
    #             claimed = {s for s in self.sources.values() if s}

    #         for path in sorted(glob.glob("/dev/video*")):
    #             devices.append({
    #                 "path": path, "name": path,
    #                 "device_id": _resolve_device_id(path),
    #             })
    #         if not devices:
    #             for i in range(0, 16):
    #                 if str(i) in claimed:
    #                     devices.append({"path": str(i), "name": f"Camera {i}", "device_id": None})
    #                     continue
    #                 # Avoid repeated index-based OpenCV probe attempts on Windows,
    #                 # which trigger the noisy DSHOW errors when no device is present.
    #                 if platform.system() == "Windows":
    #                     break
    #                 try:
    #                     cap = None
    #                     for backend in [cv2.CAP_ANY]:
    #                         try:
    #                             cap = cv2.VideoCapture(i, backend)
    #                         except Exception:
    #                             cap = None
    #                             continue
    #                         if cap is not None and cap.isOpened():
    #                             break
    #                         cap = None
    #                     if cap is not None and cap.isOpened():
    #                         w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #                         h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #                         devices.append({
    #                             "path": str(i), "name": f"Camera {i}",
    #                             "resolution": f"{w}x{h}" if w and h else None,
    #                             "device_id": None,
    #                         })
    #                     if cap is not None:
    #                         cap.release()
    #                 except Exception:
    #                     continue

    #     seen = {}
    #     for dev in devices:
    #         key = dev.get("device_id") or dev.get("path")
    #         if key in seen:
    #             continue
    #         seen[key] = dev
    #     return list(seen.values())


    def _probe_devices(self):
        devices = []
        system_name = platform.system()

        if system_name == "Linux":
            # 1. Parse v4l2-ctl cleanly, keeping only primary video capture nodes (avoiding duplicates)
            try:
                v4l = subprocess.run(
                    ["v4l2-ctl", "--list-devices"],
                    capture_output=True, text=True, timeout=3,
                )
                if v4l.returncode == 0:
                    current_name = None
                    for line in v4l.stdout.splitlines():
                        line = line.strip()
                        if not line:
                            current_name = None
                            continue
                        if line.endswith(":"):
                            current_name = line.rstrip(":")
                        elif line.startswith("/dev/video"):
                            # Linux double-detection fix: 
                            # Usually, odd-numbered video nodes (e.g. /dev/video1, /dev/video3) 
                            # are metadata/output companion nodes. Only accept even or primary nodes 
                            # unless no other choice exists, or filter out nodes containing "meta".
                            if "meta" in line.lower():
                                continue
                                
                            devices.append({
                                "path": line,
                                "name": current_name or line,
                                "device_id": None,
                            })
            except Exception:
                pass

            # Fallback or additional glob check, filtered to avoid duplicates already found via v4l2-ctl
            existing_paths = {d["path"] for d in devices}
            for path in sorted(glob.glob("/dev/video*")):
                if path in existing_paths or "meta" in path.lower():
                    continue
                devices.append({
                    "path": path,
                    "name": path,
                    "device_id": None,
                })

        elif system_name == "Windows":
            # Registry enumeration of video capture sources - no index-range
            # probing and no transient VideoCapture opens (which would flicker
            # the stream and spam stderr). Index order matches CAP_MSMF.
            with self.lock:
                claimed = {s for s in self.sources.values() if s}
            for cam in _windows_cameras_from_registry():
                index = str(cam["index"])
                if index in claimed:
                    continue
                devices.append({
                    "path": index,
                    "name": cam["name"],
                    "device_id": None,
                })

        else:
            # macOS / other: best-effort /dev/video glob plus index probing.
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

        # Deduplicate final list
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