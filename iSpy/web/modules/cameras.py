import glob
import os
import platform
import subprocess
import threading
import cv2
import time
from flask import Response, jsonify, render_template
import numpy as np
from iSpy.web.Backend.WebModule import WebModule

_STALE_EVICT_S = 10.0
_FEED_TIMEOUT_S = 15.0


class CamerasModule(WebModule):
    plugin_name = "cameras"

    def __init__(self, context: dict):
        super().__init__(context)
        self.lock = threading.Lock()
        self.frames: dict[str, "np.ndarray"] = {}
        self.dims: dict[str, tuple[int, int]] = {}
        self.last_seen: dict[str, float] = {}

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/cameras", "cameras_page", lambda: render_template("cameras.html"))
        flask_app.add_url_rule("/api/cameras", "api_cameras", self._api_cameras)
        flask_app.add_url_rule("/api/cameras/discover", "api_cameras_discover", self._discover)
        flask_app.add_url_rule("/video/<camera_name>", "video_feed", self._video_feed)

    def update(self, frame_data: dict):
        frame = frame_data.get("frame")
        if frame is None:
            return
        cameras = frame_data.get("cameras") or []
        per_cam = frame_data.get("camera_frames")
        now = time.monotonic()
        with self.lock:
            if per_cam:
                for name, f in per_cam.items():
                    if f is not None:
                        self.frames[name] = f
                        self.dims[name] = f.shape[1], f.shape[0]
                        self.last_seen[name] = now
            elif cameras:
                name = cameras[0].config.get("name", "camera_1") if hasattr(cameras[0], "config") else "camera_1"
                self.frames[name] = frame
                self.dims[name] = frame.shape[1], frame.shape[0]
                self.last_seen[name] = now
            else:
                self.frames["camera_1"] = frame
                self.dims["camera_1"] = frame.shape[1], frame.shape[0]
                self.last_seen["camera_1"] = now

            self._evict_stale(now)

    def _evict_stale(self, now: float):
        stale = [n for n, ts in self.last_seen.items() if now - ts > _STALE_EVICT_S]
        for n in stale:
            self.frames.pop(n, None)
            self.dims.pop(n, None)
            self.last_seen.pop(n, None)

    def _api_cameras(self):
        now = time.monotonic()
        with self.lock:
            self._evict_stale(now)
            cameras = [
                {"name": n, "w": d[0], "h": d[1], "age_ms": round((now - self.last_seen.get(n, now)) * 1000, 1)}
                for n, d in self.dims.items()
            ]
        return jsonify(cameras=cameras)

    def _discover(self):
        devices = []
        if platform.system() == "Linux":
            v4l = subprocess.run(
                ["v4l2-ctl", "--list-devices"],
                capture_output=True, text=True, timeout=5,
            )
            if v4l.returncode == 0:
                current_name = None
                for line in v4l.stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.endswith(":"):
                        current_name = line.rstrip(":")
                    elif line.startswith("/dev/video"):
                        devices.append({"path": line, "name": current_name or line})
        else:
            for path in sorted(glob.glob("/dev/video*")):
                devices.append({"path": path, "name": path})
            if not devices:
                for i in range(10):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        devices.append({"path": str(i), "name": f"Camera {i}", "resolution": f"{w}x{h}" if w and h else None})
                        cap.release()

        with self.lock:
            active = set(self.frames.keys())

        for dev in devices:
            dev["active"] = any(
                dev["path"] in n or n in dev["path"]
                for n in active
            )

        return jsonify(devices=devices)

    def _video_feed(self, camera_name):
        return Response(
            self._generate(camera_name),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _generate(self, camera_name):
        target_interval = 1.0 / 20
        last_frame_time = time.monotonic()
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
