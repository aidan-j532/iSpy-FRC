import threading
import cv2
import time
from flask import Response, jsonify, render_template
import numpy as np
from iSpy.web.Backend.WebModule import WebModule


class CamerasModule(WebModule):
    plugin_name = "cameras"

    def __init__(self, context: dict):
        super().__init__(context)
        self.lock = threading.Lock()
        self.frames: dict[str, "np.ndarray"] = {}
        self.dims: dict[str, tuple[int, int]] = {}

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/cameras", "cameras_page", lambda: render_template("cameras.html"))
        flask_app.add_url_rule("/api/cameras", "api_cameras", self._api_cameras)
        flask_app.add_url_rule("/video/<camera_name>", "video_feed", self._video_feed)

    def update(self, frame_data: dict):
        frame = frame_data.get("frame")
        if frame is None:
            return
        cameras = frame_data.get("cameras") or []
        # Solo mode: one camera, one frame. Multi mode: iSpy.py should set
        # frame_data["camera_frames"] = {name: frame} instead (see iSpy.py).
        per_cam = frame_data.get("camera_frames")
        with self.lock:
            if per_cam:
                for name, f in per_cam.items():
                    if f is not None:
                        self.frames[name] = f
                        self.dims[name] = f.shape[1], f.shape[0]
            elif cameras:
                name = cameras[0].config.get("name", "camera_1") if hasattr(cameras[0], "config") else "camera_1"
                self.frames[name] = frame
                self.dims[name] = frame.shape[1], frame.shape[0]
            else:
                self.frames["camera_1"] = frame
                self.dims["camera_1"] = frame.shape[1], frame.shape[0]

    def _api_cameras(self):
        with self.lock:
            names = list(self.frames.keys())
        return jsonify(cameras=[{"name": n} for n in names])

    def _video_feed(self, camera_name):
        return Response(
            self._generate(camera_name),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _generate(self, camera_name):
        target_interval = 1.0 / 20
        while True:
            t0 = time.perf_counter()
            with self.lock:
                frame = self.frames.get(camera_name)
            if frame is None:
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            elapsed = time.perf_counter() - t0
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)