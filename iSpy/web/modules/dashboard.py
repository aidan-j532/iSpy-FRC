import time
from flask import jsonify, render_template
from iSpy.web.Backend.WebModule import WebModule


class DashboardModule(WebModule):
    plugin_name = "dashboard"

    def __init__(self, context: dict, other_modules_ref=None):
        super().__init__(context)
        self._web_app = other_modules_ref  # so we can read camera names etc.
        self._latest: dict = {}
        self._start_time = time.perf_counter()

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/dashboard", "dashboard_page", lambda: render_template("dashboard.html"))
        flask_app.add_url_rule("/api/status", "api_status", self._api_status)

    def update(self, frame_data: dict):
        self._latest = {
            "fps": round(frame_data.get("fps", 0), 1),
            "vision_ms": round(frame_data.get("vision_s", 0) * 1000, 1),
            "camera_lag_ms": round(frame_data.get("camera_lag_s", 0) * 1000, 1),
            "detections": frame_data.get("detections", 0),
            "uptime_s": round(time.perf_counter() - self._start_time, 1),
        }

    def _api_status(self):
        cameras = self.context.get("cameras") or []
        cam_status = []
        for cam in cameras:
            try:
                age = cam.get_frame_age()
                name = cam.config.get("name", str(cam.source)) if hasattr(cam, "config") else str(cam.source)
                cam_status.append({"name": name, "ok": age < 1.0, "frame_age_ms": round(age * 1000, 1)})
            except Exception:
                cam_status.append({"name": "unknown", "ok": False, "frame_age_ms": None})
        return jsonify({**self._latest, "cameras": cam_status})