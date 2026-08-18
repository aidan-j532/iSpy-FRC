import time
import threading
import logging
from iSpy.plugins.bases import UtilityBase

try:
    from flask import jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


class StatusReporter(UtilityBase):
    plugin_name = "status_reporter"

    def __init__(self, context: dict):
        config = context["config"]
        flask_app = context.get("flask_app")
        self.cameras = context.get("cameras", [])
        self.logger = logging.getLogger(__name__)

        self._lock = threading.Lock()
        self._fps = 0.0
        self._vision_s = 0.0
        self._detections = 0
        self._last_tick = time.perf_counter()
        self._uptime_start = time.perf_counter()
        # stale_threshold lives in health_reporter add-on now
        self._stale_threshold = config.get_addon_setting(
            "utilities", "health_reporter", "stale_threshold", 1.0
        )
        self._network_handler = None
        self._plugins = {"trackers": [], "utilities": [], "frame_processors": []}

        if flask_app and FLASK_AVAILABLE:
            flask_app.add_url_rule("/health", "health", self._health_route)
        elif not FLASK_AVAILABLE:
            self.logger.warning("Flask not available - /health disabled.")

    def set_network_handler(self, handler):
        self._network_handler = handler

    def set_plugins(self, trackers: dict, utilities: dict, frame_processors: dict):
        self._plugins = {
            "trackers": list(trackers.keys()),
            "utilities": list(utilities.keys()),
            "frame_processors": list(frame_processors.keys()),
        }

    def update(self, frame_data: dict):
        with self._lock:
            self._fps = round(frame_data.get("fps", 0), 1)
            self._vision_s = round(frame_data.get("vision_s", 0) * 1000, 2)
            self._detections = frame_data.get("detection_count", 0)
            self._last_tick = time.perf_counter()

    def stop(self):
        pass

    def _health_route(self):
        now = time.perf_counter()
        with self._lock:
            fps, vision_ms, detections, last_tick = self._fps, self._vision_s, self._detections, self._last_tick

        stale_s = round(now - last_tick, 2)
        uptime_s = round(now - self._uptime_start, 1)

        cameras_data, all_ok = [], True
        for cam in self.cameras:
            try:
                age = cam.get_frame_age()
                ok = age < self._stale_threshold
                name = cam.config.get("name", str(cam.source)) if hasattr(cam, "config") else str(cam.source)
                cameras_data.append({"name": name, "source": str(cam.source), "ok": ok, "frame_age_ms": round(age * 1000, 1)})
                all_ok = all_ok and ok
            except Exception:
                cameras_data.append({"name": str(getattr(cam, "source", "?")), "source": "?", "ok": False, "frame_age_ms": None})
                all_ok = False

        nt_connected = None
        if self._network_handler is not None:
            try:
                nt_connected = self._network_handler.isConnected()
            except Exception:
                nt_connected = False

        healthy = stale_s < self._stale_threshold and all_ok and (nt_connected is None or nt_connected)
        payload = {
            "status": "ok" if healthy else "degraded",
            "uptime_s": uptime_s, "loop_stale_s": stale_s,
            "fps": fps, "vision_ms": vision_ms, "detections": detections,
            "cameras": cameras_data,
            "network_tables": {"enabled": self._network_handler is not None, "connected": nt_connected},
            "plugins": self._plugins,
        }
        return jsonify(payload), (200 if healthy else 503)
