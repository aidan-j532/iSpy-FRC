import time
import threading
import logging
from flask import jsonify, render_template
from iSpy.web.Backend.WebModule import WebModule

logger = logging.getLogger(__name__)


class HealthModule(WebModule):
    plugin_name = "health"

    def __init__(self, context: dict):
        super().__init__(context)
        self.logger = logging.getLogger(__name__)
        config = context["config"]
        self.cameras = context.get("cameras", [])

        self._lock = threading.Lock()
        self._fps = 0.0
        self._vision_s = 0.0
        self._detections = 0
        self._loop_count = 0
        self._last_tick = time.perf_counter()
        self._uptime_start = time.perf_counter()
        # stale_threshold moved from the global config into the health_reporter add-on
        self._stale_threshold = config.get_addon_setting(
            "utilities", "health_reporter", "stale_threshold", 1.0
        )
        self._network_handler = None  # wired externally, see iSpy.py

    def set_network_handler(self, handler):
        self._network_handler = handler

    def set_cameras(self, cameras):
        self.cameras = cameras

    def register_routes(self, flask_app):
        # minimal stable contract for watchdogs - external tooling may depend on the shape
        flask_app.add_url_rule("/health", "health", self._health_route)
        # fuller payload: uptime, loop_count, per-cam detail
        flask_app.add_url_rule("/health/detailed", "health_detailed", self._detailed_route)
        # human-facing page: everything above + live plugin status
        flask_app.add_url_rule("/health-page", "health_page", lambda: render_template("health.html"))
        flask_app.add_url_rule("/api/health", "api_health", self._api_health)

    def update(self, frame_data: dict):
        with self._lock:
            self._fps = round(frame_data.get("fps", 0), 1)
            self._vision_s = round(frame_data.get("vision_s", 0) * 1000, 2)
            self._detections = frame_data.get("detection_count", 0)
            self._last_tick = time.perf_counter()
            self._loop_count += 1

    def _camera_status(self):
        cameras_data = []
        all_ok = True
        for cam in self.cameras:
            try:
                age = cam.get_frame_age()
                ok = age < self._stale_threshold
                name = cam.config.get("name", str(cam.source)) if hasattr(cam, "config") else str(cam.source)
                cameras_data.append({
                    "name": name, "source": str(getattr(cam, "source", "?")),
                    "ok": ok, "frame_age_ms": round(age * 1000, 1),
                })
                all_ok = all_ok and ok
            except Exception:
                cameras_data.append({
                    "name": str(getattr(cam, "source", "?")), "source": "?",
                    "ok": False, "frame_age_ms": None,
                })
                all_ok = False
        return cameras_data, all_ok

    def _build_payload(self):
        now = time.perf_counter()
        with self._lock:
            fps, vision_ms, detections = self._fps, self._vision_s, self._detections
            last_tick, loop_count = self._last_tick, self._loop_count

        stale_s = round(now - last_tick, 2)
        uptime_s = round(now - self._uptime_start, 1)
        cameras_data, all_cams_ok = self._camera_status()

        nt_connected = None
        if self._network_handler is not None:
            try:
                nt_connected = self._network_handler.isConnected()
            except Exception:
                nt_connected = False

        healthy = stale_s < self._stale_threshold and all_cams_ok and (nt_connected is None or nt_connected)

        return {
            "status": "ok" if healthy else "degraded",
            "uptime_s": uptime_s,
            "loop_count": loop_count,
            "loop_stale_s": stale_s,
            "fps": fps,
            "vision_ms": vision_ms,
            "detections": detections,
            "cameras": cameras_data,
            "network_tables": {"enabled": self._network_handler is not None, "connected": nt_connected},
        }, healthy

    def _plugin_statuses(self):
        vision = self.context.get("vision_instance")
        if not vision:
            return []
        out = []
        for group, items in (
            ("tracker", vision.trackers),
            ("utility", vision.utilities),
            ("frame_processor", vision.frame_processors),
        ):
            for name, inst in items.items():
                out.append({
                    "name": name, "type": group,
                    "status": inst.get_status() if hasattr(inst, "get_status") else "unknown",
                })
        return out

    def _health_route(self):
        payload, healthy = self._build_payload()
        # keep the body minimal/stable for watchdogs
        return jsonify(status=payload["status"], uptime_s=payload["uptime_s"]), (200 if healthy else 503)

    def _detailed_route(self):
        payload, healthy = self._build_payload()
        return jsonify(payload), (200 if healthy else 503)

    def _api_health(self):
        payload, healthy = self._build_payload()
        payload["plugins"] = self._plugin_statuses()
        return jsonify(payload), (200 if healthy else 503)

    def stop(self):
        pass