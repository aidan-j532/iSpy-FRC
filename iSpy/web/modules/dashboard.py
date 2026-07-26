import time
import json
import logging
import threading
from pathlib import Path
from flask import jsonify, render_template, Response
from iSpy.web.Backend.WebModule import WebModule

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class DashboardModule(WebModule):
    plugin_name = "dashboard"

    def __init__(self, context: dict):
        super().__init__(context)
        self._latest: dict = {
            "fps": 0, "vision_ms": 0, "camera_lag_ms": 0,
            "detections": 0, "loop_s": 0,
        }
        self._vision_last_tick: float = 0
        self._start_time = time.perf_counter()
        self._model_info: dict = {}
        self._plugin_info: dict = {"trackers": [], "utilities": [], "frame_processors": []}
        self._detection_classes: dict = {}
        self._sse_lock = threading.Lock()
        self._sse_clients: list = []

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/dashboard", "dashboard_page", lambda: render_template("dashboard.html"))
        flask_app.add_url_rule("/api/status", "api_status", self._api_status)
        flask_app.add_url_rule("/api/system", "api_system", self._api_system)
        flask_app.add_url_rule("/api/events", "api_events", self._sse_stream)

    def update(self, frame_data: dict):
        self._latest = {
            "fps": round(frame_data.get("fps", 0), 1),
            "vision_ms": round(frame_data.get("vision_s", 0) * 1000, 1),
            "camera_lag_ms": round(frame_data.get("camera_lag_s", 0) * 1000, 1),
            "detections": frame_data.get("detections", 0),
            "loop_s": round(frame_data.get("loop_s", 0) * 1000, 1),
            "uptime_s": round(time.perf_counter() - self._start_time, 1),
        }
        self._vision_last_tick = time.perf_counter()

        fuel_list = frame_data.get("fuel_list", [])
        self._detection_classes = {}
        for obj in fuel_list:
            cls_name = getattr(obj, "name", None) or "unknown"
            self._detection_classes[cls_name] = self._detection_classes.get(cls_name, 0) + 1

        if not self._model_info:
            self._refresh_model_info()

        self._push_sse({
            "type": "tick",
            **self._latest,
            "detection_classes": self._detection_classes,
            "cameras": self._get_camera_status(),
            "system": self._get_system_metrics(),
        })

    def _refresh_model_info(self):
        try:
            config = self.context.get("config")
            if not config:
                return
            model_cfg = config.get("vision_model", {})
            file_path = model_cfg.get("file_path", "")
            if file_path:
                p = Path(file_path)
                size_mb = round(p.stat().st_size / (1024 * 1024), 2) if p.exists() else 0
                self._model_info = {
                    "name": p.stem,
                    "format": p.suffix.lstrip("."),
                    "path": str(p),
                    "size_mb": size_mb,
                }
        except Exception:
            self._model_info = {}

    def set_plugins(self, trackers: dict, utilities: dict, frame_processors: dict):
        self._plugin_info = {
            "trackers": list(trackers.keys()),
            "utilities": list(utilities.keys()),
            "frame_processors": list(frame_processors.keys()),
        }

    def _get_system_metrics(self) -> dict:
        if not PSUTIL_AVAILABLE:
            return {"cpu_percent": None, "memory_percent": None, "memory_used_mb": None,
                    "memory_total_mb": None, "temperature": None}

        try:
            cpu = psutil.cpu_percent(interval=0.05)
        except Exception:
            cpu = None

        try:
            mem = psutil.virtual_memory()
            mem_used = round(mem.used / (1024 * 1024), 1)
            mem_total = round(mem.total / (1024 * 1024), 1)
        except Exception:
            mem_used = mem_total = None
            mem = None

        temp = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name in ("coretemp", "cpu_thermal", "soc_thermal", "k10temp"):
                    if name in temps and temps[name]:
                        temp = round(temps[name][0].current, 1)
                        break
                if temp is None:
                    first = next(iter(temps.values()), None)
                    if first:
                        temp = round(first[0].current, 1)
        except Exception:
            pass

        return {
            "cpu_percent": cpu,
            "memory_percent": round(mem.percent, 1) if mem else None,
            "memory_used_mb": mem_used,
            "memory_total_mb": mem_total,
            "temperature": temp,
        }

    def _get_camera_status(self) -> list[dict]:
        cameras = self.context.get("cameras") or []
        cam_status = []
        for cam in cameras:
            try:
                age = cam.get_frame_age()
                name = cam.config.get("name", str(cam.source)) if hasattr(cam, "config") else str(cam.source)
                resolution = None
                try:
                    frame = cam.get_frame() if hasattr(cam, "get_frame") else None
                    if frame is not None:
                        import numpy as np
                        if isinstance(frame, np.ndarray):
                            h, w = frame.shape[:2]
                            resolution = f"{w}x{h}"
                except Exception:
                    pass
                cam_status.append({
                    "name": name,
                    "ok": age < 1.0,
                    "stale": 1.0 <= age < 3.0,
                    "frame_age_ms": round(age * 1000, 1),
                    "resolution": resolution,
                })
            except Exception:
                cam_status.append({"name": "unknown", "ok": False, "stale": False,
                                   "frame_age_ms": None, "resolution": None})
        return cam_status

    def _build_full_payload(self) -> dict:
        vision_running = (time.perf_counter() - self._vision_last_tick) < 5.0 if self._vision_last_tick else False
        return {
            **self._latest,
            "vision_running": vision_running,
            "uptime_s": round(time.perf_counter() - self._start_time, 1),
            "cameras": self._get_camera_status(),
            "system": self._get_system_metrics(),
            "model": self._model_info,
            "detection_classes": self._detection_classes,
            "plugins": self._plugin_info,
        }

    def _api_status(self):
        return jsonify(self._build_full_payload())

    def _api_system(self):
        return jsonify({
            "system": self._get_system_metrics(),
            "cameras": self._get_camera_status(),
            "uptime_s": round(time.perf_counter() - self._start_time, 1),
        })

    def _sse_stream(self):
        def generate():
            q: list = []
            with self._sse_lock:
                self._sse_clients.append(q)
            try:
                yield f"data: {json.dumps(self._build_full_payload())}\n\n"
                while True:
                    while q:
                        payload = q.pop(0)
                        yield f"data: {json.dumps(payload)}\n\n"
                    time.sleep(0.05)
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
