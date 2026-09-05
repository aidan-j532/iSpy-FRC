import time
import threading
import logging
from flask import jsonify, render_template
from iSpy.web.Backend.WebModule import WebModule

logger = logging.getLogger(__name__)

#: the only color presets a health contributor may choose. Everything else
#: falls back to green (or red when the legacy "ok": False flag is used).
COLOR_PRESETS = ("green", "yellow", "red")


class HealthModule(WebModule):
    plugin_name = "health"

    def __init__(self, context: dict):
        super().__init__(context)
        self.logger = logging.getLogger(__name__)
        config = context["config"]
        self._lock = threading.Lock()
        self.cameras = context.get("cameras", [])
        self._fps = 0.0
        self._vision_s = 0.0
        self._detections = 0
        self._loop_count = 0
        self._last_tick = time.perf_counter()
        self._uptime_start = time.perf_counter()
        # single canonical health implementation (PROMPT 5 merged the old
        # health_reporter/status_reporter add-ons into this always-on module)
        self._stale_threshold = float(config.get("health_stale_threshold", 1.0) or 1.0)

    def set_cameras(self, cameras):
        with self._lock:
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
        with self._lock:
            cams = list(self.cameras)
        for cam in cams:
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
        addon_health = self._collect_addon_health()
        addons_ok = all(item["color"] != "red" for item in addon_health)

        healthy = stale_s < self._stale_threshold and all_cams_ok and addons_ok

        return {
            "status": "ok" if healthy else "degraded",
            "uptime_s": uptime_s,
            "loop_count": loop_count,
            "loop_stale_s": stale_s,
            "fps": fps,
            "vision_ms": vision_ms,
            "detections": detections,
            "cameras": cameras_data,
            "addon_health": addon_health,
        }, healthy

    def _collect_addon_health(self) -> list:
        """Collect health rows from vision pipelines and utilities.

        Only vision pipelines and utilities may contribute to the Health tab;
        trackers and frame processors don't get a row. Each contributor
        returns a dict::

            {
                "color": "green" | "yellow" | "red",   # preset, their choice
                "state": str,                           # their own state text
                "metrics": [{"label", "value"}, ...],   # live values, cycled
            }

        The legacy {"ok", "title", "info", "rows"} shape is still honored so
        older add-ons degrade gracefully instead of breaking the page.
        Contributors that raise are reported as a red "error" row.
        """
        vision = self.context.get("vision_instance")
        collected = []
        groups = []
        if vision is not None:
            utilities = getattr(vision, "utilities", None) or {}
            for name, inst in utilities.items():
                groups.append(("utility", name, inst))
        with self._lock:
            cameras = list(self.cameras)
        for cam in cameras:
            groups.append(("pipeline", str(getattr(cam, "source", "camera")), cam))

        for group, name, inst in groups:
            fn = getattr(inst, "get_health", None)
            if not callable(fn):
                continue
            try:
                data = fn() or {}
            except Exception:
                self.logger.exception("get_health() raised for %s %s", group, name)
                data = {"ok": False, "state": "get_health() raised", "metrics": []}
            if not isinstance(data, dict):
                continue

            color = str(data.get("color", "")).lower()
            if color not in COLOR_PRESETS:
                color = "green" if bool(data.get("ok", True)) else "red"

            state = str(data.get("state")
                        or data.get("info")
                        or data.get("title")
                        or name)

            metrics = data.get("metrics") or data.get("rows") or []
            normalized = []
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue
                label = str(metric.get("label") or "")
                value = metric.get("value")
                normalized.append({
                    "label": label,
                    "value": "" if value is None else str(value),
                })

            collected.append({
                "name": name,
                "type": group,
                "color": color,
                "state": state,
                "metrics": normalized,
            })
        return collected

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