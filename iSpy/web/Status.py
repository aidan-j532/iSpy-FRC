# This will be like the HealthReporter upgraded. Remove health reporter from plugins and merge with this
# this will also display plugin status as well and lots of info.
# Status page. This is the pattern every other web/*.py file should copy:
#   1. one Blueprint
#   2. one route that renders a template (extends base.html)
#   3. one or more /api/* routes that the template's JS polls via ISPY_LIVE
#
# This replaces the old HealthReporter utility plugin - its _build_payload
# logic moves here, and health data now flows through the single shared
# Flask app instead of a route bolted onto whatever flask_app the plugin
# context happened to hand it.

import time
import logging
from flask import Blueprint, render_template, jsonify, current_app

logger = logging.getLogger(__name__)

bp = Blueprint("status", __name__, url_prefix="/status")

_state = {
    "fps": 0.0,
    "vision_ms": 0.0,
    "detections": 0,
    "last_tick": time.perf_counter(),
    "uptime_start": time.perf_counter(),
    "loop_count": 0,
}


def record_frame_data(frame_data: dict) -> None:
    """Call this from iSpy's main loop (same place HealthReporter.update()
    used to be called) to feed live numbers to /status/api/status."""
    _state["fps"] = round(frame_data.get("fps", 0), 1)
    _state["vision_ms"] = round(frame_data.get("vision_s", 0) * 1000, 2)
    _state["detections"] = frame_data.get("detections", 0)
    _state["last_tick"] = time.perf_counter()
    _state["loop_count"] += 1


@bp.route("/")
def status_page():
    return render_template("status.html", active_page="status")


@bp.route("/api/status")
def api_status():
    now = time.perf_counter()
    stale_s = round(now - _state["last_tick"], 2)
    uptime_s = round(now - _state["uptime_start"], 1)

    cameras = current_app.config.get("ISPY_CAMERAS", [])
    cameras_data = []
    all_ok = True
    for cam in cameras:
        try:
            age = cam.get_frame_age()
            ok = age < 1.0
            name = cam.config.get("name", str(cam.source)) if hasattr(cam, "config") else str(cam.source)
            cameras_data.append({"name": name, "ok": ok, "frame_age_ms": round(age * 1000, 1)})
            all_ok = all_ok and ok
        except Exception:
            cameras_data.append({"name": str(getattr(cam, "source", "unknown")), "ok": False, "frame_age_ms": None})
            all_ok = False

    healthy = stale_s < 1.0 and all_ok
    payload = {
        "status": "ok" if healthy else "degraded",
        "uptime_s": uptime_s,
        "loop_count": _state["loop_count"],
        "loop_stale_s": stale_s,
        "fps": _state["fps"],
        "vision_ms": _state["vision_ms"],
        "detections": _state["detections"],
        "cameras": cameras_data,
    }
    return jsonify(payload), (200 if healthy else 503)