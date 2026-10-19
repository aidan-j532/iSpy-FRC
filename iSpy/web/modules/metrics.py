import json
import os
import time
from datetime import datetime
from collections import deque
from pathlib import Path
from flask import jsonify, render_template
from iSpy.core import performance
from iSpy.web.Backend.WebModule import WebModule


class MetricsModule(WebModule):
    plugin_name = "metrics"

    SERIES = {
        "loop_s": ("Loop time", "ms", 1000.0),
        "vision_s": ("Vision time", "ms", 1000.0),
        "camera_lag_s": ("Camera lag", "ms", 1000.0),
    }
    # built-in, always-present breakdown parts: {key: (label, color)}.
    # Display-only, never written to the saved file. Add-ons can register extra
    # parts at runtime via set_code_parts() so any addon (tracker, utility,
    # frame processor, vision pipeline) can show up here if it wants to.
    CODE_PARTS = {
        "vision": ("Vision", "#e63946"),
        "trackers": ("Trackers", "#f4a261"),
        "utilities": ("Utilities", "#457b9d"),
        "web": ("Web", "#9b5de5"),
    }
    MAX_POINTS = 600  # ring buffer, keeps memory bounded on long runs
    # the metrics page saves on every unload (sendBeacon), so without a cap
    # a long-lived tab left open/closed repeatedly fills Outputs with
    # metrics_*.json forever. Keep only the most recent snapshots.
    MAX_SAVED_FILES = 20

    def __init__(self, context: dict):
        super().__init__(context)
        self._timeline = {k: deque(maxlen=self.MAX_POINTS) for k in self.SERIES}
        self._fps_timeline = deque(maxlen=self.MAX_POINTS)
        self._code_parts = dict(self.CODE_PARTS)
        self._code_timeline = {
            k: deque(maxlen=self.MAX_POINTS) for k in self._code_parts
        }
        self._start = time.perf_counter()
        # last accepted sample (perf_counter) for the profile's sample-rate cap
        self._last_sample = 0.0

    def set_code_parts(self, parts: dict):
        for key, (label, color) in (parts or {}).items():
            if key in self._code_parts:
                continue
            self._code_parts[key] = (label, color)
            self._code_timeline[key] = deque(maxlen=self.MAX_POINTS)

    def register_routes(self, flask_app):
        flask_app.add_url_rule(
            "/metrics", "metrics_page", lambda: render_template("metrics.html")
        )
        flask_app.add_url_rule("/api/metrics", "api_metrics", self._api_metrics)
        flask_app.add_url_rule(
            "/api/metrics/save", "api_metrics_save", self._save, methods=["POST"]
        )

    def update(self, frame_data: dict):
        profile = performance.get_profile()
        if not profile.metrics_enabled:
            return
        now = time.perf_counter()
        hz = profile.metrics_sample_hz
        if hz > 0 and (now - self._last_sample) < (1.0 / hz):
            return
        self._last_sample = now

        t = now - self._start
        for key in self.SERIES:
            val = frame_data.get(key)
            if val is not None:
                self._timeline[key].append((t, val))
        loop_s = frame_data.get("loop_s")
        if loop_s:
            self._fps_timeline.append((t, 1.0 / loop_s if loop_s > 0 else 0))
        code_times = frame_data.get("code_times") or {}
        for key in self._code_parts:
            val = code_times.get(key)
            if val is not None:
                self._code_timeline[key].append((t, val * 1000.0))

    def _api_metrics(self):
        out = {}
        for key, (label, unit, scale) in self.SERIES.items():
            pts = list(self._timeline[key])
            out[key] = {
                "label": label,
                "unit": unit,
                "x": [p[0] for p in pts],
                "y": [p[1] * scale for p in pts],
            }
        out["fps"] = {
            "label": "FPS",
            "unit": "fps",
            "x": [p[0] for p in self._fps_timeline],
            "y": [p[1] for p in self._fps_timeline],
        }
        out["code_parts"] = {
            "label": "Loop breakdown",
            "unit": "ms",
            "series": {
                key: {
                    "label": label,
                    "color": color,
                    "x": [p[0] for p in self._code_timeline[key]],
                    "y": [p[1] for p in self._code_timeline[key]],
                }
                for key, (label, color) in self._code_parts.items()
            },
        }
        return jsonify(out)

    def stop(self):
        self._save_to_disk()

    def _save(self):
        saved = self._save_to_disk()
        return jsonify({"success": True, "saved": saved})

    def _has_data(self) -> bool:
        return bool(
            self._fps_timeline
            or any(self._timeline.values())
            or any(self._code_timeline.values())
        )

    def _save_to_disk(self) -> bool:
        # level 0 (and a run where nothing was ever sampled) must not drop an
        # empty metrics_*.json into Outputs on shutdown
        if not self._has_data():
            return False
        os.makedirs("Outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join("Outputs", f"metrics_{timestamp}.json")
        data = self._build_data()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        self._prune_saved()
        return True

    def _prune_saved(self):
        try:
            saved = sorted(Path("Outputs").glob("metrics_*.json"))
        except OSError:
            return
        for stale in saved[: -self.MAX_SAVED_FILES]:
            try:
                stale.unlink()
            except OSError:
                pass

    def _build_data(self):
        data = {}
        for key, (label, unit, scale) in self.SERIES.items():
            pts = list(self._timeline[key])
            data[key] = {
                "label": label,
                "unit": unit,
                "x": [p[0] for p in pts],
                "y": [p[1] * scale for p in pts],
            }
        data["fps"] = {
            "label": "FPS",
            "unit": "fps",
            "x": [p[0] for p in self._fps_timeline],
            "y": [p[1] for p in self._fps_timeline],
        }
        return data
