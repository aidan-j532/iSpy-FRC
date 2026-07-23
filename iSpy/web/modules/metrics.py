# iSpy/web/modules/metrics.py
import json
import os
import time
from datetime import datetime
from collections import deque
from flask import jsonify, render_template
from iSpy.web.Backend.WebModule import WebModule

class MetricsModule(WebModule):
    plugin_name = "metrics"

    SERIES = {
        "loop_s": ("Loop time", "ms", 1000.0),
        "vision_s": ("Vision time", "ms", 1000.0),
        "camera_lag_s": ("Camera lag", "ms", 1000.0),
    }
    MAX_POINTS = 600  # ring buffer - keeps memory bounded on long runs

    def __init__(self, context: dict):
        super().__init__(context)
        self._timeline = {k: deque(maxlen=self.MAX_POINTS) for k in self.SERIES}
        self._fps_timeline = deque(maxlen=self.MAX_POINTS)
        self._start = time.perf_counter()

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/metrics", "metrics_page", lambda: render_template("metrics.html"))
        flask_app.add_url_rule("/api/metrics", "api_metrics", self._api_metrics)

    def update(self, frame_data: dict):
        t = time.perf_counter() - self._start
        for key in self.SERIES:
            val = frame_data.get(key)
            if val is not None:
                self._timeline[key].append((t, val))
        loop_s = frame_data.get("loop_s")
        if loop_s:
            self._fps_timeline.append((t, 1.0 / loop_s if loop_s > 0 else 0))

    def _api_metrics(self):
        out = {}
        for key, (label, unit, scale) in self.SERIES.items():
            pts = list(self._timeline[key])
            out[key] = {"label": label, "unit": unit,
                        "x": [p[0] for p in pts], "y": [p[1] * scale for p in pts]}
        out["fps"] = {
            "label": "FPS", "unit": "fps",
            "x": [p[0] for p in self._fps_timeline],
            "y": [p[1] for p in self._fps_timeline],
        }
        return jsonify(out)

    def stop(self):
        os.makedirs("Outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join("Outputs", f"metrics_{timestamp}.json")
        data = {}
        for key, (label, unit, scale) in self.SERIES.items():
            pts = list(self._timeline[key])
            data[key] = {"label": label, "unit": unit,
                         "x": [p[0] for p in pts], "y": [p[1] * scale for p in pts]}
        data["fps"] = {
            "label": "FPS", "unit": "fps",
            "x": [p[0] for p in self._fps_timeline],
            "y": [p[1] for p in self._fps_timeline],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)