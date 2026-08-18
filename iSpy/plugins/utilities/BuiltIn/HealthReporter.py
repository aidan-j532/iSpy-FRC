import time
import threading
import logging
from iSpy.plugins.bases import UtilityBase

try:
    from flask import jsonify, Response, request

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>System Health</title>
  <style>
    body { font-family: Arial; background: #111; color: #eee; margin: 0; padding: 20px; }
    h1   { text-align: center; }
    .card {
      padding: 20px; margin: 20px auto; max-width: 600px;
      border-radius: 10px; background: #222;
    }
    .ok  { color: #4caf50; }
    .bad { color: #f44336; font-weight: bold; }
  </style>
</head>
<body>
  <h1>System Health</h1>

  <div class="card">
    <h2>Status: <span id="status_text"></span></h2>
    <p><b>FPS:</b> <span id="fps"></span></p>
    <p><b>Inference:</b> <span id="vision"></span> ms</p>
    <p><b>Detections:</b> <span id="detections"></span></p>
    <p><b>Loop stale:</b> <span id="stale"></span> s</p>
    <p><b>Uptime:</b> <span id="uptime"></span> s</p>
    <p><b>Loop count:</b> <span id="loop_count"></span></p>
  </div>

  <div id="cameras-container"></div>

  <div class="card">
    <h3>NetworkTables</h3>
    <p>Enabled: <span id="nt_enabled"></span></p>
    <p>Connected: <span id="nt_connected"></span></p>
  </div>

  <script>
    async function refresh() {
      try {
        const data = await fetch('/health/detailed', {
          headers: { 'Accept': 'application/json' }
        }).then(r => r.json());

        document.getElementById('fps').textContent        = data.fps;
        document.getElementById('vision').textContent     = data.vision_ms;
        document.getElementById('detections').textContent = data.detections;
        document.getElementById('stale').textContent      = data.loop_stale_s;
        document.getElementById('uptime').textContent     = data.uptime_s;
        document.getElementById('loop_count').textContent = data.loop_count;

        const s = document.getElementById('status_text');
        s.textContent = data.status.toUpperCase();
        s.className   = data.status === 'ok' ? 'ok' : 'bad';

        const container = document.getElementById('cameras-container');
        container.innerHTML = '';
        if (data.cameras && data.cameras.length) {
          data.cameras.forEach(function(cam) {
            const card = document.createElement('div');
            card.className = 'card';
            card.innerHTML = '<h3>' + cam.name + '</h3>' +
              '<p>Source: ' + cam.source + '</p>' +
              '<p>Status: <span class="' + (cam.ok ? 'ok' : 'bad') + '">' +
              (cam.ok ? 'OK' : 'BAD') + '</span></p>' +
              '<p>Frame age: ' + (cam.frame_age_ms != null ? cam.frame_age_ms : 'N/A') + ' ms</p>';
            container.appendChild(card);
          });
        }

        document.getElementById('nt_enabled').textContent =
          data.network_tables.enabled;
        document.getElementById('nt_connected').textContent =
          data.network_tables.connected;
      } catch (e) {
        console.error('Health refresh failed', e);
      }
    }
    setInterval(refresh, 250);
    refresh();
  </script>
</body>
</html>"""

class HealthReporter(UtilityBase):
    plugin_name = "health_reporter"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "stale_threshold": {
                "type": "number",
                "label": "Stale Threshold (s)",
                "hint": "Frames older than this (s) mark a camera or the main "
                        "loop as stale/degraded.",
                "default": 1.0,
            },
        }

    def __init__(self, context: dict):
        super().__init__(context)
        flask_app = context.get("flask_app")
        self.cameras = context.get("cameras", [])
        self.logger = logging.getLogger(__name__)

        self._lock = threading.Lock()
        self._fps = 0.0
        self._vision_s = 0.0
        self._detections = 0
        self._last_tick = time.perf_counter()
        self._uptime_start = time.perf_counter()
        self._loop_count = 0
        # stale_threshold lives in this add-on's settings now
        self._stale_threshold = self.config.get("stale_threshold", 1.0)
        self._network_handler = None  # set externally after all utilities load

        if flask_app and FLASK_AVAILABLE:
            # health module registers this route at startup - re-registering breaks after
            # the first request, so only add it if it isn't already present.
            registered = any(
                rule.endpoint == "health_detailed"
                for rule in flask_app.url_map.iter_rules()
            )
            if not registered:
                try:
                    flask_app.add_url_rule(
                        "/health/detailed", "health_detailed", self._health_route
                    )
                except AssertionError:
                    self.logger.warning(
                        "/health/detailed already registered - skipping."
                    )
        elif not FLASK_AVAILABLE:
            self.logger.warning("Flask not available - /health/detailed endpoint disabled.")

    def set_network_handler(self, handler):
        self._network_handler = handler

    def update(self, frame_data: dict):
        with self._lock:
            self._fps = round(frame_data.get("fps", 0), 1)
            self._vision_s = round(frame_data.get("vision_s", 0) * 1000, 2)
            self._detections = frame_data.get("detection_count", 0)
            self._last_tick = time.perf_counter()
            self._loop_count += 1

    def stop(self):
        pass

    def _build_payload(self):
        now = time.perf_counter()
        with self._lock:
            fps = self._fps
            vision_ms = self._vision_s
            detections = self._detections
            last_tick = self._last_tick
            loop_count = self._loop_count

        stale_s = round(now - last_tick, 2)
        uptime_s = round(now - self._uptime_start, 1)

        cameras_data = []
        all_cameras_ok = True
        if self.cameras:
            for cam in self.cameras:
                try:
                    age = cam.get_frame_age()
                    cam_ok = age < self._stale_threshold
                    cam_name = (
                        cam.config.get("name", str(cam.source))
                        if hasattr(cam, "config")
                        else str(cam.source)
                    )
                    cameras_data.append({
                        "name": cam_name,
                        "source": str(cam.source),
                        "ok": cam_ok,
                        "frame_age_ms": round(age * 1000, 1),
                    })
                    if not cam_ok:
                        all_cameras_ok = False
                except Exception:
                    cameras_data.append({
                        "name": str(getattr(cam, "source", "unknown")),
                        "source": str(getattr(cam, "source", "unknown")),
                        "ok": False,
                        "frame_age_ms": None,
                    })
                    all_cameras_ok = False

        nt_connected = None
        if self._network_handler is not None:
            try:
                nt_connected = self._network_handler.isConnected()
            except Exception:
                nt_connected = False

        healthy = (
            stale_s < self._stale_threshold
            and all_cameras_ok
            and (nt_connected is None or nt_connected)
        )

        payload = {
            "status": "ok" if healthy else "degraded",
            "uptime_s": uptime_s,
            "loop_count": loop_count,
            "loop_stale_s": stale_s,
            "fps": fps,
            "vision_ms": vision_ms,
            "detections": detections,
            "cameras": cameras_data,
            "network_tables": {
                "enabled": self._network_handler is not None,
                "connected": nt_connected,
            },
        }
        return payload, healthy

    def _health_route(self):
        payload, healthy = self._build_payload()
        if "text/html" in request.headers.get("Accept", ""):
            return Response(_HTML, mimetype="text/html")
        return jsonify(payload), (200 if healthy else 503)
