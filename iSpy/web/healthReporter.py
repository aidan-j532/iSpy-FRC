import time
import threading
from iSpy.config.iSpyConfig import iSpyConfig
from flask import jsonify, Response, request

class HealthReporter:
    def __init__(self, flask_app, config: iSpyConfig):
        self._lock = threading.Lock()
        self._camera = None
        self._network_handler = None

        self._fps: float = 0.0
        self._vision_s: float = 0.0
        self._detections: int = 0
        self._last_tick: float = time.perf_counter()
        self._uptime_start: float = time.perf_counter()
        self._loop_count: int = 0
        self.stale_threshold = config.stale_threshold

        flask_app.add_url_rule("/health", "health", self._health_route)

    def set_camera(self, camera):
        self._camera = camera

    def set_network_handler(self, network_handler):
        self._network_handler = network_handler

    def tick(self, fps: float, vision_s: float, detections: int):
        with self._lock:
            self._fps = round(fps, 1)
            self._vision_s = round(vision_s * 1000, 2)
            self._detections = detections
            self._last_tick = time.perf_counter()
            self._loop_count += 1

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

        # Camera
        camera_ok = False
        frame_age_ms = None
        if self._camera is not None:
            try:
                age = self._camera.get_frame_age()
                frame_age_ms = round(age * 1000, 1)
                camera_ok = age < self.stale_threshold
            except Exception:
                camera_ok = False

        # NetworkTables
        nt_connected = None
        if self._network_handler is not None:
            try:
                nt_connected = self._network_handler.isConnected()
            except Exception:
                nt_connected = False

        healthy = (
            stale_s < self.stale_threshold
            and camera_ok
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
            "camera": {
                "ok": camera_ok,
                "frame_age_ms": frame_age_ms,
            },
            "network_tables": {
                "enabled": self._network_handler is not None,
                "connected": nt_connected,
            },
        }

        return payload, healthy

    def _render_html(self):
        return """
        <html>
        <head>
            <title>System Health</title>
            <style>
                body { font-family: Arial; background: #111; color: #eee; }
                .card { padding: 20px; margin: 20px; border-radius: 10px; background: #222; }
                .ok { color: #4caf50; }
                .bad { color: #f44336; font-weight: bold; }
                h1 {
                    text-align: center;
                }
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
            </div>

            <div class="card">
                <h3>Camera</h3>
                <p>Status: <span id="camera_status"></span></p>
                <p>Frame age: <span id="frame_age"></span> ms</p>
            </div>

            <div class="card">
                <h3>NetworkTables</h3>
                <p>Enabled: <span id="nt_enabled"></span></p>
                <p>Connected: <span id="nt_connected"></span></p>
            </div>

            <script>
            async function updateHealth() {
                try {
                    const res = await fetch('/health', {
                        headers: { 'Accept': 'application/json' }
                    });

                    const data = await res.json();

                    // Main stats
                    document.getElementById('fps').textContent = data.fps;
                    document.getElementById('vision').textContent = data.vision_ms;
                    document.getElementById('detections').textContent = data.detections;
                    document.getElementById('stale').textContent = data.loop_stale_s;
                    document.getElementById('uptime').textContent = data.uptime_s;

                    // Status
                    const statusEl = document.getElementById('status_text');
                    statusEl.textContent = data.status.toUpperCase();
                    statusEl.className = data.status === "ok" ? "ok" : "bad";

                    // Camera
                    document.getElementById('camera_status').textContent =
                        data.camera.ok ? "OK" : "BAD";
                    document.getElementById('frame_age').textContent =
                        data.camera.frame_age_ms;

                    // NetworkTables
                    document.getElementById('nt_enabled').textContent =
                        data.network_tables.enabled;
                    document.getElementById('nt_connected').textContent =
                        data.network_tables.connected;

                } catch (e) {
                    console.error("Health update failed", e);
                }
            }

            setInterval(updateHealth, 250);

            updateHealth();
            </script>
        </body>
        </html>
        """

    def _health_route(self):
        payload, healthy = self._build_payload()

        wants_html = "text/html" in request.headers.get("Accept", "")

        if wants_html:
            return Response(self._render_html(), mimetype="text/html")

        return jsonify(payload), (200 if healthy else 503)