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
<html><head><title>System Status</title>
<style>
body{font-family:Arial;background:#111;color:#eee;margin:0;padding:20px}
h1{text-align:center}
.card{padding:20px;margin:20px auto;max-width:700px;border-radius:10px;background:#222}
.ok{color:#4caf50}.bad{color:#f44336;font-weight:bold}
.pill{display:inline-block;padding:2px 8px;border-radius:10px;background:#333;margin:2px}
</style></head>
<body>
<h1>iSpy System Status</h1>
<div class="card">
  <h2>Status: <span id="status_text"></span></h2>
  <p><b>FPS:</b> <span id="fps"></span></p>
  <p><b>Inference:</b> <span id="vision"></span> ms</p>
  <p><b>Detections:</b> <span id="detections"></span></p>
  <p><b>Loop stale:</b> <span id="stale"></span> s</p>
  <p><b>Uptime:</b> <span id="uptime"></span> s</p>
</div>
<div id="cameras-container"></div>
<div class="card"><h3>NetworkTables</h3>
  <p>Enabled: <span id="nt_enabled"></span> Connected: <span id="nt_connected"></span></p>
</div>
<div class="card"><h3>Loaded Plugins</h3><div id="plugins"></div></div>
<script>
async function refresh(){
  try{
    const data = await fetch('/health', {headers:{'Accept':'application/json'}}).then(r=>r.json());
    fps.textContent = data.fps; vision.textContent = data.vision_ms;
    detections.textContent = data.detections; stale.textContent = data.loop_stale_s;
    uptime.textContent = data.uptime_s;
    const s = status_text; s.textContent = data.status.toUpperCase(); s.className = data.status==='ok'?'ok':'bad';
    cameras-container.innerHTML='';
    (data.cameras||[]).forEach(c=>{
      const d=document.createElement('div'); d.className='card';
      d.innerHTML = '<h3>'+c.name+'</h3><p>Source: '+c.source+'</p><p>Status: <span class="'+(c.ok?'ok':'bad')+'">'+(c.ok?'OK':'BAD')+'</span></p><p>Frame age: '+(c.frame_age_ms??'N/A')+' ms</p>';
      cameras-container.appendChild(d);
    });
    nt_enabled.textContent = data.network_tables.enabled;
    nt_connected.textContent = data.network_tables.connected;
    plugins.innerHTML='';
    for (const [kind, names] of Object.entries(data.plugins||{})){
      const row = document.createElement('p');
      row.innerHTML = '<b>'+kind+':</b> ' + (names.length ? names.map(n=>'<span class="pill">'+n+'</span>').join('') : '<i>none</i>');
      plugins.appendChild(row);
    }
  }catch(e){ console.error(e); }
}
setInterval(refresh, 500); refresh();
</script>
</body></html>"""


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
        self._stale_threshold = config.get("stale_threshold", 1.0)
        self._network_handler = None
        self._plugins = {"trackers": [], "utilities": [], "frame_processors": []}

        if flask_app and FLASK_AVAILABLE:
            flask_app.add_url_rule("/health", "health", self._health_route)
            flask_app.add_url_rule("/status", "status_page", self._status_page)
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
            self._detections = frame_data.get("detections", 0)
            self._last_tick = time.perf_counter()

    def stop(self):
        pass

    def _status_page(self):
        return Response(_HTML, mimetype="text/html")

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
        if "text/html" in request.headers.get("Accept", ""):
            return Response(_HTML, mimetype="text/html")
        return jsonify(payload), (200 if healthy else 503)