import glob
import os
import re
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

#: camera-name tokens that say nothing useful about which pipeline owns an
#: accelerator ("default_cam", "Camera 0", "video0", ...) - hidden from the
#: System card's hardware rows instead of cluttering them.
_GENERIC_CAMERA_TOKENS = ("default", "camera", "cam", "webcam", "video", "usb", "unnamed")


def _generic_camera_name(name: str) -> bool:
    n = str(name or "").strip().lower()
    if not n:
        return True
    if n.replace("_", " ").replace("-", " ").split() and n.split()[-1].isdigit() and len(n.split()) == 1:
        return True
    return any(token == n or n.startswith(token + " ") or n.startswith(token + "_")
               or n.startswith(token + "-") for token in _GENERIC_CAMERA_TOKENS)


#: highest value that can still be a genuine 0-100 utilization percentage.
#: Anything above it is a raw busy-clock counter from rknpu's devfreq node,
#: which must never be clamped into a fake 100%.
_NPU_PERCENT_MAX = 100

#: comma-separated override for where to read NPU load from, so users can
#: point the dashboard at a custom source without code changes.
NPU_LOAD_PATHS_ENV = "ISPY_NPU_LOAD_PATHS"

#: glob patterns covering the devfreq/debugfs layouts different RKNPU boards
#: ship with (device name varies: fdab0000.npu, fdbb0000.npu, rknpu, ...).
_NPU_LOAD_GLOBS = (
    "/sys/kernel/debug/rknpu/load",
    "/sys/class/devfreq/*npu*/load",
    "/sys/devices/platform/*npu*/devfreq/*/load",
)

#: literal fallbacks for distros where the platform path isn't glob-visible.
_NPU_LOAD_FALLBACKS = (
    "/sys/devices/platform/fdab0000.npu/devfreq/fdab0000.npu/load",
    "/sys/class/devfreq/fdab0000.npu/load",
    "/sys/devices/platform/fdab0000.npu/devfreq/rknpu/load",
    "/sys/kernel/debug/rknpu/load",
)

_CORE_LOAD_RE = re.compile(r"Core\s*(\d+)\s*:\s*(\d+)\s*%")


def _parse_npu_percent(text: str) -> int | None:
    """Return a 0-100 utilization parsed from an RKNPU load file, or None.

    Understands the formats seen in the wild:
      * per-core debugfs  "NPU load:  Core0:  0%, Core1:  5%, Core2:  0%,"
      * single aggregate "NPU load:  37%" (newer kernels)
      * devfreq "load@freq" / bare integer where load is already a percent

    Raw busy-clock counters (>100) are rejected - they are NOT percentages,
    and clamping them produces the permanent-100% bug this replaces.
    """
    text = (text or "").strip()
    if not text:
        return None
    cores = _CORE_LOAD_RE.findall(text)
    if cores:
        per_core = [int(v) for _, v in cores]
        return max(0, min(sum(per_core) // len(per_core), 100))
    if "%" in text:
        match = re.search(r"(\d+)\s*%", text)
    elif "@" in text:
        match = re.search(r"(\d+)\s*@", text)
    else:
        match = re.search(r"(\d+)", text)
    if not match:
        return None
    value = int(match.group(1))
    if 0 <= value <= _NPU_PERCENT_MAX:
        return value
    return None


def _npu_load_candidates() -> list[str]:
    """Ordered, de-duplicated list of NPU load files to try.

    Environment override wins, then glob-discovered devfreq/debugfs nodes
    (covering fdab0000.npu, fdbb0000.npu, rknpu, ...), then literal fallbacks.
    """
    override = os.environ.get(NPU_LOAD_PATHS_ENV, "").strip()
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]
    paths: list[str] = []
    seen: set[str] = set()
    for pattern in _NPU_LOAD_GLOBS:
        for matched in sorted(glob.glob(pattern)):
            if matched not in seen:
                seen.add(matched)
                paths.append(matched)
    for fallback in _NPU_LOAD_FALLBACKS:
        if fallback not in seen:
            seen.add(fallback)
            paths.append(fallback)
    return paths


class DashboardModule(WebModule):
    plugin_name = "dashboard"

    def __init__(self, context: dict):
        super().__init__(context)
        self._data_lock = threading.Lock()
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
        self._npu_load_cache: tuple[float, list[int] | None] | None = None

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/dashboard", "dashboard_page", lambda: render_template("dashboard.html"))
        flask_app.add_url_rule("/api/status", "api_status", self._api_status)
        flask_app.add_url_rule("/api/system", "api_system", self._api_system)
        flask_app.add_url_rule("/api/events", "api_events", self._sse_stream)

    def update(self, frame_data: dict):
        tick = {
            "fps": round(frame_data.get("fps", 0), 1),
            "vision_ms": round(frame_data.get("vision_s", 0) * 1000, 1),
            "camera_lag_ms": round(frame_data.get("camera_lag_s", 0) * 1000, 1),
            "detections": frame_data.get("detection_count", 0),
            "loop_s": round(frame_data.get("loop_s", 0) * 1000, 1),
            "uptime_s": round(time.perf_counter() - self._start_time, 1),
        }

        det_classes = {}
        for obj in frame_data.get("detections", []):
            cls_name = getattr(obj, "name", None) or "unknown"
            det_classes[cls_name] = det_classes.get(cls_name, 0) + 1

        with self._data_lock:
            self._latest = tick
            self._vision_last_tick = time.perf_counter()
            self._detection_classes = det_classes
            if not self._model_info:
                self._refresh_model_info_unlocked()

        vision_running = (time.perf_counter() - self._vision_last_tick) < 5.0 if self._vision_last_tick else False
        self._push_sse({
            "type": "tick",
            **tick,
            "vision_running": vision_running,
            "detection_classes": det_classes,
            "cameras": self._get_camera_status(),
            "system": self._get_system_metrics(),
        })

    def _refresh_model_info_unlocked(self):
        try:
            config = self.context.get("config")
            if not config:
                return
            # models are per-camera now - just report the first model-backed cam's
            from iSpy.config.iSpyConfig import get_pipeline_settings
            cams = config.get("camera_configs", {})
            model_cfg = None
            for cam in cams.values():
                if not isinstance(cam, dict):
                    continue
                settings = get_pipeline_settings(cam) or {}
                candidate = settings.get("vision_model")
                if isinstance(candidate, dict) and candidate.get("file_path"):
                    model_cfg = candidate
                    break
            if not model_cfg:
                return
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
        with self._data_lock:
            self._plugin_info = {
                "trackers": list(trackers.keys()),
                "utilities": list(utilities.keys()),
                "frame_processors": list(frame_processors.keys()),
            }

    def _get_system_metrics(self) -> dict:
        if not PSUTIL_AVAILABLE:
            return {"cpu_percent": None, "memory_percent": None, "memory_used_mb": None,
                    "memory_total_mb": None, "temperature": None, "hardware": self._get_hardware()}

        try:
            cpu = psutil.cpu_percent(interval=None)
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
            "hardware": self._get_hardware(),
        }

    def _get_hardware(self) -> list[dict]:
        """Aggregate each camera/pipeline's active hardware accelerator.

        Pipelines with no dedicated accelerator (or not yet loaded) are
        omitted so we don't double-report the CPU. Accelerators are grouped
        by type (multiple cameras share one NPU/GPU), and each entry carries
        a best-effort utilization reading for its progress bar. Returns
        [{"hardware": "npu"|"tpu"|"gpu"|"cpu", "cameras": [...], "load_percent": int|None}]
        """
        cams = self.context.get("cameras") or []
        grouped: dict[str, set[str]] = {}
        for cam in cams:
            try:
                if hasattr(cam, "config"):
                    name = str(cam.config.get("name", str(getattr(cam, "source", "?"))))
                else:
                    name = str(getattr(cam, "source", "?"))
                resolver = getattr(cam, "active_hardware", None)
                if resolver is None:
                    continue
                hardware = resolver()
            except Exception:
                continue
            if not hardware:
                continue
            grouped.setdefault(str(hardware).lower(), set()).add(name)

        out = []
        for hardware, names in grouped.items():
            meaningful = sorted(n for n in names if not _generic_camera_name(n))
            out.append({
                "hardware": hardware,
                "cameras": meaningful,
                "load_percent": self._read_hardware_load(hardware),
            })
        out.sort(key=lambda entry: entry["hardware"])
        return out

    def _read_hardware_load(self, hardware: str) -> int | None:
        """Best-effort utilization (0-100) for a shared accelerator.

        NPU: RKNPU sysfs exposes load through several files with different
        formats depending on the distro kernel - see :meth:`_read_npu_load`.
        GPU: NVIDIA reports via nvidia-smi (throttled to every ~2s so the
        busy dashboard SSE doesn't spawn a subprocess per tick). Returns None
        when the platform can't report utilization.
        """
        hardware = str(hardware).lower()
        if hardware == "npu":
            return self._read_npu_load()
        if hardware == "gpu":
            now = time.monotonic()
            cached = getattr(self, "_gpu_load_cache", None)
            if cached and now - cached[0] < 2.0:
                return cached[1]
            import shutil
            import subprocess
            value = None
            try:
                if shutil.which("nvidia-smi"):
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=2,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        value = max(0, min(int(result.stdout.split()[0]), 100))
            except Exception:
                value = None
            self._gpu_load_cache = (now, value)
            return value
        return None

    def _read_npu_load(self) -> int | None:
        """Best-effort RKNPU utilization (0-100), reading every source.

        Rockchip (RK3588/RK3576/...) exposes NPU load through several files
        whose format depends on the distro kernel. This reads each candidate
        and favors the most trustworthy reading instead of trusting one:

          * per-core debugfs (needs root): "NPU load:  Core0:  0%, Core1:  5%,
            Core2:  0%," - averaged, most accurate.
          * devfreq "load@freq": the value before "@" is usually already a
            percentage; a raw busy-clock counter (>>100) is rejected rather
            than clamped, since clamping is what makes the bar sit at 100%.
          * bare integer in 0-100 or "N%" aggregate.

        Results are cached for ~1s so the frequent SSE ticks don't hammer
        sysfs. Returns None when no usable source can be read.
        """
        now = time.monotonic()
        if self._npu_load_cache and now - self._npu_load_cache[0] < 1.0:
            return self._npu_load_cache[1]

        readings: list[int] = []
        core_average: int | None = None
        for path in _npu_load_candidates():
            try:
                raw = Path(path).read_text()
            except Exception:
                continue
            if not raw or not raw.strip():
                continue
            value = _parse_npu_percent(raw)
            if value is None:
                continue
            # Prefer the per-core debugfs average over devfreq aggregate
            # readings when both exist - it's the accurate one.
            if _CORE_LOAD_RE.search(raw):
                core_average = value
            else:
                readings.append(value)
        result = core_average
        if result is None and readings:
            result = round(sum(readings) / len(readings))
        if result is None:
            result = None
        else:
            result = max(0, min(int(result), 100))
        self._npu_load_cache = (now, result)
        return result

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
        with self._data_lock:
            latest = dict(self._latest)
            detection_classes = dict(self._detection_classes)
            model_info = dict(self._model_info)
            plugin_info = dict(self._plugin_info)
        return {
            **latest,
            "vision_running": vision_running,
            "uptime_s": round(time.perf_counter() - self._start_time, 1),
            "cameras": self._get_camera_status(),
            "system": self._get_system_metrics(),
            "model": model_info,
            "detection_classes": detection_classes,
            "plugins": plugin_info,
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
