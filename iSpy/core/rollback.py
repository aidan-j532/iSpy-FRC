import json
import logging
import os
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from iSpy.core import performance
from iSpy.plugins.bases import UtilityBase

_MIN_FREE_BYTES = 100 * 1024 * 1024

_CODEC = "MJPG"
_EXT = ".avi"

_READ_EXTS = (".avi", ".mp4")
_SEGMENT_PREFIXES = ("rollback_", "recording_")

_SESSION_PREFIX = "session_"
_SESSION_FILE = "session.json"
_PINNED_FILE = "PINNED"
_SIDECAR_EXT = ".json"

_LEGACY_SESSION = "_legacy"

def _sanitize_cam(name) -> str:
    keep = []
    for ch in str(name):
        keep.append(ch if (ch.isalnum() or ch in "-_") else "_")
    cleaned = "".join(keep).strip("_")
    return cleaned or "cam"

def _serialize_detections(detections) -> list:
    out = []
    for det in detections or []:
        if isinstance(det, dict):
            out.append(det)
            continue
        to_dict = getattr(det, "to_dict", None)
        if callable(to_dict):
            try:
                out.append(to_dict())
            except Exception:
                continue
    return out

def _json_default(value):
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass
    return str(value)

def _dir_bytes(path) -> int:
    total = 0
    try:
        for entry in Path(path).rglob("*"):
            if not entry.is_file():
                continue
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total

_HEADER_PREFIX_BYTES = 8192
_HEADER_KEYS = ("stem", "session", "cams", "started", "fps")

def read_sidecar_header(path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(_HEADER_PREFIX_BYTES)
    except OSError:
        return {}

    marker = '"records": ['
    pos = head.find(marker)
    if pos >= 0:
        try:
            header = json.loads(head[:pos].rstrip().rstrip(",") + "}")
            if isinstance(header, dict):
                return header
        except ValueError:
            pass

    out = {}
    for key in _HEADER_KEYS:
        key_marker = f'"{key}": '
        key_pos = head.find(key_marker)
        if key_pos < 0:
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(head, key_pos + len(key_marker))
        except ValueError:
            continue
        out[key] = value
    return out

def _sidecar_cams(path: Path) -> list:
    cams = read_sidecar_header(path).get("cams")
    return [str(c) for c in cams] if isinstance(cams, list) else []

def read_sidecar(path) -> dict:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return {}

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except ValueError:
        pass

    marker = '"records": ['
    pos = text.find(marker)
    if pos < 0:
        return {}
    try:
        header = json.loads(text[:pos].rstrip().rstrip(",") + "}")
    except ValueError:
        header = {}

    dec, i, n, records = json.JSONDecoder(), pos + len(marker), len(text), []
    while i < n:
        while i < n and text[i] in " ,\r\n\t":
            i += 1
        if i >= n or text[i] == "]":
            break
        try:
            rec, i = dec.raw_decode(text, i)
        except ValueError:
            break
        records.append(rec)
    header["records"] = records
    return header

def _iter_segments(directory):
    try:
        root = Path(directory)
        if not root.is_dir():
            return []
        entries = list(root.iterdir())
    except OSError:
        return []

    sidecars = {}
    loose = []
    for path in entries:
        if not path.is_file() or not path.name.startswith(_SEGMENT_PREFIXES):
            continue
        suffix = path.suffix.lower()
        if suffix == _SIDECAR_EXT:
            sidecars[path.stem] = path
        elif suffix in _READ_EXTS:
            loose.append(path)

    segments = []
    claimed = set()
    for stem, sidecar in sorted(sidecars.items()):
        cams = _sidecar_cams(sidecar)
        files = [sidecar]
        for cam in cams:
            avi = root / f"{stem}_{_sanitize_cam(cam)}{_EXT}"
            claimed.add(avi.stem)
            if avi.is_file():
                files.append(avi)
        segments.append({"stem": stem, "cams": cams, "files": files})

    for path in sorted(loose):
        if path.stem in claimed:
            continue
        segments.append({"stem": path.stem, "cams": [], "files": [path]})

    return sorted(segments, key=lambda seg: seg["stem"])

def _iter_sessions(data_dir):
    try:
        root = Path(data_dir)
        if not root.is_dir():
            return []
        entries = sorted(root.iterdir())
    except OSError:
        return []

    sessions = []
    loose = []
    for path in entries:
        if path.is_dir():
            if path.name.startswith(_SESSION_PREFIX):
                sessions.append(
                    {
                        "name": path.name,
                        "path": path,
                        "pinned": (path / _PINNED_FILE).is_file(),
                        "legacy": False,
                        "started": _session_started(path),
                    }
                )
        elif (
            path.is_file()
            and path.name.startswith(_SEGMENT_PREFIXES)
            and path.suffix.lower() in _READ_EXTS
        ):
            loose.append(path)

    if loose:
        sessions.append(
            {
                "name": _LEGACY_SESSION,
                "path": root,
                "pinned": (root / _PINNED_FILE).is_file(),
                "legacy": True,
                "started": None,
                "loose": loose,
            }
        )

    return sorted(sessions, key=lambda s: s["name"])

def _session_started(path: Path):
    try:
        data = json.loads((path / _SESSION_FILE).read_text())
    except Exception:
        return None
    return data.get("started") if isinstance(data, dict) else None

def _session_size(session) -> int:
    if session.get("loose"):
        total = 0
        for path in session["loose"]:
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total
    return _dir_bytes(session["path"])

def _delete_session(session) -> bool:
    try:
        if session.get("loose"):
            removed = False
            for path in session["loose"]:
                try:
                    path.unlink()
                    removed = True
                except OSError:
                    continue
            return removed
        shutil.rmtree(session["path"])
        return True
    except OSError:
        return False

class RollBack(UtilityBase):
    plugin_name = "rollback"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "enabled": {
                "type": "toggle",
                "label": "Record",
                "hint": "Record every frame for replay.",
                "default": True,
            },
            "data_dir": {
                "type": "text",
                "label": "Output Directory",
                "hint": "Where recording sessions are written.",
                "default": "VideoRecordings",
            },
            "fps": {
                "type": "number",
                "label": "Recording FPS",
                "hint": "Frames per second stored in the recording file.",
                "default": 30.0,
            },
            "max_queue": {
                "type": "number",
                "label": "Max Queue",
                "hint": "Maximum buffered frames before dropping.",
                "default": 60,
            },
            "downsample": {
                "type": "number",
                "label": "Downsample",
                "hint": "Record every Nth frame (1 = every frame).",
                "default": 1,
            },
            "segment_minutes": {
                "type": "number",
                "label": "Segment Length (minutes)",
                "hint": "Start a new file every N minutes so replay can jump.",
                "default": 5,
            },
            "max_total_mb": {
                "type": "number",
                "label": "Max Total Size (MB)",
                "hint": "Oldest unpinned sessions are deleted once the folder "
                "exceeds this.",
                "default": 2048,
            },
        }

    def __init__(self, context: dict):
        super().__init__(context)
        self.logger = logging.getLogger(__name__)
        vision = self.context.get("vision_instance")
        profile = performance.get_profile()
        self._enabled = bool(self.config.get("enabled", True)) and not bool(
            getattr(vision, "replay_mode", False)
        )
        if not profile.rollback_recorder:
            self._enabled = False
        self._disabled_by_profile = not profile.rollback_recorder
        self._video_output_dir = self.config.get("data_dir", "VideoRecordings")
        self._fps = float(self.config.get("fps", 30.0))
        self._max_queue = int(self.config.get("max_queue", 60))
        self._downsample = max(
            1,
            int(self.config.get("downsample", 1)),
            int(profile.rollback_downsample_min),
        )
        self._segment_seconds = max(1, int(self.config.get("segment_minutes", 5))) * 60
        self._max_total_bytes = (
            int(self.config.get("max_total_mb", 2048)) * 1024 * 1024
        )

        self._queue = queue.Queue(maxsize=self._max_queue)
        self._writers = {}
        self._thread = None
        self._started = False
        self._stopped = False
        self._frame_counter = 0
        self._dropped = 0
        self._size = None
        self._segment_index = 0
        self._stem = None
        self._cams = []
        self._cams_pending = None
        self._segment_opened_at = None
        self._sidecar = None
        self._sidecar_records = 0
        self._last_clean = {}
        self._disk_paused = False
        self._idle_logged_at = 0.0

        self._session_name = None
        self._session_dir = None
        self._session_started_at = None
        self._retention_thread = None
        try:
            os.makedirs(self._video_output_dir, exist_ok=True)
        except OSError as e:
            self.logger.error(
                "Rollback output directory %s is unusable: %s",
                self._video_output_dir,
                e,
            )

    def _camera_meta(self) -> dict:
        cams = {}
        vision = self.context.get("vision_instance")
        pipeline_cams = list(getattr(vision, "cameras", None) or [])
        for i, cam in enumerate(pipeline_cams):
            cfg = getattr(cam, "config", None)

            def _cfg(key, default=None):
                try:
                    value = cfg.get(key, default) if cfg is not None else default
                except Exception:
                    return default
                return default if value is None else value

            name = _cfg("name", f"Camera {i + 1}")
            try:
                source = cam.source
            except Exception:
                source = _cfg("source", "")
            cams[str(name)] = {
                "source": source,
                "pipeline": getattr(cam, "plugin_name", "unknown"),
                "x": _cfg("x", 0),
                "y": _cfg("y", 0),
                "height": _cfg("height", 0),
                "yaw": _cfg("yaw", 0),
                "pitch": _cfg("pitch", 0),
                "calibration": _cfg("calibration", {}),
            }
        return cams

    def _open_session(self) -> bool:
        try:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            name = f"{_SESSION_PREFIX}{stamp}"
            attempt = 0
            while (Path(self._video_output_dir) / name).exists():
                attempt += 1
                name = f"{_SESSION_PREFIX}{stamp}_{attempt}"
            self._session_name = name
            self._session_dir = Path(self._video_output_dir) / name
            self._session_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.logger.error(
                "Could not create rollback session in %s: %s", self._video_output_dir, e
            )
            self._session_name = None
            self._session_dir = None
            return False

        self._write_session_meta()
        self.logger.info("Rollback session started: %s", self._session_dir)
        return True

    def _write_session_meta(self) -> bool:
        if self._session_dir is None:
            return False

        try:
            from iSpy import __version__

            version = __version__
        except Exception:
            version = "unknown"

        unit = "frc"
        try:
            unit = self.context["global_config"].get("unit", "frc")
        except Exception:
            pass

        meta = {
            "started": getattr(self, "_session_started_at", None) or time.time(),
            "version": version,
            "cameras": self._camera_meta(),
            "unit": unit,
        }
        self._session_started_at = meta["started"]
        try:
            (self._session_dir / _SESSION_FILE).write_text(
                json.dumps(meta, indent=2, default=_json_default)
            )
        except OSError as e:
            self.logger.warning("Could not write %s: %s", _SESSION_FILE, e)
            return False
        return True

    def _enforce_retention(self):
        sessions = _iter_sessions(self._video_output_dir)
        if not sessions:
            return

        sizes = {s["name"]: _session_size(s) for s in sessions}
        total = sum(sizes.values())
        if total <= self._max_total_bytes:
            return

        for session in sessions:
            if total <= self._max_total_bytes:
                return
            if session["name"] == self._session_name:
                continue  # never delete the session being written
            if session["pinned"]:
                continue
            if not _delete_session(session):
                continue
            total -= sizes[session["name"]]
            self.logger.info(
                "Deleted session %s (retention cap)", session["name"]
            )

        if total > self._max_total_bytes:
            self.logger.warning(
                "Rollback is %.1f MB over the %d MB cap but everything left is "
                "pinned or the active session - delete or unpin something.",
                (total - self._max_total_bytes) / (1024 * 1024),
                int(self._max_total_bytes / (1024 * 1024)),
            )

    def _schedule_retention(self):
        if self._retention_thread is not None and self._retention_thread.is_alive():
            return

        def _run():
            try:
                self._enforce_retention()
            except Exception:
                self.logger.exception("Rollback retention pass failed")

        self._retention_thread = threading.Thread(target=_run, daemon=True)
        self._retention_thread.start()

    def _join_retention(self, timeout: float = 10.0) -> None:
        thread = self._retention_thread
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout=timeout)
        if thread.is_alive():
            self.logger.warning(
                "Rollback retention pass still running after %.0fs - it may be "
                "deleting a large session",
                timeout,
            )
        else:
            self._retention_thread = None

    def start(self):
        pass

    def wants_raw_frames(self) -> bool:
        return self._enabled and not self._disk_paused

    def update(self, frame_data: dict):
        if not self._enabled:
            return

        raw_frames = frame_data.get("raw_frames")
        if not raw_frames:
            return

        self._frame_counter += 1

        if self._started and self._frame_counter % 150 == 0:
            self._disk_paused = bool(self._disk_problem())
        if self._disk_paused:
            return

        if self._frame_counter % self._downsample != 0:
            return

        record = {
            "i": self._frame_counter,
            "t": time.time(),
            "cams": sorted(raw_frames),
            "detection_count": int(frame_data.get("detection_count", 0) or 0),
            "detections": _serialize_detections(frame_data.get("detections")),
            "robot_pose": frame_data.get("robot_pose"),
        }

        if not self._started:
            first = next(iter(raw_frames.values()))
            h, w = first.shape[:2]
            if not self._start_recorder(w, h, raw_frames.keys()):
                return
        else:
            cams = sorted(raw_frames)
            merged = sorted(set(self._cams) | set(cams))
            if merged != self._cams and merged != self._cams_pending:
                self._cams_pending = merged

        self._write({"frames": raw_frames, "record": record})

    def stop(self):
        self._stop_recorder()

    def _clean_frame(self, frame, size):
        if not isinstance(frame, np.ndarray):
            return None
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if len(frame.shape) != 3:
            return None
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if size is not None and (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)
        return np.ascontiguousarray(frame)

    def _disk_problem(self) -> str | None:
        if not Path(self._video_output_dir).is_dir():
            return f"output directory {self._video_output_dir} is not a usable folder"
        if not os.access(self._video_output_dir, os.W_OK):
            return f"output directory {self._video_output_dir} is not writable"

        try:
            usage = shutil.disk_usage(self._video_output_dir)
        except OSError as e:
            return f"cannot check free space on {self._video_output_dir} ({e})"
        if usage.free < _MIN_FREE_BYTES:
            free_mb = usage.free / (1024 * 1024)
            margin_mb = _MIN_FREE_BYTES // (1024 * 1024)
            return (
                f"only {free_mb:.0f} MB free on {self._video_output_dir} "
                f"(a {margin_mb} MB safety margin is required)"
            )
        return None

    def _start_recorder(self, width: int, height: int, cams) -> bool:
        if self._started:
            return True

        problem = self._disk_problem()
        if problem:
            now = time.monotonic()
            if now - self._idle_logged_at >= 30:
                self._idle_logged_at = now
                self.logger.warning("Rollback recorder idle: %s", problem)
            return False

        if self._session_dir is None and not self._open_session():
            return False

        self._cams = sorted(cams)
        self._cams_pending = None
        self._size = (width, height)
        self._started = True
        self._stopped = False
        self._last_clean = {}

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2)

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return True

    def _worker(self):
        try:
            self._open_segment()
            while True:
                item = self._queue.get()
                if item is None:
                    break
                try:
                    if self._stem is not None and self._should_rotate():
                        self._close_segment()
                        self._adopt_pending_cams()
                        self._open_segment()
                    if self._stem is not None:
                        self._write_segment(item)
                except Exception:
                    self.logger.exception("Rollback worker error - frame dropped")
                finally:
                    self._queue.task_done()
        except Exception as e:
            self.logger.error("Error in video worker: %s", e)
        finally:
            self._close_segment()
            if self._thread is threading.current_thread():
                self._thread = None

    def _write_segment(self, item):
        for cam, writer in self._writers.items():
            try:
                clean = self._clean_frame(item["frames"].get(cam), self._size)
                if clean is None:
                    clean = self._last_clean.get(cam)
                    if clean is None:
                        clean = np.zeros(
                            (self._size[1], self._size[0], 3), np.uint8
                        )
                self._last_clean[cam] = clean
                writer.write(clean)
            except Exception:
                self.logger.exception(
                    "Rollback could not write frame for camera %s", cam
                )

        self._write_sidecar(item["record"])

    def _clip_fps(self) -> float:
        return max(self._fps / max(1, self._downsample), 0.1)

    def _sidecar_path(self) -> Path:
        return self._session_dir / f"{self._stem}{_SIDECAR_EXT}"

    def _open_sidecar(self):
        self._sidecar_records = 0
        try:
            handle = open(self._sidecar_path(), "w", encoding="utf-8")
        except OSError as e:
            self.logger.error("Could not write sidecar %s: %s", self._sidecar_path(), e)
            self._sidecar = None
            return
        self._sidecar = handle
        header = {
            "stem": self._stem,
            "session": self._session_name,
            "cams": list(self._cams),
            "started": self._segment_opened_at,
            "fps": self._clip_fps(),
        }
        try:
            handle.write("{")
            for key, value in header.items():
                handle.write(
                    f"{json.dumps(key)}: {json.dumps(value, default=_json_default)},"
                )
            handle.write('"records": [')
        except OSError as e:
            self.logger.error("Sidecar write failed: %s", e)
            self._close_sidecar()
            return
        self.logger.info("Recording sidecar %s", self._sidecar_path().name)

    def _write_sidecar(self, record):
        if self._sidecar is None:
            return
        try:
            if self._sidecar_records:
                self._sidecar.write(",")
            self._sidecar.write(json.dumps(record, default=_json_default))
            self._sidecar_records += 1
            if self._sidecar_records % 30 == 0:
                self._sidecar.flush()
        except OSError as e:
            self.logger.error("Sidecar write failed: %s", e)
            self._close_sidecar()

    def _close_sidecar(self):
        if self._sidecar is None:
            return
        try:
            self._sidecar.write("]}")
        except OSError:
            pass
        try:
            self._sidecar.close()
        except OSError:
            pass
        self._sidecar = None
        self.logger.info("Closed sidecar with %d records", self._sidecar_records)
        self._sidecar_records = 0

    def _segment_expired(self) -> bool:
        if self._segment_opened_at is None:
            return False
        return (time.time() - self._segment_opened_at) >= self._segment_seconds

    def _should_rotate(self) -> bool:
        return self._segment_expired() or bool(self._cams_pending)

    def _adopt_pending_cams(self):
        if not self._cams_pending:
            return
        added = [c for c in self._cams_pending if c not in self._cams]
        gone = [c for c in self._cams if c not in self._cams_pending]
        if added or gone:
            self.logger.info(
                "Camera set changed, rotating: +%s -%s",
                ", ".join(added) or "none",
                ", ".join(gone) or "none",
            )
        self._cams = list(self._cams_pending)
        self._cams_pending = None

    def _open_segment(self):
        if self._size is None or self._session_dir is None or not self._cams:
            return

        self._schedule_retention()

        self._write_session_meta()

        width, height = self._size
        self._stem = self._next_segment_stem()
        clip_fps = self._clip_fps()

        for cam in self._cams:
            path = self._segment_path(cam)
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*_CODEC), clip_fps, (width, height)
            )
            if not writer.isOpened():
                self.logger.error("VideoWriter could not open %s", path)
                continue
            self._writers[cam] = writer

        self._segment_opened_at = time.time()
        self._open_sidecar()
        self.logger.info(
            "Recording segment %s (%s) at %.2f fps",
            self._stem,
            ", ".join(self._cams),
            clip_fps,
        )

    def _close_segment(self):
        for cam, writer in list(self._writers.items()):
            try:
                writer.release()
            except Exception:
                pass
            self._writers.pop(cam, None)
        self._close_sidecar()
        if self._stem is not None:
            self.logger.info("Closed segment %s", self._stem)
        self._stem = None
        self._segment_opened_at = None

    def _segment_path(self, cam) -> Path:
        return self._session_dir / f"{self._stem}_{_sanitize_cam(cam)}{_EXT}"

    def _next_segment_stem(self) -> str:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        while True:
            suffix = f"_{self._segment_index:02d}" if self._segment_index else ""
            stem = f"rollback_{stamp}{suffix}"
            taken = any(
                (self._session_dir / f"{stem}_{_sanitize_cam(cam)}{_EXT}").exists()
                for cam in self._cams
            ) or (self._session_dir / f"{stem}{_SIDECAR_EXT}").exists()
            if not taken:
                return stem
            self._segment_index += 1

    def _write(self, item):
        if not self._started or self._stopped:
            return

        try:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                    self._dropped += 1
                except queue.Empty:
                    pass
            self._queue.put_nowait(item)
        except queue.Full:
            self._dropped += 1

    def _stop_recorder(self):
        _thread = self._thread
        was_started = self._started
        self._stopped = True

        if _thread is not None and _thread.is_alive():
            while self._queue.full():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                    self._dropped += 1
                except queue.Empty:
                    break
            try:
                self._queue.put_nowait(None)
            except queue.Full:  # queue is empty - sentinel always fits
                pass
            if _thread is not threading.current_thread():
                _thread.join(timeout=15)

        self._started = False
        self._thread = None
        self._cams_pending = None
        self._join_retention()

        if was_started:
            self.logger.info(
                "Recording stopped. Frames=%d Dropped=%d",
                self._frame_counter,
                self._dropped,
            )

    def get_health(self) -> dict:
        problem = None
        try:
            problem = self._disk_problem()
        except Exception:
            problem = "could not check the output directory"

        if not self._enabled:
            color = "green"
            state = (
                "Recording off - performance profile"
                if self._disabled_by_profile
                else "Recording off"
            )
        elif problem:
            color, state = "red", "Recorder idle"
        elif self._disk_paused:
            color, state = "red", "Paused - no disk space"
        elif self._started:
            color, state = "green", "Recording"
        else:
            color, state = "yellow", "Waiting for a camera"

        metrics = [
            {"label": "Session", "value": self._session_name or "--"},
            {"label": "Frames", "value": str(self._frame_counter)},
            {"label": "Dropped", "value": str(self._dropped)},
            {"label": "Cameras", "value": ", ".join(self._cams) or "--"},
        ]
        if problem:
            metrics.append({"label": "Disk", "value": problem})

        return {
            "color": color,
            "state": state,
            "metrics": metrics,
        }
