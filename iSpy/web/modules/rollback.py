import json
import logging
import os
import math
import shlex
import shutil
import threading
from pathlib import Path

import cv2
from flask import Response, jsonify, render_template, request, send_file

from iSpy.config.iSpyConfig import unit_to_output
from iSpy.core.rollback import (
    _LEGACY_SESSION,
    _PINNED_FILE,
    _READ_EXTS,
    _SESSION_FILE,
    RollBack,
    _delete_session,
    _iter_segments,
    _iter_sessions,
    _sanitize_cam,
    _session_size,
    read_sidecar,
    read_sidecar_header,
)
from iSpy.web.Backend.WebModule import WebModule

logger = logging.getLogger(__name__)

_CAPS = {}
_FRAMES = {}
_FRAME_ORDER = []
_CAPS_LOCK = threading.RLock()
_MAX_CAPS = 4
_MAX_FRAMES = 48

_SEGMENT_FRAMES = {}
_MAX_CACHED_SEGMENTS = 512

_JPEG_QUALITY = 85

_NUMERIC_SETTINGS = {
    "fps": float,
    "max_queue": int,
    "downsample": int,
    "segment_minutes": int,
    "max_total_mb": int,
}
_TEXT_SETTINGS = ("data_dir",)
_BOOL_SETTINGS = ("enabled",)

_MINIMUMS = {
    "fps": 0.5,
    "max_queue": 1,
    "downsample": 1,
    "segment_minutes": 1,
    "max_total_mb": 1,
}

def _mb(num_bytes) -> float:
    return round(float(num_bytes or 0) / (1024 * 1024), 2)

def _segment_sidecar(seg):
    for path in seg["files"]:
        if path.suffix.lower() == ".json":
            return path
    return None

def _segment_frames(seg) -> int:
    path = _segment_sidecar(seg)
    if path is None:
        return 0
    try:
        stat = path.stat()
    except OSError:
        return 0

    key = str(path)
    cached = _SEGMENT_FRAMES.get(key)
    if cached is not None and (cached[0], cached[1]) == (stat.st_mtime_ns, stat.st_size):
        return cached[2]

    count = len(read_sidecar(path).get("records") or [])
    if len(_SEGMENT_FRAMES) >= _MAX_CACHED_SEGMENTS:
        for stale in [k for k in _SEGMENT_FRAMES if not Path(k).exists()]:
            _SEGMENT_FRAMES.pop(stale, None)
    _SEGMENT_FRAMES[key] = (stat.st_mtime_ns, stat.st_size, count)
    return count

def _segment_header(seg) -> dict:
    path = _segment_sidecar(seg)
    return read_sidecar_header(path) if path is not None else {}

def _capture_for(path: Path):
    with _CAPS_LOCK:
        cap = _CAPS.get(str(path))
        if cap is not None:
            return cap
        while len(_CAPS) >= _MAX_CAPS:
            oldest = next(iter(_CAPS))
            try:
                _CAPS.pop(oldest).release()
            except Exception:
                pass
        cap = cv2.VideoCapture(str(path))
        _CAPS[str(path)] = cap
        return cap

def _release_tree(root: Path) -> None:
    prefix = str(root)
    with _CAPS_LOCK:
        for key in [k for k in _CAPS if k.startswith(prefix)]:
            try:
                _CAPS.pop(key).release()
            except Exception:
                pass
        for key in [k for k in _FRAMES if k[0].startswith(prefix)]:
            _FRAMES.pop(key, None)
            if key in _FRAME_ORDER:
                _FRAME_ORDER.remove(key)

def _encoded(path: Path, index: int):
    key = (str(path), int(index))
    hit = _FRAMES.get(key)
    if hit is not None:
        return hit

    cap = _capture_for(path)
    if cap is None or not cap.isOpened():
        return None
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(index)):
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        return None

    good, buf = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]
    )
    if not good:
        return None
    payload = buf.tobytes()
    _FRAMES[key] = payload
    _FRAME_ORDER.append(key)
    while len(_FRAME_ORDER) > _MAX_FRAMES:
        _FRAMES.pop(_FRAME_ORDER.pop(0), None)
    return payload

def _trim_pose(pose) -> dict:
    if not isinstance(pose, dict):
        return {}

    def _f(k):
        try:
            v = float(pose.get(k, 0.0) or 0.0)
            return v if math.isfinite(v) else 0.0
        except Exception:
            return 0.0

    return {"x": _f("x"), "y": _f("y"), "heading": _f("heading")}

def _trim_detection(det: dict) -> dict:
    def _f(k, default=0.0):
        try:
            v = det.get(k, default)
            v = float(v or default)
            if not math.isfinite(v):
                return default
            return v
        except Exception:
            return default

    if not isinstance(det, dict):
        return None

    return {
        "name": str(det.get("name", "unknown")),
        "confidence": round(_f("confidence"), 3),
        "x": _f("x"),
        "y": _f("y"),
        "z": _f("z"),
        "yaw": _f("yaw"),
        "vis_type": str(det.get("vis_type", "generic")),
    }

class RollbackModule(WebModule):
    plugin_name = "rollback"

    def __init__(self, context: dict):
        super().__init__(context)
        self.logger = logging.getLogger(__name__)
        self._last_retention = None
        self._session_meta_cache = {}

    def _session_meta(self, session) -> dict:
        key = session["name"]
        path = session["path"] / _SESSION_FILE
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            stamp = None
        hit = self._session_meta_cache.get(key)
        if hit and hit[0] == stamp:
            return hit[1]
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            meta = raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            meta = {}
        self._session_meta_cache[key] = (stamp, meta)
        return meta

    def _recorder(self):
        vision = self.context.get("vision_instance")
        if vision is None:
            return None
        return (getattr(vision, "utilities", None) or {}).get("rollback")

    def _data_dir(self) -> Path:
        recorder = self._recorder()
        if recorder is not None:
            return Path(recorder._video_output_dir)
        config = self.context.get("config")
        block = {}
        if config is not None:
            raw = config.get("rollback", {})
            block = raw if isinstance(raw, dict) else {}
        return Path(block.get("data_dir") or RollBack.config_schema()["data_dir"]["default"])

    def _find_session(self, name: str):
        for session in _iter_sessions(self._data_dir()):
            if session["name"] == name:
                return session
        return None

    def _find_segment(self, session, stem: str):
        for seg in _iter_segments(session["path"]):
            if seg["stem"] == stem:
                return seg
        return None

    def _segment_cams(self, seg) -> list:
        return [c for c in seg["cams"] if self._cam_file(seg, c)]

    def _cam_file(self, seg, cam: str):
        videos = [p for p in seg["files"] if p.suffix.lower() in _READ_EXTS]
        if not videos:
            return None
        wanted = f"{seg['stem']}_{_sanitize_cam(cam)}"
        for path in videos:
            if path.stem == wanted:
                return path
        if len(videos) == 1 and not seg["cams"]:
            return videos[0]
        return None

    def _read_sidecar(self, seg) -> dict:
        for path in seg["files"]:
            if path.suffix.lower() != ".json":
                continue
            return read_sidecar(path)
        return {}

    def _segment_payload(self, seg) -> dict:
        header = _segment_header(seg)
        frames = _segment_frames(seg)
        cams = seg["cams"] or [
            p.stem.split("_")[-1] for p in seg["files"] if p.suffix.lower() in _READ_EXTS
        ]
        return {
            "stem": seg["stem"],
            "cams": cams,
            "frames": frames,
            "started": header.get("started"),
            "fps": header.get("fps"),
            "has_sidecar": bool(frames),
            "recording": seg["stem"] == self._open_stem(),
            "size_mb": _mb(sum(_file_size(p) for p in seg["files"])),
            "clips": [
                {"cam": cam, "file": p.name}
                for cam in cams
                for p in [self._cam_file(seg, cam)]
                if p is not None
            ],
        }

    def _open_stem(self):
        recorder = self._recorder()
        if recorder is None or not getattr(recorder, "_started", False):
            return None
        return getattr(recorder, "_stem", None)

    def register_routes(self, flask_app):
        flask_app.add_url_rule(
            "/rollback", "rollback_page", lambda: render_template("rollback.html")
        )
        flask_app.add_url_rule(
            "/api/rollback/status", "api_rollback_status", self._status, methods=["GET"]
        )
        flask_app.add_url_rule(
            "/api/rollback/sessions",
            "api_rollback_sessions",
            self._sessions,
            methods=["GET"],
        )
        flask_app.add_url_rule(
            "/api/rollback/settings",
            "api_rollback_settings_get",
            self._settings_get,
            methods=["GET"],
        )
        flask_app.add_url_rule(
            "/api/rollback/settings",
            "api_rollback_settings_post",
            self._settings_post,
            methods=["POST"],
        )
        flask_app.add_url_rule(
            "/api/rollback/prune",
            "api_rollback_prune",
            self._prune,
            methods=["POST"],
        )
        flask_app.add_url_rule(
            "/api/rollback/<session_name>",
            "api_rollback_session",
            self._session_detail,
            methods=["GET"],
        )
        flask_app.add_url_rule(
            "/api/rollback/<session_name>/pin",
            "api_rollback_pin",
            self._pin,
            methods=["POST"],
        )
        flask_app.add_url_rule(
            "/api/rollback/<session_name>/unpin",
            "api_rollback_unpin",
            self._unpin,
            methods=["POST"],
        )
        flask_app.add_url_rule(
            "/api/rollback/<session_name>/delete",
            "api_rollback_delete",
            self._delete,
            methods=["POST"],
        )

        flask_app.add_url_rule(
            "/api/rollback/<session_name>/segment/<stem>",
            "api_rollback_segment",
            self._segment,
            methods=["GET"],
        )
        flask_app.add_url_rule(
            "/api/rollback/<session_name>/segment/<stem>/<cam>/frame/<int:index>",
            "api_rollback_frame",
            self._frame,
            methods=["GET"],
        )
        flask_app.add_url_rule(
            "/api/rollback/<session_name>/segment/<stem>/<cam>/clip",
            "api_rollback_clip",
            self._clip,
            methods=["GET"],
        )
        flask_app.add_url_rule(
            "/api/rollback/<session_name>/replay",
            "api_rollback_replay",
            self._replay_command,
            methods=["GET"],
        )

    def _status(self):
        data_dir = self._data_dir()
        sessions = _iter_sessions(data_dir)
        sizes = {s["name"]: _session_size(s) for s in sessions}
        total = sum(sizes.values())

        recorder = self._recorder()
        config = self.context.get("config")
        vision = self.context.get("vision_instance")
        replay_mode = os.environ.get("ISPY_REPLAY_MODE") == "1" or bool(
            getattr(vision, "replay_mode", False)
        )
        block = config.get("rollback", {}) if config is not None else {}
        block = block if isinstance(block, dict) else {}
        max_total = float(block.get("max_total_mb", 2048) or 2048) * 1024 * 1024

        recording = False
        session = None
        counters = {"frames": 0, "dropped": 0, "segment": None, "cams": []}
        disk_problem = None
        if recorder is not None:
            recording = bool(getattr(recorder, "_started", False))
            session = getattr(recorder, "_session_name", None)
            counters = {
                "frames": int(getattr(recorder, "_frame_counter", 0)),
                "dropped": int(getattr(recorder, "_dropped", 0)),
                "segment": getattr(recorder, "_stem", None),
                "cams": list(getattr(recorder, "_cams", []) or []),
            }
            try:
                disk_problem = recorder._disk_problem()
            except Exception:
                disk_problem = None

        try:
            free = _mb(shutil.disk_usage(str(data_dir)).free)
        except Exception:
            free = None

        return jsonify(
            enabled=bool(getattr(recorder, "_enabled", block.get("enabled", True))),
            performance_off=bool(getattr(recorder, "_disabled_by_profile", False)),
            replay_mode=replay_mode,
            recording=recording,
            data_dir=str(data_dir),
            session=session,
            counters=counters,
            disk_problem=disk_problem,
            sessions=len(sessions),
            total_mb=_mb(total),
            max_total_mb=round(max_total / (1024 * 1024), 2),
            over_cap=total > max_total,
            free_mb=free,
            fps=float(getattr(recorder, "_fps", block.get("fps", 30.0)) or 30.0),
            downsample=int(getattr(recorder, "_downsample", block.get("downsample", 1)) or 1),
            queued=int(getattr(recorder, "_queue", None).qsize())
            if getattr(recorder, "_queue", None) is not None
            else 0,
        )

    def _replay_command(self, session_name):
        session = self._find_session(session_name)
        if session is None:
            return jsonify(error="Session not found"), 404
        segments = _iter_segments(session["path"])
        cameras = []
        for segment in segments:
            for camera in self._segment_cams(segment):
                if camera not in cameras:
                    cameras.append(camera)
        camera = request.args.get("cam", "").strip()
        if camera and camera not in cameras:
            return jsonify(error="Camera not found in session"), 404
        if not camera and cameras:
            camera = cameras[0]
        args = ["ispy-boot", "--replay", str(session["path"])]
        if camera:
            args.extend(["--replay-cam", camera])
        return jsonify(command=" ".join(shlex.quote(arg) for arg in args))

    def _settings_get(self):
        config = self.context.get("config")
        block = {}
        if config is not None:
            raw = config.get("rollback", {})
            block = raw if isinstance(raw, dict) else {}
        defaults = RollBack.config_schema()
        out = {}
        for key, spec in defaults.items():
            value = block.get(key, spec["default"])
            out[key] = {
                "value": value,
                "type": spec["type"],
                "label": spec["label"],
                "hint": spec["hint"],
                "default": spec["default"],
            }
        return jsonify(settings=out)

    def _settings_post(self):
        data = request.get_json(force=True) or {}
        config = self.context.get("config")
        if config is None:
            return jsonify(error="No config available"), 500

        block = config.get("rollback", {})
        block = dict(block) if isinstance(block, dict) else {}
        applied = {}
        rejected = []

        for key, caster in _NUMERIC_SETTINGS.items():
            if key not in data:
                continue
            try:
                value = caster(data[key])
            except (TypeError, ValueError):
                rejected.append(key)
                continue
            floor = _MINIMUMS[key]
            if value < floor:
                value = caster(floor)
            block[key] = value
            applied[key] = value

        for key in _TEXT_SETTINGS:
            if key not in data:
                continue
            value = str(data[key]).strip()
            if not value:
                rejected.append(key)
                continue
            block[key] = value
            applied[key] = value

        for key in _BOOL_SETTINGS:
            if key not in data:
                continue
            block[key] = bool(data[key])
            applied[key] = block[key]

        config.set("rollback", block)
        config.save()
        payload = {"success": True, "applied": applied, "needs_restart": True}
        if rejected:
            payload["rejected"] = rejected
        return jsonify(payload)

    def _session_row(self, session) -> dict:
        path = session["path"]
        started = session.get("started")
        meta = self._session_meta(session)
        segments = _iter_segments(path)
        return {
            "name": session["name"],
            "path": str(path),
            "legacy": session["legacy"],
            "pinned": session["pinned"],
            "started": started,
            "version": meta.get("version"),
            "unit": meta.get("unit", "frc"),
            "cameras": sorted(meta.get("cameras", {}) or {}),
            "segments": len(segments),
            "frames": sum(_segment_frames(seg) for seg in segments),
            "size_mb": _mb(_session_size(session)),
        }

    def _sessions(self):
        rows = [self._session_row(s) for s in _iter_sessions(self._data_dir())]
        rows.reverse()  # newest first reads better in a table
        return jsonify(sessions=rows, data_dir=str(self._data_dir()))

    def _session_detail(self, session_name):
        session = self._find_session(session_name)
        if session is None:
            return jsonify(error="Session not found"), 404
        row = self._session_row(session)
        row["segment_list"] = [
            self._segment_payload(seg) for seg in _iter_segments(session["path"])
        ]
        return jsonify(session=row)

    def _prune(self):
        data_dir = self._data_dir()
        config = self.context.get("config")
        block = config.get("rollback", {}) if config is not None else {}
        block = block if isinstance(block, dict) else {}
        cap = float(block.get("max_total_mb", 2048) or 2048) * 1024 * 1024

        recorder = self._recorder()
        active = getattr(recorder, "_session_name", None)
        sessions = _iter_sessions(data_dir)
        sizes = {s["name"]: _session_size(s) for s in sessions}
        total = sum(sizes.values())

        deleted = []
        for session in sessions:
            if total <= cap:
                break
            if session["name"] == active or session["pinned"]:
                continue
            _release_tree(session["path"])
            if not _delete_session(session):
                continue
            total -= sizes[session["name"]]
            deleted.append(session["name"])

        blocked = total > cap
        if blocked:
            self.logger.warning(
                "Prune could not get under the cap: %.1f MB over, everything left "
                "is pinned or the active session",
                (total - cap) / (1024 * 1024),
            )
        return jsonify(
            success=True,
            deleted=deleted,
            total_mb=_mb(total),
            over_cap=blocked,
        )

    def _pin(self, session_name):
        return self._set_pinned(session_name, True)

    def _unpin(self, session_name):
        return self._set_pinned(session_name, False)

    def _set_pinned(self, session_name, pinned: bool):
        session = self._find_session(session_name)
        if session is None:
            return jsonify(error="Session not found"), 404
        marker = session["path"] / _PINNED_FILE
        try:
            if pinned:
                marker.touch()
            elif marker.is_file():
                marker.unlink()
        except OSError as e:
            return jsonify(error=f"Could not update PINNED: {e}"), 500
        self.logger.info(
            "Session %s %s from the web UI",
            session_name,
            "pinned" if pinned else "unpinned",
        )
        return jsonify(success=True, pinned=pinned)

    def _delete(self, session_name):
        session = self._find_session(session_name)
        if session is None:
            return jsonify(error="Session not found"), 404
        recorder = self._recorder()
        active = session_name == getattr(recorder, "_session_name", None)
        if active:
            return jsonify(error="That is the session being recorded right now"), 400
        _release_tree(session["path"])
        if not _delete_session(session):
            return jsonify(error="Could not delete the session"), 500
        self._session_meta_cache.pop(session_name, None)
        self.logger.info("Deleted session %s from the web UI", session_name)
        return jsonify(success=True)

    def _segment(self, session_name, stem):
        session = self._find_session(session_name)
        if session is None:
            return jsonify(error="Session not found"), 404
        seg = self._find_segment(session, stem)
        if seg is None:
            return jsonify(error="Segment not found"), 404

        sidecar = self._read_sidecar(seg)
        records = sidecar.get("records")
        records = records if isinstance(records, list) else []
        cams = seg["cams"] or [
            p.stem.split("_")[-1] for p in seg["files"] if p.suffix.lower() in _READ_EXTS
        ]

        timeline = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            timeline.append(
                {
                    "i": rec.get("i"),
                    "t": rec.get("t"),
                    "n": int(rec.get("detection_count", 0) or 0),
                    "d": [
                        trimmed
                        for trimmed in (
                            _trim_detection(d) for d in rec.get("detections") or []
                        )
                        if trimmed is not None
                    ],
                    "p": _trim_pose(rec.get("robot_pose")),
                }
            )

        meta = self._session_meta(session)
        return jsonify(
            segment=self._segment_payload(seg),
            timeline=timeline,
            cameras=sorted(meta.get("cameras", {}) or {}),
            camera_meta=(self._camera_meta(session) or {}),
            unit=meta.get("unit", "frc"),
        )

    def _camera_meta(self, session) -> dict:
        meta = self._session_meta(session)
        cams = meta.get("cameras")
        if not isinstance(cams, dict):
            return {}
        unit = meta.get("unit", "frc")
        out = {}
        for name, cfg in cams.items():
            if not isinstance(cfg, dict):
                continue
            entry = dict(cfg)
            for key in ("x", "y", "height"):
                try:
                    entry[key] = unit_to_output(float(entry.get(key) or 0), unit)
                except (TypeError, ValueError):
                    entry[key] = 0.0
            out[name] = entry
        return out

    def _resolve_video(self, session_name, stem, cam):
        session = self._find_session(session_name)
        if session is None:
            return None, None, (jsonify(error="Session not found"), 404)
        seg = self._find_segment(session, stem)
        if seg is None:
            return None, None, (jsonify(error="Segment not found"), 404)
        path = self._cam_file(seg, cam)
        if path is None or not path.is_file():
            return None, None, (jsonify(error="Clip not found"), 404)
        return seg, path, None

    def _frame(self, session_name, stem, cam, index):
        seg, path, err = self._resolve_video(session_name, stem, cam)
        if err is not None:
            return err
        if index < 0:
            return jsonify(error="Bad frame index"), 400
        try:
            with _CAPS_LOCK:
                payload = _encoded(path, index)
        except Exception:
            self.logger.exception("Could not extract a frame from %s", path)
            return jsonify(error="Could not read that frame"), 500
        if payload is None:
            return jsonify(error="Frame not available"), 404
        return Response(
            payload, mimetype="image/jpeg", headers={"Cache-Control": "max-age=30"}
        )

    def _clip(self, session_name, stem, cam):
        seg, path, err = self._resolve_video(session_name, stem, cam)
        if err is not None:
            return err
        return send_file(str(path), as_attachment=True, download_name=path.name)

def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
