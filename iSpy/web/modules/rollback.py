import json
import logging
import math
import shutil
import threading
from pathlib import Path

import cv2
from flask import Response, jsonify, render_template, request, send_file

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
)
from iSpy.web.Backend.WebModule import WebModule

logger = logging.getLogger(__name__)

# a scrub through a segment asks for the same frames over and over, and an MJPG
# avi is cheap to decode but not free - so both the open capture and the encoded
# jpegs are held. everything here is process-wide and guarded by one lock.
# reentrant because _encoded calls _capture_for while the caller holds it
_CAPS = {}
_FRAMES = {}
_FRAME_ORDER = []
_CAPS_LOCK = threading.RLock()
_MAX_CAPS = 4
_MAX_FRAMES = 48

_JPEG_QUALITY = 85

# the rollback block is the recorder's, so the settings form is derived from its
# own schema - the UI can never drift from what the recorder accepts
_NUMERIC_SETTINGS = {
    "fps": float,
    "max_queue": int,
    "downsample": int,
    "segment_minutes": int,
    "max_total_mb": int,
}
_TEXT_SETTINGS = ("data_dir",)
_BOOL_SETTINGS = ("enabled",)

# sanity floors, so a typo in the form cannot wedge the recorder
_MINIMUMS = {
    "fps": 0.5,
    "max_queue": 1,
    "downsample": 1,
    "segment_minutes": 1,
    "max_total_mb": 1,
}


def _mb(num_bytes) -> float:
    return round(float(num_bytes or 0) / (1024 * 1024), 2)


def _capture_for(path: Path):
    with _CAPS_LOCK:
        cap = _CAPS.get(str(path))
        if cap is not None:
            return cap
        while len(_CAPS) >= _MAX_CAPS:
            _, old = _CAPS.popitem()
            try:
                old.release()
            except Exception:
                pass
        cap = cv2.VideoCapture(str(path))
        _CAPS[str(path)] = cap
        return cap


def _release_tree(root: Path) -> None:
    # windows will not unlink a file that is still open, so any cached capture
    # for a segment inside this tree has to be closed before a delete. without
    # this, deleting the session you are previewing silently fails
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
    # a seek past the end, or a segment whose writer died mid-file, just means
    # the frame is not there. do not fall back to frame 0 - that would silently
    # show the wrong picture on a scrub
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
    # jsonify emits a bare NaN token, which is not valid JSON - res.json() then
    # throws in the browser and the whole segment loads with an empty timeline.
    # the EKF pose path can produce NaN, so every number is coerced here
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
    # only what the player draws - the full record carries keypoints and rays
    # that would bloat every scrub request
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
        # session.json is written once when a session opens and never changes
        # after, so one read per session per process is enough
        self._session_meta_cache = {}

    def _session_meta(self, session) -> dict:
        key = session["name"]
        if key in self._session_meta_cache:
            return self._session_meta_cache[key]
        try:
            raw = json.loads(
                (session["path"] / _SESSION_FILE).read_text(encoding="utf-8")
            )
            meta = raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            meta = {}
        self._session_meta_cache[key] = meta
        return meta

    # lookups

    def _recorder(self):
        # the live RollBack instance lives in vision.utilities, not in the web
        # app, so live counters have to be asked for rather than held here
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
        # resolved through _iter_sessions rather than joined from the URL, so a
        # crafted name can never walk out of data_dir
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
        # a legacy flat segment has one file and no cam list - the recorded
        # camera name is unknown, so serve it under its own filename
        if len(videos) == 1 and not seg["cams"]:
            return videos[0]
        return None

    def _read_sidecar(self, seg) -> dict:
        for path in seg["files"]:
            if path.suffix.lower() != ".json":
                continue
            # crash tolerant: a sidecar cut off mid-write still lists, and
            # still plays back everything it managed to write
            return read_sidecar(path)
        return {}

    def _segment_payload(self, seg) -> dict:
        sidecar = self._read_sidecar(seg)
        records = sidecar.get("records")
        records = records if isinstance(records, list) else []
        cams = seg["cams"] or [
            p.stem.split("_")[-1] for p in seg["files"] if p.suffix.lower() in _READ_EXTS
        ]
        return {
            "stem": seg["stem"],
            "cams": cams,
            "frames": len(records),
            "started": sidecar.get("started"),
            "fps": sidecar.get("fps"),
            "has_sidecar": bool(records),
            # the open segment has no avi index yet and a cached capture never
            # sees the frames still being appended, so the player must not offer
            # it as scrubbable
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
        # the segment the live recorder is writing to, if there is one. only the
        # recorder knows which file is still open
        recorder = self._recorder()
        if recorder is None or not getattr(recorder, "_started", False):
            return None
        return getattr(recorder, "_stem", None)

    # routes

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

    # status and settings

    def _status(self):
        data_dir = self._data_dir()
        sessions = _iter_sessions(data_dir)
        sizes = {s["name"]: _session_size(s) for s in sessions}
        total = sum(sizes.values())

        recorder = self._recorder()
        config = self.context.get("config")
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
            enabled=bool(block.get("enabled", True)),
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

    # sessions

    def _session_row(self, session) -> dict:
        path = session["path"]
        started = session.get("started")
        # the cameras, version and unit all live in session.json - reading them
        # from there is what kept _session_row from having to parse every
        # sidecar in the session just to label the row
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
            "frames": sum(
                len(self._read_sidecar(seg).get("records") or [])
                for seg in segments
            ),
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
        # the same oldest-unpinned-first walk the recorder does when it opens a
        # segment, but run on demand from the tab. the recorder keeps its own
        # copy so its rotation path stays independent of the web app
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
            # same as the delete endpoint: an open capture would block the rmtree
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
        # the player may be holding this session's segments open right now
        _release_tree(session["path"])
        if not _delete_session(session):
            return jsonify(error="Could not delete the session"), 500
        self._session_meta_cache.pop(session_name, None)
        self.logger.info("Deleted session %s from the web UI", session_name)
        return jsonify(success=True)



    # playback

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

        row = self._session_row(session)
        return jsonify(
            segment=self._segment_payload(seg),
            timeline=timeline,
            cameras=row["cameras"],
            camera_meta=(self._camera_meta(session) or {}),
            unit=row["unit"],
        )

    def _camera_meta(self, session) -> dict:
        cams = self._session_meta(session).get("cameras")
        return cams if isinstance(cams, dict) else {}

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
