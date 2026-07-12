# This code is gonna let you control all the camera settings, and switch between cameras if their are multiple
# Needs to merge with CameraApp
# Camera live view + per-camera settings. Absorbs CameraApp.py's routes -
# CameraApp should stop creating its own Flask() and instead just push
# frames into a shared frame store that this blueprint reads from, the
# same way Status.record_frame_data() is fed from the main loop.
#
# TODO: port these from CameraApp.py onto this blueprint:
#   /cameras/video_feed              (MJPEG multipart stream)
#   /cameras/api/cameras             (list)
#   /cameras/api/<name>/settings     (GET/POST)
#   /cameras/api/<name>/feed         (per-camera MJPEG stream)

import logging
import threading
import cv2
from flask import Blueprint, render_template, Response, jsonify, current_app

logger = logging.getLogger(__name__)

bp = Blueprint("cameras", __name__, url_prefix="/cameras")

_lock = threading.Lock()
_frames: dict[str, "any"] = {}


def set_frame(camera_name: str, frame) -> None:
    """Call this from iSpy.py's _update_camera_app (renamed/adapted) instead
    of CameraApp.set_frame - same idea, just feeding this blueprint's store."""
    with _lock:
        _frames[camera_name] = frame


@bp.route("/")
def cameras_page():
    return render_template("cameras.html", active_page="cameras")


@bp.route("/api/cameras")
def api_list_cameras():
    cams = current_app.config.get("ISPY_CAMERAS", [])
    out = []
    for i, cam in enumerate(cams):
        name = cam.config.get("name", f"Camera {i+1}") if hasattr(cam, "config") else f"Camera {i+1}"
        out.append({"id": i, "name": name, "source": str(cam.config.get("source", "unknown")) if hasattr(cam, "config") else "unknown"})
    return jsonify(cameras=out)


@bp.route("/video_feed/<camera_name>")
def video_feed(camera_name):
    return Response(_generate(camera_name), mimetype="multipart/x-mixed-replace; boundary=frame")


def _generate(camera_name):
    import time
    while True:
        with _lock:
            frame = _frames.get(camera_name)
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")