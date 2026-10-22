import threading
import time
from pathlib import Path

import cv2

from iSpy.core.rollback import _iter_segments, _sanitize_cam, read_sidecar
from iSpy.vision.Cameras.base import CameraBase

_VIDEO_EXTS = (".avi", ".mp4")


class _SegmentCapture:
    def __init__(self, paths: list[Path], loop: bool):
        self.paths = paths
        self.loop = loop
        self.index = 0
        self.cycle = 0
        self.capture = None
        self._open_current()

    def _open_current(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(str(self.paths[self.index]))
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise ValueError(f"Could not open replay clip: {self.paths[self.index]}")

    def read(self):
        while self.capture is not None:
            ok, frame = self.capture.read()
            if ok:
                return True, frame
            self.index += 1
            if self.index >= len(self.paths):
                if not self.loop:
                    self.release()
                    return False, None
                self.index = 0
                self.cycle += 1
            self._open_current()
        return False, None

    def get(self, prop):
        return self.capture.get(prop) if self.capture is not None else 0.0

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class ReplayCamera(CameraBase):
    camera_type = "replay"
    plugin_name = "replay"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "source": {
                "type": "text",
                "label": "Replay Session",
                "hint": "Folder or video clip to replay",
                "default": "VideoRecordings",
            },
            "replay_cam": {
                "type": "text",
                "label": "Recorded Camera",
                "hint": "Which camera from the session to use",
                "default": "left",
            },
        }

    def __init__(self, camera_config, input_size=None, grayscale=False, **kwargs):
        self._replay_error = None
        try:
            self._playlist = self._resolve_playlist(camera_config)
        except (FileNotFoundError, ValueError) as exc:
            self._playlist = []
            self._replay_error = exc
        self._replay_clips = [entry["path"] for entry in self._playlist]
        self._records = [
            record
            for entry in self._playlist
            for record in entry["records"]
            if isinstance(record, dict)
        ]
        self._record_cursor = 0
        self._record_i = 0
        self._current_record = None
        self._replay_cycle = 0
        self._finished = False
        self._replay_loop = bool(
            camera_config.get("loop", camera_config.get("replay_loop", True))
        )
        try:
            self._replay_speed = float(camera_config.get("replay_speed", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("Replay speed must be a positive number.") from exc
        if self._replay_speed <= 0:
            raise ValueError("Replay speed must be a positive number.")
        super().__init__(
            camera_config,
            input_size=input_size,
            grayscale=grayscale,
            **kwargs,
        )
        if self._replay_error is not None:
            self.logger.warning("Replay camera is unavailable: %s", self._replay_error)

    @classmethod
    def _resolve_clips(cls, camera_config) -> list[Path]:
        configured = camera_config.get("replay_clips")
        if isinstance(configured, list) and configured:
            paths = [Path(p).expanduser().resolve() for p in configured]
            missing = [path for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Replay clip not found: {missing[0]}")
            return paths

        source = Path(str(camera_config.get("source", ""))).expanduser()
        if source.is_file() and source.suffix.lower() in _VIDEO_EXTS:
            return [source.resolve()]
        if not source.is_dir():
            raise FileNotFoundError(f"Replay session or clip not found: {source}")

        requested_cam = str(camera_config.get("replay_cam", "")).strip()
        clips = []
        for segment in _iter_segments(source):
            cams = segment.get("cams") or []
            matched = next(
                (cam for cam in cams if str(cam).casefold() == requested_cam.casefold()),
                None,
            )
            if matched is None and not requested_cam and cams:
                matched = cams[0]
            if matched is None:
                continue
            stem = f"{segment['stem']}_{_sanitize_cam(matched)}"
            clip = next(
                (
                    path
                    for path in segment.get("files", [])
                    if path.stem == stem and path.suffix.lower() in _VIDEO_EXTS
                ),
                None,
            )
            if clip is not None and clip.is_file():
                clips.append(clip.resolve())

        if not clips:
            label = requested_cam or "a camera"
            raise FileNotFoundError(
                f"No replay clips for {label!r} found in session {source}."
            )
        return clips

    @classmethod
    def _resolve_playlist(cls, camera_config) -> list[dict]:
        clips = cls._resolve_clips(camera_config)
        source = Path(str(camera_config.get("source", ""))).expanduser()
        requested_cam = str(camera_config.get("replay_cam", "")).strip()
        segments = _iter_segments(source) if source.is_dir() else []
        playlist = []
        for clip in clips:
            records = []
            for segment in segments:
                if clip not in segment.get("files", []):
                    continue
                sidecar = next(
                    (
                        path
                        for path in segment.get("files", [])
                        if path.suffix.lower() == ".json"
                    ),
                    None,
                )
                if sidecar is not None:
                    raw_records = read_sidecar(sidecar).get("records")
                    if isinstance(raw_records, list):
                        records = [item for item in raw_records if isinstance(item, dict)]
                break
            playlist.append({"path": clip, "records": records})
        return playlist

    def _open_camera(self):
        if not self._replay_clips:
            raise FileNotFoundError(
                str(self._replay_error or "No usable replay clips were found.")
            )
        self.cap = _SegmentCapture(self._replay_clips, self._replay_loop)

    def _reader(self):
        next_frame_time = 0.0
        try:
            while not self.stopped:
                if self.cap is None:
                    self._finished = True
                    return
                now = time.perf_counter()
                if now < next_frame_time:
                    time.sleep(min(next_frame_time - now, 0.05))
                    continue

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self._finished = True
                    return
                cycle = getattr(self.cap, "cycle", 0)
                if cycle != self._replay_cycle:
                    self._replay_cycle = cycle
                    self._record_cursor = 0
                if self._record_cursor < len(self._records):
                    self._current_record = self._records[self._record_cursor]
                    self._record_i = int(
                        self._current_record.get("i", self._record_cursor + 1)
                    )
                else:
                    self._current_record = None
                    self._record_i = self._record_cursor + 1
                self._record_cursor += 1
                self._publish(frame)

                fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
                if fps > 0:
                    interval = 1.0 / (fps * self._replay_speed)
                    if self._fps_cap > 0:
                        interval = max(interval, 1.0 / self._fps_cap)
                    next_frame_time = time.perf_counter() + interval
        finally:
            if self.stopped and self.cap is not None:
                self.cap.release()

    def _publish(self, frame):
        frame = self._apply_image_adjustments(frame)
        with self.frame_lock:
            self.frame = frame
            self.frame_timestamp = time.perf_counter()
        self._frame_event.set()

    def replay_finished(self) -> bool:
        return self._finished

    def replay_record_i(self) -> int:
        return self._record_i

    def replay_pose(self) -> dict | None:
        if not isinstance(self._current_record, dict):
            return None
        pose = self._current_record.get("robot_pose")
        return pose if isinstance(pose, dict) else None

    def get_frame_age(self) -> float:
        return 0.0

    def destroy(self):
        if getattr(self, "_destroyed", False):
            return
        self.stopped = True
        reader = getattr(self, "_reader_thread", None)
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=5.0)
        if reader is not None and reader.is_alive():
            self.logger.warning(
                "Replay reader for %s is still shutting down; its thread will "
                "release the capture after the current read completes.",
                self.source,
            )
        elif getattr(self, "cap", None) is not None:
            self.cap.release()
        self._destroyed = True
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def get_camera_info(self) -> dict:
        frame = self.get_raw_frame()
        width, height = (frame.shape[1], frame.shape[0]) if frame is not None else (0, 0)
        return {
            "width": width,
            "height": height,
            "fps": float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
            if getattr(self, "cap", None) is not None
            else 0.0,
        }
