import json
import time
from pathlib import Path

import cv2

from iSpy.core.rollback import _iter_segments, _sanitize_cam
from iSpy.vision.Cameras.base import CameraBase

# recorded clips are read back in these, in the order the segments were written
_VIDEO_EXTS = (".avi", ".mp4")


class ReplayCamera(CameraBase):
    # plays a recording back as if it were a live camera, so the whole vision
    # pipeline can be run over a past run with no robot attached
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
                "hint": "Which camera to replay out of a session folder. "
                "Leave blank to take the first one found.",
                "default": "",
            },
            "replay_speed": {
                "type": "number",
                "label": "Replay Speed",
                "hint": "1.0 is real time, 2.0 is twice as fast.",
                "default": 1.0,
            },
            "loop": {
                "type": "toggle",
                "label": "Loop",
                "hint": "Restart the recording when it reaches the end.",
                "default": True,
            },
        }

    def __init__(self, camera_config, input_size=None, grayscale=False, **kwargs):
        try:
            self._replay_speed = max(
                0.01, float(camera_config.get("replay_speed", 1.0) or 1.0)
            )
        except (TypeError, ValueError):
            self._replay_speed = 1.0
        self._replay_loop = bool(camera_config.get("loop", True))
        self._replay_cam = str(camera_config.get("replay_cam") or "")
        self._playlist = []
        self._records = []
        self._segment_ends = []
        self._replay_done = False
        self._replay_pose = None
        self._replay_i = None
        super().__init__(camera_config, input_size, grayscale, **kwargs)

    # resolving what to play

    def _resolve_files(self) -> list:
        # a folder means "replay this whole session", a file means just that clip
        source = Path(str(self.source))
        if source.is_file():
            return [source]
        if not source.is_dir():
            return []

        wanted = _sanitize_cam(self._replay_cam) if self._replay_cam else None
        files = []
        # _iter_segments is chronological by stem, which is the write order
        for seg in _iter_segments(source):
            for path in seg["files"]:
                if path.suffix.lower() not in _VIDEO_EXTS:
                    continue
                if wanted is not None and not path.stem.endswith(f"_{wanted}"):
                    continue
                files.append(path)
        return files

    def _load_records(self, path: Path) -> list:
        # the sidecar is named after the segment, not the per-camera file, so
        # rollback_<ts>_Left.avi pairs with rollback_<ts>.json
        candidates = [path.with_suffix(".json")]
        segment = path.stem.rsplit("_", 1)
        if len(segment) == 2:
            candidates.append(path.with_name(f"{segment[0]}.json"))
        for sidecar in candidates:
            if not sidecar.is_file():
                continue
            try:
                data = json.loads(sidecar.read_text())
            except (OSError, ValueError):
                continue
            records = data.get("records") or []
            if isinstance(records, list) and records:
                return records
        return []

    def _open_camera(self):
        # CameraBase.__init__ calls this and marks the camera connected on
        # success, then starts _reader. there is no device to open - the
        # recorded files are opened lazily as the reader walks the playlist
        self._playlist = []
        self._records = []
        self._segment_ends = []
        for path in self._resolve_files():
            records = self._load_records(path)
            self._playlist.append(
                {"path": path, "records": records, "cap": None, "next": 0}
            )
            self._records.extend(records)
            self._segment_ends.append(len(self._records))
        if not self._playlist:
            raise ValueError(f"no recorded video found at {self.source}")
        return None

    # reading

    def _read_frame_at(self, index: int):
        seg_no = 0
        for i, end in enumerate(self._segment_ends):
            if index < end:
                seg_no = i
                break
        seg = self._playlist[seg_no]
        start = self._segment_ends[seg_no - 1] if seg_no else 0
        # the sidecar carries one record per encoded frame, so the record index
        # is the frame number inside that segment
        frame_no = index - start

        cap = seg["cap"]
        if cap is None:
            cap = cv2.VideoCapture(str(seg["path"]))
            if not cap.isOpened():
                self.logger.error("Could not open %s for replay", seg["path"])
                return None
            seg["cap"] = cap
            seg["next"] = 0

        if frame_no != seg["next"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        seg["next"] = frame_no + 1

        if not ok:
            self.logger.warning(
                "Replay could not read frame %d of %s", frame_no, seg["path"].name
            )
            return None

        record = self._records[index] if index < len(self._records) else None
        if record is not None:
            self._replay_pose = record.get("robot_pose")
            self._replay_i = record.get("i")
        return frame

    def _publish(self, frame):
        with self.frame_lock:
            self.frame = frame
            self.frame_timestamp = time.perf_counter()
        self._frame_event.set()

    def _reader(self):
        while not self.stopped and self._playlist:
            # pace off the recorded timestamps, so a run that dropped or lagged
            # still takes the same wall-clock time to play back
            first_t = None
            if self._records:
                first_t = self._records[0].get("t")
            started = time.perf_counter()

            for index in range(len(self._records)):
                if self.stopped:
                    return
                record = self._records[index]
                stamp = record.get("t")
                if first_t is not None and stamp is not None:
                    target = started + (float(stamp) - float(first_t)) / self._replay_speed
                    delay = target - time.perf_counter()
                    if delay > 0:
                        time.sleep(min(delay, 1.0))
                frame = self._read_frame_at(index)
                if frame is not None:
                    self._publish(frame)

            if self.stopped:
                return
            self._replay_done = True
            self._release_caps()
            if not self._replay_loop:
                # hold the final frame rather than dropping to a placeholder
                return
            time.sleep(0.05)

    def _release_caps(self):
        for seg in self._playlist:
            cap = seg.get("cap")
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                seg["cap"] = None
                seg["next"] = 0

    def get_frame_age(self) -> float:
        # a replay is never "stale" - the age check would otherwise flag a
        # paused or finished recording as a dead camera
        return 0.0

    def replay_pose(self):
        # the robot pose the sidecar recorded for the frame being served, which
        # is what a replayed run should report instead of a live NT value
        return self._replay_pose

    def replay_record_i(self):
        return self._replay_i

    def replay_record(self):
        # the whole sidecar entry behind the frame being served, which is what a
        # replay-vs-recorded comparison diffs against
        i = self._replay_i
        if i is None:
            return None
        for record in self._records:
            if record.get("i") == i:
                return record
        return None

    def replay_session(self) -> str:
        # the session folder this camera is playing, so a replay can label the
        # run it came from without re-deriving it from the path
        try:
            return Path(str(self.source)).name
        except Exception:
            return ""

    def replay_finished(self) -> bool:
        return self._replay_done

    def release(self):
        # destroy() first - it sets stopped and joins the reader, otherwise the
        # reader just reopens a capture behind us and the file stays locked
        super().release()
        self._release_caps()
