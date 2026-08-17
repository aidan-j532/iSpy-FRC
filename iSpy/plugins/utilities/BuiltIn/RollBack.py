import cv2
import threading
import queue
import time
import os
import platform
import logging
import numpy as np
from datetime import datetime
from iSpy.plugins.bases import UtilityBase

logger = logging.getLogger(__name__)


def _best_codec():
    system = platform.system().lower()

    if system == "windows":
        return ("mp4v", ".mp4")
    elif system == "darwin":
        return ("mp4v", ".mp4")
    else:
        return ("MJPG", ".avi")


class RollBack(UtilityBase):
    plugin_name = "rollback"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "data_dir": {
                "type": "text",
                "label": "Output Directory",
                "hint": "Where recording files are written.",
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
                "default": 300,
            },
            "downsample": {
                "type": "number",
                "label": "Downsample",
                "hint": "Record every Nth frame (1 = every frame).",
                "default": 1,
            },
        }

    def __init__(self, context: dict):
        super().__init__(context)
        self.logger = logging.getLogger(__name__)
        # no enabled flag - being present in the config IS the switch.
        # record_mode/record_dir used to be top-level config, live here now
        self._video_output_dir = self.config.get("data_dir", "RollbackSave")
        self._fps = float(self.config.get("fps", 30.0))
        self._forced_codec = None
        self._forced_ext = None
        self._max_queue = int(self.config.get("max_queue", 300))
        self._downsample = max(1, int(self.config.get("downsample", 1)))

        self._queue = queue.Queue(maxsize=self._max_queue)
        self._writer = None
        self._thread = None
        self._started = False
        self._stopped = False
        self._frame_counter = 0
        self._dropped = 0
        self._size = None

        os.makedirs(self._video_output_dir, exist_ok=True)

    def start(self):
        pass

    def update(self, frame_data: dict):
        # This is what frame data looks like :)
        # frame_data = {
        #     "detections": detections,
        #     "detection_count": len(detections),
        #     "frame": frame,
        #     "fps": 1 / loop_s if loop_s > 0 else 0,
        #     "loop_s": loop_s,
        #     "vision_s": vision_s,
        #     "camera_lag_s": camera_lag_s,
        #     "cameras": self.cameras,
        #     "camera_frames": handler.get_camera_frames(),
        #     "code_times": code_times,
        #     "debug_data": {},
        #     "pipeline_name": "object_detection",
        #     "pipeline_settings": {...},
        #     "camera_config": {...},
        # }
        frame = frame_data.get("frame")
        if frame is None:
            return

        if not self._started:
            h, w = frame.shape[:2]
            self._start_recorder(w, h)
            if not self._started:
                return

        self._write(frame)
        
        

    def stop(self):
        self._stop_recorder()

    def _clean_frame(self, frame):
        if frame is None:
            return None

        if not isinstance(frame, np.ndarray):
            return None

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        if len(frame.shape) != 3:
            return None

        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if self._size is not None:
            frame = cv2.resize(frame, self._size)

        return np.ascontiguousarray(frame)

    def _start_recorder(self, width: int, height: int):
        if self._started:
            return

        self._size = (width, height)

        codec, ext = (self._forced_codec, self._forced_ext)
        if not codec or not ext:
            codec, ext = _best_codec()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(self._video_output_dir, f"recording_{timestamp}{ext}")

        fourcc = cv2.VideoWriter_fourcc(*codec)

        self._writer = cv2.VideoWriter(
            filename, fourcc, self._fps, (width, height)
        )

        if not self._writer.isOpened():
            logger.warning("Primary codec failed, switching to MJPG fallback")

            codec, ext = ("MJPG", ".avi")
            filename = os.path.join(
                self._video_output_dir, f"recording_{timestamp}{ext}"
            )
            fourcc = cv2.VideoWriter_fourcc(*codec)

            self._writer = cv2.VideoWriter(
                filename, fourcc, self._fps, (width, height)
            )

        if not self._writer.isOpened():
            logger.error("VideoWriter failed completely.")
            self._writer = None
            return

        self._started = True
        self._stopped = False

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        logger.info("Recording started: %s", filename)

    def _worker(self):
        while True:
            try:
                frame = self._queue.get()
                if frame is None:
                    break
                if self._writer:
                    self._writer.write(frame)
                self._queue.task_done()
            except Exception as e:
                logger.error(f"Error in video worker: {e}")

    def _write(self, frame):
        if not self._started or self._stopped:
            return

        self._frame_counter += 1

        if self._frame_counter % self._downsample != 0:
            return

        frame = self._clean_frame(frame)
        if frame is None:
            return

        try:
            if self._queue.full():
                self._queue.get_nowait()

            self._queue.put_nowait(frame)

        except queue.Full:
            self._dropped += 1

    def _stop_recorder(self):
        if not self._started:
            return

        self._stopped = True

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._thread:
            self._thread.join(timeout=15)

        time.sleep(1.0)

        if self._writer:
            self._writer.release()
            self._writer = None

        time.sleep(1.0)

        self._started = False

        logger.info(
            "Recording stopped. Frames=%d Dropped=%d",
            self._frame_counter,
            self._dropped,
        )
