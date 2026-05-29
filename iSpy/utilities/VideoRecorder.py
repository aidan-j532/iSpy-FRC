import cv2
import threading
import queue
import time
import os
import platform
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

def _best_codec():
    system = platform.system().lower()

    if system == "windows":
        return ("mp4v", ".mp4")
    elif system == "darwin":
        return ("mp4v", ".mp4")
    else:
        return ("MJPG", ".avi")

class VideoRecorder:
    def __init__(
        self,
        output_dir="VideoRecordings",
        fps=30.0,
        codec=None,
        extension=None,
        max_queue=300,
        downsample=1,
    ):
        self.output_dir = output_dir
        self.fps = fps
        self.downsample = max(1, downsample)

        self._forced_codec = codec
        self._forced_ext = extension

        self._queue = queue.Queue(maxsize=max_queue)
        self._writer = None
        self._thread = None

        self._started = False
        self._stopped = False

        self._frame_counter = 0
        self._dropped = 0

        self._size = None
        self._last_emit = 0

        os.makedirs(output_dir, exist_ok=True)

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

    def start(self, width: int, height: int):
        if self._started:
            return

        self._size = (width, height)

        codec, ext = (self._forced_codec, self._forced_ext)
        if not codec or not ext:
            codec, ext = _best_codec()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(self.output_dir, f"recording_{timestamp}{ext}")

        fourcc = cv2.VideoWriter_fourcc(*codec)

        self._writer = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))

        # fallback if needed
        if not self._writer.isOpened():
            logger.warning("Primary codec failed, switching to MJPG fallback")

            codec, ext = ("MJPG", ".avi")
            filename = os.path.join(self.output_dir, f"recording_{timestamp}{ext}")
            fourcc = cv2.VideoWriter_fourcc(*codec)

            self._writer = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))

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

    def write(self, frame):
        if not self._started or self._stopped:
            return

        self._frame_counter += 1

        if self._frame_counter % self.downsample != 0:
            return

        frame = self._clean_frame(frame)
        if frame is None:
            return

        try:
            if self._queue.full():
                # drop oldest frame instead of crashing pipeline
                self._queue.get_nowait()

            self._queue.put_nowait(frame)

        except queue.Full:
            self._dropped += 1

    def stop(self):
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
