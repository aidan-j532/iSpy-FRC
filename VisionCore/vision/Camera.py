import cv2
import numpy as np
import time
import logging
import threading
import subprocess
from VisionCore.config.VisionCoreConfig import VisionCoreCameraConfig
import platform

from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"

class Camera:
    def __init__(
        self,
        camera_config: VisionCoreCameraConfig,
        fps_cap: int,
        input_size: tuple,
        grayscale: bool,
    ):
        self.logger = logging.getLogger(__name__)

        self.fps_cap = fps_cap
        self.input_size = input_size  # (w, h)
        self.grayscale = grayscale

        self.source = camera_config["source"]
        self.stopped = False
        self.frame: np.ndarray | None = None
        self.frame_timestamp: float | None = None
        self.frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self.frame_timeout = 1.0 / max(self.fps_cap, 1)

        if isinstance(self.source, str) and self.source.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp")
        ):
            self.is_image = True
            self.image = cv2.imread(self.source)
            if self.image is None:
                self.logger.warning(
                    "Camera source image not found: %s — using placeholder.",
                    self.source,
                )
                # Load image
            self.image = cv2.imread(str(ASSETS_DIR / "camera_not_found.png"))
        else:
            self.is_image = False
            try:
                self._open_camera()
            except Exception as exc:
                self.logger.warning(
                    "Camera source '%s' could not be opened (%s) — "
                    "using 'Camera Not Found' placeholder.",
                    self.source,
                    exc,
                )
                self.is_image = True
                self.image = cv2.imread(str(ASSETS_DIR / "camera_not_found.png"))
            if not self.is_image:
                threading.Thread(
                    target=self._reader,
                    daemon=True,
                    name=f"CamReader-{self.source}",
                ).start()

    def _open_camera(self):
        is_windows = platform.system() == "Windows"

        if is_windows:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise ValueError(f"Camera failed to open: {self.source}")

        # Drain stale frames
        for _ in range(10):
            self.cap.grab()

        # v4l2-ctl is Linux only
        if not is_windows:
            device = (
                self.source
                if isinstance(self.source, str)
                else f"/dev/video{self.source}"
            )
            subprocess.run(
                [
                    "v4l2-ctl",
                    "-d",
                    device,
                    f"--set-fmt-video=width={self.input_size[0]},"
                    f"height={self.input_size[1]},pixelformat=MJPG",
                ],
                capture_output=True,
            )
            time.sleep(0.15)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.input_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.input_size[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps_cap)

        for _ in range(20):
            self.cap.grab()

        if not self.cap.isOpened():
            raise ValueError(f"Camera lost after configuration: {self.source}")

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning("Frame read failed on %s, retrying…", self.source)
                time.sleep(0.05)
                continue

            if frame.max() < 1:
                self.logger.debug("Solid-black frame skipped.")
                continue

            with self.frame_lock:
                self.frame = frame
                self.frame_timestamp = time.perf_counter()
            self._frame_event.set()

    def get_frame_age(self) -> float:
        if self.is_image:
            # Placeholder / still images are always "fresh"
            return 0.0
        with self.frame_lock:
            ts = self.frame_timestamp
        return 0.0 if ts is None else time.perf_counter() - ts

    def get_frame(self) -> np.ndarray | None:
        if self.is_image:
            return self.image.copy()
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None

    def destroy(self):
        self.stopped = True
        if not self.is_image and hasattr(self, "cap") and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def release(self):
        self.destroy()
