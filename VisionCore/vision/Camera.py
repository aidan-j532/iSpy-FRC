import cv2
import numpy as np
import time
import logging
import threading
import subprocess
from VisionCore.config.VisionCoreConfig import VisionCoreCameraConfig
import platform
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent
_ASSETS_DIR = _PACKAGE_ROOT.parent / "assets"

class Camera:

    # Resolution used for the synthetic "no camera" placeholder frame.
    _PLACEHOLDER_W = 640
    _PLACEHOLDER_H = 480

    def __init__(self, camera_config: VisionCoreCameraConfig, fps_cap: int, input_size: tuple, grayscale: bool):
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
                    "Could not read image '%s' - using synthetic placeholder frame.",
                    self.source,
                )
                self.image = self._make_placeholder_frame()
        else:
            self.is_image = False
            try:
                self._open_camera()
            except ValueError as exc:
                self.logger.warning(
                    "Camera source '%s' could not be opened (%s) - using synthetic placeholder frame.",
                    self.source,
                    exc,
                )
                # Treat the object as an "image" source backed by the placeholder so
                # the rest of the pipeline can keep running without modification.
                self.is_image = True
                self.image = self._make_placeholder_frame()
                return

            threading.Thread(
                target=self._reader,
                daemon=True,
                name=f"CamReader-{self.source}",
            ).start()

    def _make_placeholder_frame(
        self, width: int = _PLACEHOLDER_W,
        height: int = _PLACEHOLDER_H,
    ) -> np.ndarray:
        # Try to load the image first from assets/image.png
        try:
            placeholder = cv2.imread(str(_ASSETS_DIR / "camera_not_found.png"))
            if placeholder is not None:
                return cv2.resize(placeholder, (width, height))
        except Exception as exc:
            self.logger.debug(f"Could not load placeholder image: {exc}")
        
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        text = "Camera Not Found"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cx = (width - tw) // 2
        cy = (height + th) // 2
        cv2.putText(frame, text, (cx, cy), font, scale, (180, 180, 180), thickness, cv2.LINE_AA)
        return frame

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

        # v4l2-ctl is Linux-only
        if not is_windows:
            device = self.source if isinstance(self.source, str) else f"/dev/video{self.source}"
            subprocess.run(
                [
                    "v4l2-ctl", "-d", device,
                    f"--set-fmt-video=width={self.input_size[0]},height={self.input_size[1]},pixelformat=MJPG",
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
                self.logger.warning(f"Frame read failed on {self.source}, retrying…")
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
            return 0.0
        with self.frame_lock:
            ts = self.frame_timestamp
        return 0.0 if ts is None else time.perf_counter() - ts

    def get_frame(self) -> np.ndarray | None:
        if self.is_image:
            return self.image.copy() if self.image is not None else None
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None

    def destroy(self):
        self.stopped = True
        if not self.is_image and hasattr(self, "cap") and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def release(self):
        self.destroy()