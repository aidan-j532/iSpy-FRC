import cv2
import numpy as np
import time
import logging
import threading
import subprocess
from iSpy.config.iSpyConfig import iSpyCameraConfig
import platform
from pathlib import Path
from iSpy.utilities.device_id import _resolve_device_id

_PACKAGE_ROOT = Path(__file__).resolve().parent
_ASSETS_DIR = _PACKAGE_ROOT.parent / "assets"
try:
    cv2.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    # Older OpenCV versions don't have setLogLevel
    pass

class Camera:
    # Resolution used for the synthetic "no camera" placeholder frame.
    _PLACEHOLDER_W = 640
    _PLACEHOLDER_H = 480

    def __init__(self, camera_config: iSpyCameraConfig, input_size: tuple, grayscale: bool):
        self.logger = logging.getLogger(__name__)

        self.input_size = input_size
        self.grayscale = grayscale
        self._cap_w = input_size[0]
        self._cap_h = input_size[1]

        self.source = camera_config["source"]

        self.is_image = isinstance(self.source, str) and self.source.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp")
        )
        self.device_id = (
            _resolve_device_id(self.source)
            if isinstance(self.source, (str, int)) and not self.is_image
            else None
        )

        self.stopped = False
        self.frame: np.ndarray | None = None
        self.frame_timestamp: float | None = None
        self.frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self._frame_processors = []

        self.auto_brightness = camera_config.get("auto_brightness", True)
        self._brightness_target = 128.0
        self._brightness_gamma = 1.0
        self._brightness_lut: np.ndarray | None = None
        self._brightness_frame_count = 0

        # UVC exposure/gain settings. Short exposure ensures the camera can
        # deliver the requested frame rate even in moderate-to-low light.
        self._exposure_time = camera_config.get("exposure_time", 100)   # 10 ms
        self._gain = camera_config.get("gain", 200)

        if self.is_image:
            self.image = cv2.imread(self.source)
            if self.image is None:
                self.logger.warning(
                    "Could not read image '%s' - using synthetic placeholder frame.",
                    self.source,
                )
                self.image = self._make_placeholder_frame()
        else:
            try:
                self._open_camera()
            except (ValueError, Exception) as exc:
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
        sys_platform = platform.system()
        is_windows = sys_platform == "Windows"
        is_linux = sys_platform == "Linux"

        # CROSS-PLATFORM FIX: Only run v4l2-ctl configurations if specifically on Linux
        if is_linux:
            device = self.source if isinstance(self.source, str) else f"/dev/video{self.source}"

            _fmt_resolutions = [
                (self._cap_w, self._cap_h),    # model input size
                (640, 480),
                (1280, 720),
                (800, 600),
                (640, 360),
            ]
            _fmt_ok = False
            for _fw, _fh in _fmt_resolutions:
                try:
                    result = subprocess.run(
                        [
                            "v4l2-ctl", "-d", device,
                            f"--set-fmt-video=width={_fw},height={_fh},pixelformat=MJPG",
                        ],
                        capture_output=True, text=True, timeout=5,
                    )
                    if result.returncode == 0:
                        _fmt_ok = True
                        break
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    break

            if not _fmt_ok:
                self.logger.warning(
                    "v4l2-ctl format set failed for %s — skipping hardware format negotiation",
                    device,
                )

            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", device,
                     "--set-ctrl=exposure_dynamic_framerate=0",
                     "--set-ctrl=auto_exposure=1",
                     f"--set-ctrl=exposure_time_absolute={self._exposure_time}",
                     f"--set-ctrl=gain={self._gain}"],
                    capture_output=True, text=True, timeout=5,
                )
            except FileNotFoundError:
                pass
            except subprocess.TimeoutExpired:
                self.logger.warning("v4l2-ctl timed out setting UVC controls on %s", device)

            time.sleep(0.15)

        # CROSS-PLATFORM FIX: Map proper backends explicitly by OS type
        if is_windows:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        elif is_linux:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        else:
            # Fallback backend (e.g. AVFoundation on macOS, or generic CAP_ANY)
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_ANY)

        if not self.cap.isOpened():
            raise ValueError(f"Camera failed to open: {self.source}")

        for _ in range(10):
            self.cap.grab()

        # CROSS-PLATFORM FIX: Safely assign parameters based on platform capabilities
        if is_windows:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cap_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cap_h)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        elif is_linux:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cap_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cap_h)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cap_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cap_h)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            # Fallback configuration for macOS/other systems
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cap_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cap_h)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        for _ in range(20):
            try:
                self.cap.grab()
            except Exception:
                break

        if not self.cap.isOpened():
            raise ValueError(f"Camera lost after configuration: {self.source}")

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)) if fourcc else "N/A"
        self.logger.info(
            "Camera %s: capture %dx%d @ %.1f FPS (format %s)",
            self.source, actual_w, actual_h, actual_fps, fourcc_str,
        )
        if actual_w != self._cap_w or actual_h != self._cap_h:
            self.logger.warning(
                "Requested %dx%d but camera vetoed to %dx%d",
                self._cap_w, self._cap_h, actual_w, actual_h,
            )
        if fourcc_str not in ("MJPG", "JPEG"):
            self.logger.warning(
                "Camera format is %s (not MJPG). USB bandwidth may be higher.", fourcc_str
            )
            
        self._frame_processors = []

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning(f"Frame read failed on {self.source}, retrying...")
                time.sleep(0.05) # Don't starve CPU
                continue

            if frame.max() < 1:
                self.logger.debug("Solid-black frame skipped.")
                time.sleep(0.05) # Don't starve CPU
                continue

            if self.auto_brightness:
                self._brightness_frame_count += 1
                if self._brightness_frame_count % 15 == 0:
                    mean_br = np.mean(frame)
                    if abs(mean_br - self._brightness_target) > 5:
                        target_gamma = max(mean_br, 1.0) / self._brightness_target
                        target_gamma = np.clip(target_gamma, 0.3, 3.0)
                        self._brightness_gamma += (target_gamma - self._brightness_gamma) * 0.2
                        self._brightness_gamma = np.clip(self._brightness_gamma, 0.3, 3.0)
                        gamma_table = (
                            (np.arange(256, dtype=np.float32) / 255.0) ** self._brightness_gamma * 255.0
                        ).astype(np.uint8)
                        self._brightness_lut = gamma_table
                if self._brightness_lut is not None:
                    frame = cv2.LUT(frame, self._brightness_lut)

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
    
    def add_frame_processor(self, processor):
        if self.is_image:
            self.logger.warning("Cannot add frame processor to image source.")
            return
        self._frame_processors.append(processor)

    def get_frame(self) -> np.ndarray | None:
            if self.is_image:
                frame = self.image.copy() if self.image is not None else None
            else:
                with self.frame_lock:
                    frame = self.frame.copy() if self.frame is not None else None

            if frame is None:
                return None

            for processor in self._frame_processors:
                try:
                    frame = processor.process(frame)
                except Exception as exc:
                    self.logger.warning(f"Frame processor error: {exc}")
                    break

            return frame
        
    def destroy(self):
        self.stopped = True
        if not self.is_image and hasattr(self, "cap") and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def release(self):
        self.destroy()