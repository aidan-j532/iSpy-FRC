import atexit
import logging
import platform
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from iSpy.config.iSpyConfig import iSpyCameraConfig
from iSpy.vision.Cameras._device_guard import free_camera_device
from iSpy.vision.Object import Object

_PACKAGE_ROOT = Path(__file__).resolve().parent
_ASSETS_DIR = _PACKAGE_ROOT.parents[1] / "assets"

try:
    cv2.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    # Older OpenCV versions don't have setLogLevel
    pass


class CameraOpenTimeout(ValueError):
    pass


# Bound on concurrently-live driver-open threads across all cameras. A missing
# or wedged camera re-opens every few seconds; without a cap each abandoned
# retry spawns a thread that lingers until the driver returns, leaking threads
# unboundedly. The slot is only released when the worker actually exits, so
# once the cap is hit further opens fail fast and resume when a slot frees.
_OPEN_WORKER_MAX = 4
_open_worker_guard = threading.Lock()
_open_worker_live = 0


class CameraBase:
    """A camera *source*: something that produces frames.

    This is the low-level frame pipeline every concrete source (OpenCV USB
    camera, DJI Tello, RTSP stream, ...) is built on. It owns the reader
    thread, automatic reconnect while the device is missing, placeholder
    frames, image adjustment knobs and the calibration mode / heartbeat state
    shared with the web wizard.

    Concrete sources subclass :class:`CameraBase`, set a unique ``camera_type``
    and (optionally) override how the capture device is opened. The capture
    machinery keeps working unchanged - the class only needs to know how to
    *produce* the frame, exactly like a pipeline only knows how to *use* it.
    """

    camera_type = "generic"

    _PLACEHOLDER_W = 640
    _PLACEHOLDER_H = 480

    # first-frame timeout (MSMF is slow to start)
    _STREAM_START_TIMEOUT = 1.5

    # max time cv2.VideoCapture() can block before we give up
    _CAP_OPEN_TIMEOUT = 15.0

    _READ_RETRY_DELAY = 0.05
    _READ_REOPEN_AFTER = 10
    _READ_WARN_EVERY = 20

    _CALIBRATION_HEARTBEAT_TIMEOUT = 10.0

    # how long to wait between reconnect attempts when the device is missing
    _RECONNECT_RETRY_DELAY = 3.0
    # poll `stopped` at most every this long while sleeping so a leaked reader
    # never keeps polling a bogus source (e.g. /dev/video99) through interpreter
    # shutdown and races native (RKNN/OpenCV) teardown.
    _RECONNECT_POLL_S = 0.05
    # warn about the missing device at most every N attempts (~30s)
    _RECONNECT_WARN_EVERY = 10

    # reported frame age while the device is missing (JSON-safe "very stale")
    _DISCONNECTED_AGE_S = 1e6

    @staticmethod
    def _get_capture_backend_candidates(sys_platform: str | None = None):
        platform_name = (sys_platform or platform.system()).lower()
        if platform_name == "windows":
            return [cv2.CAP_MSMF]
        if platform_name == "linux":
            return [cv2.CAP_V4L2, cv2.CAP_ANY]
        return [cv2.CAP_ANY]

    @classmethod
    def config_schema(cls) -> dict:
        """Schema describing which config keys configure this camera source.

        Mirrors the pipeline config_schema() convention: {key: {...field}}.
        Consumed by the web UI to render the per-source field section.
        """
        return {}

    @classmethod
    def discover(cls, claimed_sources: set | None = None) -> list[dict]:
        """Enumerate the sources this camera type can currently see.

        Returns a list of ``{"path", "name", "device_id", ...}`` dicts suitable
        for the web discovery endpoint. Defaults to nothing - only sources that
        can actually be enumerated override it.
        """
        return []

    def __init__(
        self,
        camera_config,
        input_size: tuple | None = None,
        grayscale: bool = False,
        **kwargs,
    ):
        self.logger = logging.getLogger(__name__)

        if input_size is None:
            input_size = (self._PLACEHOLDER_W, self._PLACEHOLDER_H)
        self.input_size = input_size
        self.grayscale = bool(grayscale)
        self._cap_w = input_size[0]
        self._cap_h = input_size[1]

        self.source = camera_config["source"]

        self._is_url_source = (
            isinstance(self.source, str) and "://" in self.source
        )

        self.is_image = (
            isinstance(self.source, str)
            and self.source.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            and not self._is_url_source
        )

        self.stopped = False
        self.frame: np.ndarray | None = None
        self.frame_timestamp: float | None = None
        self.frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self._frame_processors = []

        # False until the capture device is successfully opened. While False,
        # the placeholder frame is served but the reader thread keeps polling
        # for the device so it recovers automatically when plugged back in.
        self._connected = False
        self._reconnect_failures = 0
        self._placeholder_frame = self._make_placeholder_frame()

        self.calibration_active = False
        self.calibration_last_seen = 0.0

        self._brightness = float(camera_config.get("brightness", 0) or 0)
        self._contrast = float(camera_config.get("contrast", 0) or 0)
        self._saturation = float(camera_config.get("saturation", 0) or 0)
        self._gamma = float(camera_config.get("gamma", 1.0) or 1.0)
        self._white_balance = float(camera_config.get("white_balance", 0) or 0)
        self._tint = float(camera_config.get("tint", 0) or 0)

        self._prefer_alternate_backend = False
        self._reopen_requested = False

        self._exposure_time = camera_config.get("exposure_time", 100)
        self._gain = camera_config.get("gain", 200)

        try:
            self._fps_cap = max(0, float(camera_config.get("fps_cap", 0) or 0))
        except (TypeError, ValueError):
            self._fps_cap = 0.0

        if self.is_image:
            self.image = cv2.imread(self.source)
            if self.image is None:
                self.logger.warning(
                    "Could not read image '%s' - using synthetic placeholder frame.",
                    self.source,
                )
                self.image = self._placeholder_frame
        else:
            try:
                self._open_camera()
                self._connected = True
            except Exception as exc:
                # Don't fall back to a static image forever - stay in camera
                # mode and keep searching for the device in the reader thread.
                self.logger.warning(
                    "Camera source '%s' could not be opened (%s) - showing "
                    "placeholder and searching for the device in the background.",
                    self.source,
                    exc,
                )
                self.cap = None

            self._reader_thread: threading.Thread | None = None
            self._reader_thread = threading.Thread(
                target=self._reader,
                daemon=True,
                name=f"CamReader-{self.source}",
            )
            self._reader_thread.start()

            try:
                atexit.register(self.destroy)
                print(f"[DEBUG] Camera {self.source}: atexit registered", flush=True)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Connects a frame source (a raw camera feed). Overridden per source
    # type; everything else below is source-agnostic.
    # ------------------------------------------------------------------

    def _open_capture(self, extra_backends: bool = False, prefer_dshow: bool = False):
        if self._is_url_source:
            try:
                cap = self._open_capture_bounded(None)
            except Exception as exc:
                raise ValueError(f"Camera failed to open stream: {self.source} ({exc})")
            if not self._wait_for_first_frame(cap):
                try:
                    cap.release()
                except Exception:
                    pass
                raise ValueError(f"Camera stream did not start: {self.source}")
            return cap
        backend_candidates = self._get_capture_backend_candidates(platform.system())
        if platform.system() == "Windows":
            if extra_backends and cv2.CAP_DSHOW not in backend_candidates:
                backend_candidates = backend_candidates + [cv2.CAP_DSHOW]
            if prefer_dshow and cv2.CAP_DSHOW in backend_candidates:
                backend_candidates = [cv2.CAP_DSHOW] + [
                    b for b in backend_candidates if b != cv2.CAP_DSHOW
                ]
        last_error = None
        for backend in backend_candidates:
            try:
                cap = self._open_capture_bounded(backend)
            except Exception as exc:
                last_error = exc
                continue
            try:
                self._configure_capture(cap)
                if not self._wait_for_first_frame(cap):
                    self.logger.info(
                        "Camera %s: no frame from backend %s within %.1fs - "
                        "trying the next backend.",
                        self.source, backend, self._STREAM_START_TIMEOUT,
                    )
                    raise ValueError(
                        f"Camera stream did not start with backend {backend}: {self.source}"
                    )
                return cap
            except Exception as exc:
                last_error = exc
                try:
                    cap.release()
                except Exception:
                    pass
        raise ValueError(f"Camera failed to open: {self.source} ({last_error})")

    def _open_capture_bounded(self, backend):
        global _open_worker_live
        holder = {"cap": None, "error": None, "abandoned": False}

        with _open_worker_guard:
            if _open_worker_live >= _OPEN_WORKER_MAX:
                raise CameraOpenTimeout(
                    f"Camera {self.source}: too many capture opens are hung "
                    "(driver wedged?) - deferring this retry until one exits."
                )
            _open_worker_live += 1

        def _worker():
            global _open_worker_live
            try:
                try:
                    cap = (
                        cv2.VideoCapture(self.source)
                        if backend is None
                        else cv2.VideoCapture(self.source, backend)
                    )
                except Exception as exc:
                    holder["error"] = exc
                    return
                if holder["abandoned"]:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    return
                holder["cap"] = cap
            finally:
                with _open_worker_guard:
                    _open_worker_live -= 1

        opener = threading.Thread(
            target=_worker,
            daemon=True,
            name=f"CapOpen-{self.source}-{backend}",
        )
        if self.stopped:
            raise CameraOpenTimeout(
                f"Camera {self.source}: stopping - aborting open with backend {backend}"
            )
        try:
            opener.start()
        except Exception:
            with _open_worker_guard:
                _open_worker_live -= 1
            raise
        opener.join(timeout=self._CAP_OPEN_TIMEOUT)
        if opener.is_alive():
            holder["abandoned"] = True
            self.logger.warning(
                "Camera %s: opening with backend %s exceeded %.0fs - "
                "abandoning it and trying the next backend.",
                self.source, backend, self._CAP_OPEN_TIMEOUT,
            )
            raise CameraOpenTimeout(
                f"Camera open with backend {backend} timed out after "
                f"{self._CAP_OPEN_TIMEOUT:.0f}s - the device may be held by "
                "another app or its driver may be wedged."
            )
        if holder["error"] is not None:
            raise holder["error"]
        cap = holder["cap"]
        if cap is None or not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            raise ValueError(
                f"Camera failed to open with backend {backend}: {self.source}"
            )
        return cap

    def _wait_for_first_frame(self, cap, timeout: float = _STREAM_START_TIMEOUT) -> bool:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            try:
                if cap.grab():
                    return True
            except Exception:
                pass
            time.sleep(0.05)
        return False

    def _configure_capture(self, cap):
        if self._is_url_source:
            return
        is_windows = platform.system() == "Windows"
        is_linux = platform.system() == "Linux"

        if is_windows:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cap_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cap_h)
            cap.set(cv2.CAP_PROP_FPS, 30)
        elif is_linux:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cap_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cap_h)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cap_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cap_h)
            cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cap_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cap_h)
            cap.set(cv2.CAP_PROP_FPS, 30)

        self._apply_capture_controls(cap)

    def _open_camera(self):
        sys_platform = platform.system()
        is_linux = sys_platform == "Linux"

        if is_linux and not self._is_url_source:
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
                     "--set-ctrl=auto_exposure=0",
                     f"--set-ctrl=exposure_time_absolute={self._exposure_time}",
                     f"--set-ctrl=gain={self._gain}"],
                    capture_output=True, text=True, timeout=5,
                )
            except FileNotFoundError:
                pass
            except subprocess.TimeoutExpired:
                self.logger.warning("v4l2-ctl timed out setting UVC controls on %s", device)

            time.sleep(0.15)

        try:
            self.cap = self._open_capture()
        except ValueError:
            self.cap = self._open_capture(extra_backends=True)

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

    def _attempt_reconnect(self) -> bool:
        """Try to open the capture device; keep searching if it's absent."""
        if self.stopped:
            return False
        try:
            self._open_camera()
        except Exception as exc:
            self._reconnect_failures += 1
            # A stray non-iSpy process may be holding the v4l2 devnode open,
            # which OpenCV reports as a reopen failure (e.g. "Inappropriate
            # ioctl for device" / "Camera ... still waiting"). Force it out so
            # the next attempt has a real chance - never touch iSpy's own pid.
            if (
                platform.system() == "Linux"
                and not self._is_url_source
                and not self.is_image
            ):
                try:
                    free_camera_device(self.source)
                except Exception:
                    self.logger.debug("device-guard failed; continuing", exc_info=True)
            if (
                self._reconnect_failures == 1
                or self._reconnect_failures % self._RECONNECT_WARN_EVERY == 0
            ):
                self.logger.warning(
                    "Camera %s: still waiting for device (%s) - retrying every %.0fs.",
                    self.source, exc, self._RECONNECT_RETRY_DELAY,
                )
            return False
        self._connected = True
        self._reconnect_failures = 0
        self._prefer_alternate_backend = False
        self.logger.info("Camera %s: device found - stream started.", self.source)
        return True

    def _reopen_capture(self):
        self._connected = False
        try:
            cap = getattr(self, "cap", None)
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            self.cap = self._open_capture(
                extra_backends=True,
                prefer_dshow=self._prefer_alternate_backend,
            )
            self._connected = True
            self.logger.info("Camera %s: capture re-opened.", self.source)
            return True
        except Exception as exc:
            self.cap = None
            self.logger.warning(
                "Camera %s: capture re-open failed (%s); will retry.", self.source, exc
            )
            return False

    def _sleep_stopping(self, seconds: float, poll: float) -> bool:
        """Sleep, but abort early when `stopped` so destroy() join is reliable.

        Returns False once stopping has begun (caller should return from the
        reader loop). Makes a leaked reader stop promptly instead of polling a
        bogus source (e.g. /dev/video99) through interpreter shutdown.
        """
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.stopped:
                return False
            time.sleep(min(poll, deadline - time.monotonic()))
        return not self.stopped

    def _reader(self):
        print(f"[DEBUG] Camera {self.source}: reader thread started", flush=True)
        frame_interval = 1.0 / self._fps_cap if self._fps_cap > 0 else 0.0
        next_frame_time = 0.0
        consecutive_failures = 0
        while not self.stopped:
            if self._reopen_requested:
                self._reopen_requested = False
                self._reopen_capture()
                time.sleep(0.1)
                continue
            if frame_interval > 0:
                now = time.perf_counter()
                if now < next_frame_time:
                    time.sleep(min(next_frame_time - now, 0.05))
                    continue

            if self.cap is None:
                if not self._attempt_reconnect():
                    if not self._sleep_stopping(
                        self._RECONNECT_RETRY_DELAY, self._RECONNECT_POLL_S
                    ):
                        return
                continue

            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures % self._READ_WARN_EVERY == 1:
                    self.logger.warning(
                        "Frame read failed on %s (%d consecutive) - camera busy "
                        "or stream unavailable; re-opening the capture if this persists.",
                        self.source, consecutive_failures,
                    )
                if consecutive_failures >= self._READ_REOPEN_AFTER:
                    consecutive_failures = 0
                    self._prefer_alternate_backend = not self._prefer_alternate_backend
                    self._reopen_capture()
                time.sleep(self._READ_RETRY_DELAY)  # Don't starve CPU
                continue

            if consecutive_failures > 0:
                self.logger.info("Camera %s: frame reads recovered.", self.source)
            consecutive_failures = 0

            if frame.max() < 1:
                self.logger.debug("Solid-black frame skipped.")
                time.sleep(0.05)  # Don't starve CPU
                continue

            frame = self._apply_image_adjustments(frame)

            with self.frame_lock:
                self.frame = frame
                self.frame_timestamp = time.perf_counter()
            self._frame_event.set()
            if frame_interval > 0:
                next_frame_time = time.perf_counter() + frame_interval

    # ------------------------------------------------------------------
    # Frame access (the source's contract)
    # ------------------------------------------------------------------

    def get_raw_frame(self) -> np.ndarray | None:
        if self.is_image:
            return self.image.copy() if self.image is not None else None
        if not self._connected:
            return self._placeholder_frame.copy()
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None

    def get_frame(self) -> np.ndarray | None:
        if not self.is_image and not self._connected:
            # device missing - serve the placeholder untouched
            return self._placeholder_frame.copy()
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

    def get_frame_age(self) -> float:
        if self.is_image:
            return 0.0
        if not self._connected:
            return self._DISCONNECTED_AGE_S
        with self.frame_lock:
            ts = self.frame_timestamp
        return 0.0 if ts is None else time.perf_counter() - ts

    def add_frame_processor(self, processor):
        if self.is_image:
            self.logger.warning("Cannot add frame processor to image source.")
            return
        self._frame_processors.append(processor)

    # ------------------------------------------------------------------
    # Source lifecycle
    # ------------------------------------------------------------------

    def is_ready(self) -> tuple[bool, str]:
        if self.is_image:
            return True, "ready"
        return self._connected, ("ready" if self._connected else "waiting for device")

    def start(self):
        # The reader thread is started in __init__/on reconnect - nothing to
        # do here. Kept so every source satisfies the CameraBase contract.
        pass

    def destroy(self):
        if getattr(self, "_destroyed", False):
            print(f"[DEBUG] Camera {self.source}: destroy() called again (atexit?) - skipping", flush=True)
            return
        self._destroyed = True
        print(f"[DEBUG] Camera {self.source}: destroy() called - joining reader thread", flush=True)
        self.stopped = True
        reader = getattr(self, "_reader_thread", None)
        if reader is not None and reader is not threading.current_thread():
            if not self.is_image and hasattr(self, "cap") and self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            reader.join(timeout=5.0)
            print(f"[DEBUG] Camera {self.source}: reader thread joined (alive={reader.is_alive()})", flush=True)
        if not self.is_image and hasattr(self, "cap") and self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print(f"[DEBUG] Camera {self.source}: destroy() complete", flush=True)

    def release(self):
        self.destroy()

    # ------------------------------------------------------------------
    # Image tuning (web sliders)
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_num(value, lo: float, hi: float, default: float) -> float:
        try:
            return min(hi, max(lo, float(value)))
        except (TypeError, ValueError):
            return default

    def _apply_capture_controls(self, cap=None):
        cap = cap if cap is not None else getattr(self, "cap", None)
        if cap is None or self._is_url_source or self.is_image:
            return
        if platform.system() == "Linux":
            try:
                device = self.source if isinstance(self.source, str) else f"/dev/video{self.source}"
                subprocess.run(
                    ["v4l2-ctl", "-d", device,
                     "--set-ctrl=auto_exposure=0",
                     f"--set-ctrl=exposure_time_absolute={int(self._exposure_time)}",
                     f"--set-ctrl=gain={int(self._gain)}"],
                    capture_output=True, text=True, timeout=5,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            return
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            ok_exp = cap.set(cv2.CAP_PROP_EXPOSURE, self._exposure_time)
            ok_gain = cap.set(cv2.CAP_PROP_GAIN, self._gain)
        except Exception:
            ok_exp = ok_gain = False
        if not ok_exp or not ok_gain:
            self._reopen_requested = True

    def _apply_saturation(self, frame, saturation: float) -> np.ndarray:
        factor = 1.0 + saturation / 100.0
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        except cv2.error:
            return frame
        hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _apply_color_balance(self, frame: np.ndarray, white_balance: float, tint: float) -> np.ndarray:
        if frame.ndim < 3 or (not white_balance and not tint):
            return frame
        wb = self._clamp_num(white_balance, -100, 100, 0) / 100.0
        tt = self._clamp_num(tint, -100, 100, 0) / 100.0
        blue_scale = (1.0 + wb * 0.5) * (1.0 + tt * 0.2)
        green_scale = 1.0 - tt * 0.4
        red_scale = (1.0 - wb * 0.5) * (1.0 + tt * 0.2)
        scales = np.clip(np.array([blue_scale, green_scale, red_scale]), 0.25, 4.0)
        if np.all(np.abs(scales - 1.0) < 1e-6):
            return frame
        luts = [
            np.clip(np.arange(256, dtype=np.float32) * s, 0, 255).astype(np.uint8)
            for s in scales
        ]
        return cv2.merge(
            [cv2.LUT(frame[:, :, i], luts[i]) for i in range(3)]
        )

    def _apply_image_adjustments(self, frame: np.ndarray) -> np.ndarray:
        contrast = self._contrast
        brightness = self._brightness
        if contrast or brightness:
            alpha = 1.0 + contrast / 100.0
            beta = brightness * 2.55
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        if self._white_balance or self._tint:
            frame = self._apply_color_balance(frame, self._white_balance, self._tint)
        if self._saturation:
            frame = self._apply_saturation(frame, self._saturation)
        if self._gamma != 1.0:
            gamma = np.clip(self._gamma, 0.1, 5.0)
            lut = (
                np.clip((np.arange(256, dtype=np.float32) / 255.0) ** gamma, 0.0, 1.0)
                * 255.0
            ).astype(np.uint8)
            frame = cv2.LUT(frame, lut)
        return frame

    def set_image_adjustments(self, adjustments: dict) -> dict:
        if "brightness" in adjustments:
            self._brightness = self._clamp_num(adjustments["brightness"], -100, 100, 0)
        if "contrast" in adjustments:
            self._contrast = self._clamp_num(adjustments["contrast"], -100, 100, 0)
        if "saturation" in adjustments:
            self._saturation = self._clamp_num(adjustments["saturation"], -100, 100, 0)
        if "white_balance" in adjustments:
            self._white_balance = self._clamp_num(adjustments["white_balance"], -100, 100, 0)
        if "tint" in adjustments:
            self._tint = self._clamp_num(adjustments["tint"], -100, 100, 0)
        if "gamma" in adjustments:
            self._gamma = self._clamp_num(adjustments["gamma"], 0.3, 3.0, 1.0)
        if "exposure_time" in adjustments:
            self._exposure_time = int(
                self._clamp_num(adjustments["exposure_time"], 0, 1_000_000, 100)
            )
        if "gain" in adjustments:
            self._gain = int(self._clamp_num(adjustments["gain"], 0, 4096, 200))
        self._apply_capture_controls()
        return self.get_image_adjustments()

    def get_image_adjustments(self) -> dict:
        return {
            "brightness": self._brightness,
            "contrast": self._contrast,
            "saturation": self._saturation,
            "white_balance": self._white_balance,
            "tint": self._tint,
            "gamma": self._gamma,
            "exposure_time": self._exposure_time,
            "gain": self._gain,
        }

    # ------------------------------------------------------------------
    # Calibration mode (shared with the web wizard)
    # ------------------------------------------------------------------

    def set_calibration(self, active: bool):
        self.calibration_active = bool(active)
        self.calibration_last_seen = time.monotonic() if active else 0.0

    def calibration_heartbeat(self):
        self.calibration_last_seen = time.monotonic()

    def in_calibration_mode(self) -> bool:
        if not self.calibration_active:
            return False
        return (
            time.monotonic() - self.calibration_last_seen
        ) < self._CALIBRATION_HEARTBEAT_TIMEOUT

    # ------------------------------------------------------------------
    # Placeholder frames
    # ------------------------------------------------------------------

    def _make_placeholder_frame(
        self, width: int = _PLACEHOLDER_W,
        height: int = _PLACEHOLDER_H,
    ) -> np.ndarray:
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

    # ------------------------------------------------------------------
    # Generic plugin-style hooks (bail-outs for the demo fallbacks)
    # ------------------------------------------------------------------

    def get_debug_frame(self, frame):
        return None

    def get_debug_data(self) -> dict:
        return {}

    def plot(self, frame):
        return frame

    def get_demo_objects(self, frame):
        if frame is None:
            return []
        h, w = frame.shape[:2]
        plugin_name = getattr(self, "plugin_name", "demo_object")
        return [
            Object(
                x=0.0,
                y=0.0,
                z=0.0,
                name=f"demo_{plugin_name}",
                confidence=0.5,
                vis_type="planar",
                vis_meta={"size": max(0.2, min(w, h) / 200.0)},
            )
        ]