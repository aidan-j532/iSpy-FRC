"""A DJI Tello / Tello Edu camera source.

Only differs from :class:`OpenCVCamera` in how the capture device is opened:
it first negotiates with the drone over its UDP command socket (``command`` +
``streamon``) so that the video stream is actually running, then hands the
``udp://`` source to OpenCV through the FFmpeg backend. Everything else
(reader thread, placeholder frame, automatic reconnection while the drone is
absent, image adjustments) is inherited.
"""

import logging
import socket
import time
from copy import deepcopy

import cv2

from iSpy.config.iSpyConfig import iSpyCameraConfig
from iSpy.vision.Cameras.OpenCVCamera import OpenCVCamera

# Defaults for a stock DJI Tello / Tello Edu with its onboard access point.
_TELLO_DEFAULT_IP = "192.168.10.1"
_TELLO_DEFAULT_COMMAND_PORT = 8889
_TELLO_DEFAULT_VIDEO_PORT = 11111

# Simple AT-command handshake. The drone only opens the UDP video stream
# after it has been told to (command -> streamon), so we must do that before
# OpenCV opens the udp:// source.
_COMMAND_TIMEOUT_S = 3.0
_HANDSHAKE_RETRIES = 3
_HANDSHAKE_RETRY_DELAY_S = 1.0

# The first _open_camera runs synchronously from __init__, so it uses a
# single fast handshake attempt - a full retry loop would stall startup for
# ~20s when the drone isn't there yet. Full retries happen on the reader
# thread's automatic reconnect instead.
_QUICK_HANDSHAKE_RETRIES = 1
_QUICK_COMMAND_TIMEOUT_S = 1.5


class TelloCamera(OpenCVCamera):
    """Camera source for a DJI Tello / Tello Edu.

    The video source is a URL (default ``udp://0.0.0.0:<video_port>``). The
    drone's command socket is configurable through the camera config::

        "camera_type": "tello",
        "tello_ip": "192.168.10.1",
        "tello_command_port": 8889,
        "tello_video_port": 11111
    """

    camera_type = "tello"
    plugin_name = "tello_edu"

    def _default_stream_url(self) -> str:
        return f"udp://0.0.0.0:{self._video_port}"

    @classmethod
    def config_schema(cls) -> dict:
        """Config keys that configure the Tello connection + its tuning."""
        return {
            "source": {
                "type": "text",
                "label": "Source",
                "default": f"udp://0.0.0.0:{_TELLO_DEFAULT_VIDEO_PORT}",
                "help": "Video stream URL. Auto-filled with the drone's UDP "
                        "feed; leave it unless you know what you are doing.",
            },
            "tello_ip": {
                "type": "text",
                "label": "Tello IP",
                "default": _TELLO_DEFAULT_IP,
                "help": "IP of the Tello's on-board access point.",
            },
            "tello_command_port": {
                "type": "number",
                "label": "Command Port",
                "default": _TELLO_DEFAULT_COMMAND_PORT,
                "help": "UDP port the Tello listens for AT commands on.",
            },
            "tello_video_port": {
                "type": "number",
                "label": "Video Port",
                "default": _TELLO_DEFAULT_VIDEO_PORT,
                "help": "Local UDP port the Tello's H.264 video stream arrives on.",
            },
            "fps_cap": {
                "type": "number",
                "label": "FPS Cap",
                "default": 0,
                "help": "Maximum processed frames per second (0 = uncapped).",
            },
            "brightness": {
                "type": "number",
                "label": "Brightness",
                "default": 0,
                "help": "Post-processing brightness offset (-100..100).",
            },
            "contrast": {
                "type": "number",
                "label": "Contrast",
                "default": 0,
                "help": "Post-processing contrast boost (-100..100).",
            },
            "saturation": {
                "type": "number",
                "label": "Saturation",
                "default": 0,
                "help": "Post-processing saturation boost (-100..100).",
            },
            "gamma": {
                "type": "number",
                "label": "Gamma",
                "default": 1.0,
                "help": "Post-processing gamma correction (0.3..3.0).",
            },
        }

    def __init__(self, camera_config, input_size=None, grayscale=False, **kwargs):
        self._tello_ip = str(
            camera_config.get("tello_ip", _TELLO_DEFAULT_IP) or _TELLO_DEFAULT_IP
        )
        try:
            self._command_port = int(
                str(camera_config.get("tello_command_port", _TELLO_DEFAULT_COMMAND_PORT))
            )
            self._video_port = int(
                str(camera_config.get("tello_video_port", _TELLO_DEFAULT_VIDEO_PORT))
            )
        except (TypeError, ValueError):
            self._command_port = _TELLO_DEFAULT_COMMAND_PORT
            self._video_port = _TELLO_DEFAULT_VIDEO_PORT

        # Keep a log-line handle available even if the camera config doesn't
        # carry it, so the handshake messages label the right camera.
        self._tello_source_label = str(camera_config.get("name", "tello"))

        self.logger = logging.getLogger(__name__)
        self._first_open_attempt = True

        if not isinstance(camera_config.get("source"), str) or not self._is_tello_url(
            camera_config.get("source")
        ):
            # Convenience: if the user left source as a device index or a bare
            # "tello_source", point it at the default UDP stream so the
            # subclass just works out of the box.
            merged = deepcopy(
                camera_config.data if hasattr(camera_config, "data") else dict(camera_config)
            )
            merged["source"] = self._default_stream_url()
            camera_config = iSpyCameraConfig(merged)

        super().__init__(camera_config, input_size, grayscale, **kwargs)

    @staticmethod
    def _is_tello_url(source) -> bool:
        return isinstance(source, str) and source.startswith("udp://")

    # OpenCV needs the FFmpeg backend to decode the Tello's UDP H.264 stream -
    # the default backends (MSMF / V4L2) can't open a udp:// source, and the
    # base _open_capture bypasses backends entirely for URL sources. Force the
    # FFmpeg backend explicitly for the udp stream.
    def _open_capture(self, extra_backends: bool = False, prefer_dshow: bool = False):
        try:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        except Exception as exc:
            raise ValueError(f"Tello failed to open stream: {self.source} ({exc})")
        if not self._wait_for_first_frame(cap):
            try:
                cap.release()
            except Exception:
                pass
            raise ValueError(f"Tello stream did not start: {self.source}")
        return cap

    def _send_command(self, command: str, timeout: float = _COMMAND_TIMEOUT_S) -> bool:
        """Send one AT command to the drone's UDP control socket and wait for
        its 'OK' acknowledgement. Returns True on success."""
        try:
            with socket.create_connection(
                (self._tello_ip, self._command_port), timeout=timeout
            ) as sock:
                sock.settimeout(timeout)
                sock.sendall((command + "\r\n").encode("ascii"))
                buf = b""
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    try:
                        chunk = sock.recv(1024)
                    except socket.timeout:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    if b"ok" in buf.lower():
                        return True
                self.logger.warning(
                    "Tello '%s': no 'OK' for command '%s' (got %r).",
                    self._tello_source_label, command, buf[:64],
                )
                return False
        except OSError as exc:
            self.logger.debug(
                "Tello '%s': command socket to %s:%s failed: %s",
                self._tello_source_label, self._tello_ip, self._command_port, exc,
            )
            return False

    def _ensure_stream(self, quick: bool = False) -> bool:
        """Turn the drone's video stream on so the UDP feed is live.

        Retries the handshake a few times - the drone only answers once it is
        ready, and a freshly-connected access point can be slow to respond.
        """
        retries = _QUICK_HANDSHAKE_RETRIES if quick else _HANDSHAKE_RETRIES
        timeout = _QUICK_COMMAND_TIMEOUT_S if quick else _COMMAND_TIMEOUT_S
        for attempt in range(retries):
            # "command" switches the drone from SDK-off to command mode; only
            # then does it accept "streamon".
            if self._send_command("command", timeout=timeout) and self._send_command(
                "streamon", timeout=timeout
            ):
                return True
            if quick:
                break
            if attempt + 1 < retries:
                time.sleep(_HANDSHAKE_RETRY_DELAY_S)
        self.logger.warning(
            "Tello '%s': could not start video stream after %d handshake attempts.",
            self._tello_source_label, retries,
        )
        return False

    def _open_camera(self):
        quick = self._first_open_attempt
        # Negotiate with the drone first. If we can't reach it yet, raise so the
        # base lifecycle keeps the placeholder and the reader thread keeps
        # retrying (via _attempt_reconnect) until the drone is found.
        if not self._ensure_stream(quick=quick):
            raise ValueError(
                f"Tello {self._tello_source_label}: unable to start video stream "
                f"at {self._tello_ip}:{self._command_port}."
            )

        # Video needs a moment to actually start flowing before OpenCV grabs.
        time.sleep(0.5)

        self.cap = self._open_capture()

        if not self.cap.isOpened():
            raise ValueError(f"Tello stream lost after configuration: {self.source}")

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.logger.info(
            "Tello '%s': stream %dx%d @ %.1f FPS via %s",
            self._tello_source_label, actual_w, actual_h, actual_fps, self.source,
        )
        self._frame_processors = []
        self._first_open_attempt = False

    def _attempt_reconnect(self) -> bool:
        # Re-running _open_camera re-sends command/streamon, which is what we
        # want when the drone (or its video stream) drops out.
        return super()._attempt_reconnect()