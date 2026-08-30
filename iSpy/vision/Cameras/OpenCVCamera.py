"""Any device OpenCV can read frames from - USB webcams, RTSP/HTTP streams,
static image files, camera indices.

This is the classic iSpy camera. ``source`` may be an integer device index, a
``/dev/video*`` path, an ``rtsp://``/``http://`` URL or an image file path.
"""

from iSpy.vision.Cameras import _discovery
from iSpy.vision.Cameras.base import CameraBase


class OpenCVCamera(CameraBase):
    """An OpenCV camera source (USB device, index, stream URL or image)."""

    camera_type = "opencv"
    plugin_name = "opencv"

    @classmethod
    def config_schema(cls) -> dict:
        """Config keys that configure *this* camera source + its tuning."""
        source_help = (
            "Device index (0), a /dev/video* path, an rtsp:// or http:// "
            "stream URL, or a static image file path."
        )
        return {
            "source": {
                "type": "source",
                "label": "Source",
                "default": 0,
                "help": source_help,
            },
            "device_id": {
                "type": "text",
                "label": "Device ID",
                "default": "",
                "help": "Hardware identifier reported by discovery "
                        "(auto-filled when you pick a discovered camera).",
            },
            "fps_cap": {
                "type": "number",
                "label": "FPS Cap",
                "default": 0,
                "help": "Maximum processed frames per second (0 = uncapped).",
            },
            "exposure_time": {
                "type": "number",
                "label": "Exposure Time",
                "default": 100,
                "help": "Camera exposure, arbitrary units (UVC/v4l2).",
            },
            "gain": {
                "type": "number",
                "label": "Gain",
                "default": 200,
                "help": "Camera gain, arbitrary units (UVC/v4l2).",
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
            "white_balance": {
                "type": "number",
                "label": "White Balance",
                "default": 0,
                "help": "Post-processing white balance (-100..100).",
            },
            "tint": {
                "type": "number",
                "label": "Tint",
                "default": 0,
                "help": "Post-processing tint (-100..100).",
            },
        }

    @classmethod
    def discover(cls, claimed_sources: set | None = None) -> list[dict]:
        """Enumerate the OpenCV sources currently connected."""
        return _discovery.probe_opencv_devices(claimed_sources)