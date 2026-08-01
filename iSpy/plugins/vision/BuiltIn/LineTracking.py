import logging

from iSpy.vision.Camera import Camera
from iSpy.plugins.bases import VisionBase
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig


class LineTrackingCamera(Camera, VisionBase):
    plugin_name = "line_tracking"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "line_color": {
                "type": "text",
                "label": "Line Color",
                "default": "white",
            },
            "threshold": {
                "type": "number",
                "label": "Threshold",
                "default": 0.5,
                "step": 0.1,
            },
        }

    def __init__(self, camera_config: iSpyCameraConfig, config: iSpyConfig, core_mask=None):
        self.logger = logging.getLogger(__name__)
        self.config = camera_config
        super().__init__(camera_config, (640, 480), camera_config.get("grayscale", False))

    def run(self):
        return [], None

    def destroy(self):
        pass
