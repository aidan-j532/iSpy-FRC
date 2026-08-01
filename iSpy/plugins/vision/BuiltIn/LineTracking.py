import logging
import cv2
import numpy as np

from iSpy.vision.Camera import Camera
from iSpy.plugins.bases import VisionBase
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision.Object import Object


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
        frame = self.get_frame()
        if frame is None:
            return [], None

        h, w = frame.shape[:2]
        cv2.line(frame, (10, h // 2), (w - 10, h // 2), (255, 255, 0), 3)
        cv2.putText(frame, "Line", (w // 2 - 20, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        obj = Object(
            x=0.0,
            y=0.0,
            z=0.0,
            name="demo_line_tracking",
            confidence=0.75,
            vis_type="generic",
            vis_meta={"kind": "line"},
        )
        return [obj], frame

    def plot(self, frame):
        if frame is None:
            return None
        try:
            import cv2
            overlay = frame.copy()
            cv2.putText(overlay, "Line Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            return overlay
        except Exception:
            return frame

    def destroy(self):
        pass
