import logging
import cv2
import numpy as np

from iSpy.vision.Camera import Camera
from iSpy.plugins.bases import VisionBase
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision.Object import Object


class QRCodeCamera(Camera, VisionBase):
    plugin_name = "qr_code"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "qr_size": {
                "type": "number",
                "label": "QR Size",
                "default": 0.1,
                "step": 0.01,
            },
            "decode_mode": {
                "type": "select",
                "label": "Decode Mode",
                "options": ["standard", "fast"],
                "default": "standard",
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
        center = (w // 2, h // 2)
        cv2.rectangle(frame, (center[0] - 24, center[1] - 24), (center[0] + 24, center[1] + 24), (255, 0, 0), 2)
        cv2.putText(frame, "QR", (center[0] - 12, center[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        obj = Object(
            x=0.0,
            y=0.0,
            z=0.0,
            name="demo_qr_code",
            confidence=0.85,
            vis_type="planar",
            vis_meta={"size": max(0.2, min(w, h) / 200.0), "kind": "qr"},
        )
        return [obj], frame

    def plot(self, frame):
        if frame is None:
            return None
        try:
            import cv2
            overlay = frame.copy()
            cv2.putText(overlay, "QRCode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            return overlay
        except Exception:
            return frame

    def destroy(self):
        pass