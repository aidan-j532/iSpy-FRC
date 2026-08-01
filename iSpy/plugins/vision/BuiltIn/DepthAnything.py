import logging
import cv2
import numpy as np

from iSpy.vision.Camera import Camera
from iSpy.plugins.bases import VisionBase
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision.Object import Object


class DepthAnythingCamera(Camera, VisionBase):
    plugin_name = "depth_anything"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "model_size": {
                "type": "select",
                "label": "Model Size",
                "options": ["small", "base", "large"],
                "default": "base",
            },
            "estimate_depth": {
                "type": "boolean",
                "label": "Estimate Depth",
                "default": True,
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.55, heatmap, 0.45, 0)

        center_x, center_y = w // 2, h // 2
        radius = max(12, min(w, h) // 6)
        cv2.circle(blended, (center_x, center_y), radius, (255, 255, 255), 2)
        cv2.putText(blended, "Depth", (center_x - 24, center_y + radius + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        depth_estimate = 1.0 - (normalized[center_y, center_x] / 255.0)
        obj = Object(
            x=0.0,
            y=0.0,
            z=float(depth_estimate),
            name="demo_depth",
            confidence=0.8,
            vis_type="generic",
            vis_meta={
                "kind": "depth",
                "heatmap": True,
                "depth_estimate": round(depth_estimate, 3),
                "radius": radius / max(w, h),
            },
        )
        return [obj], blended

    def plot(self, frame):
        if frame is None:
            return None
        try:
            overlay = frame.copy()
            h, w = overlay.shape[:2]
            cv2.putText(overlay, "Depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.rectangle(overlay, (8, 38), (w - 8, h - 8), (255, 255, 255), 1)
            return overlay
        except Exception:
            return frame

    def destroy(self):
        pass
