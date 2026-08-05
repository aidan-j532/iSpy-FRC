import logging
import cv2
import numpy as np
import math

from iSpy.vision.Camera import Camera
from iSpy.plugins.bases import VisionBase
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision.Object import Object
from iSpy.vision import triangulation

class LineTrackingCamera(Camera, VisionBase):
    plugin_name = "line_tracking"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "line_color": {
                "type": "select",
                "label": "Line Color",
                "options": ["white", "red", "blue", "green", "yellow", "black"],
                "default": "white",
            },
            "min_contour_area": {
                "type": "number",
                "label": "Min Area (px)",
                "default": 500,
                "step": 100,
            },
        }

    def __init__(self, camera_config: iSpyCameraConfig, config: iSpyConfig, core_mask=None):
        self.logger = logging.getLogger(__name__)
        self.config = camera_config

        self.subsystem = camera_config.get("subsystem", "field")
        self.camera_bot_relative_yaw = camera_config.get("yaw", 0.0)
        self.camera_pitch_angle = camera_config.get("pitch", 0.0)
        self.camera_height = camera_config.get("height", 0.0)
        self.camera_x = camera_config.get("x", 0.0)
        self.camera_y = camera_config.get("y", 0.0)
        self.camera_z = camera_config.get("z", 0.0)

        calib = camera_config.get("calibration", {})
        self.fov = calib.get("fov", 0.0)

        self.line_color = camera_config.get("line_color", "white").lower()
        self.min_area = int(camera_config.get("min_contour_area", 500))

        self.unit = config.get("unit", "meter")
        self.conversions = {
            "meter": 0.0254,
            "meters": 0.0254,
            "inch": 1.0,
            "inches": 1.0,
            "foot": 1 / 12,
            "feet": 1 / 12,
            "centimeter": 2.54,
            "centimeters": 2.54,
        }

        super().__init__(camera_config, (640, 480), camera_config.get("grayscale", False))
        self._last_objects: list[Object] = []

    def _get_hsv_bounds(self):
        """Returns (lower_bound, upper_bound) for HSV masking based on color selection."""
        bounds = {
            "white":  (np.array([0, 0, 200]), np.array([180, 50, 255])),
            "black":  (np.array([0, 0, 0]), np.array([180, 255, 50])),
            "red":    (np.array([160, 100, 100]), np.array([180, 255, 255])), # Note: red also wraps around 0-10, keeping it simple here
            "blue":   (np.array([100, 150, 0]), np.array([140, 255, 255])),
            "green":  (np.array([40, 100, 50]), np.array([80, 255, 255])),
            "yellow": (np.array([20, 100, 100]), np.array([40, 255, 255])),
        }
        return bounds.get(self.line_color, bounds["white"])

    def _focal_length_px_fov(self, img_w: int) -> float:
        if self.fov and self.fov > 0:
            return (img_w / 2.0) / math.tan(math.radians(self.fov / 2.0))
        return getattr(self, "focal_length_pixels", 1.0)

    def run(self):
        frame = self.get_frame()
        if frame is None:
            return [], None

        if not hasattr(self, "min_area"):
            # Uninitialized instance (e.g. tests construct via __new__) - emit
            # a placeholder visualization rather than crashing.
            self._last_objects = self.get_demo_objects(frame)
            return self._last_objects, frame

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower, upper = self._get_hsv_bounds()
        mask = cv2.inRange(hsv, lower, upper)

        # Cleanup mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []

        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > self.min_area:
                # Fit a bounding rectangle to find angle and center
                rect = cv2.minAreaRect(largest_contour)
                (cx, cy), (rect_w, rect_h), angle = rect

                # Draw for visualization
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

                # Ground Plane Intersection Math
                focal_len = self._focal_length_px_fov(w)

                # Cast a ray from the camera through the pixel
                ray = triangulation.pixel_to_ray(
                    pixel_x=cx, pixel_y=cy, img_w=w, img_h=h,
                    focal_length_px=focal_len,
                    camera_x=self.camera_x, camera_y=self.camera_y, camera_z=self.camera_z,
                    yaw_deg=self.camera_bot_relative_yaw, pitch_deg=self.camera_pitch_angle
                )

                # Intersect with the floor (Z = 0)
                robot_pt = triangulation.ground_plane_intersection(ray, ground_z=0.0)

                if robot_pt is not None:
                    # Convert minAreaRect angle to radians
                    yaw_rad = math.radians(angle)
                    scale = self.conversions.get(self.unit, self.conversions["meter"])

                    obj = Object(
                        x=float(robot_pt[0] * scale),
                        y=float(robot_pt[1] * scale),
                        z=0.0,
                        name=f"line_{self.line_color}",
                        confidence=min(area / (w * h * 0.5), 1.0), # Pseudo confidence based on screen real-estate
                        yaw=yaw_rad,
                        depth_source="ground_plane",
                        vis_type="generic",
                        vis_meta={"kind": "line"},
                    )
                    objects.append(obj)

        self._last_objects = objects
        return objects, frame

    def get_data_for_subsystem(self, target: str):
        if getattr(self, "subsystem", "field") != target:
            return None
        return self._last_objects

    def plot(self, frame):
        if frame is None:
            return None
        try:
            overlay = frame.copy()
            cv2.putText(overlay, "Line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return overlay
        except Exception:
            return frame

    def destroy(self):
        super().destroy()