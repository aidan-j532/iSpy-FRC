import cv2
import math
import numpy as np
import logging
import time

from iSpy.vision.Camera import Camera
from iSpy.plugins.bases import VisionBase
from iSpy.vision.Object import Object
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision import triangulation

class AprilTagCamera(Camera, VisionBase):
    plugin_name = "april_tag"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "tag_size_inches": {
                "type": "number",
                "label": "Tag Size (in)",
                "default": 6.5,
                "step": 0.1,
            }
        }

    def __init__(self, camera_config: iSpyCameraConfig, config: iSpyConfig, core_mask=None):
        self.logger = logging.getLogger(__name__)
        self.config = camera_config

        # 1. Load Camera & Calibration Settings
        try:
            self.subsystem = camera_config["subsystem"]
            self.camera_bot_relative_yaw = camera_config.get("yaw", 0.0)
            self.camera_pitch_angle = camera_config.get("pitch", 0.0)
            self.camera_height = camera_config.get("height", 0.0)
            self.camera_x = camera_config.get("x", 0.0)
            self.camera_y = camera_config.get("y", 0.0)
            self.camera_z = camera_config.get("z", 0.0)
            
            calib = camera_config.get("calibration", {})
            self.known_calibration_distance = calib.get("distance", 1.0)
            self.known_calibration_pixel_height = calib.get("size", 100)
            tag_size_inches = camera_config.get("tag_size_inches")
            if tag_size_inches is None:
                tag_size_inches = calib.get("game_piece_size", 6.5)
            self.tag_size_inches = float(tag_size_inches)
            self.ball_d_inches = self.tag_size_inches
            self.fov = calib.get("fov", 0.0)
            self.grayscale = camera_config.get("grayscale", False)
        except KeyError as e:
            raise ValueError(f"Missing camera config key for AprilTag: {e}")

        self.unit = config.get("unit", "meter")
        self.conversions = {
            "meter": 0.0254, "meters": 0.0254,
            "inch": 1.0, "inches": 1.0,
            "foot": 1 / 12, "feet": 1 / 12,
            "centimeter": 2.54, "centimeters": 2.54,
        }

        # Calculate focal length
        try:
            if self.known_calibration_pixel_height <= 0 or self.known_calibration_distance <= 0:
                self.focal_length_pixels = 1.0
            else:
                self.focal_length_pixels = (
                    self.known_calibration_pixel_height * self.known_calibration_distance
                ) / self.ball_d_inches
        except ZeroDivisionError:
            self.focal_length_pixels = 1.0

        # Note: We request 640x480 by default, though you can adjust this based on your camera
        super().__init__(camera_config, (640, 480), self.grayscale)

        # 2. Setup OpenCV Aruco Detector (FRC uses tag36h11)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self._last_objects: list[Object] = []
        
        # FRC AprilTags are exactly 16.5cm (~6.5 inches)
        # We work in inches internally before converting to output unit
        half = self.tag_size_inches / 2.0
        self.obj_pts = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]
        ], dtype=np.float32)

    def _focal_length_px_fov(self, img_w: int) -> float:
        if self.fov and self.fov > 0:
            return (img_w / 2.0) / math.tan(math.radians(self.fov / 2.0))
        return self.focal_length_pixels

    def _camera_point_to_robot(self, pt: tuple[float, float, float]) -> np.ndarray:
        fx, fy, fz = pt
        yaw_rad = math.radians(self.camera_bot_relative_yaw)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        
        x_rot = fz * cos_y + (-fx) * sin_y
        y_rot = fz * sin_y - (-fx) * cos_y
        scale = self.conversions.get(self.unit, self.conversions["meter"])
        
        return np.array([
            (x_rot + self.camera_x) * scale,
            (y_rot + self.camera_y) * scale,
            (-fy + self.camera_z) * scale,
        ], dtype=np.float32)

    def _rvec_to_euler(self, rvec: np.ndarray) -> tuple[float, float, float]:
        R, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0.0
        return roll, pitch, yaw

    def run(self):
        """Main pipeline step: get frame, detect tags, build Object list."""
        frame = self.get_frame()
        if frame is None:
            return [], None

        # Convert to grayscale if it isn't already
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Run detection
        corners, ids, rejected = self.detector.detectMarkers(gray)
        objects = []

        if ids is not None:
            img_h, img_w = frame.shape[:2]
            f = self._focal_length_px_fov(img_w)
            cx, cy = img_w / 2.0, img_h / 2.0
            cam_mat = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros(5, dtype=np.float64)

            for i in range(len(ids)):
                tag_id = int(ids[i][0])
                tag_corners = corners[i][0]

                # Draw bounding box and ID on the frame for debugging
                cv2.polylines(frame, [tag_corners.astype(np.int32)], True, (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {tag_id}", tuple(tag_corners[0].astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Solve PnP for Tag pose
                ok, rvec, tvec = cv2.solvePnP(
                    self.obj_pts, 
                    tag_corners, 
                    cam_mat, 
                    dist_coeffs, 
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if ok:
                    # Draw 3D axis on tag
                    cv2.drawFrameAxes(frame, cam_mat, dist_coeffs, rvec, tvec, 3.25)
                    
                    tvec = tvec.reshape(3)
                    robot_pt = self._camera_point_to_robot((tvec[0], tvec[1], tvec[2]))
                    roll, pitch, yaw = self._rvec_to_euler(rvec.reshape(3))

                    scale = self.conversions.get(self.unit, self.conversions["meter"])

                    # Map properties to iSpy Object
                    obj = Object(
                        x=float(robot_pt[0]),
                        y=float(robot_pt[1]),
                        z=float(robot_pt[2]),
                        name=f"tag_{tag_id}",
                        confidence=1.0, # AprilTags are highly confident if detected
                        roll=roll,
                        pitch=pitch,
                        yaw=yaw,
                        depth_source="pnp",
                        vis_type="planar",
                        vis_meta={
                            "tag_id": tag_id,
                            "size": self.tag_size_inches * scale,
                        },
                    )
                    
                    # Optional: Add ray origins for triangulation across multiple cameras
                    center_px = np.mean(tag_corners, axis=0)
                    ray = triangulation.pixel_to_ray(
                        center_px[0], center_px[1], img_w, img_h, f,
                        self.camera_x, self.camera_y, self.camera_z,
                        self.camera_bot_relative_yaw, self.camera_pitch_angle
                    )
                    obj.ray_origin = ray.origin
                    obj.ray_direction = ray.direction

                    objects.append(obj)

        self._last_objects = objects
        return objects, frame

    def get_data_for_subsystem(self, target: str):
        """Required method to satisfy the NetworkHandler routing subsystem data."""
        if self.subsystem != target:
            return None
        positions = self._last_objects
        if self.subsystem == "hopper":
            return len(positions) > 0
        return positions

    def plot(self, frame):
        if frame is None:
            return None
        try:
            import cv2
            overlay = frame.copy()
            cv2.putText(overlay, "AprilTag", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return overlay
        except Exception:
            return frame

    def destroy(self):
        super().destroy()