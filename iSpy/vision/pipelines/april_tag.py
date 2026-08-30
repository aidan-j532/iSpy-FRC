import cv2
import math
import numpy as np
import logging
import time

from iSpy.vision.pipelines.base import VisionPipeline
from iSpy.vision.Object import Object
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig, unit_to_inches
from iSpy.vision import triangulation
from iSpy.vision import calibration as cam_calibration

class AprilTagPipeline(VisionPipeline):
    plugin_name = "april_tag"
    # AprilTag pose is derived from the camera matrix, so the ChArUco board
    # intrinsics calibration is all that's needed.
    calibration_sections = ["charuco"]

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

        try:
            self.subsystem = camera_config["subsystem"]
            self.camera_bot_relative_yaw = camera_config.get("yaw", 0.0)
            self.camera_pitch_angle = camera_config.get("pitch", 0.0)

            _pos_unit = config.get("unit", "frc")
            # 'height' is the single mount-height field feeding triangulation
            self.camera_height = unit_to_inches(camera_config.get("height", 0.0), _pos_unit)
            self.camera_x = unit_to_inches(camera_config.get("x", 0.0), _pos_unit)
            self.camera_y = unit_to_inches(camera_config.get("y", 0.0), _pos_unit)
            _legacy_z = camera_config.get("z")
            if _legacy_z not in (None, 0, 0.0):
                self.logger.warning(
                    "Camera config key 'z' (%s) is deprecated and ignored - "
                    "'height' is the single mount-height field feeding "
                    "triangulation.",
                    _legacy_z,
                )
            
            calib = camera_config.get("calibration", {})
            self.known_calibration_distance = calib.get("distance", 1.0)
            self.known_calibration_pixel_height = calib.get("size", 100)
            tag_size_inches = camera_config.get_pipeline_setting("tag_size_inches", 6.5)
            self.tag_size_inches = float(tag_size_inches)
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
            "frc": 0.0254,
        }

        try:
            if self.known_calibration_pixel_height <= 0 or self.known_calibration_distance <= 0:
                self.focal_length_pixels = 1.0
            else:
                self.focal_length_pixels = (
                    self.known_calibration_pixel_height * self.known_calibration_distance
                ) / self.ball_d_inches
        except ZeroDivisionError:
            self.focal_length_pixels = 1.0

        super().__init__(camera_config, (640, 480), self.grayscale)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self._last_objects: list[Object] = []
        
        half = self.tag_size_inches / 2.0
        self.obj_pts = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]
        ], dtype=np.float32)

        self._set_status("ready")

    def _focal_length_px_fov(self, img_w: int) -> float:
        if self.fov and self.fov > 0:
            return (img_w / 2.0) / math.tan(math.radians(self.fov / 2.0))
        return self.focal_length_pixels

    def _camera_point_to_robot(self, pt: tuple[float, float, float]) -> np.ndarray:
        scale = self.conversions.get(self.unit, self.conversions["meter"])
        return triangulation.camera_point_to_robot(
            pt,
            self.camera_x,
            self.camera_y,
            self.camera_height,
            self.camera_bot_relative_yaw,
            self.camera_pitch_angle,
        ) * scale

    def _matrix_to_euler(self, R: np.ndarray) -> tuple[float, float, float]:
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
        frame = self.get_frame()
        if frame is None:
            return [], None
        gated = self._gate_uncalibrated(frame)
        if gated is not None:
            return gated

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        corners, ids, rejected = self.detector.detectMarkers(gray)
        objects = []

        if ids is not None:
            img_h, img_w = frame.shape[:2]
            f = self._focal_length_px_fov(img_w)
            cx, cy = img_w / 2.0, img_h / 2.0
            cam_mat = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros(5, dtype=np.float64)

            intr = cam_calibration.intrinsics_for_frame(
                self.config.get("calibration", {}), img_w, img_h
            )
            if intr is not None:
                cam_mat, dist_coeffs = intr

            for i in range(len(ids)):
                tag_id = int(ids[i][0])
                tag_corners = corners[i][0]

                cv2.polylines(frame, [tag_corners.astype(np.int32)], True, (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {tag_id}", tuple(tag_corners[0].astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                ok, rvec, tvec = cv2.solvePnP(
                    self.obj_pts, 
                    tag_corners, 
                    cam_mat, 
                    dist_coeffs, 
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if ok:
                    cv2.drawFrameAxes(frame, cam_mat, dist_coeffs, rvec, tvec, 3.25)
                    
                    tvec = tvec.reshape(3)
                    robot_pt = self._camera_point_to_robot((tvec[0], tvec[1], tvec[2]))
                    R_tag, _ = cv2.Rodrigues(rvec.reshape(3))
                    R_robot = triangulation.camera_rotation_to_robot(
                        R_tag, self.camera_bot_relative_yaw, self.camera_pitch_angle
                    )
                    roll, pitch, yaw = self._matrix_to_euler(R_robot)

                    scale = self.conversions.get(self.unit, self.conversions["meter"])

                    obj = Object(
                        x=float(robot_pt[0]),
                        y=float(robot_pt[1]),
                        z=float(robot_pt[2]),
                        name=f"tag_{tag_id}",
                        confidence=1.0, # tags are basically always right when detected
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
                    
                    center_px = np.mean(tag_corners, axis=0)
                    ray = triangulation.pixel_to_ray(
                        center_px[0], center_px[1], img_w, img_h, f,
                        self.camera_x, self.camera_y, self.camera_height,
                        self.camera_bot_relative_yaw, self.camera_pitch_angle
                    )
                    obj.ray_origin = ray.origin
                    obj.ray_direction = ray.direction

                    objects.append(obj)

        self._last_objects = objects
        return objects, frame

    def get_data_for_subsystem(self, target: str):
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
            overlay = frame.copy()
            cv2.putText(
                overlay, "AprilTag", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            return overlay
        except Exception:
            return frame

    def destroy(self):
        super().destroy()


# Backward-compatible alias: iSpy pre-restructure called pipelines '*Camera'.
AprilTagCamera = AprilTagPipeline