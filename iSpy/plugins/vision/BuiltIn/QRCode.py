import logging
import cv2
import numpy as np
import math

from iSpy.vision.Camera import Camera
from iSpy.plugins.bases import VisionBase
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision.Object import Object
from iSpy.vision import triangulation

class QRCodeCamera(Camera, VisionBase):
    plugin_name = "qr_code"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "qr_size": {
                "type": "number",
                "label": "QR Size (in configured units)",
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

        # 1. Load Camera & Calibration Settings
        try:
            self.subsystem = camera_config.get("subsystem", "field")
            self.camera_bot_relative_yaw = camera_config.get("yaw", 0.0)
            self.camera_pitch_angle = camera_config.get("pitch", 0.0)
            self.camera_height = camera_config.get("height", 0.0)
            self.camera_x = camera_config.get("x", 0.0)
            self.camera_y = camera_config.get("y", 0.0)
            self.camera_z = camera_config.get("z", 0.0)
            
            calib = camera_config.get("calibration", {})
            self.fov = calib.get("fov", 0.0)
            self.grayscale = camera_config.get("grayscale", False)
            self.qr_size = float(camera_config.get("qr_size", 0.1))
        except KeyError as e:
            raise ValueError(f"Missing camera config key for QRCode: {e}")

        self.unit = config.get("unit", "meter")
        self.conversions = {
            "meter": 0.0254, "meters": 0.0254,
            "inch": 1.0, "inches": 1.0,
            "foot": 1 / 12, "feet": 1 / 12,
            "centimeter": 2.54, "centimeters": 2.54,
        }

        super().__init__(camera_config, (640, 480), self.grayscale)

        # 2. Setup OpenCV QR Code Detector
        self.detector = cv2.QRCodeDetector()
        self._last_objects: list[Object] = []
        
        # Define the 3D corners of the QR code (assuming centered at origin)
        half = self.qr_size / 2.0
        self.obj_pts = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]
        ], dtype=np.float32)

    def _focal_length_px_fov(self, img_w: int) -> float:
        if self.fov and self.fov > 0:
            return (img_w / 2.0) / math.tan(math.radians(self.fov / 2.0))
        return getattr(self, "focal_length_pixels", 1.0)

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
        frame = self.get_frame()
        if frame is None:
            return [], None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Detect and decode
        retval, decoded_info, points, _ = self.detector.detectAndDecodeMulti(gray)
        objects = []

        if retval and points is not None:
            img_h, img_w = frame.shape[:2]
            f = self._focal_length_px_fov(img_w)
            cx, cy = img_w / 2.0, img_h / 2.0
            cam_mat = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros(5, dtype=np.float64)

            for i in range(len(points)):
                qr_corners = points[i]
                info = decoded_info[i] if i < len(decoded_info) else ""
                
                cv2.polylines(frame, [qr_corners.astype(np.int32)], True, (255, 0, 0), 2)
                if info:
                    cv2.putText(frame, info, tuple(qr_corners[0].astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Solve PnP
                ok, rvec, tvec = cv2.solvePnP(
                    self.obj_pts, 
                    qr_corners, 
                    cam_mat, 
                    dist_coeffs, 
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if ok:
                    cv2.drawFrameAxes(frame, cam_mat, dist_coeffs, rvec, tvec, self.qr_size / 2)
                    
                    tvec = tvec.reshape(3)
                    robot_pt = self._camera_point_to_robot((tvec[0], tvec[1], tvec[2]))
                    roll, pitch, yaw = self._rvec_to_euler(rvec.reshape(3))

                    scale = self.conversions.get(self.unit, self.conversions["meter"])

                    obj = Object(
                        x=float(robot_pt[0]),
                        y=float(robot_pt[1]),
                        z=float(robot_pt[2]),
                        name="qr_code",
                        confidence=1.0, 
                        roll=roll,
                        pitch=pitch,
                        yaw=yaw,
                        depth_source="pnp",
                        vis_type="planar",
                        vis_meta={
                            "payload": info,
                            "size": self.qr_size * scale,
                            "kind": "qr"
                        },
                    )
                    objects.append(obj)

        self._last_objects = objects
        return objects, frame

    def get_data_for_subsystem(self, target: str):
        if getattr(self, "subsystem", "field") != target:
            return None
        return self._last_objects

    def plot(self, frame):
        return frame # Overlays are drawn directly in the run() function

    def destroy(self):
        super().destroy()