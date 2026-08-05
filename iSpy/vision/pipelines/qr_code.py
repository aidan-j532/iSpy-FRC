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
        configured = getattr(self, "focal_length_pixels", 0.0)
        if configured and configured > 1:
            return configured
        # No calibration: assume a typical 60 deg horizontal FOV so the PnP
        # solve produces sensible distances.
        return (img_w / 2.0) / math.tan(math.radians(60.0 / 2.0))

    def _camera_point_to_robot(self, pt: tuple[float, float, float]) -> np.ndarray:
        scale = self.conversions.get(self.unit, self.conversions["meter"])
        return triangulation.camera_point_to_robot(
            pt,
            self.camera_x,
            self.camera_y,
            self.camera_z,
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

    def _decode_scales(self, gray):
        """Try decoding at several scales (upsample helps small/rotated QR
        codes). Returns (points, decoded_info, scale) scaled back to the
        original frame coordinates, or (None, [], 1.0)."""
        img_h, img_w = gray.shape[:2]
        scales = [1.0, 1.5, 2.0, 0.75]
        for scale in scales:
            if scale != 1.0:
                resized = cv2.resize(gray, (int(img_w * scale), int(img_h * scale)),
                                     interpolation=cv2.INTER_LINEAR)
            else:
                resized = gray
            try:
                retval, decoded_info, points, _ = self.detector.detectAndDecodeMulti(resized)
            except cv2.error:
                retval, decoded_info, points = False, [], None
            if retval and points is not None and len(points):
                if scale != 1.0:
                    points = [pts / scale for pts in points]
                return points, list(decoded_info), scale
            # Single-QR fallback - detectAndDecodeMulti can miss lone codes.
            try:
                info, corners, _straight = self.detector.detectAndDecode(resized)
            except cv2.error:
                info, corners = "", None
            if corners is not None and len(corners):
                if scale != 1.0:
                    corners = corners / scale
                return [corners], [info], scale
        return None, [], 1.0

    def run(self):
        frame = self.get_frame()
        if frame is None:
            return [], None

        detector = getattr(self, "detector", None)
        if detector is None:
            # Uninitialized instance (e.g. tests construct via __new__) - emit
            # a placeholder visualization rather than crashing.
            self._last_objects = self.get_demo_objects(frame)
            return self._last_objects, frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        objects = []
        points, decoded_info, _ = self._decode_scales(gray)
        if points is not None:
            objects = self._build_objects(frame, points, decoded_info)

        # Temporal persistence: keep the last decoded objects for a few
        # frames so a single dropped/missed frame doesn't make the code
        # flicker out of existence.
        if objects:
            self._stable = objects
            self._stability = getattr(self, "stability_frames", 3)
        elif getattr(self, "_stability", 0) > 0:
            self._stability -= 1
            objects = self._stable

        self._last_objects = objects
        return objects, frame

    def _build_objects(self, frame, points, decoded_info):
        objects = []
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
                # solvePnP output is in the units of `obj_pts` (configured
                # unit), but `_camera_point_to_robot` works in the codebase's
                # internal inch convention, so convert configured unit ->
                # inches first (mirrors ObjectDetectionCamera).
                to_inches = 1.0 / self.conversions.get(self.unit, self.conversions["meter"])
                robot_pt = self._camera_point_to_robot(
                    (tvec[0] * to_inches, tvec[1] * to_inches, tvec[2] * to_inches)
                )
                R_tag, _ = cv2.Rodrigues(rvec.reshape(3))
                R_robot = triangulation.camera_rotation_to_robot(
                    R_tag, self.camera_bot_relative_yaw, self.camera_pitch_angle
                )
                roll, pitch, yaw = self._matrix_to_euler(R_robot)

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
                        "size": self.qr_size,
                        "kind": "qr"
                    },
                )
                objects.append(obj)
        return objects

    def get_data_for_subsystem(self, target: str):
        if getattr(self, "subsystem", "field") != target:
            return None
        return self._last_objects

    def plot(self, frame):
        if frame is None:
            return None
        try:
            overlay = frame.copy()
            h, w = overlay.shape[:2]
            cv2.putText(overlay, "QR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.rectangle(overlay, (8, 38), (w - 8, h - 8), (255, 0, 0), 1)
            return overlay
        except Exception:
            return frame

    def destroy(self):
        super().destroy()