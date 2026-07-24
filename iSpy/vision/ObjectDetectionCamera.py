from pathlib import Path
from iSpy.vision.Object import Object
import cv2
import math
import numpy as np
import time
import logging
import threading
import queue
from iSpy.vision.Camera import Camera
from iSpy.plugins.bases import VisionBase
from iSpy.vision.genericYolo import Box, Results, GenericYolo
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig

class ObjectDetectionCamera(Camera, VisionBase):
    plugin_name = "object_detection"

    def __init__(
        self,
        camera_config: iSpyCameraConfig,
        config: iSpyConfig,
        core_mask=None,
    ):
        self.logger = logging.getLogger(__name__)

        self.config = camera_config

        try:
            self.known_calibration_distance = camera_config["calibration"]["distance"]
            self.ball_d_inches = camera_config["calibration"]["game_piece_size"]
            self.known_calibration_pixel_height = camera_config["calibration"]["size"]
            self.fov = camera_config["calibration"]["fov"]
            self.grayscale = camera_config.get("grayscale", False)
            self.subsystem = camera_config["subsystem"]

            self.camera_bot_relative_yaw = camera_config["yaw"]
            self.camera_pitch_angle = camera_config["pitch"]
            self.camera_height = camera_config["height"]
            self.camera_x = camera_config["x"]
            self.camera_y = camera_config["y"]
            self.camera_z = camera_config["z"]
        except KeyError as e:
            raise ValueError(f"Missing camera config key: {e}")

        # Model architecture fields come exclusively from metadata sidecars,
        # not from config.  Config holds only user-preference fields.
        from iSpy.vision.ModelInspector import fill_missing_config
        vm_filled = fill_missing_config(dict(config["vision_model"]))
        self.margin = vm_filled.get("margin", config["vision_model"].get("margin", 0))
        self.min_confidence = float(vm_filled.get("min_conf", config["vision_model"].get("min_conf", 0.5)))
        self.yolo_model_file = vm_filled["file_path"]
        self.input_size = tuple(vm_filled["input_size"])
        self.quantized = vm_filled.get("quantized", config["vision_model"].get("quantized", False))
        self.frame_sync = config.get("frame_sync", False)
        if self.frame_sync:
            self.logger.warning("Frame sync is enabled. This may introduce latency in detection (you probaly don't want this).")
        self.core_mask = core_mask
        self.unit = config["unit"]
        self.debug_mode = config["debug_mode"]
        self.gui_available = False

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

        try:
            if self.known_calibration_pixel_height <= 0 or self.known_calibration_distance <= 0:
                self.logger.info("Calibration values must be positive, defaulting focal length to 1")
                self.focal_length_pixels = 1.0
            else:
                self.focal_length_pixels = (
                    self.known_calibration_pixel_height * self.known_calibration_distance
                ) / self.ball_d_inches
        except ZeroDivisionError:
            self.logger.warning(
                "Calibration game_piece_size is 0, defaulting focal length to 1"
            )
            self.focal_length_pixels = 1.0

        super().__init__(camera_config, self.input_size, self.grayscale)

        self.model = GenericYolo(
            vm_filled,
            self.core_mask,
            iSpy_config=config,
        )

        self._class_names: dict[int, str] = {0: "object"}
        try:
            from iSpy.vision.metadata import read_metadata
            meta = read_metadata(Path(self.yolo_model_file))
            if meta and isinstance(meta.get("names"), dict):
                self._class_names = {int(k): str(v) for k, v in meta["names"].items()}
        except Exception:
            pass

        self._preproc_q: queue.Queue = queue.Queue(maxsize=1)
        self._use_pipeline = self.model.model_type in ("rknn", "onnx", "tflite")

        self._last_result: Results | None = None
        self._last_frame: np.ndarray | None = None
        self.last_time = time.perf_counter()
        self._pipeline_timeout = 0.1
        self._last_objects: list[Object] = []
        
        if self._use_pipeline:
            threading.Thread(
                target=self._preprocess_worker,
                daemon=True,
                name=f"PreProc-{self.source}",
            ).start()

    def _letterbox(self, img: np.ndarray, target_size: tuple) -> tuple:
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        left = pad_w // 2
        padded = cv2.copyMakeBorder(
            resized,
            top,
            pad_h - top,
            left,
            pad_w - left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return padded, scale, left, top

    def _letterbox_into(
        self, img: np.ndarray, dst: np.ndarray, target_size: tuple
    ) -> None:
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        dst[:] = 114
        dst[top : top + new_h, left : left + new_w] = resized

    def _preprocess_worker(self):
        last_ts = None
        while not self.stopped:
            if self.is_image:
                frame = self.get_frame()
                ts = 0
                
                time.sleep(1 / 100) # Simulate a 100 fps camera
            else:
                with self.frame_lock:
                    frame = self.frame
                    ts = self.frame_timestamp
                if frame is None or ts == last_ts:
                    self._frame_event.wait(timeout=0.05)
                    self._frame_event.clear()
                    continue

            if not self.frame_sync and not self._preproc_q.empty():
                time.sleep(0.001)
                continue

            last_ts = ts
            orig_shape = frame.shape
            preprocessed = self.model._preprocess_frame(frame)
            self._preproc_q.put((preprocessed, frame, orig_shape))

    def _filter_box(self, box: Box, img_w: int, img_h: int) -> bool:
        x1, y1, x2, y2 = box.xyxy
        w_px = x2 - x1
        h_px = y2 - y1
        if (
            x1 < self.margin
            or y1 < self.margin
            or x2 > (img_w - self.margin)
            or y2 > (img_h - self.margin)
        ):
            return False
        if h_px == 0:
            return False
        aspect = w_px / h_px # Aspect is calculate but I won't use it because
        # I want it to continue detections partial objectcs/rectangles
        return True
        # return 0.8 <= aspect <= 1.2

    def _box_to_robot_point(
        self, box: Box, img_w: int, img_h: int
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = box.xyxy
        avg_px = ((x2 - x1) + (y2 - y1)) / 2.0
        if avg_px <= 0:
            return None
        cx = (x1 + x2) / 2.0
        distance_los = (self.ball_d_inches * self.focal_length_pixels) / avg_px
        return self._pixel_to_robot_coordinates(
            cx, (y1 + y2) / 2.0, distance_los, img_w, img_h
        )

    def _pixel_to_robot_coordinates(
        self,
        pixel_x: float,
        pixel_y: float,
        distance_los: float,
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        pixel_offset_x = pixel_x - img_w / 2.0
        horizontal_angle_rad = math.atan(pixel_offset_x / self.focal_length_pixels)

        if self.camera_height > 0 and distance_los > self.camera_height:
            true_horiz = math.sqrt(distance_los**2 - self.camera_height**2)
        else:
            true_horiz = distance_los * math.cos(math.radians(self.camera_pitch_angle))

        left_right = true_horiz * math.sin(horizontal_angle_rad)
        forward = true_horiz * math.cos(horizontal_angle_rad)

        yaw_rad = math.radians(self.camera_bot_relative_yaw)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        x_rot = forward * cos_y + left_right * sin_y
        y_rot = forward * sin_y - left_right * cos_y

        scale = self.conversions.get(self.unit, self.conversions["meter"])
        return np.array(
            [(x_rot + self.camera_x) * scale, (y_rot + self.camera_y) * scale],
            dtype=np.float32,
        )

    def get_yolo_data(self) -> tuple[Results | None, np.ndarray | None]:
        if self._use_pipeline:
            try:
                preprocessed, orig_frame, orig_shape = self._preproc_q.get(
                    timeout=None if self.frame_sync else self._pipeline_timeout
                )
            except queue.Empty:
                return self._last_result, self._last_frame

            results = self.model.predict_preprocessed(preprocessed, orig_shape)
            annotated_frame = orig_frame.copy()
            self._last_result = results
            self._last_frame = annotated_frame
        else:
            frame = self.get_frame()
            if frame is None:
                self.logger.warning("No frame available.")
                return None, None
            clean_frame = frame.copy()  # keep clean copy before prediction
            results = self.model.predict(frame, orig_shape=frame.shape)
            annotated_frame = clean_frame  # use untouched frame
            self._last_result = results
            self._last_frame = annotated_frame

        if annotated_frame is not None:
            annotated_frame = results.plot(annotated_frame.copy())
            if self.debug_mode:
                new_time = time.perf_counter()
                fps = 1 / max(new_time - self.last_time, 1e-6)
                self.last_time = new_time
                cv2.putText(
                    annotated_frame,
                    f"FPS: {int(fps)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                if self.gui_available:
                    cv2.imshow("YOLO Detections", annotated_frame)
                    cv2.waitKey(1)
            self._last_frame = annotated_frame

        return results, annotated_frame

    def _pnp_to_robot_coordinates(
        self, tvec: tuple[float, float, float]
    ) -> np.ndarray:
        fx, fy, fz = tvec
        yaw_rad = math.radians(self.camera_bot_relative_yaw)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        x_rot = fz * cos_y + (-fx) * sin_y
        y_rot = fz * sin_y - (-fx) * cos_y
        scale = self.conversions.get(self.unit, self.conversions["meter"])
        return np.array(
            [
                (x_rot + self.camera_x) * scale,
                (y_rot + self.camera_y) * scale,
                (-fy + self.camera_z) * scale,
            ],
            dtype=np.float32,
        )

    def _box_to_object(self, box: Box, img_w: int, img_h: int) -> Object|None:
        if box.translation is not None:
            pt = self._pnp_to_robot_coordinates(box.translation)
        else:
            pt = self._box_to_robot_point(box, img_w, img_h)
        if pt is None:
            return None
        roll, pitch, yaw = 0.0, 0.0, 0.0
        if box.rotation is not None:
            roll, pitch, yaw = box.rotation
        z = float(pt[2]) if len(pt) > 2 else 0.0
        class_name = self._class_names.get(box.cls_id, f"class_{box.cls_id}")
        return Object(float(pt[0]), float(pt[1]), z=z, roll=roll, pitch=pitch, yaw=yaw, name=class_name, confidence=box.conf)

    def run(self):
        data, frame = self.get_yolo_data()
        if data is None or frame is None:
            return [], None

        img_h, img_w = frame.shape[:2]
        objects: list[Object] = []
        for box in data.boxes:
            if not self._filter_box(box, img_w, img_h):
                continue
            obj = self._box_to_object(box, img_w, img_h)
            if obj is not None:
                objects.append(obj)
        self._last_objects = objects
        return objects, frame
 
    def run_with_supplied_data(self, data: Results) -> list[Object]:
        img_h, img_w = data.orig_shape[:2]
        objects: list[Object] = []
        for box in data.boxes:
            if not self._filter_box(box, img_w, img_h):
                continue
            obj = self._box_to_object(box, img_w, img_h)
            if obj is not None:
                objects.append(obj)
        return objects
 

    def get_data_for_subsystem(self, target: str):
        if self.subsystem != target:
            return None
        positions = self._last_objects
        if self.subsystem == "hopper":
            return len(positions) > 0
        return positions

    def get_subsystem(self) -> str:
        return self.subsystem

    def destroy(self):
        self.stopped = True
        if not self.is_image and hasattr(self, "cap") and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def release(self):
        self.destroy()