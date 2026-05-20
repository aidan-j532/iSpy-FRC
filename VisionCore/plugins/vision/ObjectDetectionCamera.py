import cv2
import math
import numpy as np
import time
import logging
import threading
import queue
from VisionCore.vision.Camera import Camera
from VisionCore.plugins.bases import VisionBase
from VisionCore.vision.genericYolo import Box, Results, GenericYolo
from VisionCore.config.VisionCoreConfig import VisionCoreConfig, VisionCoreCameraConfig


class ObjectDetectionCamera(Camera, VisionBase):
    plugin_name = "object_detection"

    def __init__(
        self,
        camera_config: VisionCoreCameraConfig,
        config: VisionCoreConfig,
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
            self.fps_cap = camera_config.get("fps_cap", 30)
            self.subsystem = camera_config["subsystem"]

            self.camera_bot_relative_yaw = camera_config["yaw"]
            self.camera_pitch_angle = camera_config["pitch"]
            self.camera_height = camera_config["height"]
            self.camera_x = camera_config["x"]
            self.camera_y = camera_config["y"]
        except KeyError as e:
            raise ValueError(f"Missing camera config key: {e}")

        self.margin = config["vision_model"].get("margin", 0)
        self.min_confidence = config["vision_model"].get("min_conf", 0.5)
        self.yolo_model_file = config["vision_model"]["file_path"]
        self.input_size = tuple(config["vision_model"]["input_size"])
        self.quantized = config["vision_model"].get("quantized", False)
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
                logger.info("Calibration values must be positive, defaulting focal length to 1")
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

        super().__init__(camera_config, self.fps_cap, self.input_size, self.grayscale)

        model_config = dict(config["vision_model"])
        if "file_path" not in model_config:
            model_config["file_path"] = self.yolo_model_file
        if "input_size" not in model_config:
            model_config["input_size"] = self.input_size
        if "min_conf" not in model_config:
            model_config["min_conf"] = self.min_confidence
        if "quantized" not in model_config:
            model_config["quantized"] = self.quantized

        self.model = GenericYolo(
            model_config,
            self.core_mask,
            visioncore_config=config,
        )

        self._preproc_q: queue.Queue = queue.Queue(maxsize=1)
        self._use_pipeline = (not self.is_image) and (self.model.model_type == "rknn")

        self._last_result: Results | None = None
        self._last_frame: np.ndarray | None = None
        self.last_time = time.perf_counter()
        self.frame_timeout = 1.0 / max(self.fps_cap, 1)

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
        h, w = self.input_size[1], self.input_size[0]
        bufs = [
            np.empty((1, h, w, 3), dtype=np.uint8),
            np.empty((1, h, w, 3), dtype=np.uint8),
        ]
        buf_idx = 0

        while not self.stopped:
            with self.frame_lock:
                frame = self.frame
                ts = self.frame_timestamp

            if frame is None or ts == last_ts:
                self._frame_event.wait(timeout=0.05)
                self._frame_event.clear()
                continue

            if not self._preproc_q.empty():
                time.sleep(0.005)
                continue

            last_ts = ts
            orig_shape = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.grayscale:
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            self._letterbox_into(img_rgb, bufs[buf_idx][0], self.input_size)
            self._preproc_q.put((bufs[buf_idx], frame, orig_shape))
            buf_idx = 1 - buf_idx

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
        aspect = w_px / h_px
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
                    timeout=self.frame_timeout
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

        if self.debug_mode and annotated_frame is not None:
            annotated_frame = results.plot(annotated_frame.copy())
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
            self._last_frame = annotated_frame
            if self.gui_available:
                cv2.imshow("YOLO Detections", annotated_frame)
                cv2.waitKey(1)

        return results, annotated_frame

    def run(self):
        data, frame = self.get_yolo_data()
        if data is None or frame is None:
            return np.empty((0, 2)), None

        img_h, img_w = frame.shape[:2]
        map_points = []
        passed_boxes = []  # was missing

        for box in data.boxes:
            if not self._filter_box(box, img_w, img_h):
                continue
            pt = self._box_to_robot_point(box, img_w, img_h)
            if pt is not None:
                map_points.append(pt)
                passed_boxes.append(box)

        return (np.array(map_points) if map_points else np.empty((0, 2))), frame

    def run_with_supplied_data(self, data: Results) -> np.ndarray:
        img_h, img_w = data.orig_shape[:2]
        map_points = []
        for box in data.boxes:
            if not self._filter_box(box, img_w, img_h):
                continue
            pt = self._box_to_robot_point(box, img_w, img_h)
            if pt is not None:
                map_points.append(pt)
        return np.array(map_points) if map_points else np.empty((0, 2))

    def get_data_for_subsystem(self, target: str):
        if self.subsystem != target:
            return None
        positions, _ = self.run()
        if self.subsystem == "hopper":
            return positions.shape[0] > 0
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
