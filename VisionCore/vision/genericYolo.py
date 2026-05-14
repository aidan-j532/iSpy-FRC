import logging
import cv2
import numpy as np
from ultralytics import YOLO
from VisionCore.config.AutoOpt import recommend_format
import os
from pathlib import Path
from ultralytics import YOLO

try:
    from rknnlite.api import RKNNLite
    RKNN_FOUND = True
except ImportError:
    RKNN_FOUND = False

class Box:
    def __init__(self, xyxy, conf):
        self.xyxy = xyxy
        self.conf = conf

class Results:
    def __init__(self, boxes: list[Box], orig_shape):
        self.boxes = boxes
        self.orig_shape = orig_shape

    def plot(self, frame):
        for box in self.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def __str__(self):
        s = f"Results(orig_shape={self.orig_shape}, num_boxes={len(self.boxes)})\n"
        for i, box in enumerate(self.boxes):
            s += f"  Box {i}: xyxy={box.xyxy}, conf={box.conf:.3f}\n"
        return s

class GenericYolo:
    def __init__(self, model_file: str, core_mask, input_size=(640, 640), quantized: bool = False, min_conf: float = 0.5):
        self.model_file = model_file
        self.input_size = input_size
        self.core_mask = core_mask
        self.model_type = None
        self.logger = logging.getLogger(__name__)

        self._output_fmt = None
        self._needs_sigmoid = None

        self.quantized = quantized
        self.min_conf = min_conf

        if model_file.endswith(".pt"):
            self.logger.info(".pt model provided at runtime; skipping automatic conversion. Ensure boot-time conversion was performed if needed.")
            # keep model_file as-is; ultralytics backend can still load .pt directly
            self.model_file = model_file

        if model_file.endswith(".rknn"):
            if not RKNN_FOUND:
                self.logger.error("rknnlite not found but .rknn model was specified.")
                raise ImportError("rknnlite not installed.")

            self.model_type = "rknn"
            self.model = RKNNLite()

            ret = self.model.load_rknn(self.model_file)
            if ret != 0:
                raise ValueError(f"Failed to load RKNN model: {self.model_file}")

            ret = self.model.init_runtime(core_mask=core_mask)
            if ret != 0:
                raise ValueError(f"Failed to init RKNN runtime: {self.model_file}")

            h, w = self.input_size[1], self.input_size[0]
            self._input_buf = np.empty((1, h, w, 3), dtype=np.uint8)

        elif (model_file.endswith(".onnx") or model_file.endswith(".pt")
              or "openvino_model" in model_file or model_file.endswith(".mlpackage")):
            self.model_type = "yolo"
            self.model = YOLO(self.model_file, verbose=False, task="detect")

        elif model_file.endswith(".tflite"):
            self.model_type = "tflite"
            self._load_tflite(model_file)

        else:
            raise ValueError(f"Unsupported model file type: {self.model_file}")

        self.logger.info(f"YoloWrapper loaded: {self.model_file} as {self.model_type}")

    def _load_tflite(self, model_file: str):
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate
            delegates = []
            try:
                delegates = [load_delegate("libedgetpu.so.1")]
                self.logger.info("Coral Edge TPU delegate loaded")
            except Exception:
                self.logger.info("No Edge TPU delegate, running TFLite on CPU")
            self.model = Interpreter(model_path=model_file, experimental_delegates=delegates)
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
            self.model = Interpreter(model_path=model_file)
        self.model.allocate_tensors()
        self._tflite_inp = self.model.get_input_details()[0]
        self._tflite_out = self.model.get_output_details()

    def _preprocess_for_rknn(self, frame: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._letterbox_into(img_rgb, self._input_buf[0], self.input_size)
        return self._input_buf

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
            resized, top, pad_h - top, left, pad_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return padded, scale, left, top

    def _letterbox_into(self, img: np.ndarray, dst: np.ndarray, target_size: tuple) -> None:
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        dst[:] = 114
        dst[top:top + new_h, left:left + new_w] = resized

    def _run_rknn(self, preprocessed: np.ndarray, orig_shape) -> Results:
        raw_outputs = self.model.inference(inputs=[preprocessed])
        if raw_outputs is None:
            return Results([], orig_shape)

        output_tensor = raw_outputs[0]

        if self._output_fmt is None:
            _, d1, d2 = output_tensor.shape
            self._output_fmt = "end2end" if d2 == 6 else "no_nms"
            self.logger.info(f"RKNN output format: {self._output_fmt}, shape={output_tensor.shape}")

        if output_tensor.dtype == np.int8:
            output_tensor = output_tensor.astype(np.float32) / 128.0
        elif output_tensor.dtype == np.uint8:
            output_tensor = output_tensor.astype(np.float32) / 255.0

        if self._output_fmt == "end2end":
            return self._convert_rknn_end2end_outputs(output_tensor[0], orig_shape)
        else:
            return self._convert_rknn_outputs(output_tensor[0], orig_shape)

    def _run_tflite(self, frame: np.ndarray, orig_shape) -> Results:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        padded, scale, pad_x, pad_y = self._letterbox(img_rgb, self.input_size)

        dtype = self._tflite_inp["dtype"]
        inp = padded[np.newaxis].astype(np.uint8 if dtype == np.uint8 else np.float32)
        if dtype != np.uint8:
            inp /= 255.0

        self.model.set_tensor(self._tflite_inp["index"], inp)
        self.model.invoke()
        raw = [self.model.get_tensor(d["index"]) for d in self._tflite_out]

        # TFLite from ultralytics export is end2end (N, 6) like ONNX
        output_tensor = raw[0]
        if self._output_fmt is None:
            last = output_tensor.shape[-1]
            self._output_fmt = "end2end" if last == 6 else "no_nms"
        if self._output_fmt == "end2end":
            return self._convert_rknn_end2end_outputs(output_tensor[0], orig_shape)
        return self._convert_rknn_outputs(output_tensor[0], orig_shape)

    def predict_preprocessed(self, preprocessed: np.ndarray, orig_shape) -> Results:
        if self.model_type != "rknn":
            raise RuntimeError("predict_preprocessed is only valid for RKNN models.")
        return self._run_rknn(preprocessed, orig_shape)

    def predict(self, frame_or_frames, orig_shape=None) -> "Results | list[Results]":
        is_list = isinstance(frame_or_frames, list)
        frames = frame_or_frames if is_list else [frame_or_frames]
        results_list = []

        if self.model_type == "rknn":
            for frame in frames:
                target_shape = orig_shape if orig_shape is not None else frame.shape
                preprocessed = self._preprocess_for_rknn(frame)
                results_list.append(self._run_rknn(preprocessed, target_shape))

        elif self.model_type == "tflite":
            for frame in frames:
                target_shape = orig_shape if orig_shape is not None else frame.shape
                results_list.append(self._run_tflite(frame, target_shape))

        else:
            for frame in frames:
                frame_copy = frame.copy()
                result = self.model(
                    frame_copy,
                    verbose=False,
                    show=False,
                    imgsz=(self.input_size[1], self.input_size[0]),
                    conf=self.min_conf,
                )
                # discard result[0].orig_img — it's been drawn on
                result[0].orig_img = None
                results_list.append(self._convert_ultralytics_to_results(result[0], frame))

            return results_list if is_list else results_list[0]
        # For RKNN and TFLite backends we built results_list above; return it
        return results_list if is_list else results_list[0]

    def _convert_rknn_outputs(self, frame_output: np.ndarray, orig_shape) -> Results:
        if frame_output.ndim == 3:
            frame_output = frame_output[0]

        if frame_output.shape[0] == 5 and frame_output.shape[1] > 5:
            frame_output = frame_output.T

        if self._needs_sigmoid is None:
            sample = frame_output[:, 4]
            self._needs_sigmoid = bool(sample.min() < -0.1 or sample.max() > 1.1)

        if self._needs_sigmoid:
            confs = 1 / (1 + np.exp(-np.clip(frame_output[:, 4], -88, 88)))
        else:
            confs = frame_output[:, 4].copy()

        conf_mask = confs >= self.min_conf
        frame_output = frame_output[conf_mask]
        confs = confs[conf_mask]

        if len(frame_output) == 0:
            return Results([], orig_shape)

        valid_mask = (
            ~np.isinf(frame_output).any(axis=1)
            & ~np.isnan(frame_output).any(axis=1)
        )
        frame_output = frame_output[valid_mask]
        confs = confs[valid_mask]

        if len(frame_output) == 0:
            return Results([], orig_shape)

        orig_h, orig_w = orig_shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_x = (target_w - new_w) / 2
        pad_y = (target_h - new_h) / 2

        x_c = (frame_output[:, 0] - pad_x) / scale
        y_c = (frame_output[:, 1] - pad_y) / scale
        w   = frame_output[:, 2] / scale
        h   = frame_output[:, 3] / scale

        x1s = np.clip((x_c - w / 2).astype(int), 0, orig_w)
        y1s = np.clip((y_c - h / 2).astype(int), 0, orig_h)
        x2s = np.clip((x_c + w / 2).astype(int), 0, orig_w)
        y2s = np.clip((y_c + h / 2).astype(int), 0, orig_h)

        size_mask = (x2s - x1s > 0) & (y2s - y1s > 0)
        x1s, y1s, x2s, y2s, confs = (
            x1s[size_mask], y1s[size_mask],
            x2s[size_mask], y2s[size_mask],
            confs[size_mask],
        )

        if len(x1s) == 0:
            return Results([], orig_shape)

        boxes = [Box([x1, y1, x2, y2], float(c)) for x1, y1, x2, y2, c in zip(x1s, y1s, x2s, y2s, confs)]
        scores = confs.tolist()
        nms_boxes = [[b.xyxy[0], b.xyxy[1], b.xyxy[2] - b.xyxy[0], b.xyxy[3] - b.xyxy[1]] for b in boxes]
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores, score_threshold=0.5, nms_threshold=0.3)
        indices = indices.flatten() if len(indices) > 0 else []

        return Results([boxes[i] for i in indices], orig_shape)

    def _convert_ultralytics_to_results(self, ultralytics_result, original_frame):
        boxes = []
        for b in ultralytics_result.boxes:
            xyxy = np.asarray(b.xyxy)
            if xyxy.ndim > 1:
                xyxy = xyxy[0]
            conf = np.asarray(b.conf).item()
            boxes.append(Box(xyxy.tolist(), float(conf)))
        return Results(boxes, ultralytics_result.orig_shape)

    def _convert_rknn_end2end_outputs(self, detections: np.ndarray, orig_shape) -> Results:
        orig_h, orig_w = orig_shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_x = (target_w - new_w) / 2
        pad_y = (target_h - new_h) / 2

        boxes = []
        for det in detections:
            x1, y1, x2, y2, conf = det[0], det[1], det[2], det[3], det[4]
            if float(conf) < self.min_conf:
                continue
            x1 = max(0, int((x1 - pad_x) / scale))
            y1 = max(0, int((y1 - pad_y) / scale))
            x2 = min(orig_w, int((x2 - pad_x) / scale))
            y2 = min(orig_h, int((y2 - pad_y) / scale))
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue
            boxes.append(Box([x1, y1, x2, y2], float(conf)))

        return Results(boxes, orig_shape)

    def release(self):
        if self.model_type == "rknn":
            self.model.release()