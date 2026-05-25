import logging
import math

import cv2
import numpy as np
from ultralytics import YOLO
from VisionCore.vision.ModelInspector import fill_missing_config
import threading
import queue
import torch

try:
    from rknnlite.api import RKNNLite

    RKNN_FOUND = True
except ImportError:
    RKNN_FOUND = False

class _GPUInferencePool:
    def __init__(self, model_file: str, task: str, devices: list[int], input_size: tuple, min_conf: float):
        # Removed maxsize to prevent deadlocks when submitting batches larger than the queue size
        self._in_q  = queue.Queue()
        self._out_q = queue.Queue()
        self._input_size = (input_size[1], input_size[0])  # ultralytics wants (h, w)
        self._min_conf = min_conf
        self._n = len(devices)

        self.logger = logging.getLogger(__name__)

        for device in devices:
            model = YOLO(model_file, task=task, verbose=False)
            model.to(f"cuda:{device}")
            threading.Thread(
                target=self._worker,
                args=(model, device),
                daemon=True,
                name=f"GPU{device}-Infer",
            ).start()

        self.logger.info("Multi-GPU inference pool: %d device(s) %s", len(devices), devices)

    def _worker(self, model, device):
        # 1. Prime the model directly in the worker thread to guarantee this GPU warms up
        dummy_frame = np.zeros((self._input_size[1], self._input_size[0], 3), dtype=np.uint8)
        model(dummy_frame, verbose=False, show=False, imgsz=self._input_size, device=device)

        # 2. Main execution loop
        while True:
            item = self._in_q.get()
            if item is None:
                break
            
            idx, frame = item  # Unpack the index and the frame
            
            result = model(
                frame,
                verbose=False,
                show=False,
                imgsz=self._input_size,
                conf=self._min_conf,
                device=device,
            )
            result[0].orig_img = None
            
            # Put the index back with the result so we can sort them later
            self._out_q.put((idx, result[0]))

    def infer_batch(self, frames: list[np.ndarray]):
        num_frames = len(frames)
        
        # 1. Push all frames into the queue with their original index
        for idx, frame in enumerate(frames):
            self._in_q.put((idx, frame))

        # 2. Collect all results (they will likely come in out-of-order)
        results = [None] * num_frames
        for _ in range(num_frames):
            idx, res = self._out_q.get()
            results[idx] = res  # Slot it into the correct position

        return results

    def stop(self):
        for _ in range(self._n):
            self._in_q.put(None)

def normalize_model_config(model_config: dict) -> dict:
    cfg = dict(model_config)
    for key in ("file_path", "task", "num_classes", "input_size", "output"):
        if key not in cfg:
            raise ValueError(f"model_config must include '{key}'.")

    task = cfg["task"]
    if task not in ("detect", "pose"):
        raise ValueError(f"Unsupported task '{task}'. Use 'detect' or 'pose'.")

    num_classes = int(cfg["num_classes"])
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1.")

    out = dict(cfg["output"])
    _validate_output_block(out, task, num_classes)
    cfg["output"] = out

    if "input" in cfg:
        cfg["input"] = _validate_input_block(dict(cfg["input"]))

    return cfg


def _validate_input_block(inp: dict) -> dict:
    layout = inp.get("layout")
    if layout not in ("nhwc", "nchw"):
        raise ValueError("input.layout must be 'nhwc' or 'nchw'.")
    dtype = inp.get("dtype")
    if dtype not in ("uint8", "float32"):
        raise ValueError("input.dtype must be 'uint8' or 'float32'.")
    if "letterbox" not in inp:
        raise ValueError("input.letterbox is required (true or false).")
    if inp["letterbox"] and "pad_value" not in inp:
        raise ValueError("input.pad_value is required when input.letterbox is true.")
    if "normalize" not in inp:
        raise ValueError("input.normalize is required (true or false).")
    if inp["normalize"] and "scale" not in inp:
        raise ValueError("input.scale is required when input.normalize is true.")
    return inp


def _validate_output_block(out: dict, task: str, num_classes: int) -> None:
    fmt = out.get("format")
    if fmt not in ("hardware_nms", "raw"):
        raise ValueError("output.format must be 'hardware_nms' or 'raw'.")
    if "layout" not in out:
        raise ValueError(
            "output.layout is required ('anchors_first' or 'features_first')."
        )
    if out["layout"] not in ("anchors_first", "features_first"):
        raise ValueError("output.layout must be 'anchors_first' or 'features_first'.")
    if "quantization" not in out:
        raise ValueError(
            "output.quantization is required ('none', 'int8', or 'uint8')."
        )
    if out["quantization"] not in ("none", "int8", "uint8"):
        raise ValueError("output.quantization must be 'none', 'int8', or 'uint8'.")
    if out["quantization"] in ("int8", "uint8") and "quant_scale" not in out:
        raise ValueError(
            "output.quant_scale is required when output.quantization is int8 or uint8."
        )
    if fmt == "hardware_nms":
        return
    for key in (
        "box_format",
        "score_mode",
        "scores_are_logits",
        "apply_software_nms",
        "nms_iou",
    ):
        if key not in out:
            raise ValueError(f"output.{key} is required for raw format.")
    if out["box_format"] not in ("cxcywh", "xyxy"):
        raise ValueError("output.box_format must be 'cxcywh' or 'xyxy'.")
    if out["score_mode"] not in ("multi_class", "objectness"):
        raise ValueError("output.score_mode must be 'multi_class' or 'objectness'.")
    if out["score_mode"] == "objectness" and num_classes != 1:
        raise ValueError("output.score_mode 'objectness' requires num_classes == 1.")
    if task == "pose":
        for key in ("num_keypoints", "keypoint_dims", "keypoint_scores_are_logits"):
            if key not in out:
                raise ValueError(f"output.{key} is required for task='pose'.")


class Box:
    def __init__(self, xyxy, conf, cls_id=0, translation=None, rotation=None):
        self.xyxy = xyxy
        self.conf = conf
        self.cls_id = cls_id
        # PnP results, both None for detect-only models
        self.translation = translation  # (x, y, z) metres in camera frame
        self.rotation = rotation  # (roll, pitch, yaw) radians in camera frame


class Results:
    def __init__(
        self, boxes: list[Box], orig_shape, keypoints: list[np.ndarray] | None = None
    ):
        self.boxes = boxes
        self.orig_shape = orig_shape
        self.keypoints = keypoints if keypoints is not None else []

    def plot(self, frame):
        for box in self.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if box.rotation is not None:
                roll, pitch, yaw = box.rotation
                label = (
                    f"R:{math.degrees(roll):.0f} "
                    f"P:{math.degrees(pitch):.0f} "
                    f"Y:{math.degrees(yaw):.0f}"
                )
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
        for kpt_set in self.keypoints:
            for kpt in kpt_set:
                x, y, conf = kpt
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
        return frame

    def __str__(self):
        return f"Results(boxes={len(self.boxes)}, keypoints={len(self.keypoints)})"


class GenericYolo:
    def __init__(self, model_config: dict, core_mask=None, visioncore_config=None):
        self.logger = logging.getLogger(__name__)
        self._visioncore_config = visioncore_config
        model_config = fill_missing_config(model_config)

        if visioncore_config is not None:
            visioncore_config.set("vision_model", model_config)
            visioncore_config.save(quiet=True)

        cfg = normalize_model_config(model_config)
        self.device = cfg.get("device", 0)
        requested_device = cfg.get("device", 0)

        try:
            import torch
            cuda_ok = (
                torch.cuda.is_available()
                and isinstance(requested_device, int)
                and requested_device < torch.cuda.device_count()
            )
        except Exception:
            cuda_ok = False

        if not cuda_ok and requested_device != "cpu":
            self.logger.info(
                "Device %r not available (CUDA=%s, count=%d) — falling back to CPU",
                requested_device,
                torch.cuda.is_available() if 'torch' in dir() else False,
                torch.cuda.device_count() if 'torch' in dir() else 0,
            )
            self.device = "cpu"
        else:
            self.device = requested_device

        self.model_file = cfg["file_path"]
        self.task = cfg["task"]
        self.num_classes = int(cfg["num_classes"])
        self.input_size = tuple(cfg["input_size"])
        self.min_conf = float(cfg["min_conf"]) if "min_conf" in cfg else 0.25
        self.output = cfg["output"]
        self.pnp_config = cfg.get("pnp")
        self.input = cfg.get("input")
        self._preprocess_buf: np.ndarray | None = None

        self.has_hardware_nms = self.output["format"] == "hardware_nms"
        self.model_type = None

        if self.model_file.endswith(".rknn"):
            if not RKNN_FOUND:
                raise ImportError(
                    "rknnlite not installed but .rknn model was specified."
                )
            self._require_input_block()
            self.model_type = "rknn"
            self.model = RKNNLite()
            if self.model.load_rknn(self.model_file) != 0:
                raise ValueError(f"Failed to load RKNN model: {self.model_file}")
            if self.model.init_runtime(core_mask=(core_mask if core_mask is not None else 0)) != 0:
                raise ValueError(f"Failed to init RKNN runtime: {self.model_file}")

        elif self.model_file.endswith(".onnx"):
            self._require_input_block()
            self.model_type = "onnx"
            self._load_onnx(self.model_file)

        elif self.model_file.endswith(".tflite"):
            self._require_input_block()
            self.model_type = "tflite"
            self._load_tflite(self.model_file)

        elif (
            self.model_file.endswith(".pt")
            or "openvino_model" in self.model_file
            or self.model_file.endswith(".mlpackage")
            or self.model_file.endswith(".engine")
        ):
            self.model_type = "yolo"
            self.model = YOLO(self.model_file, task=self.task, verbose=False)
            if self.model_file.endswith(".pt"):
                self.model.to(f"cuda:{self.device}")

            # after self.model_type = "yolo" and self.model = YOLO(...)
            self._pool: _GPUInferencePool | None = None

            num_gpus = cfg.get("num_gpus", 1)
            if num_gpus == "auto":
                try:
                    import torch
                    num_gpus = torch.cuda.device_count()
                except Exception:
                    num_gpus = 1

            if self.model_type == "yolo" and num_gpus > 1:
                try:
                    import torch
                    available = torch.cuda.device_count()
                    devices = list(range(min(num_gpus, available)))
                    if len(devices) > 1:
                        self._pool = _GPUInferencePool(
                            model_file=self.model_file,
                            task=self.task,
                            devices=devices,
                            input_size=self.input_size,
                            min_conf=self.min_conf,
                        )
                except Exception as e:
                    self.logger.warning("Multi-GPU pool failed, falling back to single GPU: %s", e)
        else:
            raise ValueError(f"Unsupported model file type: {self.model_file}")

        self.logger.info(
            "GenericYolo loaded: %s  type=%s  task=%s  output=%s",
            self.model_file,
            self.model_type,
            self.task,
            self.output["format"],
        )

    def _require_input_block(self) -> None:
        if self.input is None:
            raise ValueError(
                "model_config.input is required for RKNN, ONNX, and TFLite models."
            )

    def _feature_width(self) -> int:
        out = self.output
        if out["format"] == "hardware_nms":
            return 6
        if self.task == "pose":
            score_cols = 1 if out["score_mode"] == "objectness" else self.num_classes
            return 4 + score_cols + out["num_keypoints"] * out["keypoint_dims"]
        score_cols = 1 if out["score_mode"] == "objectness" else self.num_classes
        return 4 + score_cols

    def _load_onnx(self, model_file: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime is required for .onnx models.") from exc

        providers = []
        try:
            available = ort.get_available_providers()
            for ep in (
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ):
                if ep in available:
                    providers.append(ep)
        except Exception:
            providers = ["CPUExecutionProvider"]

        self.model = ort.InferenceSession(model_file, providers=providers)
        self._onnx_inp_name = self.model.get_inputs()[0].name
        self._onnx_out_names = [o.name for o in self.model.get_outputs()]
        self.logger.info("ONNX providers: %s", self.model.get_providers())

    def _load_tflite(self, model_file: str):
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate

            delegates = []
            try:
                delegates = [load_delegate("libedgetpu.so.1")]
                self.logger.info("Coral Edge TPU delegate loaded.")
            except Exception:
                self.logger.info("No Edge TPU delegate - running TFLite on CPU.")
            self.model = Interpreter(
                model_path=model_file, experimental_delegates=delegates
            )
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter

            self.model = Interpreter(model_path=model_file)

        self.model.allocate_tensors()
        self._tflite_inp = self.model.get_input_details()[0]
        self._tflite_out = self.model.get_output_details()

    def _letterbox_into(self, img, dst, target_size, pad_value=114):
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        dst[:] = pad_value
        dst[top : top + new_h, left : left + new_w] = cv2.resize(img, (new_w, new_h))

    def _alloc_preprocess_buffer(self) -> np.ndarray:
        inp = self.input
        target_w, target_h = self.input_size
        shape = (
            (1, target_h, target_w, 3)
            if inp["layout"] == "nhwc"
            else (1, 3, target_h, target_w)
        )
        buf_dtype = np.uint8 if inp["dtype"] == "uint8" else np.float32
        return np.empty(shape, dtype=buf_dtype)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        inp = self.input
        target_w, target_h = self.input_size
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if inp["letterbox"]:
            if self._preprocess_buf is None:
                self._preprocess_buf = self._alloc_preprocess_buffer()
            pad_value = int(inp["pad_value"])
            if inp["layout"] == "nhwc":
                self._letterbox_into(
                    img_rgb, self._preprocess_buf[0], self.input_size, pad_value
                )
            else:
                nhwc = np.empty((target_h, target_w, 3), dtype=np.uint8)
                self._letterbox_into(img_rgb, nhwc, self.input_size, pad_value)
                self._preprocess_buf[0][:] = np.transpose(nhwc, (2, 0, 1))
            tensor = self._preprocess_buf
        else:
            resized = cv2.resize(img_rgb, (target_w, target_h))
            tensor = (
                np.transpose(resized, (2, 0, 1))[np.newaxis]
                if inp["layout"] == "nchw"
                else resized[np.newaxis]
            )

        if inp["dtype"] == "float32":
            out = tensor.astype(np.float32, copy=False)
            if inp["normalize"]:
                out = out / float(inp["scale"])
            return out

        if tensor.dtype != np.uint8:
            return np.clip(tensor, 0, 255).astype(np.uint8)
        return tensor

    def predict_preprocessed(self, preprocessed: np.ndarray, orig_shape) -> Results:
        if self.model_type != "rknn":
            raise RuntimeError("predict_preprocessed is only valid for RKNN models.")
        return self._run_rknn(preprocessed, orig_shape)

    def predict(self, frame_or_frames, orig_shape=None) -> "Results | list[Results]":
        is_list = isinstance(frame_or_frames, list)
        frames = frame_or_frames if is_list else [frame_or_frames]
        
        if self.model_type == "yolo" and self._pool is not None:
            raw_results = self._pool.infer_batch(frames)
            results_list = [self._convert_ultralytics_to_results(raw) for raw in raw_results]
            return results_list if is_list else results_list[0]

        results_list = []
        for frame in frames:
            target_shape = orig_shape if orig_shape is not None else frame.shape
            if self.model_type == "rknn":
                results_list.append(self._run_rknn(self._preprocess_frame(frame), target_shape))
            elif self.model_type == "onnx":
                results_list.append(self._run_onnx(frame, target_shape))
            elif self.model_type == "tflite":
                results_list.append(self._run_tflite(frame, target_shape))
            else:
                # Single GPU fallback
                result = self.model(
                    frame,
                    verbose=False,
                    show=False,
                    imgsz=(self.input_size[1], self.input_size[0]),
                    conf=self.min_conf,
                    device=self.device,
                )
                result[0].orig_img = None
                result = self._convert_ultralytics_to_results(result[0])
                results_list.append(result)
                
        return results_list if is_list else results_list[0]

    def _dequantize_tensor(self, tensor: np.ndarray) -> np.ndarray:
        q = self.output["quantization"]
        if q == "none":
            return tensor.astype(np.float32) if tensor.dtype != np.float32 else tensor
        scale = float(self.output["quant_scale"])
        return tensor.astype(np.float32) / scale

    def _run_rknn(self, preprocessed: np.ndarray, orig_shape) -> Results:
        raw_outputs = self.model.inference(inputs=[preprocessed])
        if raw_outputs is None:
            return Results([], orig_shape)

        tensor = self._dequantize_tensor(raw_outputs[0])

        if not hasattr(self, '_rknn_fmt_verified'):
            self._rknn_fmt_verified = True
            t = tensor[0] if tensor.ndim == 3 else tensor
            actual_fmt = "hardware_nms" if t.shape[-1] == 6 or t.shape[0] == 6 else "raw"

            if actual_fmt != self.output["format"]:
                self.logger.warning(
                    "RKNN output shape %s says format should be %r but config has %r "
                    "— correcting and saving to config.",
                    tensor.shape, actual_fmt, self.output["format"],
                )
                self.output["format"] = actual_fmt
                self.has_hardware_nms = actual_fmt == "hardware_nms"

                if self._visioncore_config is not None:
                    self._visioncore_config.set("vision_model", "output", "format", actual_fmt)
                    self._visioncore_config.save(quiet=True)

        return self.postprocess([tensor], orig_shape)

    def _run_onnx(self, frame: np.ndarray, orig_shape) -> Results:
        inp = self._preprocess_frame(frame)
        raw = self.model.run(self._onnx_out_names, {self._onnx_inp_name: inp})
        raw[0] = self._dequantize_tensor(raw[0])
        return self.postprocess(raw, orig_shape)

    def _run_tflite(self, frame: np.ndarray, orig_shape) -> Results:
        inp = self._preprocess_frame(frame)
        self.model.set_tensor(self._tflite_inp["index"], inp)
        self.model.invoke()
        raw = [self.model.get_tensor(d["index"]) for d in self._tflite_out]
        raw[0] = self._dequantize_tensor(raw[0])
        return self.postprocess(raw, orig_shape)

    def postprocess(self, raw_outputs, orig_shape) -> Results:
        tensor = raw_outputs[0]
        tensor = self._prepare_output_tensor(tensor)
        if tensor.size == 0:
            return Results([], orig_shape)
        if self.output["format"] == "hardware_nms":
            return self._parse_hardware_nms(tensor, orig_shape)
        if self.task == "detect":
            return self._parse_raw_detect(tensor, orig_shape)
        return self._parse_raw_pose(tensor, orig_shape)

    def _prepare_output_tensor(self, tensor: np.ndarray) -> np.ndarray:
        while isinstance(tensor, (list, tuple)) and len(tensor) > 0:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D output tensor, got shape {tensor.shape}.")
        if self.output["format"] == "hardware_nms":
            return tensor

        feat_w = self._feature_width()
        if self.output["layout"] == "features_first":
            if tensor.shape[0] == feat_w:
                tensor = tensor.T
            elif tensor.shape[1] != feat_w:
                raise ValueError(
                    f"Tensor shape {tensor.shape} vs feature width {feat_w}."
                )
        elif tensor.shape[1] != feat_w:
            if tensor.shape[0] == feat_w:
                raise ValueError(
                    "output.layout is 'anchors_first' but feature dim is on axis 0. "
                    "Set output.layout to 'features_first'."
                )
            raise ValueError(f"Tensor shape {tensor.shape} vs feature width {feat_w}.")
        return tensor

    def _apply_score_activation(
        self, scores: np.ndarray, are_logits: bool
    ) -> np.ndarray:
        if scores.size == 0 or not are_logits:
            return scores
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -88.0, 88.0)))

    def _boxes_from_encoding(self, tensor: np.ndarray) -> np.ndarray:
        if self.output["box_format"] == "xyxy":
            return tensor[:, :4].astype(np.float32)
        cx, cy, w, h = tensor[:, 0], tensor[:, 1], tensor[:, 2], tensor[:, 3]
        return np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def _scores_from_tensor(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = self.output
        if out["score_mode"] == "objectness":
            raw = tensor[:, 4]
            confs = self._apply_score_activation(raw, out["scores_are_logits"])
            return confs, np.zeros(len(confs), dtype=np.int32)
        class_scores = tensor[:, 4 : 4 + self.num_classes]
        class_scores = self._apply_score_activation(
            class_scores, out["scores_are_logits"]
        )
        confs = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        return confs, class_ids

    def _scale_coords(
        self, coords: np.ndarray, orig_shape, is_kpts=False
    ) -> np.ndarray:
        orig_h, orig_w = orig_shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w / orig_w, target_h / orig_h)
        pad_x = (target_w - int(orig_w * scale)) / 2
        pad_y = (target_h - int(orig_h * scale)) / 2
        scaled = coords.copy().astype(np.float32)
        if not is_kpts:
            scaled[[0, 2]] = np.clip((scaled[[0, 2]] - pad_x) / scale, 0, orig_w)
            scaled[[1, 3]] = np.clip((scaled[[1, 3]] - pad_y) / scale, 0, orig_h)
        else:
            scaled[:, 0] = (scaled[:, 0] - pad_x) / scale
            scaled[:, 1] = (scaled[:, 1] - pad_y) / scale
        return scaled

    def _rvec_to_euler(self, rvec: np.ndarray) -> tuple[float, float, float]:
        R, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:  # gimbal lock
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0.0
        return roll, pitch, yaw

    def _solve_pnp(
        self, keypoints: np.ndarray
    ) -> tuple[tuple[float, float, float] | None, np.ndarray | None]:
        if not self.pnp_config:
            return None, None

        object_points = np.asarray(self.pnp_config["object_points"], dtype=np.float64)
        camera_matrix = np.asarray(self.pnp_config["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.asarray(
            self.pnp_config.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        min_kpt_conf = float(self.pnp_config.get("min_keypoint_conf", 0.5))

        image_points, model_points = [], []
        for i, pt in enumerate(keypoints):
            if i >= len(object_points):
                break
            if pt[2] < min_kpt_conf:
                continue
            image_points.append([float(pt[0]), float(pt[1])])
            model_points.append(object_points[i])

        if len(image_points) < 4:
            return None, None

        image_points = np.asarray(image_points, dtype=np.float64)
        model_points = np.asarray(model_points, dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None, None

        euler = self._rvec_to_euler(rvec.reshape(3))
        return euler, tvec.reshape(3)

    def _apply_software_nms(
        self,
        boxes_xyxy: np.ndarray,
        confs: np.ndarray,
        class_ids: np.ndarray,
        orig_shape,
        kpts_raw: np.ndarray | None = None,
    ) -> Results:
        mask = confs >= self.min_conf
        boxes_xyxy = boxes_xyxy[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]
        if kpts_raw is not None:
            kpts_raw = kpts_raw[mask]

        if len(boxes_xyxy) == 0:
            return Results([], orig_shape)

        if not self.output["apply_software_nms"]:
            return self._pack_detections(
                boxes_xyxy, confs, class_ids, orig_shape, kpts_raw, None
            )

        nms_iou = float(self.output["nms_iou"])
        nms_input = [
            [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
            for b in boxes_xyxy
        ]
        indices = cv2.dnn.NMSBoxes(nms_input, confs.tolist(), self.min_conf, nms_iou)
        indices = indices.flatten() if len(indices) > 0 else []
        return self._pack_detections(
            boxes_xyxy, confs, class_ids, orig_shape, kpts_raw, indices
        )

    def _pack_detections(
        self,
        boxes_xyxy: np.ndarray,
        confs: np.ndarray,
        class_ids: np.ndarray,
        orig_shape,
        kpts_raw: np.ndarray | None,
        indices,
    ) -> Results:
        if indices is None:
            indices = range(len(boxes_xyxy))

        final_boxes: list[Box] = []
        final_kpts: list[np.ndarray] = []
        num_kpts = self.output.get("num_keypoints", 0) if kpts_raw is not None else 0
        kpt_dims = self.output.get("keypoint_dims", 3) if kpts_raw is not None else 3

        for i in indices:
            xyxy = self._scale_coords(boxes_xyxy[i], orig_shape, is_kpts=False)
            translation: list | None = None
            rotation: tuple | None = None
            kpt_scaled: np.ndarray | None = None

            if kpts_raw is not None and num_kpts > 0:
                kpt_set = kpts_raw[i].reshape(num_kpts, kpt_dims)
                kpt_scaled = self._scale_coords(kpt_set, orig_shape, is_kpts=True)

                if self.pnp_config:
                    euler, tvec = self._solve_pnp(kpt_scaled)
                    if tvec is not None:
                        translation = tvec.tolist()
                    if euler is not None:
                        rotation = euler  # (roll, pitch, yaw) in radians

            final_boxes.append(
                Box(
                    xyxy.tolist(),
                    float(confs[i]),
                    int(class_ids[i]),
                    translation,
                    rotation,
                )
            )
            if kpt_scaled is not None:
                final_kpts.append(kpt_scaled)

        return Results(
            final_boxes,
            orig_shape,
            keypoints=final_kpts if kpts_raw is not None else None,
        )

    def _parse_hardware_nms(self, tensor: np.ndarray, orig_shape) -> Results:
        if tensor.shape[1] < 6:
            return Results([], orig_shape)
        confs = tensor[:, 4]
        valid = tensor[confs >= self.min_conf]
        boxes = []
        for det in valid:
            xyxy = self._scale_coords(det[:4], orig_shape, is_kpts=False)
            cls_id = int(det[5]) if det.shape[0] > 5 else 0
            boxes.append(Box(xyxy.tolist(), float(det[4]), cls_id))
        return Results(boxes, orig_shape)

    def _parse_raw_detect(self, tensor: np.ndarray, orig_shape) -> Results:
        boxes_xyxy = self._boxes_from_encoding(tensor)
        confs, class_ids = self._scores_from_tensor(tensor)
        return self._apply_software_nms(boxes_xyxy, confs, class_ids, orig_shape)

    def _parse_raw_pose(self, tensor: np.ndarray, orig_shape) -> Results:
        boxes_xyxy = self._boxes_from_encoding(tensor)
        confs, class_ids = self._scores_from_tensor(tensor)

        score_cols = (
            1 if self.output["score_mode"] == "objectness" else self.num_classes
        )
        kpts_start = 4 + score_cols
        kpts_raw = tensor[:, kpts_start:]
        expected = self.output["num_keypoints"] * self.output["keypoint_dims"]
        if kpts_raw.shape[1] != expected:
            raise ValueError(
                f"Pose keypoint columns {kpts_raw.shape[1]} != expected {expected}."
            )

        if self.output["keypoint_scores_are_logits"]:
            kd = self.output["keypoint_dims"]
            for k in range(2, kpts_raw.shape[1], kd):
                kpts_raw[:, k] = self._apply_score_activation(kpts_raw[:, k], True)

        return self._apply_software_nms(
            boxes_xyxy, confs, class_ids, orig_shape, kpts_raw=kpts_raw
        )

    def _convert_ultralytics_to_results(self, ultralytics_result) -> Results:
        boxes = []
        for b in ultralytics_result.boxes:
            xyxy = b.xyxy
            if hasattr(xyxy, "cpu"):
                xyxy = xyxy.cpu().numpy()
            xyxy = np.asarray(xyxy)
            xyxy = xyxy[0] if xyxy.ndim > 1 else xyxy
            conf = b.conf
            if hasattr(conf, "cpu"):
                conf = conf.cpu().numpy()
            conf = float(np.asarray(conf).item())
            cls_id = b.cls if hasattr(b, "cls") else 0
            if hasattr(cls_id, "cpu"):
                cls_id = cls_id.cpu().numpy()
            cls_id = int(np.asarray(cls_id).item())
            boxes.append(Box(xyxy.tolist(), conf, cls_id))

        keypoints_list = []
        if (
            hasattr(ultralytics_result, "keypoints")
            and ultralytics_result.keypoints is not None
        ):
            kpt_data = ultralytics_result.keypoints.data
            if hasattr(kpt_data, "cpu"):
                kpt_data = kpt_data.cpu().numpy()
            # Run PnP on Ultralytics keypoints too if pnp_config is set
            for kpt_set in kpt_data:
                kpt_arr = np.asarray(kpt_set)
                if self.pnp_config and len(boxes) == len(kpt_data):
                    idx = len(keypoints_list)
                    euler, tvec = self._solve_pnp(kpt_arr)
                    if euler is not None:
                        boxes[idx].rotation = euler
                    if tvec is not None:
                        boxes[idx].translation = tvec.tolist()
                keypoints_list.append(kpt_arr)

        return Results(boxes, ultralytics_result.orig_shape, keypoints_list or None)

    def release(self):
        if self.model_type == "rknn":
            self.model.release()
        if self._pool is not None:
            self._pool.stop()