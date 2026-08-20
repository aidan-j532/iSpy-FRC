# 04 — Vision Loop and Pipelines

> The core vision loop in iSpy.py, pipeline architecture, and each pipeline implementation.

---

## Vision Loop (iSpy/iSpy.py)

The iSpy class orchestrates the entire vision pipeline. It runs a continuous loop that processes camera frames, runs inference, applies trackers, and pushes data to utilities and the web server.

### Initialization

```python
iSpy(config, cameras=None, web_app=None)
```

1. Loads config
2. Builds cameras from config if not provided:
   - Validates model organization
   - For each camera_config, instantiates the appropriate VisionPipeline subclass
3. Starts plugins:
   - Discovers plugin classes via load_plugins()
   - Instantiates enabled plugins from config
4. Creates Flask web app (if app_mode is True)
5. Wires health module to NetworkTable handler
6. Attaches frame processors to cameras
7. Starts Flask server in background thread

### Solo Mode (1 camera)

```python
iSpy.run_solo_mode()
```

Loop body (_run_loop_body_solo):
1. Get raw frame age (camera lag)
2. Run vision: `camera.run()` returns (detections, frame)
3. Get robot pose from utilities (e.g., NetworkTables)
4. Run trackers: each tracker.update(detections, pose)
5. Build frame_data dict with all metrics
6. Optionally run debug_frame and plot
7. Update utilities with frame_data
8. Update web with frame_data
9. Sleep if max_fps is set

### Multi Mode (2+ cameras)

```python
iSpy.run_multi_mode()
```

Uses MultipleCameraHandler which:
1. Runs each camera's inference in its own thread
2. Merges detections via triangulation
3. Returns combined frame

Loop body is similar to solo but uses handler.predict() and handler.get_combined_frame().

### Pause/Resume

The vision loop checks `self.pause_event` each tick. When paused:
- Last frame_data is frozen and re-sent to utilities/web with fps=0
- Loop sleeps 50ms per iteration
- Camera inference is skipped

Pause/resume is controlled via stdin commands (PAUSE, RESUME, SHUTDOWN) sent by the service_daemon.

---

## Vision Pipelines (iSpy/vision/pipelines/)

### Base Classes (base.py)

**VisionPipeline**: Synchronous pipeline base.
```python
class VisionPipeline(AddonBase):
    def run(self): ...          # returns (detections, frame)
    def destroy(self): ...
    def is_ready(self): ...
    def get_frame(self): ...
    def get_raw_frame(self): ...
    def get_frame_age(self): ...
    def in_calibration_mode(self): ...
    def add_frame_processor(self, processor): ...
```

**BackgroundPreparedPipeline**: Extends VisionPipeline. Loads models in a background thread so the constructor returns immediately. `is_ready()` returns True once loading is complete.

### get_pipeline_classes()

Returns a dict mapping pipeline names to their classes:
```python
{
    "object_detection": ObjectDetectionPipeline,
    "april_tag": AprilTagPipeline,
    "qr_code": QrCodePipeline,
    "yolo_world": YoloWorldPipeline,
    "depth_anything": DepthAnythingPipeline,
    "optical_flow": OpticalFlowPipeline,
}
```

---

## Object Detection Pipeline (object_detection.py, 941 lines)

The main YOLO detection pipeline. This is the most complex pipeline.

### Initialization

1. Reads pipeline settings from camera config (vision_model, calibration, tuning)
2. Creates GenericYolo inference engine
3. Sets up ground-plane intersection for depth estimation
4. Configures PnP if pose model is available

### run() flow

1. Get frame from Camera thread
2. Run frame processors on the frame
3. Feed frame to GenericYolo.infer()
4. Post-process detections:
   - Filter by confidence (min_conf)
   - Apply margin exclusion (pixels near image edges)
   - Convert pixel positions to field coordinates:
     a. If camera has calibration + known distance → ground-plane intersection
     b. If no calibration → size-based distance fallback
   - If pose model → run PnP for 6DoF pose
5. Annotate frame (draw bboxes, labels, FPS)
6. Return (detections, annotated_frame)

### Ground-Plane Intersection

Converts a 2D bounding box center to a 3D field coordinate:
1. Compute ray from camera through pixel
2. Intersect ray with ground plane (z=0 or z=camera_height)
3. Result is (x, y) in camera frame
4. Transform to robot frame using camera mount position + yaw

### Size-Based Distance Fallback

When no calibration data is available:
1. Uses the game piece size and FOV to estimate distance
2. `distance = (real_size * focal_pixels) / pixel_size`
3. Less accurate than ground-plane but works without calibration

---

## GenericYolo (vision/genericYolo.py, 1386 lines)

The central inference engine. Supports all backend formats.

### Model Loading

```
Backend selection:
  1. Check for optimized format (e.g., .rknn, .onnx, .tflite)
  2. If auto_opt is True and optimized exists → use it
  3. Otherwise → use Ultralytics for .pt loading
```

Supported backends:
| Format | Library | Hardware |
|--------|---------|----------|
| .pt | ultralytics | CPU/GPU |
| .rknn | rknn-toolkit2 | Rockchip NPU |
| .onnx | onnxruntime | CPU/GPU |
| .tflite | tflite-runtime | ARM CPU |
| .engine | tensorrt | NVIDIA GPU |
| .xml/.bin | openvino | Intel GPU/NPU/CPU |
| .mlmodel | coreml | Apple Silicon |
| .hef | hailo | Hailo NPU |
| .tflite+edgetpu | pycoral | Google Coral TPU |

### infer() flow

1. Preprocess: resize to input_size, letterbox if needed, normalize
2. Run inference via the loaded backend
3. Post-process:
   - If backend does NMS internally → parse output directly
   - If backend returns raw output → apply NMS via ultralytics
4. Apply non-max suppression
5. Map output coordinates back to original image size
6. Return list of raw detections (bbox, confidence, class)

### PnP Solving

If a pose model is configured:
1. Run object detection to get bounding boxes
2. For each detection, extract keypoints
3. Solve PnP with known 3D object points → get rotation + translation
4. Attach pose data to the detection Object

---

## Other Pipelines

### AprilTag (april_tag.py, 226 lines)

1. Convert frame to grayscale
2. Detect ArUco markers with cv2.aruco
3. For each detected marker, solve PnP to get pose
4. Return detections with tag_id and 6DoF pose

### QR Code (qr_code.py, 259 lines)

1. Multi-scale QR detection (pyzbar)
2. For each detected QR code, solve PnP
3. Decode QR content
4. Return detections with decoded data and pose

### YOLO-World (yolo_world.py, 601 lines)

Open-vocabulary detection:
1. Downloads YOLO-World weights if not present
2. Reparameterizes with custom class names
3. Runs inference like standard YOLO but with text-prompted classes
4. Supports quantized/optimized models

### Depth Anything (depth_anything.py, 1256 lines)

Monocular depth estimation:
1. Loads Depth Anything V2 (HuggingFace or ONNX)
2. Runs depth inference → per-pixel depth map
3. Converts relative depth to metric using camera calibration
4. Supports multiple backends (ONNX, RKNN, TensorRT, CoreML, TFLite, TPU)

### Optical Flow (optical_flow.py, 393 lines)

Velocity estimation:
1. Farneback dense optical flow
2. Lucas-Kanade sparse point tracking
3. Converts pixel velocity to real-world velocity using depth
4. Attachs velocity data to detections (depth_source="optical_flow")

---

## Camera (vision/Camera.py, 650 lines)

Threaded camera reader using OpenCV VideoCapture.

### Key features:
- Reads frames in a background thread at the camera's native FPS
- Stores frames in a ring buffer (latest frame always available)
- Thread-safe get_raw_frame() with lock
- Image adjustment properties (brightness, contrast, etc.)
- V4L2 backend on Linux, MSMF on Windows
- Calibration heartbeat (periodic reconnection)
- Frame age tracking

### Methods:

| Method | Description |
|--------|-------------|
| get_raw_frame() | Latest frame (thread-safe) |
| get_frame() | Latest frame (same as get_raw_frame) |
| get_frame_age() | Seconds since last frame was captured |
| set_property(prop, value) | Set OpenCV VideoCapture property |
| in_calibration_mode() | Whether calibration mode is active |
| destroy() | Stop reader thread, release capture |
| add_frame_processor(proc) | Attach a frame processor |

---

## Object (vision/Object.py, 110 lines)

Data class for a single detection:

```python
class Object:
    # Position (field coordinates, after ground-plane intersection)
    x, y, z

    # Rotation (Euler angles, from PnP)
    roll, pitch, yaw

    # Bounding box (pixel coordinates)
    bbox  # (x1, y1, x2, y2)

    # Metadata
    confidence
    class_name
    class_id

    # 3D keypoints (from PnP)
    keypoints_3d

    # Ray (from camera through detection center)
    ray_origin, ray_direction

    # Tracking
    track_id
    lifetime, decay

    # Velocity (from optical flow)
    velocity_x, velocity_y, velocity_z
```

---

## Triangulation (vision/triangulation.py, 133 lines)

Math utilities for coordinate conversion:

| Function | Description |
|----------|-------------|
| pixel_to_ray(u, v, camera_matrix) | Convert pixel to 3D ray |
| ground_plane_intersection(ray, camera_height) | Intersect ray with ground |
| closest_point_between_two_rays(ray1, ray2) | Stereo triangulation |
| camera_to_robot(point, camera_offset, camera_yaw) | Transform to robot frame |
| robot_to_field(point, robot_pose) | Transform to field frame |
