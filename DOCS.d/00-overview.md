# 00 — System Overview

> High-level architecture, entry points, data flow, threading model, and
> project structure for the entire iSpy-FRC codebase.

---

## What is iSpy-FRC?

iSpy-FRC is a real-time computer vision pipeline for the FIRST Robotics Competition (FRC). It runs on any hardware — Orange Pi 5 with RK3588 NPU, Raspberry Pi, Jetson, x86 Linux/Windows/macOS — detects game pieces and field elements with YOLO models, converts pixel detections into robot-relative and field-relative 3D coordinates, and publishes them to the robot over NetworkTables v4.

**Core capabilities:**

- YOLO inference across 8+ backend formats (RKNN, ONNX, TFLite, TensorRT, OpenVINO, CoreML, Hailo HEF, Google TPU)
- Automatic hardware detection and model conversion at boot
- Camera calibration via ChArUco boards, chessboard patterns, and known-object focal length
- 6DoF pose estimation via solvePnP
- Multi-camera stereo triangulation
- Monocular depth estimation (Depth Anything V2)
- Optical flow velocity estimation
- Live MJPEG web dashboard with 12+ modules
- 3D field viewer with overlay API
- Plugin architecture (trackers, utilities, frame processors)
- NetworkTables publish to roboRIO / AdvantageKit
- Video recording with rollback support

---

## Project Structure

```
iSpy-FRC/
├── iSpy/                              # Main Python package
│   ├── __init__.py                    # Sets OPENCV_LOG_LEVEL=ERROR globally
│   ├── iSpy.py                        # Core vision loop orchestrator (509 lines)
│   ├── config/                        # Configuration system
│   │   ├── iSpyConfig.py              # iSpyConfig, iSpyCameraConfig, iSpyAddonConfig (757 lines)
│   │   └── AutoOpt.py                 # Hardware detection, format recommendation (249 lines)
│   ├── boot/                          # Boot, install, service management
│   │   ├── boot.py                    # Main boot entry point (first-run setup)
│   │   ├── first_boot.py              # Systemd oneshot for first boot
│   │   ├── service_daemon.py          # Flask service manager (start/stop/pause via stdin)
│   │   ├── setup_service.py           # Cross-platform service installer
│   │   ├── watchdog.py                # Crash-restart loop
│   │   ├── announce.py                # UDP board discovery (every 5s broadcast)
│   │   └── opencv_fix.py              # GStreamer auto-fix for Jetson/CSI cameras
│   ├── core/                          # CLI entry points and shared data structures
│   │   ├── game_loop.py               # Configures logging, creates iSpyConfig, launches iSpy.run() (62 lines)
│   │   ├── ispy.py                    # Runs boot then game_loop
│   │   └── frame_data.py              # TypedFrameData TypedDict and build_frame_data() (97 lines)
│   ├── vision/                        # Vision system
│   │   ├── Camera.py                  # Threaded camera reader with VideoCapture (650 lines)
│   │   ├── Object.py                  # Detection data class (110 lines)
│   │   ├── calibration.py             # ChArUco, chessboard, focal length math (542 lines)
│   │   ├── triangulation.py           # pixel_to_ray, ground_plane_intersection, stereo (133 lines)
│   │   ├── genericYolo.py             # Central inference engine — all backends (1386 lines)
│   │   ├── ModelInspector.py          # Model file introspection and config filling (946 lines)
│   │   ├── QuantizedModel.py          # ONNX quantization utilities
│   │   ├── optimizer.py               # Model conversion pipeline (1422 lines)
│   │   ├── metadata.py                # YAML sidecar read/write (222 lines)
│   │   └── pipelines/                 # Vision pipeline implementations
│   │       ├── __init__.py            # BUILTIN_PIPELINES dict + get_pipeline_classes() (40 lines)
│   │       ├── base.py                # VisionPipeline, BackgroundPreparedPipeline (102 lines)
│   │       ├── object_detection.py    # Main YOLO pipeline (941 lines)
│   │       ├── april_tag.py           # AprilTag + solvePnP (226 lines)
│   │       ├── qr_code.py             # QR code detection + decode (259 lines)
│   │       ├── yolo_world.py          # Open-vocabulary YOLO-World (601 lines)
│   │       ├── depth_anything.py      # Monocular depth estimation (1256 lines)
│   │       └── optical_flow.py        # Dense + sparse optical flow (393 lines)
│   ├── plugins/                       # Plugin system
│   │   ├── bases.py                   # AddonBase, TrackerBase, UtilityBase, FrameProcessorBase, VisionBase (153 lines)
│   │   ├── _loader.py                 # Dynamic plugin discovery via importlib (56 lines)
│   │   ├── trackers/                  # Detection post-processing plugins
│   │   │   ├── BuiltIn/
│   │   │   │   ├── ObjectTracker.py   # EMA position smoothing + stale tracking (120 lines)
│   │   │   │   └── PathPlanner.py     # DBSCAN clustering (63 lines)
│   │   │   └── example_tracker.py     # Template for new trackers
│   │   ├── utilities/                 # Side-effect plugins
│   │   │   ├── BuiltIn/
│   │   │   ├── NetworkHandler.py  # NetworkTables publish + robot pose read (250 lines)
│   │   │   └── RollBack.py        # Video recording with async writer (228 lines)
│   │   │   └── example_utility.py     # Template for new utilities
│   │   ├── frame_processors/          # Frame manipulation plugins
│   │   │   └── example_frame_processor.py
│   │   └── pipelines/                 # User-authored pipeline plugins (scanned at startup)
│   ├── algorithms/                    # Math helpers
│   │   └── CustomDBScan.py            # DBSCAN clustering wrapper
│   ├── web/                           # Flask web application
│   │   ├── Backend/
│   │   │   ├── WebApp.py              # Flask app factory, module registry (107 lines)
│   │   │   ├── WebModule.py           # Abstract module base class
│   │   │   ├── PluginStatus.py        # Addon CRUD API (479 lines)
│   │   │   ├── Settings.py            # Global config API
│   │   │   └── save_store.py          # JSON persistence
│   │   ├── modules/                   # Web modules (one per feature)
│   │   │   ├── cameras.py             # Camera CRUD, feeds, calibration wizard (1824 lines)
│   │   │   ├── viewer3d.py            # 3D viewer + overlay API
│   │   │   ├── dashboard.py           # SSE metrics stream
│   │   │   ├── health.py              # Health endpoints
│   │   │   ├── models.py              # Model management
│   │   │   ├── datasets.py            # Dataset management
│   │   │   ├── metrics.py             # Time-series metrics
│   │   │   ├── logs.py                # Log tailing
│   │   │   ├── onboarding.py          # Tour/onboarding flag
│   │   │   └── recommendations.py     # Hardware recommendations
│   │   ├── templates/                 # Jinja2 HTML templates
│   │   └── static/                    # CSS, vendor JS (Three.js, Chart.js)
│   ├── utilities/                     # Runtime utilities
│   │   ├── MultipleCameraHandler.py   # Multi-camera thread + triangulation
│   │   └── fake_nt.py                 # Test NT4 server
│   ├── validations/                   # Validation, benchmarking, model org checks
│   │   ├── model_validator.py         # enforce_model_organization()
│   │   ├── benchmarking.py            # Model benchmark entry point
│   │   └── tests/                     # Model accuracy comparison
│   └── dataset/                       # Dataset download/scraping
├── tests/                             # pytest test suite
├── Config/                            # Runtime config (config.json)
├── YoloModels/                        # Model files organized by format/size
│   ├── pytorch/                       # .pt source models
│   ├── onnx/                          # Converted ONNX artifacts
│   ├── rknn/                          # Converted RKNN artifacts
│   └── ...
├── QuantizeDataset/                   # Calibration images for quantization
├── Outputs/                           # Logs, metrics, recordings
├── weights/                           # Pre-trained weights
├── VideoRecordings/                   # Recorded video output
├── tools/                             # Development utilities
├── pyproject.toml                     # Package metadata, dependencies, entry points
└── install.bat / install.sh           # Quick-start installers
```

---

## Entry Points

Defined in `pyproject.toml` under `[project.scripts]`:

| Command | Module | Description |
|---------|--------|-------------|
| `ispy-run` | `iSpy.core.game_loop:main` | Main entry: loads config, builds cameras, runs vision loop |
| `ispy-boot` | `iSpy.boot.boot:main` | First-time setup: creates dirs, copies models, installs service |
| `ispy-bench` | `iSpy.validations.benchmarking:main` | Benchmarks model inference across backends |
| `ispy-stats` | `iSpy.validations.tests.compare_models:main` | Compares base vs optimized model accuracy |
| `ispy-test-web` | `iSpy.web.test_web:main` | Dev server with fake cameras for UI testing |

### Primary flow: `ispy-run`

```
ispy-run
  └── iSpy.core.game_loop.main()                  # game_loop.py:52
        │
        ├── _configure_quiet_logging()             # game_loop.py:9-41
        │   ├── Sets up StreamHandler to stdout (iSpy-* loggers only)
        │   ├── Sets up FileHandler to Outputs/log.txt
        │   └── Silences all non-iSpy loggers to WARNING
        │
        ├── iSpyConfig(config_path)                # iSpyConfig.py:186
        │   ├── Creates default_config dict        # iSpyConfig.py:192-270
        │   ├── load_from_file(file_path)          # iSpyConfig.py:396-427
        │   │   ├── Reads JSON with utf-8-sig encoding (BOM-safe)
        │   │   ├── Rejects legacy top-level "vision_model" layout
        │   │   └── Calls _update_config(data)     # iSpyConfig.py:549-580
        │   │       ├── Deep-merges into self.config
        │   │       ├── For camera_configs: normalize_camera_entry() + ensure_camera_entries_ready()
        │   │       └── For nested dicts: recursive merge
        │   ├── _check_config()                    # iSpyConfig.py:300-306 (ensures required keys)
        │   ├── _migrate_addons()                  # iSpyConfig.py:308-355 (legacy format upgrade)
        │   ├── _migrate_camera_configs()          # iSpyConfig.py:357-373 (normalizes all cameras)
        │   ├── _rebuild_camera_configs()          # iSpyConfig.py:387-394 (wraps in iSpyCameraConfig)
        │   └── _configure_logging()               # iSpyConfig.py:582-615 (re-applies log level from config)
        │
        ├── iSpy(config=config)                    # iSpy.py:32
        │   ├── _build_cameras_from_config()       # iSpy.py:138-168
        │   │   ├── enforce_model_organization()   # Validates YoloModels/ layout
        │   │   ├── get_pipeline_classes()          # Discovers built-in + user pipelines
        │   │   └── cls(cam_config, config) for each camera in config
        │   │       (e.g., ObjectDetectionCamera, AprilTagCamera, etc.)
        │   │
        │   ├── Plugin loading (3 categories):
        │   │   ├── load_plugins("trackers")       # iSpy.py:72-80
        │   │   ├── load_plugins("utilities")      # iSpy.py:82-94
        │   │   └── load_plugins("frame_processors") # iSpy.py:96-107
        │   │   Each: discover classes → filter enabled in config → instantiate with context
        │   │
        │   ├── create_app(cameras, config)         # WebApp.py — Flask app factory
        │   │   └── Registers 13 web modules (cameras, models, datasets, viewer3d, etc.)
        │   │
        │   ├── threading.Thread(web_app.run, daemon=True).start()   # iSpy.py:120
        │   │
        │   ├── Wire health module → NetworkTable handler            # iSpy.py:110-113
        │   ├── Attach frame processors to all cameras              # iSpy.py:128-131
        │   └── Signal handlers: SIGINT, SIGTERM → _handle_shutdown()  # iSpy.py:44-45
        │
        └── iSpy.run()                             # iSpy.py:280
            ├── Start stdin-reader thread          # iSpy.py:285-287
            ├── Optional duration timer thread     # iSpy.py:289-295
            ├── 1 camera  → run_solo_mode()        # iSpy.py:419
            └── 2+ cameras → run_multi_mode()      # iSpy.py:463
```

---

## Per-Tick Data Flow (Solo Mode)

The `run_solo_mode()` method at `iSpy.py:419` runs a continuous `while` loop. Each iteration processes one frame through the entire pipeline:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Single Vision Loop Tick                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. PAUSE CHECK                                                              │
│     if pause_event.is_set():                                                 │
│       re-send frozen last_frame_data with fps=0                              │
│       sleep(50ms), continue                                                  │
│                                                                              │
│  2. CAMERA READ (Camera.py:392-445)                                          │
│     Camera._reader thread continuously:                                      │
│       cap.read() → apply_image_adjustments() → lock → self.frame = frame     │
│     Main thread:                                                             │
│       camera.get_frame() → lock → copy frame → run frame_processors          │
│       camera.get_frame_age() → time since last frame capture                 │
│                                                                              │
│  3. VISION INFERENCE                                                         │
│     ObjectDetectionCamera.run()  (object_detection.py:864-892)               │
│       ├── get_frame() from Camera                                            │
│       ├── get_yolo_data():                                                   │
│       │   ├── If HW pipeline (RKNN/ONNX/TFLite):                            │
│       │   │   get preprocessed frame from _preproc_q queue                   │
│       │   │   model.predict_preprocessed() → Results                         │
│       │   └── Else:                                                          │
│       │       model.predict(frame) → Results                                 │
│       ├── For each Box in Results.boxes:                                     │
│       │   ├── _filter_box(): margin + aspect ratio check                     │
│       │   └── _box_to_object():                                              │
│       │       ├── _pixel_ray() → triangulation.Ray                           │
│       │       ├── _box_to_robot_point():                                     │
│       │       │   ├── ground_plane_intersection(ray) → 3D point              │
│       │       │   └── fallback: _size_based_point() → 2D point              │
│       │       ├── If PnP: _pnp_to_robot_coordinates(box.translation)         │
│       │       └── Return Object(x, y, z, roll, pitch, yaw, ...)             │
│       └── Return (detections: list[Object], annotated_frame: ndarray)         │
│                                                                              │
│  4. ROBOT POSE                                                               │
│     _get_pose() (iSpy.py:213-219)                                           │
│       Iterates utilities; first one with get_robot_pose() returns Pose2d      │
│       (typically NetworkTableHandler reads from AdvantageKit/RealOutputs)     │
│                                                                              │
│  5. TRACKER UPDATE                                                           │
│     for each tracker:                                                        │
│       detections = tracker.update(detections, pose.X(), pose.Y(),            │
│                                   -pose.rotation().radians(), 0.0)           │
│       ObjectTracker.update():                                                │
│         ├── Age all tracked_objects, remove destroyed                        │
│         ├── Convert new detections to robot frame via relative_to()           │
│         ├── _merge(): for each new det, find closest tracked_obj:            │
│         │   ├── If within distance_threshold: EMA smooth position/rotation   │
│         │   └── Else: add as new tracked object with alive_time              │
│         └── Return tracked_objects list                                       │
│                                                                              │
│  6. BUILD frame_data (frame_data.py:57-97)                                   │
│     build_frame_data(detections, frame, fps, loop_s, vision_s, ...)          │
│     Optional: debug_data, debug_frame from camera                            │
│                                                                              │
│  7. UTILITY UPDATE                                                           │
│     for each utility:                                                        │
│       util.update(frame_data)                                                │
│       ├── NetworkTableHandler.update():                                      │
│       │   ├── For each publish entry: _resolve_source() → _publish_entry()   │
│       │   ├── _send_detections(): create FuelStruct[], publish as struct[]   │
│       │   ├── inst.flush()                                                   │
│       │   └── _update_viewer_overlay(): push robot box to 3D viewer          │
│       ├── HealthModule.update() [core web module]:                            │
│       │   └── Updates thread-safe metrics (fps, vision_s, detections, etc.)  │
│       └── RollBack.update():                                                 │
│           └── _write() → clean_frame → queue.put → _worker writes to disk    │
│                                                                              │
│  8. WEB UPDATE                                                               │
│     web_app.update(frame_data)                                               │
│       └── Each web module receives frame_data for SSE, metrics, etc.         │
│                                                                              │
│  9. FPS LIMIT                                                                │
│     if max_fps > 0: sleep(max(0, 1/max_fps - elapsed))                       │
│                                                                              │
│  10. PRINT FPS                                                               │
│      print(f"\rFPS: {actual_fps:.1f}   ", end="")                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## The `frame_data` Dict

Defined as a `TypedDict` in `iSpy/core/frame_data.py:35-55` and constructed by `build_frame_data()` at line 57. This is the central data structure passed through every subsystem on every tick.

```python
frame_data = {
    # ── Core (always present) ─────────────────────────────────────────────
    "detections":       list[Object],       # All detections after tracker processing
    "detection_count":  int,                # len(detections)
    "frame":            np.ndarray | None,  # Annotated BGR image (or raw frame)
    "fps":              float,              # Measured loop frequency (Hz)
    "loop_s":           float,              # Total loop iteration time (seconds)
    "vision_s":         float,              # Vision/inference time (seconds)
    "camera_lag_s":     float,              # Time since camera captured the frame (seconds)
    "cameras":          list[VisionPipeline],  # All camera instances
    "code_times":       dict[str, float],   # Per-stage timing breakdown:
        # "vision":    float,               #   Vision inference
        # "pose":      float,               #   Robot pose retrieval
        # "trackers":  float,               #   All tracker updates
        # "utilities": float,               #   All utility updates (added after _update_utilities)
        # "web":       float,               #   Web app update (added after _update_web)
    "debug_data":       dict[str, Any],     # Pipeline-specific debug info
    "pipeline_name":    str,                # e.g. "object_detection" or "april_tag,object_detection" (multi)
    "pipeline_settings": dict[str, Any],    # Pipeline config subset (vision_model, calibration, etc.)
    "camera_config":    dict[str, Any],     # Full camera config dict for the active camera
    "robot_pose":       {"x": float, "y": float, "heading": float},  # Robot field pose

    # ── Conditional ───────────────────────────────────────────────────────
    "debug_frame":      np.ndarray | None,     # Alternative debug visualization (present when camera outputs one)
    "camera_frames":    dict[str, np.ndarray], # Per-camera frames (multi-camera mode only)
}
```

### How `frame_data` flows:

```
_build_loop_body_solo()
  → build_frame_data(...)
  → code_times["utilities"] added after _update_utilities()
  → code_times["web"] added after _update_web()
  → passed to every UtilityBase.update()
  → passed to every WebModule.update()
```

---

## Threading Model

The iSpy system runs multiple concurrent threads. Here is every thread and its purpose:

| Thread | Created At | Purpose |
|--------|-----------|---------|
| **Main** | `iSpy.run()` → `run_solo_mode()` / `run_multi_mode()` | The vision loop: reads frames, runs inference, updates trackers, utilities, and web. All timing measurements happen here. |
| **Camera reader** (per camera) | `Camera.__init__()` → `threading.Thread(target=self._reader, daemon=True)` (`Camera.py:118-123`) | Continuously reads frames from `cv2.VideoCapture` in a tight loop. Writes to `self.frame` under `self.frame_lock`. Handles FPS capping, auto-reopen on failure, black-frame skipping, and image adjustments. |
| **Flask server** | `iSpy.__init__()` → `threading.Thread(target=self.web_app.run, daemon=True)` (`iSpy.py:120`) | Runs the Flask development server in the background. Serves the web dashboard, camera feeds, calibration wizards, and all API endpoints. |
| **stdin-reader** | `iSpy.run()` → `threading.Thread(target=self._stdin_reader, daemon=True)` (`iSpy.py:285-287`) | Reads commands from `sys.stdin` line by line. Recognized commands: `PAUSE`, `RESUME`, `SHUTDOWN`. Used by the service daemon to control the vision loop without killing the process. |
| **UDP announcer** | `boot/announce.py` (started by boot/service) | Broadcasts UDP discovery packets every 5 seconds so iSpy boards can be found on the network. |
| **Preprocess worker** (per camera, HW pipelines) | `ObjectDetectionCamera.__init__()` → `threading.Thread(target=self._preprocess_worker, daemon=True)` (`object_detection.py:194-199`) | Runs for RKNN/ONNX/TFLite backends. Reads raw frames, applies letterbox + normalization, puts preprocessed tensors into `_preproc_q` for the main thread to consume. |
| **Prepare thread** (background pipelines) | `BackgroundPreparedPipeline.prepare()` → `threading.Thread(target=self._prepare, daemon=True)` (`base.py:90-95`) | Runs heavy model loading/optimization in a background thread so the constructor returns immediately. `is_ready()` reports status. |
| **Optimize runner** (per camera, when optimizing) | `ObjectDetectionCamera.__init__()` → `threading.Thread(target=self._optimize_runner, daemon=True)` (`object_detection.py:210-214`) | Builds the optimized model artifact (RKNN/ONNX/TFLite/Engine) in a background thread so the app keeps running during conversion. |
| **Video recorder** | `RollBack._start_recorder()` → `threading.Thread(target=self._worker, daemon=True)` (`RollBack.py:161-162`) | Async video writer: pulls frames from a `queue.Queue` and writes them to disk via `cv2.VideoWriter`. Prevents blocking the vision loop on disk I/O. |
| **Detection worker** (calibration feed) | `CamerasModule._generate_calibration()` (inside `cameras.py`) | Runs ChArUco/chessboard detection on a downscaled frame copy for the live calibration overlay. Result is cached under a `result_lock`. |

---

## Concurrency Controls

| Control | Type | Location | Purpose |
|---------|------|----------|---------|
| `shutdown_event` | `threading.Event` | `iSpy.py:39` | Signals all threads to stop. Set by SIGINT/SIGTERM handlers or SHUTDOWN stdin command. Checked by `while not self.shutdown_event.is_set()` in both solo and multi modes. |
| `pause_event` | `threading.Event` | `iSpy.py:40` | When set, the vision loop pauses: skips inference, re-sends the last `frame_data` with `fps=0`, and sleeps 50ms per tick. Flask server stays alive. Set/cleared by PAUSE/RESUME stdin commands. |
| `Camera.frame_lock` | `threading.Lock` | `Camera.py:71` | Protects `Camera.frame` and `Camera.frame_timestamp`. The reader thread writes under this lock; the main thread reads under this lock. |
| `Camera._frame_event` | `threading.Event` | `Camera.py:72` | Signaled by the reader thread when a new frame is available. Used by the preprocess worker to avoid busy-waiting. |
| `HealthModule._lock` | `threading.Lock` | `health.py:17` | Protects the health metrics (`_fps`, `_vision_s`, `_detections`, `_last_tick`, `_loop_count`) which are written by the vision loop and read by the Flask `/health/detailed` route. |
| `result_lock` (calibration) | `threading.Lock` | In `cameras.py` calibration feed | Protects the detection worker's cached result which is read by the feed generator. |
| `BackgroundPreparedPipeline._prep_lock` | `threading.Lock` | `base.py:81` | Ensures `prepare()` only starts the background thread once. |

---

## Package Dependencies

### Core (always installed, from `pyproject.toml`):

```
opencv-python>=4.8.0         # Image processing, VideoCapture, all inference backends
numpy>=1.26.0                # Array operations, tensor math
ultralytics>=8.3.0           # YOLO model loading (fallback for .pt and Ultralytics-managed formats)
onnxruntime>=1.16.0          # ONNX inference
flask>=3.0                   # Web server
pyntcore>=2024.0.0           # NetworkTables v4 client
robotpy>=2024.0.0            # WPILib math (Pose2d, Rotation2d)
scikit-learn>=1.3.0          # DBSCAN clustering (PathPlanner)
scipy>=1.10.0                # Camera calibration (Levenberg-Marquardt)
psutil>=7.2.0                # System metrics (CPU, memory)
Pillow>=12.0.0               # Image I/O for calibration board PDF generation
requests>=2.31.0             # HTTP client (model downloads)
protobuf>=3.20.2             # Serialization
tqdm>=4.66.0                 # Progress bars (model conversion)
ruamel.yaml>=0.19.0          # YAML metadata sidecar read/write
plotly>=5.0                  # Charts (metrics module)
transformers                 # HuggingFace (Depth Anything V2 models)
wpiutil                      # WPILib struct serialization (FuelStruct for NT)
```

### Dev-only (optional):

```
torch, torchvision           # Model training, CUDA inference
tensorflow                   # TFLite conversion
onnx, onnxslim               # ONNX optimization/simplification
tensorrt                     # NVIDIA GPU inference (TensorRT engine)
```

### Hardware-specific backends (auto-installed by `boot.py`):

```
rknn-toolkit2 / rknnlite     # Rockchip NPU (Orange Pi 5, RK3588)
openvino                     # Intel GPU/NPU/CPU
tflite-runtime               # ARM CPU TFLite inference
coremltools                  # Apple Silicon (macOS)
pycoral / edgetpu            # Google Coral USB TPU
torch_xla                    # Google TPU (Cloud TPU / bare metal)
```

---

## Key Files by Line Count

| File | Lines | Role |
|------|-------|------|
| `web/modules/cameras.py` | 1824 | Camera management, live feeds, full calibration wizard |
| `vision/genericYolo.py` | 1386 | Central inference engine — all 8+ backends, NMS, PnP |
| `vision/pipelines/depth_anything.py` | 1256 | Monocular depth estimation pipeline |
| `vision/optimizer.py` | 1422 | Model conversion pipeline (pt→onnx/rknn/tflite/engine/etc.) |
| `vision/ModelInspector.py` | 946 | Model file introspection, config auto-fill, metadata handling |
| `vision/pipelines/object_detection.py` | 941 | Main YOLO object detection pipeline |
| `iSpy/config/iSpyConfig.py` | 757 | Configuration system (load/save/migrate/CRUD) |
| `vision/Camera.py` | 650 | Threaded camera reader with V4L2/MSMF backends |
| `plugins/utilities/BuiltIn/NetworkHandler.py` | 250 | NetworkTables publish, robot pose read, viewer overlay |
| `web/modules/health.py` | 140 | Core HealthModule: /health + /api/health (absorbed health_reporter/status_reporter) |
| `plugins/utilities/BuiltIn/RollBack.py` | 228 | Video recording with async queue-based writer |
| `vision/calibration.py` | 542 | ChArUco/chessboard detection, Levenberg-Marquardt calibration, focal length math |
| `iSpy/iSpy.py` | 509 | Vision loop orchestrator (solo + multi mode) |
| `vision/triangulation.py` | 133 | Pixel-to-ray, ground-plane intersection, stereo triangulation |
| `vision/Object.py` | 110 | Detection data class with relative_to() transform |
| `plugins/bases.py` | 153 | All plugin base classes (AddonBase, TrackerBase, UtilityBase, etc.) |
| `core/frame_data.py` | 97 | TypedDict contract for frame_data dict |

---

## Quick Reference: Startup to Steady State

```
1.  User runs `ispy-run`
2.  game_loop.main() configures logging
3.  iSpyConfig("Config/config.json") loads + migrates config
4.  iSpy(config) is constructed:
    a. Cameras built from config (ObjectDetectionCamera instances)
    b. Each camera opens its VideoCapture + starts reader thread
    c. Plugins discovered and instantiated
    d. Flask web server started in background thread
    e. Frame processors attached to cameras
5.  iSpy.run() is called:
    a. stdin-reader thread started
    b. run_solo_mode() enters the main loop
6.  Each tick:
    a. Camera reader has frames ready (continuous background)
    b. Main thread grabs frame, runs vision pipeline
    c. Tracker processes detections
    d. Utilities receive frame_data (NT publish, health, recording)
    e. Web modules receive frame_data (SSE, metrics, feeds)
    f. FPS limit sleep if configured
7.  On shutdown (SIGINT/SIGTERM/SHUTDOWN command):
    a. shutdown_event is set
    b. Loop exits
    c. All plugins stopped (video writer flushed, etc.)
    d. Web server stopped
    e. Camera capture released
```
