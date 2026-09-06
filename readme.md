# iSpy

> FRC vision pipeline for object detection and field mapping - runs on Orange Pi with Rockchip NPU, supports RKNN, ONNX, OpenVINO, TFLite, and CoreML backends.

---

## What It Does

iSpy is a plug-and-play computer vision system for FRC robots. You point a camera at the field, it detects game pieces, converts pixel positions into field-relative coordinates, and sends them to your robot over NetworkTables - all automatically.

- Detects objects with a YOLO model (any size, any format)
- Converts detections to real-world field coordinates using camera calibration
- Tracks objects across frames with EMA smoothing and DBSCAN clustering
- Publishes positions and diagnostics to NetworkTables
- Streams live annotated video over a local web server
- Auto-selects the fastest model format for whatever hardware you're running on
- Survives crashes via a watchdog that restarts the pipeline automatically

---

## Hardware

**Recommended deploy target:** Orange Pi 5 / 5 Pro (RK3588 NPU)

Also runs on:
- Any aarch64 Linux board (Raspberry Pi, Jetson) - uses TFLite
- x86 Linux - uses ONNX or OpenVINO
- macOS (Apple Silicon) - uses CoreML
- Windows - uses ONNX

---

## Quick Start - Flash and Go

This is the zero-config path. Flash the pre-built image, plug in ethernet, power on.

### 1. Download the image

Go to the [Releases](../../releases) page and download the latest `orangepi.img` file.

### 2. Flash it

Use [balenaEtcher](https://etcher.balena.io/) or `dd`:

```bash
sudo dd if=orangepi.img of=/dev/sdX bs=4M status=progress
```

### 3. Power on with ethernet connected

The board will boot, connect to the internet, clone this repo, install all dependencies, and start the vision pipeline automatically. Watch progress over serial or SSH:

```bash
journalctl -u first-boot -f
```

Once complete, the pipeline runs as a systemd service on every boot:

```bash
journalctl -u iSpy -f   # live logs
systemctl restart iSpy  # restart
systemctl stop iSpy     # stop
```

### 4. Configure

Edit `/etc/iSpy/config.json` on the board, then restart the service. See [Configuration](#configuration) below.

---

## Manual Install (No Image)

If you have a board already running Ubuntu/Debian:

```bash
git clone https://github.com/aidan-j532/iSpy-FRC
cd iSpy-FRC
chmod +x install.sh
./install.sh
```

Or run the full provisioner in one line:

```bash
curl -fsSL https://raw.githubusercontent.com/aidan-j532/iSpy-FRC/main/Image/provision.sh | bash
```

---

## Dev Setup (x86 / Laptop)

Use this to train models, convert formats, or modify the pipeline on a regular computer.

```bash
git clone https://github.com/aidan-j532/iSpy-FRC
cd iSpy-FRC
chmod +x install.sh
./install.sh dev
```

Run the pipeline locally (uses a webcam or image file):

```bash
iSpy-run
```

Run the boot sequence (downloads a default model, sets up service):

```bash
iSpy-boot
```

---

## Configuration

The config file lives at `Config/config.json` (or `/etc/iSpy/config.json` on deployed boards).

```json
{
    "vision_model": {
        "file_path": "YoloModels/pytorch/your_model.pt",
        "input_size": [640, 640],
        "min_conf": 0.5,
        "margin": 10
    },
    "unit": "meter",
    "auto_opt": true,
    "debug_mode": false,
    "plugins": {
        "trackers": {
            "object_tracker": {
                "distance_threshold": 0.5,
                "stale_threshold": 1.0
            },
            "path_planner": {
                "epsilon": 0.3,
                "min_samples": 3
            }
        },
        "utilities": {
            "network_table_handler": {
                "network_tables_ip": "10.TE.AM.2"
            },
            "video_recorder": {
                "record_dir": "VideoRecordings"
            }
        },
        "frame_processors": {}
    },
    "camera_configs": {
        "front_cam": {
            "name": "front_cam",
            "source": "/dev/video0",
            "pipeline": "object_detection",
            "fps_cap": 30,
            "yaw": 0,
            "pitch": 0,
            "height": 0.5,
            "x": 0.2,
            "y": 0,
            "subsystem": "field",
            "calibration": {
                "distance": 1.0,
                "game_piece_size": 3.5,
                "size": 120,
                "fov": 70
            }
        }
    }
}
```

### Key settings

| Key | What it does |
|-----|-------------|
| `auto_opt` | Automatically converts your `.pt` model to the fastest format for the current hardware |
| `unit` | Output coordinate unit: `frc` (meters - WPILib convention), `meter`, `inch`, `foot`, `centimeter` |
| `debug_mode` | Draws bounding boxes and FPS on the video feed |
| `margin` | Pixels to ignore at image edges (filters partial detections) |

### Add-ons (plugins)

Add-ons are enabled by their presence: `plugins.<type>.<name>` being present in
the config (even as `{}`) enables it - remove the entry to disable it. No
separate `enabled` flag exists. Each add-on's settings live inside its own
entry; missing settings fall back to defaults declared by the add-on's schema.

| Add-on | Default settings | What it does |
|--------|------------------|-------------|
| `trackers.object_tracker` | `distance_threshold: 0.5`, `stale_threshold: 1.0` | Stitches detections into a single object per camera; drops stale detections |
| `trackers.path_planner` | `epsilon: 0.3`, `min_samples: 3` | DBSCAN clustering of tracked objects into game-piece piles |
| `utilities.network_table_handler` | `network_tables_ip: "10.0.0.2"` | Publishes vision output to the robot over NetworkTables |
| `utilities.rollback` | `data_dir: "VideoRecordings"`, `fps: 30.0`, `max_queue: 300`, `downsample: 1` | Ring-buffer video recorder for reviewing past footage |

Health reporting is **not** an add-on: it is the always-on core web module
(`iSpy/web/modules/health.py`, `/health` + `/api/health`). Tune its stale-frame
threshold with the top-level config key `health_stale_threshold` (Settings →
Advanced).

All add-ons (enabled state, settings) are managed from the **Add-ons page** in
the web UI.

### Camera calibration

To get accurate distances, measure these values with your actual camera and game piece:

| Calibration field | How to measure |
|---|---|
| `game_piece_size` | Diameter or height of the game piece in inches |
| `distance` | Distance from camera to the game piece during calibration (same unit as game piece) |
| `size` | Pixel height of the game piece bounding box at that calibration distance |
| `fov` | Camera field of view in degrees (check your camera's spec sheet) |

---

## Model Setup

Models live in `YoloModels/<format>/`, one folder per backend format. A `[size]`
subfolder (e.g. `pytorch/nano/`) is optional and useful when you keep many
models of one format. Actual build-tree layout:

```
YoloModels/
  pytorch/_default_detect.pt            # stock defaults downloaded on first use
  pytorch/_default_pose.pt
  pytorch/_default_pose_metadata.yaml
  pytorch/world/yolov8s-worldv2.pt
  onnx/depth_anything_v2_small.onnx
  openvino/<model>_openvino_model/      # IR: <name>.xml + <name>.bin + metadata.yaml
  tflite/
  engine/                               # TensorRT .engine
  coreml/
  huggingface/                          # HuggingFace model cache (depth_anything, ...)
```

Converted artifacts and calibration data automatically land in these folders and
are described by a small YAML **metadata sidecar** next to the file
(`_default_pose_metadata.yaml`) - iSpy reads the output format, task, class
names and input size from that sidecar instead of guessing from the extension.

With `auto_opt: true`, iSpy converts your `.pt` model at boot time and caches
the result. Supported formats: `rknn`, `onnx`, `openvino`, `tflite`, `coreml`,
`engine` (TensorRT), `hef` (Hailo).

To convert manually on a dev machine:

```python
from iSpy.vision.optimizer import convert_model

convert_model("my_model.pt", target_format="rknn", input_size=(640, 640))
```

---

## Web Interface

When the pipeline is running, open a browser and go to `http://<board-ip>:5000`.

| Endpoint | What you get |
|---|---|
| `/` | Live annotated camera feed |
| `/health` | System health dashboard (browser) or JSON (API) |
| `/api/cameras` | List of connected cameras |
| `/api/camera/<name>/feed` | Stream for a specific camera |
| `/api/camera/<name>/settings` | GET or POST camera settings |

The health endpoint returns `200 OK` when everything is healthy, `503` when degraded. Useful for robot code that wants to know if vision is alive.

---

## NetworkTables Output

iSpy publishes to the `VisionData` table with a single universal detection
schema: every pipeline (object_detection, april_tag, qr_code, optical_flow,
depth, ...) flattens to the same per-detection dict keys, handed to the robot
as one compact JSON string so the format works across all pipelines:

```json
[{"name":"object","vis_type":"box","x":1.2,"y":0.4,"confidence":0.93}, ...]
```

| Default topic | Type | Description |
|---|---|---|
| `VisionData/vision_data` | string (JSON) | Array of detected object entries (field x, y in `unit`) |
| `VisionData/fps` | double | Current pipeline FPS |
| `VisionData/num_detections` | double | Number of active tracked objects |
| `VisionData/camera_lag` | double | Camera frame age in seconds |

The legacy `FuelStruct[]` struct-array form is still available through the
add-on's `data_type` dropdown (`struct[]`) on the Add-ons page for
back-compatibility; the JSON string is the default and is pipeline-agnostic.
Robot-side consumers that hardcoded the old `FuelStruct` schema must switch to
parsing the JSON topic.

---

## Plugin System

iSpy uses a plugin architecture. Drop a file into the right folder and it loads automatically.

### Custom tracker

```python
# iSpy/plugins/trackers/my_tracker.py
from iSpy.plugins.bases import TrackerBase

class MyTracker(TrackerBase):
    plugin_name = "my_tracker"

    @classmethod
    def config_schema(cls):
        return {
            "merge_radius": {
                "type": "number",
                "label": "Merge Radius",
                "default": 0.5,
                "hint": "Radius to merge nearby detections",
            },
        }

    def __init__(self, context: dict):
        super().__init__(context)
        # context["config"] is your OWN settings (iSpyAddonConfig view,
        # schema defaults merged in); context["global_config"],
        # context["cameras"], context["flask_app"], context["vision_instance"]
        # give access to the rest of iSpy.
        self.merge_radius = self.config.get("merge_radius", 0.5)

    def update(self, detections, robot_x, robot_y, robot_yaw):
        # filter, smooth, or modify detections here
        return detections
```

Then add `"my_tracker": {}` (or with your settings) to `plugins.trackers` in
your config - its presence enables it:

### Custom utility (Flask route, side effect, etc.)

```python
# iSpy/plugins/utilities/my_utility.py
from iSpy.plugins.bases import UtilityBase

class MyUtility(UtilityBase):
    plugin_name = "my_utility"

    def __init__(self, context: dict):
        flask_app = context["flask_app"]
        if flask_app:
            flask_app.add_url_rule("/my-route", "my_route", self._route)

    def update(self, frame_data: dict):
        # called every loop with fps, detection_count, frame, detections, etc.
        pass

    def _route(self):
        return "hello from my plugin"
```

Then add `"my_utility": {}` to `plugins.utilities` in your config - its
presence enables it. Settings declared in `config_schema()` are editable from
the Add-ons page in the web UI and arrive in `context["config"]`.

---

## Validation

Run before deploying to catch config or model issues:

```bash
# Unit tests
python -m iSpy.validations.ez

# Check model organization
python -m iSpy.validations.model_validator check-org

# Full system validation (tests + model + config checks)
python -c "from iSpy.validations.validate_system import validate_system; validate_system()"

# Config recommendations
python -c "from iSpy.validations.validate_system import get_recommendations; print(get_recommendations())"
```

---

## Architecture

```
game_loop.py
  └── iSpy
        ├── ObjectDetectionCamera (per camera)
        │     ├── Camera (threaded frame reader)
        │     └── GenericYolo (RKNN / ONNX / TFLite / Ultralytics)
        ├── MultipleCameraHandler (merges multi-camera detections)
        ├── Trackers (object_tracker -> path_planner -> your plugins)
        ├── Utilities (rollback, network_handler, your plugins)
        └── CameraApp (Flask web server)
```

The main loop runs at whatever FPS the camera and model allow. On an Orange Pi 5 with a nano RKNN model, expect 30–60 FPS.

Benchmarking I'VE tested with default models (pip install iSpy-frc, iSpy-boot -f, iSpy-run):
| Pose (Yolov8 Nano)          | Detect (Yolov8 Nano)             | Detect (Yolov26 Nano)       |
|-----------------------------|----------------------------------|-----------------------------|
| Orange Pi (RK3588): ~30 fps | Orange Pi (RK3588):   Not Tested | Orange Pi (RK3588): ~60 fps |
| Colab (2 T4's):     ~180 fps| Colab (2 T4's):       Not Tested | Colab (2 T4's):  Not tested |
Colab v5e1 Yolov8 nano pose 47 fps, detect is 48, and fuel is 

---

## Known Limitations & Security

- **Field-network assumptions.** iSpy is designed for an isolated FRC field
  LAN (robot + coprocessor + DS). It is not a hardened multi-tenant server and
  serves HTTP (no TLS).
- **Admin routes are local-only by default.** The mutating admin endpoints
  (plugin upload/create/delete, service control in `service_daemon.py`) reject
  non-local requests unless you set `ISPY_ADMIN_TOKEN` environment on the board;
  when set, remote callers must present the matching `X-iSpy-Admin-Token` header
  (see `require_local_or_token` in `iSpy/web/Backend/PluginStatus.py`). Live
  feeds and health endpoints are open - do not expose port 5000 to an
  untrusted network (i.e. the internet).
- **Camera pressure.** Opening cameras is bounded: at most 4 concurrent open
  attempts at once; a wedged driver cannot spawn unbounded retry threads.
- **NetworkTables startup is non-blocking.** The handler connects in the
  background and publishes JSON to `VisionData` once connected; a missing
  robot/table never stalls the vision loop.
- **Model resolution is graceful.** A missing/truncated model file falls back
  to "no detection" instead of crashing the pipeline.
- **RKNN conversion needs a known target platform.** When the SoC cannot be
  detected the converter prints a loud warning, defaults to `rk3588`, and
  stamps `target_platform` / `target_platform_detected` (plus `warning`) in the
  artifact metadata sidecar. Override with `ISPY_RKNN_TARGET_PLATFORM=<chip>`.
- **RKNN wheels are distributed via GitHub releases** (not PyPI) because the
  hardware-specific wheels are large. The converter downloads the matching
  `rknn_toolkit2` wheel for your arch/Python from the `RKNN_Wheels` release;
  `rknn_toolkit_lite2` wheels come from a local `rknn_wheels/` folder if present.
- **Tested hardware coverage.** The team tests on Orange Pi 5/5 Pro and x86
  Linux/Windows. Other boards (Jetson, macOS) work from the same code paths but
  see less field time.

---

## License

**PolyForm Noncommercial License 1.0.0** (source-available) - see
[LICENSE](LICENSE).

iSpy's own code is licensed under the PolyForm Noncommercial License 1.0.0: it
is free for noncommercial use (which covers student teams, FRC use, and hobby
projects). Commercial use requires a separate license from the authors. See
[LICENSE](LICENSE) for the full terms.

The stock default checkpoints (`_default_detect.pt`, `_default_pose.pt`) are
NOT distributed with iSpy - they are downloaded on first use from Ultralytics'
own release assets and remain under their original AGPL-3.0 terms. If your team
distributes them separately you must meet the AGPL-3.0 obligations for those
model files or retrain/replace them; see
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). The default
`_default_v26_detect_for_fuel.pt` model is iSpy's own and is licensed under the
same PolyForm Noncommercial terms as the rest of iSpy.