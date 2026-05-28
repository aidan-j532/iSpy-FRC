# VisionCore

> FRC vision pipeline for object detection and field mapping - runs on Orange Pi with Rockchip NPU, supports RKNN, ONNX, OpenVINO, TFLite, and CoreML backends.

---

## What It Does

VisionCore is a plug-and-play computer vision system for FRC robots. You point a camera at the field, it detects game pieces, converts pixel positions into field-relative coordinates, and sends them to your robot over NetworkTables - all automatically.

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
journalctl -u visioncore -f   # live logs
systemctl restart visioncore  # restart
systemctl stop visioncore     # stop
```

### 4. Configure

Edit `/etc/visioncore/config.json` on the board, then restart the service. See [Configuration](#configuration) below.

---

## Manual Install (No Image)

If you have a board already running Ubuntu/Debian:

```bash
git clone https://github.com/aidan-j532/VisionCore-Deploy
cd VisionCore-Deploy
chmod +x install-deploy.sh
./install-deploy.sh
```

Or run the full provisioner in one line:

```bash
curl -fsSL https://raw.githubusercontent.com/aidan-j532/VisionCore-Deploy/main/Image/provision.sh | bash
```

---

## Dev Setup (x86 / Laptop)

Use this to train models, convert formats, or modify the pipeline on a regular computer.

```bash
git clone https://github.com/aidan-j532/VisionCore-Deploy
cd VisionCore-Deploy
chmod +x install-dev.sh
./install-dev.sh
```

Run the pipeline locally (uses a webcam or image file):

```bash
visioncore-run
```

Run the boot sequence (downloads a default model, sets up service):

```bash
visioncore-boot
```

---

## Configuration

The config file lives at `Config/config.json` (or `/etc/visioncore/config.json` on deployed boards).

```json
{
    "vision_model": {
        "file_path": "YoloModels/pytorch/nano/your_model.pt",
        "input_size": [640, 640],
        "min_conf": 0.5,
        "margin": 10
    },
    "unit": "meter",
    "auto_opt": true,
    "debug_mode": false,
    "use_network_tables": true,
    "network_tables_ip": "10.TE.AM.2",
    "stale_threshold": 1.0,
    "distance_threshold": 0.5,
    "dbscan": {
        "elipson": 0.3,
        "min_samples": 3
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
| `unit` | Output coordinate unit: `meter`, `inch`, `foot`, `centimeter` |
| `network_tables_ip` | Robot IP - typically `10.TE.AM.2` where TEAM is your 4-digit team number |
| `stale_threshold` | Seconds before a detection is considered stale (default `1.0`) |
| `distance_threshold` | Merge radius for the object tracker in your chosen unit (default `0.5`) |
| `debug_mode` | Draws bounding boxes and FPS on the video feed |
| `margin` | Pixels to ignore at image edges (filters partial detections) |

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

Models live in `YoloModels/[format]/[size]/`. Example structure:

```
YoloModels/
  pytorch/nano/my_model.pt
  rknn/nano/my_model.rknn
  openvino/nano/my_model_openvino_model/
```

With `auto_opt: true`, VisionCore converts your `.pt` model at boot time and caches the result. Supported formats: `rknn`, `onnx`, `openvino`, `tflite`, `coreml`.

To convert manually on a dev machine:

```python
from VisionCore.utilities.laptop.AllInOneConvert import convert_model

convert_model("my_model.pt", format="rknn", task="detect")
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

VisionCore publishes to the `VisionData` table:

| Key | Type | Description |
|---|---|---|
| `VisionData/vision_data` | `FuelStruct[]` | Array of detected object positions (x, y) in field coordinates |
| `VisionData/fps` | `double` | Current pipeline FPS |
| `VisionData/num_detections` | `double` | Number of active tracked objects |
| `VisionData/camera_lag` | `double` | Camera frame age in seconds |
| `VisionData/timestamp_ms` | `double` | Unix timestamp of last update |

---

## Plugin System

VisionCore uses a plugin architecture. Drop a file into the right folder and it loads automatically.

### Custom tracker

```python
# VisionCore/plugins/trackers/my_tracker.py
from VisionCore.plugins.bases import TrackerBase

class MyTracker(TrackerBase):
    plugin_name = "my_tracker"

    def update(self, fuel_list, robot_x, robot_y, robot_yaw):
        # filter, smooth, or modify detections here
        return fuel_list
```

Then add `"my_tracker"` to `plugins.trackers` in your config.

### Custom utility (Flask route, side effect, etc.)

```python
# VisionCore/plugins/utilities/my_utility.py
from VisionCore.plugins.bases import UtilityBase

class MyUtility(UtilityBase):
    plugin_name = "my_utility"

    def __init__(self, context: dict):
        flask_app = context["flask_app"]
        if flask_app:
            flask_app.add_url_rule("/my-route", "my_route", self._route)

    def update(self, frame_data: dict):
        # called every loop with fps, detections, frame, fuel_list, etc.
        pass

    def _route(self):
        return "hello from my plugin"
```

Then add `"my_utility"` to `plugins.utilities` in your config.

---

## Validation

Run before deploying to catch config or model issues:

```bash
# Unit tests
python -m VisionCore.validations.ez

# Check model organization
python -m VisionCore.validations.model_validator check-org

# Full system validation (tests + model + config checks)
python -c "from VisionCore.validations.validate_system import validate_system; validate_system()"

# Config recommendations
python -c "from VisionCore.validations.validate_system import get_recommendations; print(get_recommendations())"
```

---

## Architecture

```
game_loop.py
  └── VisionCore
        ├── ObjectDetectionCamera (per camera)
        │     ├── Camera (threaded frame reader)
        │     └── GenericYolo (RKNN / ONNX / TFLite / Ultralytics)
        ├── MultipleCameraHandler (merges multi-camera detections)
        ├── Trackers (object_tracker → path_planner → your plugins)
        ├── Utilities (health_reporter, video_recorder, network_handler, your plugins)
        └── CameraApp (Flask web server)
```

The main loop runs at whatever FPS the camera and model allow. On an Orange Pi 5 with a nano RKNN model, expect 30–60 FPS.

Benchmarking I'VE tested with default models (pip install visioncore-frc, visioncore-boot -f, visioncore-run):
| Pose (Yolov8 Nano)          | Detect (Yolov8 Nano)          | Detect (Yolov26 Nano)       |
|-----------------------------|-------------------------------|-----------------------------|
| Orange Pi (RK3588): ~30 fps | Orange Pi (RK3588): ~32 fps   | Orange Pi (RK3588): ~60 fps |
| Colab (2 T4's):     ~200 fps| Colab (2 T4's):     ~203 fps  | Colab (2 T4's): Not tested  |

---

## License

GPL-3.0 - see [LICENSE](LICENSE).