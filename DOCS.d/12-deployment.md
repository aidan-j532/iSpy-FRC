# 12 — Deployment

> Installation, configuration, calibration, running, and troubleshooting iSpy in production.

---

## Installation Methods

### Method 1: Flash Image (Recommended)

Download the pre-built Orange Pi image from Releases and flash with balenaEtcher
or `dd`:

```bash
sudo dd if=orangepi.img of=/dev/sdX bs=4M status=progress
```

On first boot with ethernet:
1. Connects to internet
2. Clones this repo
3. Installs all dependencies
4. Starts vision pipeline as systemd service

Monitor with:
```bash
journalctl -u first-boot -f   # first boot progress
journalctl -u iSpy -f         # vision pipeline logs
```

### Method 2: Manual Install (Linux)

```bash
git clone https://github.com/aidan-j532/iSpy-FRC
cd iSpy-FRC
chmod +x install.sh
./install.sh
```

Or the one-liner provisioner:
```bash
curl -fsSL https://raw.githubusercontent.com/aidan-j532/iSpy-FRC/main/Image/provision.sh | bash
```

### Method 3: Dev Install (x86/Mac/Windows)

```bash
git clone https://github.com/aidan-j532/iSpy-FRC
cd iSpy-FRC
chmod +x install.sh
./install.sh dev
```

Installs with dev dependencies (torch, tensorflow, onnx, etc.).

### pip Install

```bash
pip install ispy-frc
```

---

## Running

### Quick Start

```bash
ispy-boot    # first-time setup (creates dirs, copies models, installs service)
ispy-run     # start vision pipeline
```

### Combined Boot + Run

```bash
python -m iSpy.core.ispy [-s] [-f] [-w]
```

Flags:
- `-s` / `--service`: Install as system service after boot
- `-f` / `--fresh`: Wipe all generated state and start fresh
- `-w` / `--wait`: Wait for pipeline readiness before starting vision

### As a Service (Linux)

After `ispy-boot -s`:
```bash
systemctl start iSpy      # start
systemctl stop iSpy       # stop
systemctl restart iSpy    # restart
systemctl status iSpy     # status
journalctl -u iSpy -f     # live logs
```

### Service Manager (REST API)

If running via `service_daemon.py`:
```bash
# Status
curl http://localhost:5050/service/status
# Returns: {"status": "running", "pid": 12345, "last_error": null}

# Start
curl -X POST http://localhost:5050/service/start
# Returns: {"ok": true, "pid": 12345}

# Stop
curl -X POST http://localhost:5050/service/stop
# Returns: {"ok": true}

# Restart
curl -X POST http://localhost:5050/service/restart
# Returns: {"ok": true, "pid": 12346}

# Pause
curl -X POST http://localhost:5050/service/pause
# Returns: {"ok": true}

# Resume
curl -X POST http://localhost:5050/service/resume
# Returns: {"ok": true}
```

### Pause/Resume via stdin

When running in foreground, send commands via stdin:
```
PAUSE
RESUME
SHUTDOWN
```

---

## Configuration

### Config File Locations

| Context | Path |
|---------|------|
| Development | `Config/config.json` |
| Deployed (Linux) | `/etc/iSpy/config.json` |
| Override | `ISPY_CONFIG` env variable |

### Config Search Order

1. `ISPY_CONFIG` environment variable (if set)
2. `Config/config.json` (default)
3. Any non-`config.json` file in `Config/` (named configs from web UI)

### Essential Settings

```json
{
    "unit": "frc",
    "auto_opt": true,
    "debug_mode": false,
    "app_mode": true,
    "max_fps": 0,
    "vision_model": {
        "file_path": "YoloModels/pytorch/nano/your_model.pt",
        "input_size": [640, 640],
        "min_conf": 0.5,
        "margin": 10
    },
    "plugins": {
        "trackers": {
            "object_tracker": {
                "enabled": true,
                "settings": {
                    "distance_threshold": 0.5,
                    "stale_threshold": 2.0
                }
            },
            "path_planner": {
                "enabled": false,
                "settings": {
                    "dbscan_epsilon": 0.3,
                    "dbscan_min_samples": 3
                }
            }
        },
        "utilities": {
            "network_table_handler": {
                "enabled": true,
                "settings": {
                    "network_tables_ip": "10.0.0.2"
                }
            },
            "video_recorder": {
                "enabled": false,
                "settings": {}
            },
            "rollback": {
                "enabled": false,
                "settings": {}
            }
        },
        "frame_processors": {}
    },
    "camera_configs": {
        "front_cam": {
            "name": "front_cam",
            "source": "/dev/video0",
            "pipeline": {
                "name": "object_detection",
                "settings": {
                    "vision_model": {
                        "file_path": "YoloModels/pytorch/nano/your_model.pt",
                        "input_size": [640, 640],
                        "min_conf": 0.5
                    }
                }
            },
            "fps_cap": 30,
            "yaw": 0,
            "pitch": 0,
            "height": 0.5,
            "x": 0.2,
            "y": 0,
            "subsystem": "field"
        }
    }
}
```

### Settings Key Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `unit` | string | `"frc"` | Unit system (`"frc"` = inches, `"metric"` = meters) |
| `auto_opt` | bool | `true` | Auto-convert models to hardware-optimal format |
| `debug_mode` | bool | `false` | Enable debug overlays and verbose logging |
| `app_mode` | bool | `true` | Enable web UI and REST API |
| `max_fps` | int | `0` | FPS cap (0 = unlimited) |
| `log_level` | string | `"INFO"` | Logging level |
| `log_file` | bool | `true` | Enable file logging to `Outputs/log.txt` |
| `num_gpus` | int | `0` | Number of GPUs to use |
| `device` | string | `"cpu"` | Inference device |

### Camera Config Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | required | Camera identifier |
| `source` | string/int | required | Device path (`/dev/video0`) or index (`0`) |
| `pipeline.name` | string | `"object_detection"` | Vision pipeline type |
| `pipeline.settings` | object | `{}` | Pipeline-specific settings |
| `fps_cap` | int | `30` | Maximum frames per second |
| `yaw` | float | `0` | Camera horizontal angle (degrees) |
| `pitch` | float | `0` | Camera vertical angle (degrees) |
| `height` | float | `0.5` | Camera height from ground (meters or inches) |
| `x` | float | `0` | Camera X offset from robot center |
| `y` | float | `0` | Camera Y offset from robot center |
| `subsystem` | string | `"field"` | FRC subsystem assignment |
| `csi` | bool | `false` | Whether camera uses CSI interface (Jetson) |

### Configuration via Web UI

All settings can be managed from the web interface:
- **Settings page** (`/settings`): Global config (unit, auto_opt, debug_mode, etc.)
- **Cameras page** (`/cameras`): Camera CRUD, calibration wizard
- **Add-ons page** (`/addons`): Plugin enable/disable, per-plugin settings
- **Models page** (`/models`): Upload/select YOLO models

---

## Model Setup

### Directory Structure

```
YoloModels/
  pytorch/
    nano/          # small, fast models (< 5MB)
    small/         # medium accuracy/speed
    medium/        # higher accuracy
    large/         # highest accuracy, slowest
    _default_pose.pt  # stock default, downloaded on first use
  rknn/            # converted for Rockchip NPU
  onnx/            # converted for ONNX Runtime
  openvino/        # converted for Intel
  tflite/          # converted for ARM/TFLite
  coreml/          # converted for Apple Silicon
  engine/          # converted for TensorRT
```

### Auto-Conversion

With `auto_opt: true`, iSpy automatically:
1. Detects hardware at boot (reads device tree for Rockchip, checks for CUDA, etc.)
2. Finds the best inference format for the detected hardware
3. Converts `.pt` to that format (if not already done)
4. Caches the result in the format-specific directory

### Manual Conversion

```python
from iSpy.utilities.laptop.AllInOneConvert import convert_model
convert_model("my_model.pt", format="rknn", task="detect")
```

Or via the optimizer directly:
```python
from iSpy.vision.optimizer import convert_model
artifact = convert_model(
    "YoloModels/pytorch/my_model.pt",
    target_format="onnx",
    input_size=[640, 640],
    quantize=True,
    force=False,
)
```

### Model File Naming

Models should follow the naming convention:
```
{model_name}.pt          # PyTorch source
{model_name}.onnx        # ONNX export
{model_name}.rknn        # RKNN conversion
{model_name}_metadata.yaml  # Sidecar metadata
```

The metadata sidecar contains:
```yaml
task: detect
nc: 3  # number of classes
class_names: ["cone", "cube", "note"]
input_size: [640, 640]
calibration_keywords: ["frc game piece", "frc 2025"]
```

---

## Camera Calibration

### Step-by-Step (via Web UI)

1. Open Cameras page (`/cameras`), click a camera
2. Click "Calibrate" to open the wizard

#### Step 1: Focal Length

- Measure game piece size (inches)
- Place game piece at known distance
- Enter: `game_piece_size`, `distance`, `pixel_height`, `camera_fov`
- Click Calculate

**What it computes:**
```
focal_length_pixels = (pixel_height * distance) / game_piece_size
FOV = 2 * atan(image_width / (2 * focal_length))
```

#### Step 2: ChArUco Intrinsics

- Select board size (e.g., 7x5)
- Print the generated ChArUco board
- Hold board in front of camera
- Auto-capture collects ~20 frames (diverse poses)
- Click "Compute Intrinsics" when done

**What it computes:**
- Camera matrix (3x3): focal lengths `fx`, `fy` and principal point `cx`, `cy`
- Distortion coefficients: radial (`k1`, `k2`, `k3`) and tangential (`p1`, `p2`)
- RMS reprojection error (quality metric, should be < 1.0)

**Algorithm:**
1. Detect ChArUco corners in each captured frame
2. Check for degenerate poses (all frames identical)
3. Run scipy Levenberg-Marquardt optimization
4. Fall back to OpenCV calibrateCameraCharuco if scipy fails

#### Step 3: PnP Pose

- Place ChArUco board at a known position on the field
- Capture a frame
- The system solves PnP for camera extrinsics
- Save the pose model

**What it computes:**
- Camera rotation (3x3 rotation matrix or Rodrigues vector)
- Camera translation (3D vector from robot center)
- Combined into the camera's extrinsic transform

### Via API

```bash
# Get focal length data
GET /api/cameras/<name>/calibration/focal

# Start ChArUco feed (MJPEG stream)
GET /api/cameras/<name>/calibration/charuco/feed

# Capture frame (stores for calibration)
POST /api/cameras/<name>/calibration/charuco/capture

# Get detection status
GET /api/cameras/<name>/calibration/charuco/status

# Compute intrinsics from captured frames
POST /api/cameras/<name>/calibration/charuco/intrinsics

# Get intrinsics result
GET /api/cameras/<name>/calibration/charuco/intrinsics

# PnP calibration
POST /api/cameras/<name>/calibration/pnp
```

---

## NetworkTables Output

The robot subscribes to the VisionData table:

### Published Keys

| Key | Type | Description |
|-----|------|-------------|
| `VisionData/vision_data` | `FuelStruct[]` | Array of detected positions |
| `VisionData/fps` | `double` | Pipeline FPS |
| `VisionData/num_detections` | `double` | Number of tracked objects |
| `VisionData/camera_lag` | `double` | Camera frame age in seconds |
| `VisionData/timestamp_ms` | `double` | Unix timestamp of last update |

### FuelStruct Format

Each element in the `vision_data` array contains:

| Field | Type | Description |
|-------|------|-------------|
| `x` | `double` | X position (inches from robot center) |
| `y` | `double` | Y position (inches from robot center) |
| `z` | `double` | Z position (height from ground, inches) |
| `roll` | `double` | Roll angle (degrees) |
| `pitch` | `double` | Pitch angle (degrees) |
| `yaw` | `double` | Yaw angle (degrees) |

### Robot-Side (Java)

```java
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.networktables.StructArrayPublisher;

var inst = NetworkTableInstance.getDefault();
var pub = inst.getStructArrayTopic("VisionData/vision_data", FuelStruct.class).publish();

// In your periodic loop:
var table = inst.getTable("VisionData");
var visionData = table.getEntry("vision_data").getAtomic();

// Parse FuelStruct[] from visionData
// Each struct contains x, y, z, roll, pitch, yaw
```

### NetworkTables Connection

- Default IP: `10.0.0.2` (roboRIO)
- Configured via `plugins.utilities.network_table_handler.settings.network_tables_ip`
- Uses NT4 protocol (WPILib 2024+)

---

## Web Interface

When running, open browser to `http://<board-ip>:5000`.

### Pages

| Page | URL | Purpose |
|------|-----|---------|
| Dashboard | `/` | Main dashboard, stats, metrics, FPS |
| Cameras | `/cameras` | Camera management, calibration wizard |
| Add-ons | `/addons` | Plugin enable/disable, settings |
| 3D Viewer | `/viewer3d` | 3D detection visualization |
| Models | `/models` | YOLO model library, upload, select |
| Datasets | `/datasets` | Image datasets for training/quantization |
| Settings | `/settings` | App configuration |
| Health | `/health` | System health, temperatures, memory |
| Logs | `/logs` | Log viewer with filtering |
| Metrics | `/metrics` | Performance metrics, latency graphs |

### Key API Endpoints

```bash
# Camera management
GET    /api/cameras                    # list cameras
POST   /api/cameras                    # create camera
PUT    /api/cameras/<name>             # update camera
DELETE /api/cameras/<name>             # delete camera

# Calibration
GET    /api/cameras/<name>/calibration/focal
POST   /api/cameras/<name>/calibration/focal
GET    /api/cameras/<name>/calibration/charuco/feed
POST   /api/cameras/<name>/calibration/charuco/capture
GET    /api/cameras/<name>/calibration/charuco/status
POST   /api/cameras/<name>/calibration/charuco/intrinsics
POST   /api/cameras/<name>/calibration/pnp

# Models
GET    /api/models                     # list models
POST   /api/models/upload              # upload model
DELETE /api/models/<name>              # delete model

# Plugins
GET    /api/plugins/available          # list available plugins
POST   /api/plugins/<name>/toggle      # enable/disable
POST   /api/plugins/<name>/settings    # update settings
GET    /api/plugins/<name>/source      # view source code

# Service control
GET    /service/status
POST   /service/start
POST   /service/stop
POST   /service/restart
POST   /service/pause
POST   /service/resume
```

---

## Hardware Recommendations

| Platform | Performance | Notes |
|----------|------------|-------|
| Orange Pi 5 / 5 Pro (RK3588) | 30–60 FPS | Recommended. RKNN NPU acceleration |
| Raspberry Pi 5 | 15–30 FPS | TFLite backend |
| Jetson Orin Nano | 30–60 FPS | TensorRT backend |
| x86 Desktop (Intel i7) | 20–40 FPS | ONNX/OpenVINO backend |
| MacBook (Apple Silicon) | 20–40 FPS | CoreML backend |

### Minimum Requirements

- CPU: 4 cores (ARM or x86_64)
- RAM: 2GB
- Storage: 4GB
- Network: Ethernet recommended, WiFi supported
- Camera: USB or CSI

### Recommended Configuration

- **Orange Pi 5 Pro** with RK3588 SoC
- 8GB RAM
- 64GB eMMC or SD card
- USB camera (Logitech C920/C922) or CSI camera
- Ethernet connection to roboRIO switch

---

## Troubleshooting

### Camera not found

```bash
# Linux: list video devices
v4l2-ctl --list-devices

# Check permissions
ls -la /dev/video*
sudo usermod -aG video $USER

# Test camera directly
python -c "import cv2; c = cv2.VideoCapture(0); print(c.read())"

# Check if camera is busy
sudo fuser /dev/video0
```

### Model not loading

```bash
# Check model organization
python -m iSpy.validations.model_validator check-org

# Check if auto_opt converted successfully
ls YoloModels/rknn/
ls YoloModels/onnx/

# Verify model is loadable
python -c "from ultralytics import YOLO; m = YOLO('YoloModels/pytorch/model.pt'); print(m.names)"

# Check metadata sidecar
cat YoloModels/pytorch/model_metadata.yaml
```

### NetworkTables not connecting

```bash
# Verify roboRIO IP
ping 10.0.0.2

# Check config
cat Config/config.json | grep network_tables_ip

# Test NT connection
python -c "
from networktables import NetworkTables
NetworkTables.initialize(server='10.0.0.2')
print(NetworkTables.isConnected())
"

# Check firewall
sudo ufw status
```

### Web UI not accessible

```bash
# Check if service is running
systemctl status iSpy

# Check port 5000
ss -tlnp | grep 5000

# Check logs
journalctl -u iSpy -n 50

# Test locally
curl http://localhost:5000

# Check if app_mode is enabled
cat Config/config.json | grep app_mode
```

### OpenCV GStreamer issues (Jetson/CSI)

```bash
# Check GStreamer support
python -c "import cv2; print(cv2.getBuildInformation())"

# Re-run OpenCV fix
python -m iSpy.boot.opencv_fix

# Manual fix
sudo apt-get install python3-opencv
```

### Performance issues

```bash
# Check CPU usage
top -bn1 | head -20

# Check temperature
cat /sys/class/thermal/thermal_zone*/temp

# Reduce FPS cap
# Edit Config/config.json: "max_fps": 15

# Use smaller model
# Switch to nano variant in camera config

# Check GPU/NPU utilization (RKNN)
cat /sys/kernel/debug/rknpu/load
```

### Fresh install corrupted

```bash
# Wipe everything and start fresh
ispy-boot -f

# Or manually
rm -rf YoloModels Config Outputs QuantizeDataset
ispy-boot -f
```

### Logs location

```bash
# Runtime log
tail -f Outputs/log.txt

# Systemd logs
journalctl -u iSpy -f

# First boot logs
journalctl -u ispy-first-boot -f
```
