# iSpy-FRC: A Free Vision Pipeline for FRC — Full Breakdown

Hey everyone,

A few people have been asking about iSpy so I figured I'd write up a proper post explaining what it is, how it works, and where it stands compared to what most teams are using. This is going to be long but I wanted to be thorough since there's a lot to cover.

---

## What Is iSpy?

iSpy is a computer vision pipeline for FRC robots. It's free for noncommercial use (source-available under the PolyForm Noncommercial 1.0.0 license), and it runs on basically any hardware you want — Orange Pi, Raspberry Pi, Jetson, x86 laptops, Macs, Windows, whatever. It takes a camera feed, runs YOLO object detection on it, converts the pixel detections into real-world field coordinates, and publishes them to NetworkTables so your robot code can use them.

The short version: you point a camera at the field, iSpy tells your robot where the game pieces are.

But that's underselling it. Here's the full picture.

---

## What It Actually Does

### Hardware Agnostic

This is probably the biggest differentiator. iSpy doesn't care what board you're running. It detects your hardware at boot and picks the fastest inference backend automatically:

- **Orange Pi 5 / 5 Pro** (RK3588) — uses RKNN, runs directly on the NPU. This is the recommended target. ~60fps with YOLOv26 Nano, ~30fps with YOLOv8 Nano pose.
- **Raspberry Pi / any aarch64 Linux** — uses TFLite
- **NVIDIA Jetson** — uses TensorRT
- **x86 Linux** — uses ONNX or OpenVINO
- **macOS (Apple Silicon)** — uses CoreML
- **Windows** — uses ONNX
- **Coral TPU / Hailo** — supported as well

So you're not locked into a specific $350 box with a specific chip. You train a `.pt` model in PyTorch like normal, point iSpy at it, and it handles the rest. At boot it auto-converts your model to whatever format runs fastest on your hardware — RKNN, ONNX, OpenVINO, TFLite, CoreML, TensorRT, Hailo HEF — and caches the result so it only converts once.

The conversion logic lives in `iSpy/vision/optimizer.py` and the auto-detection logic is in `iSpy/config/AutoOpt.py`. If you want to convert manually:

```python
from iSpy.vision.optimizer import convert_model

convert_model("my_model.pt", target_format="rknn", input_size=(640, 640))
```

But you shouldn't need to — with `auto_opt: true` in the config (which is the default), it just works at boot time.

### Six Vision Pipelines

iSpy isn't just an object detector. It has six built-in pipeline types that you can assign per-camera:

1. **Object Detection** — the main one. Runs any YOLO model for bounding box detection. Outputs field-relative x/y coordinates.
2. **Pose Estimation** — YOLOv8/v11 pose models with PnP solve. Outputs 6-DOF position and rotation of detected objects. This is something neither Limelight nor PhotonVision does.
3. **AprilTag Detection** — detects AprilTags, solves their pose via PnP, gives you exact position and orientation.
4. **QR Code Detection** — same as AprilTag but for arbitrary QR codes.
5. **Optical Flow** — dense optical flow for velocity estimation.
6. **Depth Estimation** — runs Depth Anything V2 for monocular depth. No stereo camera needed.
7. **YOLO-World** — zero-shot detection. You type what you want to detect in a text prompt and it finds it without any training.

Each camera in the config gets assigned one of these pipelines. You can run different pipelines on different cameras simultaneously.

### Universal Output Schema

Every pipeline — whether it's object detection, AprilTag, pose, whatever — flattens detections to the same JSON schema:

```json
[
  {
    "id": 1,
    "name": "coral",
    "confidence": 0.93,
    "x": 1.2,
    "y": 0.4,
    "z": 0.0,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "depth_source": "pnp",
    "vis_type": "planar",
    "vis_meta": {"tag_id": 42, "size": 0.152},
    "keypoints_3d": null,
    "ray_origin": [0.2, 0.0, 0.5],
    "ray_direction": [0.1, 0.0, -0.99]
  }
]
```

This gets published to NetworkTables as a single JSON string on `VisionData/vision_data`. Your robot code parses one format regardless of what pipeline is running. The legacy `FuelStruct[]` struct array format is still available as an opt-in through the web UI for teams that haven't migrated yet.

Other NT topics published:
- `VisionData/fps` — current pipeline framerate
- `VisionData/num_detections` — number of tracked objects
- `VisionData/camera_lag` — camera frame age in seconds

The NetworkTables connect is non-blocking now, so if your robot isn't on the network the vision loop never stalls.

### Multi-Camera Support

You can run multiple cameras simultaneously. Each one gets its own pipeline thread. When multiple cameras detect the same object, iSpy merges detections using ray-triangulation — not just "averaging the coordinates" but actual geometric intersection of camera rays in 3D space.

This is configured per-camera in `config.json`:

```json
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
        "calibration": {
            "distance": 1.0,
            "game_piece_size": 3.5,
            "size": 120,
            "fov": 70
        }
    },
    "rear_cam": {
        "name": "rear_cam",
        "source": "/dev/video2",
        "pipeline": "april_tag",
        "fps_cap": 30,
        "yaw": 180,
        "height": 0.5,
        "x": -0.2,
        "y": 0,
        "calibration": {
            "distance": 1.0,
            "game_piece_size": 3.5,
            "size": 120,
            "fov": 70
        }
    }
}
```

### Object Tracking

iSpy has a plugin-based tracker system. The built-in tracker (`ObjectTracker`) stitches detections across frames using EMA (exponential moving average) smoothing, so you get stable, filtered positions instead of jittery per-frame noise. It also handles stale detection cleanup — if an object hasn't been seen for a configurable duration, it gets dropped.

There's also a `PathPlanner` plugin that uses DBSCAN clustering to group nearby detections into "piles" of game pieces. Useful if you need to know "there's a cluster of 3 coral in this zone" rather than individual positions.

### Camera Calibration

For accurate field coordinates, iSpy needs calibration. The calibration wizard lives in the web UI and supports two methods:

1. **Focal Length / FOV** — measure the pixel height of a known-size game piece at a known distance. iSpy calculates the focal length from that. Quick and dirty.
2. **ChArUco Board** — print a ChArUco board, hold it in front of the camera, iSpy captures frames and solves for the full camera matrix and distortion coefficients. More accurate.

Each pipeline declares which calibration it needs. Object detection and AprilTag need at least focal length or ChArUco intrinsics. YOLO-World and Depth Anything don't need calibration at all — they just give you the raw frame or depth map without field coordinates.

If calibration isn't configured, the pipeline degrades gracefully — it just outputs an empty detection list and the raw frame instead of crashing or giving you garbage coordinates.

### Web Dashboard

When the pipeline is running, there's a Flask web server on port 5000. Open `http://<board-ip>:5000` in a browser and you get:

| Page | What It Does |
|------|-------------|
| **Dashboard** | Live annotated camera feed, FPS, detection count |
| **Cameras** | Manage cameras, change settings, run calibration wizard |
| **Models** | View loaded models, convert to different formats, benchmark |
| **Datasets** | Manage training datasets, view captures |
| **3D Viewer** | Interactive 3D visualization of detections on a virtual field |
| **Metrics** | Performance graphs, latency, throughput |
| **Logs** | Live log viewer |
| **Health** | System health dashboard — CPU, memory, camera status, pipeline status |
| **Recommendations** | Auto-generated config suggestions based on your hardware |
| **Onboarding** | Setup wizard for first-time configuration |
| **Settings** | Global config editor |
| **Add-ons** | Enable/disable and configure plugins from the browser |

The health endpoint at `/health` returns `200 OK` when everything is healthy, `503` when degraded. Useful for robot-side health checks.

The web UI also has a plugin management page — you can upload, create, and delete plugins directly from the browser (gated behind an admin token for security).

### Plugin Architecture

iSpy uses a plugin system. You drop a Python file into the right folder and it loads automatically at startup. Three plugin types:

**Trackers** — process detections after the pipeline runs:

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
        self.merge_radius = self.config.get("merge_radius", 0.5)

    def update(self, detections, robot_x, robot_y, robot_yaw):
        # filter, smooth, or modify detections here
        return detections
```

**Utilities** — run side effects every tick (NetworkTables publishing, video recording, custom Flask routes, etc.):

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
        pass

    def _route(self):
        return "hello from my plugin"
```

**Frame Processors** — modify the raw frame before it reaches the pipeline (e.g., color filtering, augmentation).

Plugins are enabled by their presence in config — if `plugins.trackers.my_tracker` exists (even as `{}`), it's enabled. Remove the entry to disable. No separate `enabled` flag. Settings are editable from the Add-ons page in the web UI.

Built-in plugins:
- `ObjectTracker` — EMA smoothing and stale detection cleanup
- `PathPlanner` — DBSCAN clustering of game piece positions
- `NetworkTableHandler` — publishes to NetworkTables (configurable topics, data types, custom publish entries)
- `RollBack` — ring-buffer video recorder, stores the last N seconds of footage for reviewing what happened

### Boot and Service Management

There's a full boot system (`iSpy-boot` / `iSpy-run` commands):

```bash
# Install
git clone https://github.com/aidan-j532/iSpy-FRC
cd iSpy-FRC
chmod +x install.sh
./install.sh          # full install on a board
./install.sh dev      # dev install on x86/laptop
```

After install:
```bash
iSpy-boot     # downloads default model, sets up systemd service, runs first-boot setup
iSpy-run      # starts the pipeline (uses webcam or config-specified camera)
iSpy-bench    # benchmarks models on current hardware
iSpy-stats    # compares model performance
```

When deployed as a systemd service:
```bash
journalctl -u iSpy -f      # live logs
systemctl restart iSpy      # restart
systemctl stop iSpy         # stop
```

There's also a flash image for Orange Pi. Download from releases, flash with balenaEtcher or `dd`, plug in ethernet, power on. The board boots, connects to WiFi/ethernet, clones the repo, installs everything, and starts the pipeline automatically. You can watch progress with:

```bash
journalctl -u first-boot -f
```

The board advertises itself via mDNS as `ispy-<6 hex characters>.local` so two boards on the same network don't collide. You can override this with `ISPY_MDNS_HOSTNAME`.

### Hardware Auto-Detection and Optimization

The `AutoOpt` module probes your system at startup:

- Checks for RK3588 NPU via `/sys/kernel/rknpu/version`
- Checks for Jetson via `/etc/nv_tegra_release` or device tree
- Checks for Coral TPU via `lsusb`
- Checks for NVIDIA GPU via `nvidia-smi`
- Falls back to CPU ONNX/OpenVINO/TFLite

Then `optimizer.py` converts your `.pt` model to the best format. The conversion includes metadata sidecars (`_metadata.yaml`) that describe the output format, task type, class names, and input size. iSpy reads these sidecars instead of guessing from the file extension — which means you can have multiple models of different formats and they all get handled correctly.

### Safety and Reliability

- **Watchdog**: if the pipeline crashes, a watchdog process restarts it automatically
- **Bounded camera opens**: at most 4 concurrent camera open attempts — a wedged USB driver can't spawn unbounded retry threads
- **Per-camera isolation**: each camera boots inside its own try/except — one bad camera config can't take down the whole pipeline
- **Graceful degradation**: missing model files, bad calibration, disconnected robot — all handled without crashing. The pipeline keeps running with reduced functionality.
- **Admin token gating**: mutating web routes (plugin upload, service control) reject non-local requests unless `ISPY_ADMIN_TOKEN` is set on the board
- **Atomic config saves**: settings are saved via temp-file + rename, so a crash mid-write can't corrupt `config.json`

### Unit System

iSpy supports multiple output coordinate units:
- `frc` — meters, WPILib convention (default)
- `meter`
- `inch`
- `foot`
- `centimeter`

Change the `unit` field in config and all detections are published in that unit.

---

## How It Compares

| | **Limelight 4** | **PhotonVision** | **iSpy** |
|---|---|---|---|
| **Price** | $350+ hardware + camera | Free (software only) | Free (software only) |
| **Hardware** | All-in-one (Hailo-8 NPU + camera) | Orange Pi / RPi / Jetson | Orange Pi / RPi / Jetson / x86 / Mac / Windows |
| **Setup** | Plug and play (pre-flashed) | Manual install | Flash image OR manual install |
| **AI Backend** | Hailo-8 HEF (fixed) | Ultralytics YOLO / OpenCV | RKNN, ONNX, OpenVINO, TFLite, CoreML, TensorRT, Hailo, Coral |
| **Model Support** | Limelight-trained only | Any YOLO `.pt` | Any YOLO `.pt` + auto-convert to 8+ formats |
| **Field Coordinates** | Yes (MegaTag2) | Yes (triangulation) | Yes (ground-plane ray + multi-cam triangulation) |
| **Pose Estimation** | No | Limited | Yes (PnP with keypoints) |
| **Depth Estimation** | No | No | Yes (Depth Anything V2) |
| **Web Dashboard** | Basic (crosshair, tuning) | Moderate (camera view, tuning) | 12+ modules (cameras, models, datasets, 3D viewer, metrics, health, logs, etc.) |
| **AprilTag Tracking** | Yes | Yes | Yes |
| **Multi-Camera** | Yes (hardware sync) | Limited | Yes (threaded, ray-triangulation merge) |
| **Open Source** | No | Yes (GPLv3) | Source-available (PolyForm Noncommercial 1.0.0) |
| **Plugin System** | No | No | Yes (trackers, utilities, frame processors) |
| **Zero-Shot Detection** | No | No | Yes (YOLO-World) |
| **Pose PnP** | No | No | Yes |
| **Model Benchmarking** | No | No | Yes (`iSpy-bench`) |

### Where iSpy Wins

- **Hardware agnostic** — run on literally anything, not just one specific NPU
- **Deepest model pipeline** — auto-detect, auto-convert, auto-benchmark, background optimization
- **Most feature-rich web dashboard** — 12+ modules vs 1-2 for competitors
- **Multi-camera ray triangulation** — first-class, geometric, not just averaging
- **Pose estimation via PnP** — no competitor does this
- **Plugin architecture** — extensible without touching core
- **YOLO-World zero-shot** — point camera at anything, type what you want to detect
- **Depth Anything V2** — monocular depth estimation, unique capability

### Where iSpy Loses

- **Not plug-and-play** — even with the flash image, you're spending some time on setup. Limelight is "buy box, plug in, done"
- **Community size** — PhotonVision has more teams using it, more docs, more YouTube videos, more forum posts
- **Brand recognition** — "Use a Limelight" is a verb in FRC. "Use iSpy" doesn't exist yet
- **Documentation** — this post is probably the most thorough single document about iSpy and that's a problem. Competitors have full tutorial series, wiring guides, code examples
- **Robot-side library** — Limelight has `LimelightHelpers.java`. iSpy publishes raw JSON and you figure out parsing

---

## Getting Started

### Flash Image (Recommended)

1. Download the latest `orangepi.img` from [Releases](../../releases)
2. Flash to SD card with [balenaEtcher](https://etcher.balena.io/) or `dd`
3. Plug in ethernet, power on
4. Wait for first-boot to finish: `journalctl -u first-boot -f`
5. Open `http://<board-ip>:5000` in a browser

### Manual Install

```bash
git clone https://github.com/aidan-j532/iSpy-FRC
cd iSpy-FRC
chmod +x install.sh
./install.sh
```

Or one-liner provisioner:
```bash
curl -fsSL https://raw.githubusercontent.com/aidan-j532/iSpy-FRC/main/Image/provision.sh | bash
```

### Dev Setup (Laptop)

```bash
git clone https://github.com/aidan-j532/iSpy-FRC
cd iSpy-FRC
chmod +x install.sh
./install.sh dev
iSpy-run
```

This uses your webcam or a specified image file. Good for testing models, tweaking config, developing plugins.

### Config

The config file lives at `Config/config.json` (or `/etc/iSpy/config.json` on deployed boards). Key settings:

```json
{
    "vision_model": {
        "file_path": "YoloModels/pytorch/your_model.pt",
        "input_size": [640, 640],
        "min_conf": 0.5,
        "margin": 10
    },
    "unit": "frc",
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
            "video_recorder": {}
        }
    },
    "camera_configs": {
        "front_cam": {
            "name": "front_cam",
            "source": "/dev/video0",
            "pipeline": "object_detection",
            "fps_cap": 30,
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

| Setting | What It Does |
|---------|-------------|
| `auto_opt` | Auto-convert model to fastest format for your hardware |
| `unit` | Output coordinate unit: `frc`, `meter`, `inch`, `foot`, `centimeter` |
| `debug_mode` | Draws bounding boxes and FPS on the video feed |
| `margin` | Pixels to ignore at image edges (filters partial detections) |
| `min_conf` | Minimum confidence threshold for detections |

---

## What's Coming Next

We've got a roadmap but the honest priority list is:

1. **Java/C++ robot-side library** — an `iSpyClient` with auto-discovery, typed structs, and example commands so you don't have to parse raw JSON yourself
2. **Zero-config first boot** — LED ring status indicators and a web-based setup wizard so you never need SSH
3. **Camera auto-discovery** — plug in a USB camera and it shows up in the UI automatically with recommended settings
4. **Pose PnP wizard** — step-by-step UI for keypoint annotation instead of hand-editing JSON
5. **AdvantageKit integration** — structured vision logging for teams that use AdvantageKit
6. **Alert system** — configurable alerts ("camera offline", "FPS dropped below 15") pushed to your dashboard

The long-term vision is to make iSpy the "FRC Vision Operating System" — not just a camera driver or model runner but the entire stack from "I bought a camera" to "I have field-tested vision at competition."

---

## Honest State of Things

This is a side project by a small team. It works and it's been tested on hardware, but:

- The documentation is thin. There are docs in the repo (`DOCS.d/`) but no tutorial videos, no step-by-step guides for common setups.
- We've tested primarily on Orange Pi 5 Pro and x86. Other boards (Jetson, RPi, Mac) work from the same code paths but see less field time.
- There are probably thread safety issues we haven't hit yet. The web modules and health reporter access shared state in ways that might not hold up under heavy concurrent load.
- The first-boot experience is still rough. Even with the flash image, you're staring at a terminal for a while and then SSH-ing in.
- We don't have a robot-side library yet. You'll be parsing `VisionData/vision_data` JSON yourself in Java or C++.

---

## Get Involved

- **GitHub**: [aidan-j532/iSpy-FRC](https://github.com/aidan-j532/iSpy-FRC)
- **License**: PolyForm Noncommercial 1.0.0 (source-available; see LICENSE)
- **Python**: 3.10+
- **Install**: `pip install iSpy-frc` or clone and `./install.sh`

If you're a team that takes vision seriously and you're hitting the limits of what Limelight or PhotonVision can do — or you just want pose estimation, multi-camera, or YOLO-World — give it a look. If you find bugs, open an issue. If you want to contribute, the plugin system makes it easy to add functionality without touching core code.

If you have questions about setup, hardware compatibility, or how something works, post here or open a GitHub issue. Happy to help.

---

*Tested on: Orange Pi 5 Pro (RK3588 NPU), x86 Ubuntu/Windows. YOLOv26 Nano, YOLOv8 Nano, YOLOv8 Nano Pose, YOLO-World v2, Depth Anything V2. Python 3.10-3.12.*
