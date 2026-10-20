# iSpy

> FRC vision pipeline for object detection and field mapping - runs on Orange Pi with Rockchip NPU and supports RKNN, ONNX, OpenVINO, TFLite, TensorRT, and CoreML backends.

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
journalctl -u ispy-first-boot -f
```

Once complete, the pipeline runs as a systemd service on every boot:

```bash
journalctl -u iSpy -f   # live logs
systemctl restart iSpy  # restart
systemctl stop iSpy     # stop
```

### 4. Configure

Edit `Config/config.json` on the board, then restart the service. See [Configuration](#configuration) below.

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
ispy-run
```

Run the boot sequence (downloads default models and sets up the service):

```bash
ispy-boot
```

---

## Configuration

The config file lives at `Config/config.json`.

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
            "FRC/object_tracker": {
                "distance_threshold": 0.5,
                "stale_threshold": 1.0
            },
            "path_planner": {
                "epsilon": 0.3,
                "min_samples": 3
            }
        },
        "utilities": {
            "FRC/network_table_handler": {
                "network_tables_ip": "10.TE.AM.2"
            }
        },
        "frame_processors": {}
    },
    "rollback": {
        "enabled": true,
        "data_dir": "VideoRecordings",
        "fps": 30.0,
        "max_queue": 60,
        "downsample": 1,
        "segment_minutes": 5,
        "max_total_mb": 2048
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
| `trackers.FRC/object_tracker` | `distance_threshold: 0.5`, `stale_threshold: 1.0` | Stitches detections into a single object per camera; drops stale detections |
| `trackers.path_planner` | `epsilon: 0.3`, `min_samples: 3` | DBSCAN clustering of tracked objects into game-piece piles |
| `utilities.FRC/network_table_handler` | `network_tables_ip: "10.0.0.2"` | Publishes vision output to the robot over NetworkTables |

### HSV color detection

Set `"pipeline": "hsv_detection"` on a camera to detect game pieces by color
instead of training a model. It uses only OpenCV and NumPy, so it runs on the
board with no model file and no extra dependencies. The settings live in the
camera's `pipeline.settings` block and are also editable on the Cameras page.

| Setting | Default | What it does |
|---------|---------|-------------|
| `preset` | `custom` | `orange`, `yellow`, `red`, `green`, `blue` or `white` take over the HSV ranges at runtime; `custom` keeps the ranges below |
| `hue_range` | `[5, 25]` | OpenCV hue band (0-179). With `wrap`, a low end above the high end picks reds across 0 |
| `sat_range` | `[110, 255]` | Saturation band. Lower values admit washed-out pixels |
| `val_range` | `[80, 255]` | Brightness band. Raise the low end to drop shadows |
| `lighting_normalize` | `off` | `clahe` or `auto_gamma` flatten lighting on V before thresholding |
| `clahe_clip` | `3.0` | Contrast limit when `lighting_normalize` is `clahe` |
| `blur_kernel` | `0` | Odd blur size applied to S/V; `0` is off |
| `morph_open_iters` | `0` | Erosion passes that remove specks |
| `morph_close_iters` | `1` | Dilation passes that fill pinholes |
| `morph_kernel_size` | `5` | Odd kernel used by both passes |
| `morph_kernel_shape` | `ellipse` | `ellipse` or `rect` |
| `process_scale` | `1.0` | Threshold on a smaller frame, then scale boxes back. Lower is faster |
| `roi_top_ignore_pct` | `0.0` | Skip a top band of the frame, e.g. a colored ceiling |
| `min_area_px` | `150` | Ignore blobs smaller than this |
| `max_area_px` | `250000` | Ignore blobs larger than this, e.g. a colored wall |
| `min_circularity` | `0.0` | Reject ragged blobs (`1.0` is a circle) |
| `min_solidity` | `0.0` | Reject blobs with bites taken out of them |
| `min_aspect` / `max_aspect` | `1.0` / `6.0` | Long side over short side; drops cables and lane lines |
| `shape_hint` | `any` | `circle` or `rect` picks which shape term feeds confidence and depth |
| `max_detections` | `10` | Keep the N biggest blobs |
| `merge_distance_px` | `0` | Bridge blobs closer than this; `0` is off |
| `object_name` | `hsv_object` | Name reported for each detection |
| `object_size_in` | `4.0` | Real width in inches, used by `size_based` depth |
| `z_mode` | `ground_plane` | `ground_plane` casts a ray onto the object plane; `size_based` uses the known size |
| `plane_height_in` | `0.0` | Height of the plane the object center sits on (`0` is the floor) |
| `view_mode` | `outline` | `outline` keeps the camera picture; `mask` shows only thresholded pixels |
| `mask_style` | `color` | In mask view, `color` keeps source pixels and `binary` draws white on black |
| `draw_labels` | `true` | Draw name and confidence on detections |

Open a camera's lightbox from the Cameras page for **live tuning**. The HSV
panel writes changes to the running pipeline immediately (no restart) and saves
them to the config. The **Eyedropper** button samples the pixel you click and
offers a range widened by the hue/saturation/value tolerances shown next to it -
handy for picking a game piece color under field lighting. **Reset** restores
the schema defaults. The card header also gets buttons for the current
pipeline's actions, such as switching the view between outline and mask.

Because HSV detection is model-free, `vision_model` may be omitted entirely
when every camera uses a pipeline that does not need a model.

### Rollback (recording and replay)

Rollback is **not** an add-on either. The recorder is always-on core code
(`iSpy/core/rollback.py`) with the always-on core web module
(`iSpy/web/modules/rollback.py`, `/rollback` + `/api/rollback`), so it has no
entry in the Add-ons page. Every run is recorded automatically.

| Setting | Default | What it does |
|---------|---------|-------------|
| `enabled` | `true` | Record each run. Turn off to stop writing video |
| `data_dir` | `"VideoRecordings"` | Where sessions are written |
| `fps` | `30.0` | Frame rate of the recording, independent of your cameras |
| `max_queue` | `60` | Frames buffered before the newest are dropped under load |
| `downsample` | `1` | Write every Nth frame. `2` halves the size and the rate |
| `segment_minutes` | `5` | Roll to a new video file this often, bounding file size |
| `max_total_mb` | `2048` | Cap on everything under `data_dir`. Oldest unpinned sessions go first |

Each iSpy start is one **session**: a folder holding one `.avi` per camera plus a
JSON sidecar with per-frame detections and robot pose. The **Rollback** page
lists them, and pinning a session exempts it from the size cap forever.

To re-run the whole pipeline over an old session with no robot attached:

```bash
ispy-boot --replay "VideoRecordings/session_2026-01-01_18-30-00"
```

`--replay-speed` (default `1.0`), `--replay-cam` (one recorded camera) and
`--replay-no-loop` are also available. Without `--replay-cam`, all recorded
cameras are replayed through the currently configured vision pipeline; with
it, only that camera is selected. `--replay-no-loop` exits when the clips end.
A replay disables recording and NetworkTables publishing. The Rollback page
shows the exact command rather than launching it, since swapping the camera
stack out from under a live robot is not safe.

Health reporting is **not** an add-on: it is the always-on core web module
(`iSpy/web/modules/health.py`, `/health` + `/api/health`). Tune its stale-frame
threshold with the top-level config key `health_stale_threshold` (Settings →
Advanced).

All add-ons (enabled state, settings) are managed from the **Add-ons page** in
the web UI.

### Offline mode

`offline_mode` (top-level config key, Settings → Advanced → Offline Mode)
blocks traffic to the **public internet only**. Everything that talks to the
robot keeps working: NetworkTables, the dashboard, loopback, mDNS and the
Tello's `192.168.10.1`. Allowed destinations are loopback, RFC1918 (`10/8`,
`172.16/12`, `192.168/16`), link-local (`169.254/16`) and the IPv6 private
ranges (`::1`, `fc00::/7`, `fe80::/10`). Anything else is refused - DNS lookups
included - and `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` are set so Hugging Face
models load from their local cache instead of retrying over the network.

**Game Mode** forces offline on for its one-shot run. That override lives in
memory only - it is never written to `config.json`, so a crash mid-run can
never leave offline mode stuck on. The service daemon and every subprocess
(RKNN converter worker, yolo-world reparameterizer, pip) inherit it through
`ISPY_OFFLINE=1`.

What happens to add-ons while offline:

- an add-on that **cannot do its job** offline (a critical requirement, e.g.
  YOLO World weights that were never downloaded) is **skipped** with a warning
  in the log, a yellow entry on Health and a `skipped: offline` badge on the
  Add-ons page - boot and the run loop continue normally, nothing raises;
- an add-on that only loses an optional feature keeps running and gets a
  `partially offline` badge.

An add-on declares its needs itself:

```python
from iSpy.plugins.bases import TrackerBase
from iSpy.core.network_policy import NetworkRequirement


class MyTracker(TrackerBase):
    plugin_name = "me/my_tracker"

    # shortcut: the whole add-on needs the internet
    needs_internet = True

    # or declare precisely (this wins over needs_internet) - only when a
    # setting makes it need the net, key off the current settings:
    @classmethod
    def network_requirements(cls, settings):
        reqs = []
        if settings.get("web_sync"):
            reqs.append(
                NetworkRequirement(
                    name="web sync", reason="posts to a server", critical=False
                )
            )
        return reqs
```

At runtime `self.network_allowed` is False while offline,
`self.require_network("feature")` raises `OfflineError`, and
`self.try_online(feature, fn, fallback)` records one "feature disabled" note
and returns `fallback` when the internet is missing or flaky. A config-schema
field marked `"requires_network": True` is forced back to its default while
offline so a stale "check for updates"-style toggle cannot keep reaching out.

### Performance profiles

`performance_level` (top-level config key, Settings → General → Performance
profile) is a 5-stop slider that trades background work and luxury features
for FPS. It applies at the **next restart** - the picker works from an
already-running web UI even at levels that would not have started one.

| Slider | Level | What it changes |
|--------|-------|-----------------|
| Max FPS (left) | `0` | **Web UI off for the run.** Rollback recording off (no new sessions), vision metrics off, FPS console print off, annotated/debug frames off |
| Performance | `1` | UI polling 4x slower (paused while the tab is hidden), web payloads skipped while nobody is watching, previews 10 fps/q50 capped at 960px, recording keeps every 4th frame at most, metrics sampled at 1 Hz, system metrics cached 5s |
| Balanced | `2` | Half of Performance: 2x polling, previews 15 fps/q60, recording keeps every 2nd frame, metrics 5 Hz, cache 2s |
| Smooth | `3` | Polling 1.5x slower, web modules updated at most 20x/s, previews/recording untouched |
| Full Features (right) | `4` | **Default.** Nothing trimmed - exactly the behavior of an install without the key |

What **no** level ever trims: core detection and its thresholds, trackers,
NetworkTables publishing (rate and content are unchanged), robot pose reads,
the watchdog/service control channel, and calibration correctness. Camera
capture settings are not performance knobs either.

Level `0` is the only level that takes the web UI away, so it is special:

- the Settings slider makes you sit through a confirmation countdown listing
  what turns off and what stays on before it saves;
- the boot log prints a loud warning with the recovery commands;
- `GET /api/performance` and the dashboard banner show the active level, and
  a `POST /api/performance` of level 0 without `confirm_max: true` is
  rejected - the raw JSON editor cannot set it by accident;
- the run-time override is **in memory only** - `app_mode` in `config.json`
  is never rewritten, so a crash can't leave the UI gone.

To come back out of Max FPS without the UI:

```bash
ispy performance 4        # writes performance_level = 4, takes effect next start
ispy start --web          # or: set ISPY_FORCE_WEB=1 - forces the UI on for this run only
```

The same rules on a one-shot run: **Game Mode** and the performance profile
both apply, whichever turns the web off wins, and a `--web` start overrides
both for that run.

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
    pytorch/_default_detect.pt            # stock defaults downloaded on first boot
    pytorch/_default_pose.pt              # optional stock pose model
    pytorch/_default_v26_detect_for_fuel.pt # optional AGPL-derived fuel model
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
the result. Supported formats are `rknn`, `onnx`, `openvino`, `tflite`,
`coreml`, and `engine` (TensorRT). Hailo/HEF conversion is not currently
enabled.

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
| `/` | Dashboard |
| `/health` | System health dashboard (browser) or JSON (API) |
| `/api/cameras` | List of connected cameras |
| `/video/<name>` | Live annotated stream for a specific camera |
| `/api/cameras/config/<name>` | GET, PUT or DELETE a camera's settings |
| `/api/cameras/<name>/pipeline_settings` | GET or POST live-tunable pipeline settings |
| `/api/cameras/<name>/sample_hsv` | Sample HSV at a clicked frame pixel |
| `/api/cameras/<name>/actions/<action_id>` | POST a per-camera action button |

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

### Action buttons

Add-ons and vision pipelines can put buttons on the camera card header. Return
one entry per action from `get_actions()`; `kind` is `button` or `toggle`, and
`value` is the current toggle state. When the button is pressed, iSpy POSTs to
`/api/cameras/<name>/actions/<id>` and calls `invoke_action(id, value)` with the
toggle state (or `None` for a plain button). Reply with `{"ok": true}` to show
the action worked.

```python
class MyTracker(TrackerBase):
    plugin_name = "my_tracker"

    def get_actions(self) -> list[dict]:
        return [
            {
                "id": "record",
                "label": "Record",
                "kind": "toggle",
                "value": self.recording,
                "title": "Save a snapshot of the current detections",
            }
        ]

    def invoke_action(self, action_id: str, value=None) -> dict:
        if action_id == "record":
            self.recording = bool(value)
            return {"ok": True, "actions": self.get_actions()}
        return {"ok": False, "error": "unsupported"}
```

Only ids returned by `get_actions()` are accepted by the route, and returning
`actions` from `invoke_action` updates the button state without a page reload.
Changes made by a pipeline action are persisted through the same config path as
the Settings page.

### Schema field types

`config_schema()` fields can be `number`, `toggle`, `text`, `select`, `range`
or `list`. A `range` is a two-value slider that arrives as `[low, high]`:

```python
"hue_range": {
    "type": "range",
    "label": "Hue range",
    "min": 0,
    "max": 179,
    "step": 1,
    "default": [5, 25],
    "wrap": True,
    "gradient": "hue",
    "help": "OpenCV hue 0-179",
}
```

Values are clamped to `min`/`max`. Without `wrap`, a low end above the high end
is rejected and the schema default is used; with `wrap`, the band is allowed to
run through the end of the scale and back (reds on the hue wheel). `gradient:
"hue"` colors the slider track with the hue wheel, and omitting it leaves the
normal track.

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
        │     └── GenericYolo (dependency-free .pt / RKNN / ONNX / TFLite)
        ├── MultipleCameraHandler (merges multi-camera detections)
        ├── Trackers (FRC/object_tracker -> path_planner -> your plugins)
        ├── RollBack (always-on session recorder, core/rollback.py)
        ├── Utilities (network_handler, your plugins)
        └── CameraApp (Flask web server)
```

The main loop runs at whatever FPS the camera and model allow. On an Orange Pi 5 with a nano RKNN model, expect 30–60 FPS.

Benchmark results are hardware- and model-dependent. The Orange Pi 5 Pro
(RK3588) has been tested at roughly 30 FPS for YOLO pose and roughly 60 FPS
for the YOLO fuel detector. Treat these as indicative measurements, not
guarantees; run `ispy-bench` with your own camera, model, and calibration data
before relying on a number for competition planning.

`ispy-bench` reports its normal throughput for the complete model-prediction
call, including input preparation and result processing. Where supported, it
also prints a `vision-only` measurement for PyTorch, ONNX, and TensorRT. It
prepares inputs before timing and measures the backend/model forward call,
excluding iSpy preprocessing and postprocessing. Use `vision-only` when
comparing model execution speed; use the normal `predict` FPS to estimate the
cost of running inference through iSpy. Device synchronization is included so
accelerator timings account for completed work.

Linux ARM64 installs constrain NumPy below 2 because the supported PyTorch and
RKNN runtime wheels on those boards are built against the NumPy 1.x ABI.

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

**PolyForm Noncommercial License 1.0.0** applies to iSpy-authored source code -
see [LICENSE](LICENSE). Third-party models, build tools, and hardware SDKs
remain under their own terms; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

iSpy's own code is licensed under the PolyForm Noncommercial License 1.0.0: it
is free for noncommercial use (which covers student teams, FRC use, and hobby
projects). Commercial use requires a separate license from the authors. See
[LICENSE](LICENSE) for the full terms.

The stock default checkpoints (`_default_detect.pt`, `_default_pose.pt`, and
`_default_v26_detect_for_fuel.pt`) are downloaded on demand and are not bundled
in the source tree. The first two are Ultralytics AGPL-3.0 checkpoints. The v26
fuel model was trained by the project owner using Ultralytics code and is an
Ultralytics-derived AGPL-3.0 model, not a PolyForm asset. The owner's training
contribution is released under AGPL-3.0 with that checkpoint. Commercial or
redistributed use must satisfy the applicable model and build-tool licenses.
