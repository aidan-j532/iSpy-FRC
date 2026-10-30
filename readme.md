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
- Survives crashes via a process supervisor that keeps `http://ispy.local:5000` answering and restarts the pipeline automatically

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

`http://ispy.local:5000` stays reachable through the process supervisor. While
iSpy boots, restarts, crashes, or runs with the web UI off, it shows a small
status page instead of a connection error.

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
# x86_64 installs the dev extras automatically
./install.sh
```

Run the pipeline locally (uses a webcam or image file):

```bash
ispy-run
```

Run the supervised boot sequence (downloads default models, starts the web UI,
and keeps port 5000 available while iSpy boots):

```bash
ispy start
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
    "allow_remote_restart": true,
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
ispy start --replay "VideoRecordings/session_2026-01-01_18-30-00"
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
| Max FPS (left) | `0` | **Vision web UI off for the run.** Port 5000 still serves the supervisor's web-off status page with a normal-settings restart button. Rollback recording off (no new sessions), vision metrics off, FPS console print off, annotated/debug frames off |
| Performance | `1` | UI polling 4x slower (paused while the tab is hidden), web payloads skipped while nobody is watching, previews 10 fps/q50 capped at 960px, recording keeps every 4th frame at most, metrics sampled at 1 Hz, system metrics cached 5s |
| Balanced | `2` | Half of Performance: 2x polling, previews 15 fps/q60, recording keeps every 2nd frame, metrics 5 Hz, cache 2s |
| Smooth | `3` | Polling 1.5x slower, web modules updated at most 20x/s, previews/recording untouched |
| Full Features (right) | `4` | **Default.** Nothing trimmed - exactly the behavior of an install without the key |

What **no** level ever trims: core detection and its thresholds, trackers,
NetworkTables publishing (rate and content are unchanged), robot pose reads,
the supervisor/service control channel, and calibration correctness. Camera
capture settings are not performance knobs either.

Level `0` is the only level that takes the web UI away, so it is special:

- the Settings slider makes you sit through a confirmation countdown listing
  what turns off and what stays on before it saves;
- the boot log prints a loud warning with the recovery commands;
- `GET /api/performance` and the dashboard banner show the active level, and
  a `POST /api/performance` of level 0 without `confirm_max: true` is
  rejected - the raw JSON editor cannot set it by accident;
- the run-time override is **in memory only** - `app_mode` in `config.json`
  is never rewritten, so a crash can't leave the UI gone;
- the supervisor's web-off page on port 5000 has a **Restart with normal
  settings** button that writes `performance_level = 4` and restarts iSpy.

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
`coreml`, and `engine` (TensorRT). Hailo/HEF conversion is not auto-enabled; it
requires a manually installed vendor toolchain (see below).

### Conversion environments

Conversion toolchains are installed into persistent, conversion-only Python
environments outside the project tree and outside the main virtual environment.
They are created lazily on first use, reused across runs/reboots/benchmarks, and
removed only when their identity changes (Python ABI, architecture, toolkit or
dependency-set version) or when you explicitly remove them. Inference packages
(`rknn-toolkit2-lite`, `onnxruntime`, `tflite_runtime`, OpenVINO, TensorRT,
`torch_xla`) stay in the main environment.

| Format | Environment | Notes |
|---|---|---|
| `rknn` | `rknn_env` | Full `rknn-toolkit2` (NumPy 1.x, SciPy, protobuf 4.25.x, `onnxruntime`, `fast-histogram`, `pkg_resources`); downloaded from the `RKNN_Wheels` GitHub release. |
| `tflite` | `tflite_env` | Build-only TensorFlow 2.19 / tf-keras / onnx2tf stack. Requires CPython 3.10-3.12 on `aarch64` or `x86_64`; runtime `tflite_runtime` remains in the main env. YOLO `.pt` export needs Linux x86_64 or macOS because Ultralytics uses the LiteRT exporter; on `aarch64` the env is still used by Depth Anything's ONNX-to-TFLite worker. |
| `hailo` | `hailo_env` | Registry-visible but manual-only. iSpy never pip-installs the vendor Dataflow Compiler; pre-convert with Hailo's toolchain and place the `.hef` in `YoloModels/hailo/`. An existing `.hef` can still be used or benchmarked. |

Depth Anything's ONNX-to-TFLite conversion also runs in `tflite_env`; a cached
`.tflite` loads without the build environment being present.

Formats with no separate environment are `engine`, `openvino`, `onnx`, `qnn`,
`tpu`, and `coreml`. TensorRT engines are deliberately built with the same
runtime TensorRT version that must deserialize them; ONNX/OpenVINO/QNN use the
normal export path; TPU runs the `.pt` through `torch_xla`; CoreML conversion
uses `coremltools` on macOS.

The cache root defaults to the platform cache directory (for example
`%LOCALAPPDATA%\iSpy-FRC\Cache` on Windows or `~/.cache/ispy-frc` on Linux).
Override it with `ISPY_CONVERTER_ENV_HOME`; the older
`ISPY_RKNN_CONVERTER_HOME` is still accepted as a fallback. Each format has its
own `<format>_env/<identity-hash>/` directory and build lock, so RKNN and TFLite
environments can be provisioned concurrently.

```bash
ispy env list              # format, needs_env, path, disk usage, identity, last verify
ispy env path tflite       # print the current environment directory
ispy env rebuild rknn      # force a clean rebuild (requires internet)
ispy env remove tflite     # delete the cached environment
```

Environments are verified after creation and re-checked cheaply once a day
before reuse. A broken or identity-mismatched environment is rebuilt rather
than reused. In offline mode, a missing environment fails cleanly with an
actionable error and the original `.pt` is used.

To convert manually on a dev machine:

```python
from iSpy.vision.optimizer import convert_model

convert_model("my_model.pt", target_format="rknn", input_size=(640, 640))
```

### Optimizing models with `ispy-optimize`

`ispy-optimize` is a standalone CLI that converts the `.pt` checkpoints you
already manage into every backend artifact iSpy can run (rknn / onnx / openvino /
tflite / engine / qnn / hailo / tpu passthrough). It runs each conversion in the
same isolated environments as the web UI optimizer, but in bulk or one at a time,
with explicit control over quantization, calibration data, input size, and
post-build actions.

```bash
# list all .pt models and what artifacts exist / are stale / are missing
ispy-optimize --list

# convert a single model to the recommended format (auto)
ispy-optimize fuel_detection_v26

# convert everything in YoloModels/pytorch/ to every format this machine can build
ispy-optimize --all --all-formats

# convert a specific model to rknn (int8) and onnx, using a real calibration set
ispy-optimize my_model --format rknn,onnx --calib-dataset QuantizeDataset/coco128

# convert the model used by camera "cam0" with that camera's own settings
ispy-optimize --camera cam0

# convert all configured cameras' models, then update those cameras to use the artifacts
ispy-optimize --configured --apply

# verify every artifact loads and runs one dummy frame; delete failures
ispy-optimize --all --verify --remove-bad

# dry-run: print the exact job plan without converting
ispy-optimize --all --dry-run

# machine-readable results (JSON on stdout with `-`)
ispy-optimize --all --json -
```

| Flag | Meaning |
|---|---|
| `MODEL ...` | Paths, bare names (`fuel_detection_v26`), or globs. Resolved against `YoloModels/pytorch/` then CWD. |
| `--all` | Every `.pt` directly in `YoloModels/pytorch/` (deduped by file fingerprint). Excludes `world/` and `*.part`. |
| `--camera NAME` | Optimize the model a configured camera uses (repeatable). |
| `--configured` | Same as `--camera` for every model-backed camera. |
| `--exclude PATTERN` | Glob on filename to skip (repeatable). |
| `--skip-defaults` | Skip stock `_default_*.pt` checkpoints. |
| `--format FMT[,FMT...]` | Target format(s): `auto`, `rknn`, `onnx`, `openvino`, `tflite`, `engine`, `qnn`, `hailo`, `coreml`, `tpu`. Repeatable, comma-separated. |
| `--all-formats` | Every format this machine can build/run (hardware gates from `ispy-bench`) plus `onnx`. Never includes `coreml`/`hailo` unless explicitly named. |
| `--quantize` / `--no-quantize` | Force int8 on/off (default: rknn/tflite/hailo int8 on, others off). |
| `--calib-dataset PATH` | Calibration images. Falls back to camera setting, a real `QuantizeDataset/<name>/`, COCO128 download, then synthetic noise (warns loudly). |
| `--input-size N|WxH` | Override conversion input size (default: sidecar, else 640). |
| `--rknn-target-platform CHIP` | Sets `ISPY_RKNN_TARGET_PLATFORM` for child processes. |
| `--force` | Rebuild even when a valid cached artifact exists. |
| `--only-missing` | Default: skip fresh artifacts (mtime + sidecar quantize flag match). |
| `--jobs N` | Concurrent builds (default 1; RKNN always serial). |
| `--fail-fast` | Stop at first failure (default: continue). |
| `--timeout SECONDS` | Per-job timeout; kills the child on expiry. |
| `--install-deps` | Best-effort install of build dependencies for planned formats. |
| `--no-disk-check` | Skip free-space pre-flight (default: on, uses each planned converter env's own floor). |
| `--verify` / `--no-verify` | Default on: load each fresh artifact with GenericYolo + dummy predict. |
| `--remove-bad` | Delete artifacts that fail `--verify`. |
| `--compare` | Surface verdict/speedup from `Outputs/optimized_model_report.json` (worker already ran it). |
| `--apply` | Point matching camera configs at the new artifacts (never implied; only on success). |
| `--bench` | Quick predict pass per artifact (2s), print FPS. |
| `--clean` | Remove orphaned artifacts + sidecars + `*_rknn_ready.onnx` leftovers (dry-run with `--dry-run`). |
| `--list` | Table of models × formats: model, task, nc, input size, MB, and per-format state (built/stale/missing/unsupported) with artifact size. |
| `--dry-run` | Print the exact job plan and exit 0. |
| `--json [PATH]` | Machine-readable results. Default: `Outputs/optimize_results.json`. `-` = stdout. |
| `-v` / `-q` | Verbose (shows worker output) / quiet (summary + errors only). |

**Exit codes**: `0` all jobs OK or skipped-as-fresh, `1` at least one job FAILED/VERIFY_FAILED, `2` usage/selection error, `130` interrupted (Ctrl-C).

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

The health endpoint returns `200 OK` when everything is healthy, `503` when
degraded. Useful for robot code that wants to know if vision is alive.

### Pipeline Editor

Open `/editor` (the **Editor** link in the web UI) for a visual view of the
same executable config used by the Cameras and Add-ons pages. The editor is
**additive**: it never replaces those pages, and it never changes a running
pipeline in place. Save validates the graph, writes the normal config keys, and
shows a **restart required** banner. Restart iSpy from the Health page (or the
service manager) to apply the change.

#### Flow and node rules

The canvas is a strict left-to-right lane-based pipeline. Columns are always:

| Column | Label | Rules |
|---|---|---|
| 1 | **Camera** | Starts every lane. At least one camera lane is required. Camera source/type settings stay in sync with the Cameras page. |
| 2 | **Frame Processor** | Exactly one per camera lane. Settings are shared by plugin name across all lanes that use it. Inspector shows "shared with N lanes". |
| 3 | **Pipeline** | Exactly one per camera lane. Required settings come from the real pipeline schema; calibration-requiring pipelines show a warning until the camera is calibrated. |
| 4 | **Tracker** | Exactly one per camera lane. Settings are shared by plugin name across lanes. Tracker output may fan out to the Utility column. |
| 5 | **Utility** | Global column. Only one utility total. Vertically ordered. Accepts multiple tracker inputs (one per camera) and is invalid with zero. Settings are shared (only one instance). |

Cameras are horizontal lanes; each lane owns exactly one node in every column.
Utilities are global cards in the last column, vertically ordered.

No free node dragging. Lock X/Y. Only reorder actions are allowed:
- **Camera handle** (left gripper on the Camera node) — drag vertically to reorder lanes.
- **Utility handle** (left gripper on the Utility node) — drag vertically to reorder utilities.
- No arrow-key nudges.

Ports: left input / right output. Camera has no input port; Utility has no output port.
Edges are horizontal forward-only cubic Bézier curves that never pass through other nodes.

#### Visual system

Flat GitHub-style dark cards using `design.css` tokens:
- Muted edges (`--ed-line`), alternating column bands (subtle `alt` tint).
- Sticky column headers with live counts; lane index labels on the left.
- Category pills (`.node-kind`) use the column label text only.
- Quiet status indicator (6 px dot): green = complete, amber = needs required fields.
- Accent colour **only** for selection, hover, and drag handles. No neon, glows, gradients, glass, rings, badges, emoji, or oversized pills.
- Wheel pan (vertical), Shift+wheel (horizontal), Ctrl+wheel (zoom at pointer), **F** to fit.
- Narrow-screen (< 960 px): canvas stacks above inspector, horizontal panning works naturally.

#### Editing and validation

- Click a palette item, or drag it onto a target lane (for Frame Processor / Pipeline / Tracker) or the canvas (for Camera / Utility), to add a node. The palette and drop indicator explain why an item is locked when the preceding column is incomplete.
- Select a node to edit its schema-driven inspector. Browse, source, model, select, range, list, nullable, conditional (`gated_by`), and dynamic publish source fields all use the same schema/coercion paths as the rest of the web UI.
- Connections are generated from the fixed column order and utility inputs. Change a utility's source checkboxes or remove a node to rewire. Drag-to-connect is not supported; the graph topology is derived, not drawn.
- Edge hover highlights **both endpoints** (source and target ports) and the full flow into the same utility.
- Completion progress (corner badge) uses only **required** settings: red dashed with **×** when incomplete, solid green with **✓** when complete, partial fill when some required fields are filled. A node with zero required fields is complete immediately.
- **Validate** checks the graph without writing config. **Save** validates first and either writes the complete change or leaves the config untouched. The inspector shows node-specific server errors and warnings.
- The Add-ons and Cameras pages share the same config. **Reload** discards unsaved editor state and re-reads changes made there; returning to the tab refreshes automatically when there are no unsaved edits. Because **Save** writes the complete graph, reload after editing the other pages before saving. Reloading or leaving with unsaved edits asks for confirmation first.

#### Required-field rule

The backend helper in `iSpy/config/schema_fields.py` is the source of truth; the
browser mirrors it live for progress/gating. Unless a schema explicitly sets
`required`:

- `nullable: true` fields are optional.
- `model`, `source`, and `browse` fields are **always required** because they point at a real artifact, even when their default is an empty value.
- A field **without a default** is required.
- A field **with a blank default** (empty string, empty list, malformed range) is optional.
- A field **with a meaningful default** is required, but the default **never satisfies completion** until the user confirms it (Enter/click on the ghost field, or uses "Use Defaults" in the inspector, or "Use Defaults for all nodes" in the toolbar). Defaults are shown as ghost placeholders, not saved values.
- A field behind an inactive `gated_by` controller is not counted.
- `toggle: false` is a real value and satisfies the field.
- `required: true` can make an otherwise optional field count; `required: false` can exempt a field. `required: null` is treated as if the rule were absent.
- Zero active required fields means `100%` complete.

The same Python result is used by `GET /api/editor/graph`, `Validate`, and `Save`. The browser recomputes the same rule after every edit so the canvas does not wait for a server round trip.

#### Ghost defaults & inspector

- Ghost fields show a dashed border and "default" badge; the placeholder shows the default value.
- **Click / Enter** on a ghost field writes the default explicitly (confirming it).
- **Typing** in a ghost field replaces the ghost with the typed value on change.
- Inspector "Use Defaults" writes all ghost defaults for the selected node.
- Toolbar "Use Defaults for all nodes" writes all ghost defaults globally.
- API returns explicit settings and defaults separately; runtime merges defaults on load.

#### Config persistence

The editor writes the existing executable structures, not a parallel model:

- Camera assignment and lane order are stored in `camera_configs` key order. `camera_configs.<name>.trackers` (array of tracker plugin names) is added; absent means legacy global trackers; explicit `[]` means no trackers for that lane.
- Pipeline settings stay in `camera_configs.<name>.pipeline.settings`.
- Tracker/utility settings stay under `plugins.trackers.<name>` and `plugins.utilities.<name>`. Presence in those buckets means enabled, matching the Add-ons page.
- Frame-processor settings stay under `plugins.frame_processors.<name>` (shared by plugin name).
- Top-level `utility_sources` maps utility plugin name → array of camera names. Missing key means "all cameras"; explicit `[]` means zero inputs (invalid for Save).
- Only canvas metadata lives under `editor_graph`: `viewport`, `lane_order`, and `utility_order`. No per-node x/y. Edges are reconstructed from executable config on load, not stored separately.
- Saves use the normal atomic `iSpyConfig.save()` path. Add-ons and Cameras edits and editor edits therefore read/write the same config, and the existing local/admin-token guard protects editor write routes (`restart_allowed` + `allow_remote_restart`). Frontend sends `X-iSpy-Admin-Token` header; server errors are displayed in the inspector.

#### Camera palette rules

- **Detected devices only** (via OpenCV discovery). Claimed devices are filtered; cache with 5 s TTL; **Refresh** button re-probes.
- **Add-once** behavior: clicking a palette camera item adds it once; dragging allows placement at a specific lane index.
- **Autofill**: device `source`/`device_id` are pre-filled; name is auto-generated and unique.
- **Replay camera**: at most one. Uses a session picker (populated from Rollback `data_dir`).
- **Tello EDU**: only appears in palette and is only creatable when `config.admin == true`. Existing Tello cameras are preserved with a warning; new saves/creates are rejected without admin.
- **Detected-device enforcement**: new physical cameras must exist in the detected list; URLs/files and existing cameras bypass. Zero detected devices shows an empty state message.

#### Keyboard and mouse

| Input | Action |
|---|---|
| `Ctrl`/`Cmd` + `S` | Validate and save |
| `Ctrl`/`Cmd` + `Z` | Undo |
| `Ctrl`/`Cmd` + `Shift` + `Z` or `Ctrl` + `Y` | Redo |
| `Ctrl`/`Cmd` + `A` | Select all nodes |
| `Shift`/`Ctrl`/`Cmd` + click node | Toggle multi-select |
| `Shift` + drag empty canvas | Marquee-select nodes |
| `Delete` / `Backspace` | Remove selected nodes (confirmation when needed) |
| `F` | Fit graph |
| `+` / `-` | Zoom |
| `?` | Show shortcuts |
| Wheel / trackpad | Pan vertically |
| `Shift` + wheel | Pan horizontally |
| `Ctrl`/`Cmd` + wheel | Zoom at pointer |
| Drag empty canvas | Pan |
| Drag camera handle | Reorder lane |
| Drag utility handle | Reorder utility |

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

## Process supervisor

`ispy start` runs a small, stdlib-only supervisor. It owns port 5000 first,
then launches the real vision process as a hidden `ispy _vision` child. When
the child's web server is ready it asks the supervisor to close the status
listener and takes over the same port - a real port hand-off, not a reverse
proxy, so client IPs and MJPEG/SSE streams stay direct. If the child crashes,
the supervisor immediately serves the crash page, logs the exit, and rebinds
port 5000 while it restarts.

- `GET /api/supervisor/state` is available from both the supervisor and the
  running child. It reports the state (`idle`, `fresh_boot`, `booting`,
  `running`, `restarting`, `stopping`, `crashed`,
  `crash_loop_stopped`), the restart attempt out of the limit, the current
  child phase, and the last 20 lines of `Outputs/log.txt`.
- `POST /api/supervisor/restart`, `/start`, and `/stop` are the lifecycle
  controls. The Beta restart button in Settings → Advanced calls restart.
- The supervisor allows loopback and RFC1918/link-local callers by default.
  Set `"allow_remote_restart": false` to require localhost or
  `ISPY_ADMIN_TOKEN`; `X-Forwarded-For` is never trusted.
- Runtime state lives in `.ispy-run/` at the project root, outside the folders
  `boot -f` wipes. It is gitignored and safe to delete when iSpy is stopped.
- The restarting page and fresh-boot page read
  `iSpy/assets/ui_messages.json`, so you can edit the rotating messages without
  touching Python. Add `iSpy/assets/restarting.png` to show your own image; the
  `image` key can point at another path relative to the project or assets
  folder.
- `ispy setup` still runs inline and exits. It serves the fresh-boot page on a
  best-effort basis, but never binds a port when `ISPY_NO_SUPERVISOR=1`, in
  CI, or in a detected chroot/image build.
- `ispy-run`, `python -m iSpy.boot.boot`, and direct `game_loop` users do not
  need the supervisor. The Linux systemd unit uses `ispy start`; the legacy
  `iSpy/boot/service_daemon.py` and `watchdog.py` remain functional for existing
  service units and the Windows/macOS `-s` installers.

---

## Architecture

```
ispy start -> iSpy.supervisor (owns :5000; status pages + crash loop)
  └── ispy _vision
        └── game_loop.py
              └── iSpy
                    ├── ObjectDetectionCamera (per camera)
                    │     ├── Camera (threaded frame reader)
                    │     └── GenericYolo (dependency-free .pt / RKNN / ONNX / TFLite)
                    ├── MultipleCameraHandler (merges multi-camera detections)
                    ├── Trackers (FRC/object_tracker -> path_planner -> your plugins)
                    ├── RollBack (always-on session recorder, core/rollback.py)
                    ├── Utilities (network_handler, your plugins)
                    └── CameraApp (Flask web server; takes :5000 back from the supervisor)
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
- **Admin routes are local-only by default.** The mutating plugin admin
  endpoints (upload/create/delete) reject non-local requests unless you set
  `ISPY_ADMIN_TOKEN` environment on the board; when set, remote callers must
  present the matching `X-iSpy-Admin-Token` header (see
  `require_local_or_token` in `iSpy/web/Backend/PluginStatus.py`). Live feeds
  and health endpoints are open - do not expose port 5000 to an untrusted
  network (i.e. the internet).
- **Restart endpoints allow the robot LAN by default.** The Beta restart/start/
  stop controls accept loopback and RFC1918/link-local clients, because that is
  the FRC field network. Set top-level `"allow_remote_restart": false` to require
  localhost or `ISPY_ADMIN_TOKEN`. `X-Forwarded-For` is deliberately ignored;
  authorization always uses the socket peer address, so a reverse proxy does not
  let arbitrary clients claim to be localhost.
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
- **Converter environments need disk and internet on first use.** RKNN needs
  about 6 GiB free; the TFLite stack needs about 5 GiB. These are per-format
  floors, not a global 8 GiB rule; override them with
  `ISPY_RKNN_MIN_FREE_GIB` / `ISPY_TFLITE_MIN_FREE_GIB` if the filesystem is
  unusual. pip's cache is purged once and the install is retried if the disk
  fills. In offline mode a missing environment fails cleanly and conversion
  falls back to the `.pt`. Pre-build on a connected machine or copy the cache
  directory to avoid this.
- **TFLite conversion is narrower than TFLite inference.** The build environment
  supports CPython 3.10-3.12 on `aarch64`/`x86_64` because that is what
  TensorFlow 2.19 publishes. YOLO `.pt` export additionally requires Linux
  x86_64 or macOS: Ultralytics 8.4.83+ uses the LiteRT exporter, which refuses
  Linux ARM64. On ARM64, `ispy-bench`/`ispy-optimize` therefore skip TFLite
  export and benchmark the next best format; Depth Anything's ONNX-to-TFLite
  worker still uses `tflite_env`, and a cached `.tflite` can still be
  benchmarked or run on any platform.
- **Hailo conversion is manual-only.** The Hailo Dataflow Compiler is not on
  PyPI and has its own supported Python/OS combinations. `ispy env list` shows
  the `hailo` entry so the requirement is visible, but iSpy will raise the
  vendor-install hint instead of attempting a pip install.
- **TensorRT has no converter environment by design.** An `.engine` is tied to
  the TensorRT version that built it; building in a separate env could produce
  an engine the runtime cannot deserialize.
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
