# 07 -- Utilities and Plugins

> Built-in utility plugins: NetworkTables handler and rollback. Also covers
> the MultipleCameraHandler core utility, the core Health web module (which
> absorbed the old health_reporter add-on), and plugin base classes.
> Every function, class, method, data flow, and code path is documented.

---

## Table of Contents

1. [NetworkTableHandler](#networktablehandler)
2. [Health (core web module)](#health-core-web-module--formerly-the-healthreporter-add-on)
3. [RollBack (Video Recorder)](#rollback-video-recorder)
4. [MultipleCameraHandler](#multiplecamerahandler)
5. [Plugin Base Classes](#plugin-base-classes)
6. [Example Utility](#example-utility)

---

## NetworkTableHandler

**File:** `iSpy/plugins/utilities/BuiltIn/NetworkHandler.py` (243 lines)

The primary integration point between iSpy and the FRC robot. Publishes
vision data to NetworkTables so the robot code can consume it in real-time.

### FuelStruct (lines 11-19)

A WPI struct dataclass for publishing detection data:

```python
@wpiutil.wpistruct.make_wpistruct(name="Fuel")
@dataclasses.dataclass
class FuelStruct:
    x: float
    y: float
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
```

This is the wire format for detection arrays published to NetworkTables.
The `@make_wpistruct` decorator registers it with the WPILib struct serialization
system so robot code can deserialize it directly.

### DEFAULT_PUBLISH (lines 22-27)

Default publish entries:

```python
DEFAULT_PUBLISH = [
    {"name": "fps",            "data_type": "number",   "source": "fps",             "nt_topic": "fps"},
    {"name": "num_detections", "data_type": "number",   "source": "detection_count",  "nt_topic": "num_detections"},
    {"name": "camera_lag",     "data_type": "number",   "source": "camera_lag_s",     "nt_topic": "camera_lag"},
    {"name": "vision_data",    "data_type": "struct[]", "source": "detections",       "nt_topic": "vision_data"},
]
```

Each entry maps a `source` key in `frame_data` to an NT topic. The `data_type`
determines how the value is serialized.

### `config_schema()` (lines 33-68)

Defines two settings:

```python
{
    "network_tables_ip": {
        "type": "text",
        "label": "Robot IP",
        "hint": "IP address of the robot's NetworkTables server (usually the roboRIO).",
        "default": "10.0.0.2",
    },
    "publish": {
        "type": "list",
        "label": "Publish to NetworkTables",
        "hint": "Data entries published every tick.",
        "default": DEFAULT_PUBLISH,
        "fields": {
            "name":     {"type": "text",   "label": "Name"},
            "data_type": {"type": "select", "label": "Type", "options": ["number", "boolean", "string", "struct[]"]},
            "source":   {"type": "text",   "label": "Source"},
            "nt_topic": {"type": "text",   "label": "NT Topic"},
        },
    },
}
```

The `publish` setting is a list type, rendered as an editable table in the
web UI. Users can add/remove entries, change source keys, and modify topic names.

### `__init__(context)` (lines 70-92)

Connection flow:

```python
ip = self.config.get("network_tables_ip", "10.0.0.2")
self.inst = ntcore.NetworkTableInstance.getDefault()
self.inst.setServer(ip)
self.inst.startClient4("iSpy")
```

Then retries connection up to 15 times (1 second apart):
```python
for i in range(15):
    if self.inst.isConnected():
        break
    self.logger.warning("NetworkTables not connected, retrying... (%d/15)", i + 1)
    time.sleep(1)
else:
    self.logger.error("NetworkTables could not connect after 15s.")
```

The `else` clause on the `for` loop fires if the loop completes without
`break` (i.e., never connected). After 15 seconds, it logs the error but
does not crash -- the handler simply won't publish until connected.

**Instance variables:**
- `self._subscribers: dict` -- Cached NT publishers/subscribers (keyed by
  `"pub/<table>/<topic>"` or subscriber key)
- `self._tables: dict` -- Cached NT table references
- `self._viewer` -- Reference to Viewer3DModule for overlay pushes

### `isConnected()` (line 93-94)

Delegates to `self.inst.isConnected()`.

### `get_robot_pose()` (lines 96-109)

Reads the robot's odometry pose from AdvantageKit's NetworkTables layout:

```python
sub_key = "AdvantageKit/RealOutputs/Odometry/Robot"
table = self._get_table("AdvantageKit/RealOutputs/Odometry")
self._subscribers[sub_key] = table.getStructTopic("Robot", Pose2d).subscribe(Pose2d())
return self._subscribers[sub_key].get()
```

Uses lazy subscription: the subscriber is created on first call and cached.
Returns a `Pose2d()` (origin) on error or disconnection.

### `update(frame_data)` (lines 111-129)

Called every vision tick. Flow:

1. Early return if not connected
2. Reads configured `publish` entries (falls back to `DEFAULT_PUBLISH`)
3. For each entry, calls `_publish_entry(entry, frame_data)`
4. Checks cameras for hopper data: `cam.get_data_for_subsystem("hopper")`
5. Publishes hopper boolean to `"VisionData"` table
6. Calls `self.inst.flush()` to send all pending data
7. Calls `_update_viewer_overlay(frame_data)` to push robot position

### `_publish_entry(entry, frame_data)` (lines 131-155)

Publishes a single configured entry:

```python
value = self._resolve_source(source, frame_data)
if data_type == "struct[]":
    self._send_detections(value)
elif data_type == "number":
    self._send_data(float(value), nt_topic, "VisionData")
elif data_type == "boolean":
    self._send_data(bool(value), nt_topic, "VisionData")
elif data_type == "string":
    self._send_data(str(value), nt_topic, "VisionData")
```

### `_resolve_source(source, frame_data)` (lines 157-173)

Resolves a dotted source key against frame_data:

```python
# Special case: raw detection list
if source == "detections":
    return frame_data.get("detections", [])

# Dotted path: "debug_data.fps" -> frame_data["debug_data"]["fps"]
parts = source.split(".")
obj = frame_data
for part in parts:
    if isinstance(obj, dict):
        obj = obj.get(part)
    else:
        return None
return obj
```

### `_send_detections(detections)` (lines 182-200)

Converts detection objects to FuelStruct array:

```python
structs = [
    FuelStruct(
        x=float(d.get_position_normally()[0]),
        y=float(d.get_position_normally()[1]),
        z=float(d.get_position_normally()[2]),
        roll=d.roll,
        pitch=d.pitch,
        yaw=d.yaw,
    )
    for d in detections
]
```

Uses lazy publisher creation:
```python
pub_key = "pub/VisionData/vision_data"
if pub_key not in self._subscribers:
    self._subscribers[pub_key] = table.getStructArrayTopic(
        "vision_data", FuelStruct
    ).publish()
self._subscribers[pub_key].set(structs)
```

### `_send_data(value, data_name, table_name)` (lines 202-215)

Auto-types based on Python type:

```python
if isinstance(value, bool):
    pub = table.getBooleanTopic(data_name).publish()
elif isinstance(value, (int, float)):
    pub = table.getDoubleTopic(data_name).publish()
elif isinstance(value, str):
    pub = table.getStringTopic(data_name).publish()
```

Publishers are cached in `self._subscribers` with key `"pub/<table>/<topic>"`.

### `_update_viewer_overlay(frame_data)` (lines 222-240)

Pushes a robot position overlay to the 3D viewer:

```python
pose = frame_data.get("robot_pose")
self._viewer.add_overlay("robot", {
    "type": "box",
    "x": pose["x"],
    "y": pose["y"],
    "z": 0.15,              # Half-height of robot
    "roll": 0, "pitch": 0,
    "yaw": pose["heading"],
    "color": "#4c8bf5",
    "label": "Robot",
    "data": {"width": 0.76, "height": 0.30, "depth": 0.69},
})
```

The robot dimensions (0.76m wide x 0.30m tall x 0.69m deep) are approximate
for a standard FRC robot. The z=0.15 places the box center at half-height
so it sits on the ground plane.

---

## Health (core web module — formerly the `HealthReporter` add-on)

**File:** `iSpy/web/modules/health.py` (`HealthModule`)

The standalone health utilities (`health_reporter`, `status_reporter`) were
merged into this always-on web module so `/health` is registered exactly once
and enabling every add-on can never trigger a Flask duplicate-endpoint
collision. See [02-plugins.md](02-plugins.md) for the endpoint/payload spec.

Key points:

- **Stale threshold:** top-level config key `health_stale_threshold`
  (default `1.0`), editable in Settings → Advanced. Legacy
  `utilities.health_reporter.stale_threshold` values are migrated there by
  `_migrate_addons()` in `iSpyConfig.py`.
- **Wiring:** the vision loop calls `update(frame_data)` each tick;
  `iSpy/iSpy.py` connects NetworkTables status via
  `health_mod.set_network_handler(nt)`.
- **Payloads:** `/health` (minimal watchdog contract, 200/503),
  `/health/detailed` and `/api/health` (full payload + live plugin statuses).

---

## RollBack (Video Recorder)

**File:** `iSpy/plugins/utilities/BuiltIn/RollBack.py` (228 lines)

Despite its name (`plugin_name = "rollback"`), this is actually a video
recording utility. It saves annotated video frames to disk using OpenCV's
`VideoWriter`.

### `_best_codec()` (lines 15-23)

Platform-specific codec selection:
- **Windows**: `("mp4v", ".mp4")`
- **macOS**: `("mp4v", ".mp4")`
- **Linux**: `("MJPG", ".avi")`

### `config_schema()` (lines 29-56)

```python
{
    "data_dir":    {"type": "text",   "default": "VideoRecordings", "label": "Output Directory"},
    "fps":         {"type": "number", "default": 30.0,              "label": "Recording FPS"},
    "max_queue":   {"type": "number", "default": 300,               "label": "Max Queue"},
    "downsample":  {"type": "number", "default": 1,                 "label": "Downsample"},
}
```

### `__init__(context)` (lines 58-77)

Instance variables:
```python
self._video_output_dir = self.config.get("data_dir", "RollbackSave")
self._fps = float(self.config.get("fps", 30.0))
self._max_queue = int(self.config.get("max_queue", 300))
self._downsample = max(1, int(self.config.get("downsample", 1)))
self._queue = queue.Queue(maxsize=self._max_queue)
self._writer = None          # cv2.VideoWriter
self._thread = None          # Background writer thread
self._started = False
self._stopped = False
self._frame_counter = 0
self._dropped = 0
self._size = None            # (width, height) of output video
```

### `update(frame_data)` (lines 82-94)

Flow:
1. Extracts `frame` from frame_data
2. On first frame: calls `_start_recorder(w, h)` to initialize VideoWriter
3. Calls `_write(frame)` to queue the frame

### `_clean_frame(frame)` (lines 100-119)

Normalizes a frame for the VideoWriter:
1. Rejects non-numpy arrays
2. Converts non-uint8 to uint8
3. Rejects non-3-channel frames
4. Converts BGRA (4-channel) to BGR
5. Resizes to `self._size` if set (for downsampling)
6. Returns contiguous array

### `_start_recorder(width, height)` (lines 121-164)

1. Sets `self._size = (width, height)`
2. Gets codec from `_best_codec()` (or forced override)
3. Creates filename: `VideoRecordings/recording_YYYY-MM-DD_HH-MM-SS.mp4`
4. Creates `cv2.VideoWriter` with the codec
5. If writer fails to open: falls back to `("MJPG", ".avi")`
6. If fallback also fails: sets `self._writer = None`
7. Starts background writer thread

### `_worker()` (lines 166-176)

Background thread that dequeues frames and writes to disk:
```python
def _worker(self):
    while True:
        frame = self._queue.get()
        if frame is None:      # Sentinel = stop
            break
        if self._writer:
            self._writer.write(frame)
        self._queue.task_done()
```

### `_write(frame)` (lines 178-198)

Queues a frame for writing:
1. Increments frame counter
2. Applies downsample: `if self._frame_counter % self._downsample != 0: return`
3. Cleans frame via `_clean_frame()`
4. If queue is full: drops oldest frame (`queue.get_nowait()`)
5. Puts frame into queue
6. Increments `_dropped` on `queue.Full` exception

### `_stop_recorder()` (lines 200-228)

Graceful shutdown:
1. Sets `self._stopped = True`
2. Sends sentinel (`None`) to queue
3. Joins writer thread (15s timeout)
4. Sleeps 1s to flush
5. Releases VideoWriter
6. Logs final frame/dropped counts

---

## MultipleCameraHandler

**File:** `iSpy/utilities/MultipleCameraHandler.py` (159 lines)

Manages multiple cameras running in separate threads and merges their
detections via stereo triangulation.

### `__init__(cameras, config)` (lines 14-34)

```python
self.cameras = cameras                     # List[VisionPipeline]
self._max_residual = config.get("triangulation_max_residual", 0.5)
self._match_gate = config.get("triangulation_match_distance", 2.0)
self._objects: list[list[Object]] = [[] for _ in cameras]
self._frames = [None] * len(cameras)
self._locks = [threading.Lock() for _ in cameras]
self._fresh = [threading.Event() for _ in cameras]
self._stop_events = [threading.Event() for _ in cameras]
```

Starts one daemon thread per camera (lines 28-34).

### `_camera_loop(i, camera)` (lines 36-56)

Per-camera thread function:
```python
while not self._stopped and not stop_event.is_set():
    if camera.in_calibration_mode():
        objects, frame = [], camera.get_raw_frame()
    else:
        objects, frame = camera.run()
    with self._locks[i]:
        self._objects[i] = objects if objects is not None else []
        self._frames[i] = frame
    self._fresh[i].set()
```

Key behavior:
- In calibration mode: skips `run()` and just reads raw frames
- On exception: logs warning, uses fallback frame, sleeps 50ms to avoid
  CPU starvation

### `predict()` (lines 58-69)

Waits for fresh data from all cameras, then merges:

```python
for event in self._fresh:
    if not event.wait(timeout=0.2):  # 200ms timeout per camera
        self.logger.debug("Camera timed out waiting for fresh frame")
    event.clear()

per_camera = [list(self._objects[i]) for i in range(len(cameras))]
return self._merge_with_triangulation(per_camera)
```

### `_merge_with_triangulation(per_camera)` (lines 71-110)

Stereo triangulation merge algorithm:

1. For each detection in camera A:
   a. If it has a `ray_origin` (stereo-capable pipeline)
   b. For each detection in cameras B..N with the same `name`:
      - Check rough 2D distance < `_match_gate` (default 2.0m)
      - Compute `closest_point_between_rays()` with `max_residual` (default 0.5m)
      - Track best (lowest residual) match
2. If best match found:
   - Update detection coordinates with 3D triangulated point
   - Mark `depth_source = "triangulated"`
   - Mark matched detection as used
3. Unmatched detections are included as-is

### `get_combined_frame(display_width=640)` (lines 112-143)

Stitches camera frames horizontally:
1. Gets latest frame from each camera (thread-safe)
2. If multiple frames: resizes to common height, horizontally stacks
3. If final width > display_width: scales down proportionally

### `get_camera_frames()` (lines 145-154)

Returns dict of `{camera_name: frame}` for all cameras.

### `destroy()` (lines 156-159)

Sets `_stopped = True` and calls `cam.destroy()` on each camera.

---

## Plugin Base Classes

**File:** `iSpy/plugins/bases.py` (referenced throughout)

All plugin types inherit from base classes that define the lifecycle:

### UtilityBase

```python
class UtilityBase:
    plugin_name = "base"

    def __init__(self, context: dict):
        self.context = context
        self.config = context["config"].get_addon_settings("utilities", self.plugin_name) or {}

    @classmethod
    def config_schema(cls) -> dict:
        return {}

    def update(self, frame_data: dict):
        pass

    def stop(self):
        pass
```

Key attributes available in `__init__`:
- `self.context` -- Shared context dict
- `self.config` -- Plugin-specific settings from config

### TrackerBase

```python
class TrackerBase:
    plugin_name = "base"

    def update(self, detections, robot_x, robot_y, robot_yaw, robot_z=0.0):
        return detections
```

### FrameProcessorBase

```python
class FrameProcessorBase:
    plugin_name = "base"

    def process(self, frame, detections):
        return frame, detections
```

### VisionBase

```python
class VisionBase:
    plugin_name = "base"

    def run(self):
        return detections, frame
```

---

## Example Utility

**File:** `iSpy/plugins/utilities/example_utility.py`

Template for writing new utility plugins. Demonstrates the complete plugin
interface:

```python
class ExampleUtility(UtilityBase):
    plugin_name = "example_utility"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "my_setting": {
                "type": "text",
                "label": "My Setting",
                "hint": "A sample text setting.",
                "default": "hello",
            },
        }

    def __init__(self, context: dict):
        super().__init__(context)
        # Access Flask app for route registration
        flask_app = context.get("flask_app")
        if flask_app:
            flask_app.add_url_rule(
                "/api/example", "api_example", self._api_handler
            )

    def update(self, frame_data: dict):
        # Called every vision tick with frame_data dict
        frame = frame_data.get("frame")
        detections = frame_data.get("detections", [])
        pass

    def _api_handler(self):
        from flask import jsonify
        return jsonify(message=self.config.get("my_setting", ""))

    def stop(self):
        pass
```

This demonstrates:
- `config_schema()` with typed settings
- `__init__` with context access and Flask route registration
- `update()` receiving frame_data
- Custom API endpoint
- `stop()` cleanup

---

## Data Flow: Vision Tick -> Utilities

```
Vision loop tick
  |-- frame_data = {
  |     "frame": np.ndarray,
  |     "detections": [Object, ...],
  |     "fps": float,
  |     "vision_s": float,
  |     "detection_count": int,
  |     "camera_lag_s": float,
  |     "loop_s": float,
  |     "robot_pose": {"x", "y", "heading"},
  |     "cameras": [Camera, ...],
  |     "camera_frames": {"name": frame, ...},
  |     "code_times": {"vision": s, "trackers": s, ...},
  |   }
  |
  |-- For each utility plugin:
  |     plugin.update(frame_data)
  |       |-- NetworkTableHandler:
  |       |     _publish_entry() -> _resolve_source() -> _send_data() / _send_detections()
  |       |     _update_viewer_overlay() -> viewer.add_overlay()
  |       |     inst.flush()
  |       |
  |       |-- HealthModule (web module, via web_app.update):
  |       |     Updates fps, vision_s, detections, last_tick, loop_count
  |       |
  |       |-- RollBack:
  |             _start_recorder() on first frame
  |             _write() -> _clean_frame() -> queue.put()
  |             Worker thread -> VideoWriter.write()
  |
  |-- web_app.update(frame_data)
        (see 05-web.md)
```
