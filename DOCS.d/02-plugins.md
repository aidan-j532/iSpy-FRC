# 02 — Plugin System

> Every base class, loader mechanism, built-in plugin, and user extension
> point in the iSpy-FRC plugin architecture.

---

## Overview

Plugins are discovered at startup from two directories:
1. **Built-in**: `iSpy/plugins/<category>/BuiltIn/` — shipped with the project
2. **User**: `iSpy/plugins/<category>/` — user-authored (must subclass the right base)
3. **Pipelines**: `iSpy/plugins/pipelines/` — user-authored vision pipelines

Categories: `trackers`, `utilities`, `frame_processors`, `pipelines`.

All plugins receive a **context dict** at construction time, giving them access to the
global config, cameras, Flask app, and even the iSpy vision instance.

---

## Base Classes (`iSpy/plugins/bases.py` — 153 lines)

### `AddonBase` (line 6)

Abstract base for all plugin types. Provides the common lifecycle methods.

```python
class AddonBase(ABC):
    def __init__(self, name: str, context: dict):
        self.name = name
        self.context = context

    def pre_run(self) -> None:          # Called once before vision loop starts (optional)
    def post_run(self) -> None:         # Called once after vision loop ends (optional)
    def update(self, frame_data) -> None: # Called every tick (optional, override in subclasses)
    def on_camera_added(self, camera) -> None:   # When a camera is added at runtime (optional)
    def on_camera_removed(self, camera) -> None: # When a camera is removed at runtime (optional)
```

`AddonBase` is **not used directly** — it's the parent of `TrackerBase`, `UtilityBase`, etc.

### `TrackerBase(AddonBase)` (line 44)

Post-processing plugins that run on detection lists each tick.

```python
class TrackerBase(AddonBase):
    @abstractmethod
    def update(self, detections: list[Object], robot_x: float,
               robot_y: float, robot_rotation: float,
               robot_timestamp: float) -> list[Object]:
```

- **Input**: raw detections from the vision pipeline, robot pose
- **Output**: transformed/filtered/sorted `list[Object]`
- Called once per tick per camera

### `UtilityBase(AddonBase)` (line 56)

Side-effect plugins that receive the full `frame_data` dict each tick.

```python
class UtilityBase(AddonBase):
    @abstractmethod
    def update(self, frame_data: dict) -> None:
```

Typical uses: NetworkTables publish, health reporting, video recording.

### `FrameProcessorBase(AddonBase)` (line 68)

Frame manipulation plugins that run between frame capture and vision inference.

```python
class FrameProcessorBase(AddonBase):
    @abstractmethod
    def update(self, frame: np.ndarray) -> np.ndarray:
```

- **Input**: BGR frame from camera reader
- **Output**: modified BGR frame (can replace or augment in-place)
- Attached per-camera in `iSpy.py:128-131`

### `VisionBase(AddonBase)` (line 80)

For custom vision pipelines that fully replace the default YOLO pipeline.

```python
class VisionBase(AddonBase):
    @abstractmethod
    def update(self, detections, robot_pose, camera_pose, frame):
```

Used for entirely custom detection strategies (not shown in built-in plugins).

---

## Plugin Loader (`iSpy/plugins/_loader.py` — 56 lines)

### `discover_classes(directory)` (line 8)

Walks a directory tree, imports every `.py` file, and yields all classes that
subclass `AddonBase`:

```python
def discover_classes(directory):
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.py') and not filename.startswith('_'):
                filepath = os.path.join(dirpath, filename)
                module_name = os.path.splitext(os.path.relpath(filepath, parent))[0]
                module_name = module_name.replace(os.sep, '.')
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, AddonBase) and obj is not AddonBase:
                        yield obj
```

Key behaviors:
- Skips `__init__.py` and files starting with `_`
- Registers modules in `sys.modules` to prevent duplicate imports
- Yields the **class itself**, not an instance

### `load_plugins(addon_type, context)` (line 38)

Main entry point. Called 3 times at startup (trackers, utilities, frame_processors):

```python
def load_plugins(addon_type, context):
    plugins = []
    discovered = {}

    # 1. Discover built-in plugins
    builtin_dir = Path(f"iSpy/plugins/{addon_type}/BuiltIn")
    if builtin_dir.exists():
        for cls in discover_classes(builtin_dir):
            discovered[cls.__name__] = cls

    # 2. Discover user plugins
    user_dir = Path(f"iSpy/plugins/{addon_type}")
    if user_dir.exists():
        for cls in discover_classes(user_dir):
            discovered[cls.__name__] = cls

    # 3. Filter by enabled config
    for name, cls in discovered.items():
        settings = context["config"].get_addon_settings(addon_type, name)
        if settings is not None:  # presence == enabled
            plugins.append(cls(name, context))

    # 4. Run pre_run hooks
    for plugin in plugins:
        plugin.pre_run()

    return plugins
```

---

## Context Dict (Passed to All Plugins)

Created in `iSpy.__init__()` at lines 70-94:

```python
context = {
    "config":         iSpyAddonConfig,   # Plugin-specific settings wrapper
    "global_config":  iSpyConfig,        # Full system config
    "vision_instance": iSpy,             # The iSpy vision loop instance
    "flask_app":      Flask,             # Flask web app (for adding routes)
    "cameras":        list[VisionPipeline], # All camera instances
    "viewer3d":       Viewer3DModule,    # 3D viewer (for updating overlay)
    "global_logger":  logging.Logger,    # iSpy.plugins logger
}
```

---

## Built-In Tracker: ObjectTracker (`iSpy/plugins/trackers/BuiltIn/ObjectTracker.py` — 120 lines)

EMA (Exponential Moving Average) position smoothing with stale-object tracking.

### Schema (line 12-18)

```python
PLUGIN_SCHEMA = {
    "min_detections": {
        "type": "int", "description": "Minimum detections before emitting",
        "default": 2, "min": 1, "max": 100,
    },
    "distance_threshold": {
        "type": "float", "description": "Max distance to match a detection to a tracked object",
        "default": 2.0, "min": 0.01, "max": 100.0, "step": 0.01,
    },
    "stale_threshold": {
        "type": "int", "description": "Seconds without detection before destroying a tracked object",
        "default": 2, "min": 1, "max": 30,
    },
    "smoothing_factor": {
        "type": "float", "description": "EMA alpha (0=no smoothing, 1=instant)",
        "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01,
    },
}
```

### `__init__` (line 25-40)

Initializes from config:
```python
self.min_detections = settings.get("min_detections", 2)
self.distance_threshold = settings.get("distance_threshold", 2.0)
self.stale_threshold = settings.get("stale_threshold", 2)
self.smoothing_factor = settings.get("smoothing_factor", 0.4)
self.tracked_objects = {}       # id → {"object": Object, "alive_time": float}
self.next_id = 0
```

### `update(detections, robot_x, robot_y, robot_rotation, robot_timestamp)` (line 42-120)

**Step 1: Age tracked objects** (lines 47-53)
```python
for obj_id in list(self.tracked_objects.keys()):
    entry = self.tracked_objects[obj_id]
    entry["alive_time"] += 0.04    # ~25Hz tick assumption
    if entry["alive_time"] > self.stale_threshold:
        del self.tracked_objects[obj_id]
```

**Step 2: Convert detections to robot frame** (lines 55-58)
```python
for det in detections:
    det.relative_to(robot_x, robot_y, robot_rotation)
```

**Step 3: Match new detections to tracked objects** (lines 60-118)

For each new detection:
1. Find closest tracked object by Euclidean distance in `(x, y)` space
2. If within `distance_threshold` → **merge** via `_merge()`
3. If no match → **create new tracked object**

```python
def _merge(self, obj_id, detection, timestamp):
    existing = self.tracked_objects[obj_id]["object"]
    alpha = self.smoothing_factor

    existing.x     = alpha * detection.x     + (1 - alpha) * existing.x
    existing.y     = alpha * detection.y     + (1 - alpha) * existing.y
    existing.z     = alpha * detection.z     + (1 - alpha) * existing.z
    existing.roll  = alpha * detection.roll  + (1 - alpha) * existing.roll
    existing.pitch = alpha * detection.pitch + (1 - alpha) * existing.pitch
    existing.yaw   = alpha * detection.yaw   + (1 - alpha) * existing.yaw

    self.tracked_objects[obj_id]["alive_time"] = 0.0
    self.tracked_objects[obj_id]["object"].confidence = detection.confidence
```

**Step 4: Emit** (line 118)
```python
return [
    entry["object"]
    for entry in self.tracked_objects.values()
    if entry["alive_time"] == 0.0  # only recently-updated objects
]
```

---

## Built-In Tracker: PathPlanner (`iSpy/plugins/trackers/BuiltIn/PathPlanner.py` — 63 lines)

DBSCAN-clustering tracker for grouping detections into field-level clusters.

### `__init__` (line 15-26)

```python
self.settings = context["config"]
self.cluster_distance = self.settings.get("cluster_distance", 1.5)    # meters
self.min_samples = self.settings.get("min_samples", 3)
self.tracked_paths = {}
```

### `update(detections, robot_x, robot_y, robot_rotation, robot_timestamp)` (line 28-63)

1. **Cluster** detections via `_cluster(detections)`
2. **Match clusters** to `self.tracked_paths` by centroid proximity
3. **Emit** the cluster as a single aggregated `Object`

#### `_cluster(detections)` (line 46-60)

Uses `sklearn.cluster.DBSCAN` with Euclidean metric:
```python
from sklearn.cluster import DBSCAN

points = [[d.x, d.y] for d in detections]
clustering = DBSCAN(eps=self.cluster_distance, min_samples=self.min_samples)
labels = clustering.fit_predict(points)

# Group detections by label, compute centroids
for label in set(labels):
    if label == -1: continue  # noise
    members = [d for d, l in zip(detections, labels) if l == label]
    centroid_x = sum(m.x for m in members) / len(members)
    centroid_y = sum(m.y for m in members) / len(members)
    ...
```

---

## Built-In Utility: NetworkTableHandler (`iSpy/plugins/utilities/BuiltIn/NetworkHandler.py` — 250 lines)

Publishes detections and robot state to NetworkTables v4.

### `__init__` (line 12-62)

```python
def __init__(self, name, context):
    self.config = context["config"]
    self.global_config = context["global_config"]
    self.cameras = context["cameras"]
    self.vision_instance = context["vision_instance"]
    self.viewer3d = context["viewer3d"]

    # NT4 connection
    server = self.config.get("server", "127.0.0.1")
    port = self.config.get("port", 5810)
    self.inst = NetworkTableInstance.getDefault()
    self.inst.startClient4(f"ispy-{gethostname()}")
    self.inst.setServer(server, port)
    self.inst.startDSClient()  # fallback to DS if server unreachable
    self.inst.setNetworkTablesAutoThrottle(True)
    self.inst.setServerTeam(self.config.get("team_number", 0))

    # Subtables
    self.main_table = self.inst.getTable("iSpy")
    self.robot_table = self.inst.getTable("FMSInfo")  # or robot-specific

    # Pose support (WPILib struct arrays)
    self.robot_table.putValue("RobotPose", StructArrayData(Pose2d))
```

### `update(frame_data)` (line 82-120)

Main tick handler:
1. Publishes all configured entries via `_publish_entry()`
2. Calls `_send_detections()` to publish detection struct array
3. Calls `_update_robot_pose()` to read robot pose from NT
4. Calls `_update_viewer_overlay()` to push robot box to 3D viewer
5. Flushes all pending writes

### `_resolve_source(source_key, frame_data, robot_pose)` (line 122-150)

Resolves a `source` string to a value:
- `"detection_count"` → `len(frame_data["detections"])`
- `"fps"` → `frame_data["fps"]`
- `"vision_s"` → `frame_data["vision_s"]`
- `"robot_pose"` → formatted string
- `"detections"` → handled separately
- Or any other key from frame_data

### `_send_detections(detections, timestamp, table)` (line 152-198)

Converts detections to WPILib `FuelStruct[]` and publishes:
```python
structs = []
for detection in detections:
    structs.append(FuelStruct([
        Float(detection.x),
        Float(detection.y),
        Float(detection.z),
        Float(detection.confidence),
        Float(detection.class_confidence),
        Float(detection.yaw),
        Float(detection.width),
        Float(detection.length),
        Float(detection.height),
        UInt8(detection.id),
        UInt8(detection.target_class),
        UInt8(0),  # padding
    ]))
table.putValue("Detections", StructArrayData(FuelStruct, structs))
```

### `_update_robot_pose(robot_pose)` (line 223-250)

Reads robot pose from NT and creates a `Pose2d` object:
```python
data = self.robot_table.getValue("RobotPose", None)
if data:
    x, y, heading = data[0][0], data[0][1], data[0][2]
    self.vision_instance.robot_pose = Pose2d(x, y, heading)
    self.robot_table.putValue(
        "RobotPose",
        StructArrayData(Pose2d, [[self.vision_instance.robot_pose]])
    )
```

### Config Schema

```python
PLUGIN_SCHEMA = {
    "team_number": {"type": "int", "default": 0, "description": "FRC team number"},
    "server": {"type": "str", "default": "127.0.0.1", "description": "NT server IP"},
    "port": {"type": "int", "default": 5810, "description": "NT server port"},
    "publish": {
        "type": "dict",
        "default": {"detections": {"source": "detections"}},
        "description": "Table of entries: {name: {source: str, ...}}",
    },
}
```

---

## Health Reporting (core web module, not an add-on)

The old `HealthReporter` utility was merged into the always-on core
`HealthModule` (`iSpy/web/modules/health.py`) along with the legacy
`StatusReporter`. There is deliberately no opt-in health add-on anymore — this
keeps `/health` registered exactly once so enabling every add-on can never
cause a Flask duplicate-endpoint collision.

### Endpoints (registered once at boot)

#### `GET /health`
Minimal watchdog contract: `{"status": "ok" | "degraded"}` with 200/503.

#### `GET /health/detailed`
Full payload: uptime_s, loop_count, loop_stale_s, fps, vision_ms,
detections, per-camera `{name, source, ok, frame_age_ms}` and
network_tables `{enabled, connected}`. 503 when degraded.

#### `GET /api/health`
Everything above plus live `plugins` statuses pulled from the vision instance.

### Stale threshold
Configured by the **top-level** config key `health_stale_threshold`
(default `1.0`, exposed in Settings → Advanced). The vision loop feeds the
module via `update(frame_data)`; NetworkTables connectivity comes from
`set_network_handler(nt)` wiring done in `iSpy/iSpy.py`.

---

## Built-In Utility: RollBack (`iSpy/plugins/utilities/BuiltIn/RollBack.py` — 228 lines)

Video recording with async queue-based writer to avoid blocking the vision loop.

### Config Schema

```python
PLUGIN_SCHEMA = {
    "file_path": {
        "type": "str",
        "default": "VideoRecordings",
        "description": "Directory for recorded videos",
    },
    "codec": {
        "type": "str",
        "default": "MJPG",
        "description": "Video codec (MJPG, XVID, H264)",
    },
    "frame_limit": {
        "type": "int",
        "default": 500,
        "description": "Max frames to buffer before force-flushing",
    },
}
```

### `__init__` (line 10-30)

```python
self._recording = False
self._writer = None
self._writer_thread = None
self._queue = queue.Queue()
self._file_path = self.settings.get("file_path", "VideoRecordings")
self._codec = self.settings.get("codec", "MJPG")
self._frame_limit = self.settings.get("frame_limit", 500)
```

### `update(frame_data)` (line 32-60)

If recording is active, puts the frame into the async queue:
```python
def update(self, frame_data):
    if not self._recording:
        return
    frame = frame_data.get("frame")
    if frame is not None:
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass  # drop frame if buffer full
```

### `_start_recording(frame)` (line 62-85)

1. Creates output filename with timestamp: `VideoRecordings/record_2024-01-15_12-00-00.avi`
2. Creates `cv2.VideoWriter` with the configured codec, 30 FPS
3. Starts `_worker` thread
4. Puts the first frame into the queue

### `_worker()` (line 87-105)

Background thread that writes frames from the queue to disk:
```python
def _worker(self):
    while self._recording or not self._queue.empty():
        try:
            frame = self._queue.get(timeout=0.1)
            self._writer.write(frame)
        except queue.Empty:
            continue
    self._writer.release()
```

### `_stop_recording()` (line 107-118)

1. Sets `_recording = False`
2. Joins the worker thread (waits for queue to drain)
3. Releases the VideoWriter

---

## Frame Processors

### `FrameProcessorBase` Usage in the Vision Loop

In `iSpy.py:128-131`, frame processors are attached to cameras:
```python
for processor in self.frame_processors:
    for camera in self.cameras:
        camera.add_frame_processor(processor)
```

In `ObjectDetectionCamera.run()` (object_detection.py:880-881), they are applied
before inference:
```python
for processor in self._frame_processors:
    frame = processor.update(frame)
```

### Example Frame Processor (`iSpy/plugins/frame_processors/example_frame_processor.py`)

Template provided in the repo — shows how to:
1. Subclass `FrameProcessorBase`
2. Implement `update(frame)` returning the modified frame
3. Apply brightness/contrast adjustments or draw overlays

---

## Plugin Lifecycle Summary

```
BOOT
  │
  ├── discover_classes() → finds all plugin classes
  ├── Filter: is_addon_enabled(type, name)?
  ├── Instantiate: cls(name, context)
  ├── pre_run() on each plugin
  └── Return plugins list

VISION LOOP (each tick)
  │
  ├── Frame Processors:
  │   └── frame = processor.update(frame)    [before inference]
  │
  ├── Vision Pipeline:
  │   └── detections, annotated = pipeline.run()
  │
  ├── Trackers:
  │   └── detections = tracker.update(detections, pose...)
  │
  └── Utilities:
      └── util.update(frame_data)

SHUTDOWN
  │
  ├── post_run() on each plugin
  └── Resources released (video writer, NT client, etc.)
```

---

## Writing a Custom Plugin

### Tracker Example

```python
# iSpy/plugins/trackers/my_tracker.py

from iSpy.plugins.bases import TrackerBase

PLUGIN_SCHEMA = {
    "max_distance": {
        "type": "float",
        "default": 3.0,
        "description": "Maximum match distance",
        "min": 0.1,
        "max": 10.0,
    },
}

class MyTracker(TrackerBase):
    def __init__(self, name, context):
        super().__init__(name, context)
        self.max_distance = context["config"].get("max_distance", 3.0)
        self.my_state = {}

    def update(self, detections, robot_x, robot_y, robot_rotation, robot_timestamp):
        # Your tracking logic here
        return detections  # return filtered/modified list
```

### Utility Example

```python
# iSpy/plugins/utilities/my_utility.py

from iSpy.plugins.bases import UtilityBase

PLUGIN_SCHEMA = {
    "message": {
        "type": "str",
        "default": "Hello!",
        "description": "Message to log",
    },
}

class MyUtility(UtilityBase):
    def __init__(self, name, context):
        super().__init__(name, context)
        self.message = context["config"].get("message", "Hello!")
        context["global_logger"].info(f"MyUtility initialized: {self.message}")

    def update(self, frame_data):
        count = frame_data["detection_count"]
        if count > 0:
            self.context["global_logger"].debug(
                f"{self.message} — saw {count} detections"
            )
```

### Enabling a Plugin

Add an entry in `config.json` under the appropriate `plugins` section:
```json
{
    "plugins": {
        "trackers": {
            "MyTracker": {
                "max_distance": 2.5
            }
        },
        "utilities": {
            "MyUtility": {
                "message": "Detection alert"
            }
        }
    }
}
```

Presence of the key = enabled. Remove the key to disable.

---

## User Pipeline Plugins (`iSpy/plugins/pipelines/`)

User pipelines follow the same pattern as built-in pipelines (subclasses of
`VisionBase` or directly of `AddonBase`) and are discovered by `get_pipeline_classes()`
in `iSpy/vision/pipelines/__init__.py`.

To create a user pipeline:
1. Create `iSpy/plugins/pipelines/my_pipeline.py`
2. Subclass the appropriate base
3. Implement the required interface
4. Reference it in a camera's `pipeline.name` config field

```python
# iSpy/plugins/pipelines/my_pipeline.py
from iSpy.plugins.bases import VisionBase

class MyCustomPipeline(VisionBase):
    def __init__(self, cam_config, config):
        # cam_config: iSpyCameraConfig for this camera
        # config: global iSpyConfig
        super().__init__(cam_config, config)
        # Initialize your model / detector

    def run(self):
        # Return (detections: list[Object], annotated_frame: ndarray)
        ...
```

---

## Plugin Configuration Schema

Every plugin can declare a `PLUGIN_SCHEMA` dict at module level. This is consumed
by the web UI's PluginStatus module (`iSpy/web/Backend/PluginStatus.py`) to render
dynamic settings forms.

Schema format:
```python
PLUGIN_SCHEMA = {
    "setting_name": {
        "type": "int" | "float" | "str" | "bool" | "dict",
        "default": <value>,
        "description": "Human-readable description",
        "min": <number>,       # optional, for int/float
        "max": <number>,       # optional, for int/float
        "step": <number>,      # optional, for float
    },
}
```

The web UI uses this to render:
- Text inputs for `str`
- Number inputs with min/max/step for `int`/`float`
- Toggle switches for `bool`
- Nested forms for `dict`

---

## Viewer3D Integration

The `context["viewer3d"]` object provides a `draw_robot()` method that plugins
can call to push overlay data to the 3D viewer. The `NetworkTableHandler` uses
this to draw the robot's position on the field:

```python
self.viewer3d.draw_robot(
    position=Pose2d(x, y, heading),
    dimensions=(18, 24, 6),  # L × W × H in inches
    color="blue"
)
```

The overlay data is served via `GET /api/3dviewer/overlay` as JSON for the
Three.js-based 3D viewer module.
