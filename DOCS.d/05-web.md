# 05 -- Web Backend

> Flask application factory, module system, shared context, and every web API
> route in the iSpy backend. This document covers every function, class, method,
> data flow, and code path in the web layer.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [WebApp -- Application Factory](#webapp----application-factory)
3. [WebModule -- Abstract Base](#webmodule----abstract-base)
4. [PluginStatus -- Add-on Lifecycle](#pluginstatus----add-on-lifecycle)
5. [Settings -- Global Config](#settings----global-config)
6. [cameras.py -- Camera Management](#cameraspy----camera-management)
7. [dashboard.py -- SSE & Metrics](#dashboardpy----sse--metrics)
8. [health.py -- Health Checks](#healthpy----health-checks)
9. [viewer3d.py -- 3D Viewer Backend](#viewer3dpy----3d-viewer-backend)
10. [metrics.py -- Time-Series Ring Buffer](#metricspy----time-series-ring-buffer)
11. [models.py -- YOLO Model Management](#modelspy----yolo-model-management)
12. [datasets.py -- Image Dataset Management](#datasetspy----image-dataset-management)
13. [logs.py -- Log Viewer](#logspy----log-viewer)
14. [onboarding.py -- First-Run Tour](#onboardingpy----first-run-tour)
15. [recommendations.py -- Config Health](#recommendationspy----config-health)
16. [Support Files](#support-files)
17. [Request Lifecycle](#request-lifecycle)

---

## Architecture Overview

```
iSpyWebApp (WebApp.py, 107 lines)
  |-- Flask app (template_folder, static_folder)
  |-- Shared context dict
  |     |-- config, cameras, flask_app, vision_instance, dashboard_module
  |-- Module registry (dict[str, WebModule])
  |     |-- cameras       (1824 lines)  -- Camera CRUD, calibration, feeds
  |     |-- dashboard     (235 lines)   -- SSE stream, system metrics
  |     |-- health        (141 lines)   -- Health endpoints
  |     |-- viewer3d      (130 lines)   -- 3D detection/overlay API
  |     |-- metrics       (105 lines)   -- Time-series ring buffer
  |     |-- models        (202 lines)   -- YOLO .pt management
  |     |-- datasets      (181 lines)   -- Quantization datasets
  |     |-- logs          (48 lines)    -- Log tail
  |     |-- onboarding    (34 lines)    -- First-run tour flag
  |     |-- recommendations (16 lines)  -- Config health checks
  |     |-- settings      (113 lines)   -- Global config GET/POST
  |     |-- setup_wizard  (75 lines)    -- First-boot inline wizard
  |     |-- plugin_status (465 lines)   -- Add-on lifecycle
  |-- Global route: "/" -> dashboard.html
```

The `create_app(cameras, config)` factory at `WebApp.py:106` instantiates
`iSpyWebApp` and returns it. Every module receives the same shared `context`
dict so they can cross-reference each other (e.g., `context["dashboard_module"]`
at line 64, `context["viewer3d"]` via the module registry).

---

## WebApp -- Application Factory

**File:** `iSpy/web/Backend/WebApp.py` (107 lines)

### `iSpyWebApp.__init__(self, cameras, config)` (lines 24-65)

1. Creates a `Flask` app with `template_folder` pointing to `iSpy/web/templates/`
   and `static_folder` to `iSpy/web/static/` (line 28-32). The web root is
   resolved at module level:
   ```python
   _WEB_ROOT = Path(__file__).resolve().parent.parent  # line 20
   ```

2. Builds the **shared context dict** (lines 33-38):
   ```python
   context = {
       "config": config,
       "cameras": cameras or [],
       "flask_app": flask_app,
       "vision_instance": None,  # set later via set_vision_instance()
   }
   ```

3. Instantiates all **13 WebModule subclasses** (lines 42-56), passing the
   shared context:
   ```python
   self.modules: dict[str, WebModule] = {
       "cameras": CamerasModule(context),
       "models": ModelsModule(context),
       "datasets": DatasetsModule(context),
       "viewer3d": Viewer3DModule(context),
       "dashboard": DashboardModule(context),
       "health": HealthModule(context),
       "logs": LogsModule(context),
       "metrics": MetricsModule(context),
       "settings": SettingsModule(context),
       "onboarding": OnboardingModule(context),
       "setup_wizard": SetupWizardModule(context),
       "recommendations": RecommendationsModule(context),
       "plugin_status": PluginStatusModule(context),
   }
   ```

4. Iterates all modules and calls `register_routes(flask_app)` (lines 58-62).
   Any exception during registration is logged but does not crash the server.

5. Stores a back-reference to the dashboard module in the context (line 64):
   ```python
   self.context["dashboard_module"] = self.modules.get("dashboard")
   ```

6. Registers the root route `/` which renders `dashboard.html` (line 65):
   ```python
   self.flask_app.add_url_rule("/", "root", lambda: render_template("dashboard.html"))
   ```

### `update(self, frame_data)` (lines 67-72)

Called every vision tick. Iterates all modules and calls `mod.update(frame_data)`
with try/except per module so one module's failure does not crash others.

### `set_vision_instance(self, vision)` (lines 74-75)

Stores the vision loop reference in `self.context["vision_instance"]`. This
allows modules like `PluginStatus` and `Dashboard` to query the live vision
instance for plugin statuses, camera lists, etc.

### `set_cameras(self, cameras)` (lines 77-84)

Updates `self.context["cameras"]` and propagates to every module that has a
`set_cameras` method via duck-typing (`hasattr(mod, "set_cameras")`). Currently
`HealthModule` and `CamerasModule` implement this.

### `run(self, host, port)` (lines 86-88)

Calls `self.start()` then `self.flask_app.run(host=host, port=port,
threaded=True)`. The `threaded=True` flag enables Flask's built-in threaded
mode so MJPEG streams and SSE endpoints run in parallel.

### `start(self)` (lines 90-96)

Calls `mod.start()` on any module that has a `start` attribute. Currently
only `CamerasModule` uses this to launch the auto-discover background thread.

### `stop(self)` (lines 98-103)

Calls `mod.stop()` on all modules for cleanup.

### `create_app(cameras, config)` (lines 106-107)

Factory function. Returns a fully initialized `iSpyWebApp` instance.

---

## WebModule -- Abstract Base

**File:** `iSpy/web/Backend/WebModule.py` (17 lines)

```python
class WebModule(ABC):
    plugin_name = "base_web_module"     # line 5

    def __init__(self, context: dict):  # line 7
        self.context = context

    def register_routes(self, flask_app):  # line 10
        pass

    def update(self, frame_data: dict):    # line 13
        pass

    def stop(self):                        # line 16
        pass
```

Despite inheriting from `ABC`, no methods are decorated `@abstractmethod` --
subclasses are free to override only the methods they need. The `plugin_name`
class attribute is used for logging but not for routing.

---

## PluginStatus -- Add-on Lifecycle

**File:** `iSpy/web/Backend/PluginStatus.py` (465 lines)

This is the largest backend support file. It manages the full lifecycle of
add-ons: discovery, enable/disable, settings editing, source code viewing,
creation, upload, and deletion.

### Module-Level Constants

#### `_TYPE_MAP` (lines 18-23)

Maps plugin type strings to tuples of `(subdir, base_class, base_class_name,
lifecycle_method)`:

```python
_TYPE_MAP = {
    "tracker": ("trackers", TrackerBase, "TrackerBase", "update"),
    "utility": ("utilities", UtilityBase, "UtilityBase", "update"),
    "frame_processor": ("frame_processors", FrameProcessorBase, "FrameProcessorBase", "process"),
    "vision_pipeline": ("pipelines", VisionBase, "VisionBase", "run"),
}
```

Each subdir is relative to `iSpy/plugins/`. The `base_name` is used for source
code validation (line 361-365).

#### `_NAME_RE` (line 25)

Regex `^[a-zA-Z_][a-zA-Z0-9_]*$` -- validates plugin filenames to prevent
path traversal.

#### `_ADDON_TYPES_FROM_PTYPE` (lines 27-30)

Maps `ptype` to config key for `config.disable_addon()`:
```python
{"tracker": "trackers", "utility": "utilities", "frame_processor": "frame_processors"}
```

### `_coerce_setting_value(value, defn)` (lines 33-68)

Validates and casts a single setting value against its schema definition.

**Type handling:**
- `"number"`: Rejects booleans, casts to `float`, preserves `int` if the
  original was an `int` (line 47)
- `"toggle"`: Accepts string ("1"/"true"/"yes"/"on") or bool
- `"text"`: Casts to `str`
- `"list"`: Validates it's a list of dicts, recursively coerces each field
  using the nested `fields` definition (lines 54-67)
- Default: returns value unchanged

### `_build_vision_pipeline_payloads()` (lines 71-102)

Discovers all vision pipeline classes from `iSpy.vision.pipelines.get_pipeline_classes()`
and builds API payloads with: name, class_name, config_schema, show_common_fields,
show_calibration, beta, recommended_format.

### `PluginStatusModule.register_routes()` (lines 108-120)

Registers 10 routes:

| Route | Method | Handler | Purpose |
|-------|--------|---------|---------|
| `/addons` | GET | lambda | Renders `addons.html` |
| `/plugins` | GET | lambda | Back-compat alias for `/addons` |
| `/api/plugins/status` | GET | `_status` | Lists enabled plugins + their live status |
| `/api/plugins/available` | GET | `_available` | Lists all discovered plugins |
| `/api/plugins/toggle` | POST | `_toggle` | Enable/disable a plugin |
| `/api/plugins/settings` | POST | `_save_settings` | Save plugin settings |
| `/api/plugins/upload` | POST | `_upload` | Upload .py plugin file |
| `/api/plugins/create` | POST | `_create` | Create plugin from pasted code |
| `/api/plugins/<ptype>/<name>` | DELETE | `_delete` | Delete a plugin file |
| `/api/plugins/<ptype>/<name>/source` | GET | `_source` | Read plugin source code |

### `_status()` (lines 124-140)

Reads from `vision_instance.trackers`, `.utilities`, `.frame_processors` and
returns a JSON array of `{name, type, status}` objects. Returns empty array if
no vision instance.

### `_available()` (lines 145-193)

Iterates all four plugin types. For each:
1. For `vision_pipeline`: reads from `get_pipeline_classes()` static registry,
   marks them `builtin=True`
2. For others: calls `load_plugins(_PLUGIN_ROOT / subdir, base_cls)` to discover
   `.py` files in the plugins directory
3. Checks config for enabled state via `config.get_addon_settings()`
4. Reads `config_schema()` from the class
5. Marks files under `BuiltIn/` as `builtin=True` (not deletable)

### `_toggle()` (lines 241-284)

POST handler. Flow:
1. Validates `name`, `type` against `_TYPE_MAP`
2. Discovers plugins to verify the name exists
3. Maps ptype to config section: `{"tracker": "trackers", ...}`
4. If enabling: reads schema defaults, calls `config.enable_addon()` with
   defaults pre-populated so the UI shows values immediately (lines 265-278)
5. If disabling: calls `config.disable_addon()`
6. Returns `{success, enabled, needs_restart: True}`

### `_save_settings()` (lines 288-343)

POST handler. Flow:
1. Validates name/type, rejects `vision_pipeline` (configured per-camera)
2. Discovers plugin class to get `config_schema()`
3. Verifies the plugin is currently enabled
4. Iterates all submitted keys, validates each against schema (rejects unknown
   keys for typo-proofing)
5. Coerces values via `_coerce_setting_value()`
6. Calls `config.update_addon_settings()` and `config.save()`
7. Returns `{success, settings, needs_restart: True}`

### `_validate_addon_source()` (lines 360-370)

Checks uploaded/pasted code for:
- Contains the base class name (e.g., `TrackerBase`)
- Contains `plugin_name` class attribute
- Under 200KB

### `_create()` (lines 372-403)

POST handler for creating from pasted code:
1. Validates type, filename, code
2. Calls `_validate_addon_source()`
3. Resolves safe path via `_resolve_safe_path()`
4. Rejects if file already exists (409)
5. Creates parent dirs, writes file

### `_upload()` (lines 405-433)

POST handler for file upload:
1. Reads `type` from form data, `file` from `request.files`
2. Validates .py extension, source code, safe path
3. Rejects if file exists (409)
4. Writes file

### `_delete()` (lines 435-465)

DELETE handler:
1. Resolves file path, rejects if not found
2. **Blocks deletion of BuiltIn add-ons** (line 443-444)
3. If plugin is enabled in config, disables it first to prevent dangling refs
4. Deletes the file

### `_source()` (lines 214-237)

GET handler:
1. Validates ptype, rejects `vision_pipeline` (bundled source not exposed)
2. Resolves file via `_filename_for()`
3. Reads and returns file content

### `_resolve_safe_path()` (lines 347-358)

Validates a filename: must match `_NAME_RE`, must resolve within the target
subdir, never touches `BuiltIn/`.

### `_filename_for()` (lines 195-212)

Searches for the `.py` file containing a given plugin name by scanning
`*.py` files in the subdir and checking for the quoted plugin_name string.

---

## Settings -- Global Config

**File:** `iSpy/web/Backend/Settings.py` (113 lines)

### `_RESTART_REQUIRED_KEYS` (lines 7-10)

Set of top-level config keys that require a vision restart when changed:
```python
{"unit", "debug_mode", "frame_sync", "optimize", "log_level",
 "metrics", "plugins", "camera_configs", "device", "num_gpus"}
```

### `SettingsModule.register_routes()` (lines 16-22)

| Route | Method | Handler | Purpose |
|-------|--------|---------|---------|
| `/settings` | GET | lambda | Renders `settings.html` |
| `/api/settings` | GET | `_get` | Returns full config + defaults |
| `/api/settings` | POST | `_post` | Updates config, detects changes |
| `/api/settings/compare` | POST | `_compare` | Dry-run diff without saving |
| `/api/settings/snapshot` | POST | `_snapshot` | Saves config snapshot |
| `/api/settings/restore` | POST | `_restore` | Restores snapshot (one-shot) |

### `_get()` (lines 24-26)

Returns `config.config` (current) and `config.default_config` as JSON.

### `_post()` (lines 50-81)

The most complex route. Flow:
1. Deep-copies old config
2. Calls `config._update_config(config_data)` then `config.save()`
3. Computes `changed_keys` via `_find_changed_keys()`
4. Checks if any changed key is in `_RESTART_REQUIRED_KEYS` or frontend-provided
   `restart_keys`
5. Runs `get_structured_recommendations()`, pushes critical warnings to the
   dashboard via SSE in real-time (lines 68-75):
   ```python
   if critical and dash:
       dash._push_sse({
           "type": "config_warning",
           "messages": [r["message"] for r in critical],
       })
   ```
6. Returns `{success, needs_restart, changed}`

### `_compare()` (lines 83-100)

Dry-run version of `_post()`: deep-copies config, applies changes to the copy,
computes diff, but never saves.

### `_snapshot()` / `_restore()` (lines 28-48)

Uses `save_store.write()` / `save_store.read()` to persist a snapshot in
`Save/config_snapshot.json`. Restore is one-shot: after restoring, the snapshot
is consumed (set to `{config: None, taken: False}`).

On restore (line 44-46), calls `ensure_camera_entries_ready()` and
`config._rebuild_camera_configs()` to normalize the restored config.

### `_find_changed_keys(old, new, prefix)` (lines 102-113)

Recursive dict diff. Returns a set of dotted key paths (e.g.,
`"unit"`, `"plugins.trackers.my_tracker"`). Both old and new must be dicts to
recurse; otherwise compares with `!=`.

---

## cameras.py -- Camera Management

**File:** `iSpy/web/modules/cameras.py` (1824 lines)

The largest module by far. Handles camera CRUD, live MJPEG feeds, image tuning,
and the full calibration wizard (focal length, ChArUco intrinsics, PnP).

### Module-Level Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `COCO17_OBJECT_POINTS` | lines 29-47 | Default 3D keypoints for COCO17 pose models |
| `_STALE_EVICT_S` | 10.0 | Evict camera frames older than this |
| `_FEED_TIMEOUT_S` | 15.0 | Break MJPEG generator after this idle time |
| `_AUTO_CAPTURE_MIN_SOLVE` | 6 | Min captured frames before rolling solve |
| `_AUTO_COVERAGE_MIN` | 0.6 | Min board coverage ratio for auto-capture |
| `_AUTO_DIVERSITY_PX` | 12.0 | Min pixel distance between captures |
| `_DETECT_MAX_DIM` | 1280 | Downscale detection work copy to this |
| `_FEED_MAX_DIM` | 1280 | Cap feed width for JPEG encode |
| `_AUTO_SOLVE_INTERVAL_S` | 1.5 | Throttle interval for rolling solve |
| `_AUTO_SOLVE_MAX_CAPTURES` | 10 | Max frames used in live RMS preview |
| `_MAX_OVERLAYS_DRAWN` | 20 | Max captured overlays per streamed frame |
| `_TUNING_KEYS` | tuple | brightness, contrast, saturation, etc. |
| `_TUNING_DEFAULTS` | dict | Default values for tuning knobs |

### Helper Functions

#### `_to_float(value, default)` (line 102-106)
Safe float conversion with default fallback.

#### `_camera_calibrated(entry)` (lines 109-115)
Returns `True` if the camera entry has calibration data (camera_matrix +
dist_coeffs, focal_length_pixels, or fov > 0).

#### `_decode_base64_frame(image_b64)` (lines 118-126)
Decodes a base64-encoded image (with optional `data:` prefix) into a numpy array.

#### `_pipeline_schema_keys(pipeline_name)` (lines 129-141)
Returns the set of valid keys for a pipeline's config schema.

#### `_prune_stale_pipeline_settings(entry)` (lines 144-168)
Removes pipeline settings that belong to a different pipeline type (prevents
"ghost" settings from old configs). Also removes aliased keys.

#### `_vision_model_target_format(settings)` (lines 171-179)
Returns the target optimization format (onnx, etc.) from pipeline settings.

#### `_resolve_vision_model_files(settings)` (lines 192-206)
Ensures `source_pt` and `file_path` point to the correct files, checking for
optimized artifacts.

### Windows Camera Discovery

#### `_windows_cameras_from_registry()` (lines 209-246)
Reads `HKLM\SYSTEM\CurrentControlSet\Control\DeviceClasses\{e5323777-...}`
to enumerate USB cameras. Deduplicates by stripping `&MI_XX` from hardware IDs
(UVC cameras expose two interfaces: video + metadata).

#### `_windows_camera_name(iface)` (lines 249-320)
Reads friendly name from registry, falling back to parent USB node if the
interface name is generic.

### Linux Camera Discovery

#### `_v4l2_caps(video_path)` (lines 336-354)
Reads V4L2 capability bits via `ioctl(VIDIOC_QUERYCAP)` to determine if a
device is a capture device.

#### `_linux_is_capture_node(video_path)` (lines 374-381)
Returns True/False/None based on V4L2 capabilities.

#### `_linux_device_groups()` (lines 413-440)
Groups `/dev/video*` nodes by physical device using either `v4l2-ctl --list-devices`
or sysfs device path.

### CamerasModule Class

#### `__init__()` (lines 445-461)

Instance variables:
```python
self.lock = threading.Lock()          # Guards self.frames, self.dims, etc.
self.frames: dict[str, np.ndarray]    # Latest frame per camera name
self.dims: dict[str, tuple]           # (width, height) per camera
self.last_seen: dict[str, float]      # monotonic timestamp per camera
self.sources: dict[str, str]          # device_id or source per camera
self.live_cameras: dict[str, object]  # Camera instances from vision loop
self.calib_sessions: dict[str, dict]  # Per-camera calibration session state
self.calib_lock = threading.Lock()    # Guards calib_sessions
self._sse_clients: list               # SSE subscriber queues
```

#### `register_routes()` (lines 463-492)

Registers 28 routes. Full table:

| Route | Method | Handler | Lines |
|-------|--------|---------|-------|
| `/cameras` | GET | lambda -> cameras.html | 464 |
| `/api/cameras` | GET | `_api_cameras` | 465 |
| `/api/cameras/discover` | GET | `_discover` | 466 |
| `/video/<camera_name>` | GET | `_video_feed` | 467 |
| `/api/cameras/config` | POST | `_add_camera` | 468 |
| `/api/cameras/config/<cam_name>` | GET | `_get_camera` | 469 |
| `/api/cameras/config/<cam_name>` | PUT | `_update_camera` | 470 |
| `/api/cameras/config/<cam_name>` | DELETE | `_remove_camera` | 471 |
| `/api/cameras/profile/<device_id>` | GET | `_get_profile` | 472 |
| `/api/cameras/tuning/<cam_name>` | GET | `_tuning_get` | 473 |
| `/api/cameras/tuning/<cam_name>` | POST | `_tuning_set` | 474 |
| `/api/vision_pipelines` | GET | `_vision_pipelines` | 475 |
| `/api/cameras/calibration/<cam_name>` | GET | `_calibration_get` | 476 |
| `/api/cameras/calibration/<cam_name>` | DELETE | `_calibration_reset` | 477 |
| `/api/cameras/calibration/<cam_name>/focal` | POST | `_calibration_focal` | 478 |
| `/api/cameras/calibration/<cam_name>/charuco/capture` | POST | `_charuco_capture` | 479 |
| `/api/cameras/calibration/<cam_name>/charuco/status` | GET | `_charuco_status` | 480 |
| `/api/cameras/calibration/<cam_name>/charuco` | DELETE | `_charuco_clear` | 481 |
| `/api/cameras/calibration/<cam_name>/charuco/finish` | POST | `_charuco_finish` | 482 |
| `/api/cameras/calibration/<cam_name>/feed` | GET | `_calibration_feed` | 483 |
| `/api/cameras/calibration/<cam_name>/mode` | POST | `_calibration_mode` | 484 |
| `/api/cameras/calibration/<cam_name>/heartbeat` | POST | `_calibration_heartbeat` | 485 |
| `/api/cameras/calibration/<cam_name>/auto` | POST | `_auto_set` | 486 |
| `/api/cameras/calibration/<cam_name>/auto/status` | GET | `_auto_status` | 487 |
| `/api/cameras/calibration/<cam_name>/pnp` | GET | `_pnp_get` | 488 |
| `/api/cameras/calibration/<cam_name>/pnp` | POST | `_pnp_save` | 489 |
| `/api/cameras/calibration/<cam_name>/pnp` | DELETE | `_pnp_clear` | 490 |
| `/api/cameras/calibration/board` | GET | `_calibration_board_pdf` | 491 |
| `/api/cameras/events` | GET | `_sse_stream` | 492 |

#### `update(frame_data)` (lines 678-724)

Called every vision tick. Flow:
1. Extracts `frame` and `cameras` from frame_data
2. Builds `cam_by_name` mapping using `_camera_display_name()`
3. Stores `self.live_cameras` for tuning/calibration access
4. Processes `camera_frames` dict (multi-camera) or falls back to single frame
5. Matches frames to cameras using aliases (name, source, device_id)
6. Updates `self.frames`, `self.dims`, `self.last_seen`, `self.sources`
7. Calls `_evict_stale()` to clean up cameras not seen for 10+ seconds

#### `_generate(camera_name)` (lines 1801-1824)

MJPEG generator. Runs at ~20 FPS target. Yields JPEG-encoded frames in the
MJPEG multipart format. Times out after `_FEED_TIMEOUT_S` (15s) of no frames.

#### `_generate_calibration(cam_name, overlay, pattern, dict_id)` (lines 1123-1269)

The most complex generator. For ChArUco calibration feeds:
1. Spawns a `detection_worker` thread that runs board detection on downscaled
   frames (max 1280px) at ~10 Hz
2. Maps detected corners back to full-resolution coordinate space
3. Triggers auto-capture via `_auto_consider()`
4. Draws captured overlay layers via `_draw_captured_overlays()`
5. Serves the composite frame as MJPEG

#### Calibration Flow

The full calibration wizard supports three modes:

**Focal Length** (`_calibration_focal`, lines 869-896):
1. User enters: real object size, distance, measured pixel height, frame width
2. Computes `focal_px = cam_calibration.focal_from_object()`
3. Computes `fov_deg = cam_calibration.fov_from_focal()`
4. Saves to camera calibration entry

**ChArUco Intrinsics** (`_charuco_capture`, `_charuco_finish`, lines 1306-1400):
1. Auto-capture loop detects ChArUco boards, checks coverage and diversity
2. Captures are accumulated in `calib_sessions[key]["charuco_captures"]`
3. Rolling solve runs periodically to show live RMS
4. User clicks "Calibrate" to run final `cam_calibration.calibrate_charuco()`
5. Saves camera matrix, dist_coeffs, resolution, RMS, FOV, focal length

**PnP Pose** (`_pnp_get`, `_pnp_save`, lines 581-666):
1. Reads model metadata for keypoint count
2. User enters 3D object points (auto-filled for COCO17 default pose models)
3. Saves PnP config into the vision model settings

### `_find_camera_entry(cam_name)` (lines 1480-1488)

Resolves a camera name to its config entry. Tries:
1. Direct dict lookup by `cam_name`
2. Fuzzy match on `name` or `source` fields

### `_add_camera()` (lines 1402-1478)

Creates a new camera config entry. Flow:
1. Validates name/source uniqueness
2. Checks device_id not already claimed
3. Parses pipeline settings (nested or flat format)
4. Carries over calibration from camera profiles if available
5. Resolves vision model files
6. Calls `ensure_camera_entries_ready()` and `_prune_stale_pipeline_settings()`
7. Saves to config and camera profiles

### `_update_camera()` (lines 1496-1587)

Updates an existing camera entry. Handles:
- Name changes (key rename in config dict)
- Pipeline changes (validates against known pipeline classes)
- Model-backed pipeline switches (ensures vision_model block exists)
- Core vs. pipeline setting routing

### Auto-Discovery

#### `_auto_discover_loop()` (lines 507-527)
Background thread that probes devices every 10 seconds. Pushes SSE events when
the device set changes.

#### `_probe_devices()` (lines 1685-1793)
Platform-specific device enumeration:
- **Linux**: Groups `/dev/video*` by physical device, filters non-capture nodes
- **Windows**: Reads registry for USB camera interfaces
- **macOS**: Falls back to index probing via OpenCV

---

## dashboard.py -- SSE & Metrics

**File:** `iSpy/web/modules/dashboard.py` (235 lines)

### `DashboardModule.__init__()` (lines 21-34)

Tracks:
- `_latest`: dict with fps, vision_ms, camera_lag_ms, detections, loop_s
- `_vision_last_tick`: timestamp of last vision update
- `_model_info`: model name, format, path, size
- `_plugin_info`: lists of active trackers/utilities/frame_processors
- `_sse_clients`: list of subscriber queues

### `update(frame_data)` (lines 42-70)

Every tick:
1. Builds a `tick` dict with fps, vision_ms, camera_lag_ms, detections, loop_s
2. Counts detection classes from `frame_data["detections"]`
3. Pushes tick data to all SSE subscribers
4. First tick triggers `_refresh_model_info_unlocked()` to read model metadata

### Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/dashboard` | GET | Renders `dashboard.html` |
| `/api/status` | GET | Full status payload (one-shot) |
| `/api/system` | GET | System metrics + camera status |
| `/api/events` | GET | SSE stream (text/event-stream) |

### SSE Stream (`_sse_stream`, lines 210-230)

On connect: immediately sends the full payload via `_build_full_payload()`.
Then polls the subscriber queue every 50ms. Clients are removed from the list
on `GeneratorExit`.

### `_get_system_metrics()` (lines 112-151)

Uses `psutil` for:
- CPU percent
- Memory percent/used/total
- Temperature (tries coretemp, cpu_thermal, soc_thermal, k10temp)

Returns `None` values if psutil is unavailable.

---

## health.py -- Health Checks

**File:** `iSpy/web/modules/health.py` (141 lines)

### Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/health` | GET | Minimal: `{status, uptime_s}` (200/503) |
| `/health/detailed` | GET | Full payload (200/503) |
| `/health-page` | GET | Renders `health.html` |
| `/api/health` | GET | Full payload + plugin statuses (200/503) |

### `_build_payload()` (lines 78-107)

Computes health status:
- `stale_s`: time since last vision tick
- `healthy` = `stale_s < threshold AND all_cameras_ok AND (nt_connected OR None)`
- Per-camera: `ok` = `frame_age < stale_threshold`
- NetworkTables: queries `network_handler.isConnected()`

### `update(frame_data)` (lines 47-53)

Thread-safe update of fps, vision_s, detections, last_tick, loop_count.

---

## viewer3d.py -- 3D Viewer Backend

**File:** `iSpy/web/modules/viewer3d.py` (130 lines)

### Detection Storage

`_latest_objects`: list of dicts, rebuilt every tick from `frame_data["detections"]`.

Each dict contains: id, x, y, z, roll, yaw, pitch, name, confidence, num_keypoints, vis_type, vis_meta, and optionally keypoints_3d.

### Overlay API

```python
def add_overlay(self, overlay_id: str, overlay: dict) -> None:  # line 40
    overlay["id"] = overlay_id
    self._overlays[overlay_id] = overlay

def remove_overlay(self, overlay_id: str) -> None:  # line 46
    self._overlays.pop(overlay_id, None)
```

Any plugin with a reference to `context["viewer3d"]` (via module registry) can
push/remove overlays. Overlays persist until explicitly removed.

### Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/viewer3d` | GET | Renders `viewer3d.html` |
| `/api/detections/latest` | GET | Returns `{objects: [...]}` |
| `/api/overlays` | GET | Returns `{overlays: [...]}` |

---

## metrics.py -- Time-Series Ring Buffer

**File:** `iSpy/web/modules/metrics.py` (105 lines)

### Constants

- `SERIES`: Maps frame_data keys to `(label, unit, scale_factor)`:
  - `loop_s` -> ms (x1000)
  - `vision_s` -> ms (x1000)
  - `camera_lag_s` -> ms (x1000)
- `CODE_PARTS`: Pipeline stage labels with colors
- `MAX_POINTS = 600`: Ring buffer size

### `update(frame_data)` (lines 39-52)

Records timestamped values for each series, FPS (computed as `1/loop_s`),
and code_times (per-stage ms).

### Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/metrics` | GET | Renders `metrics.html` |
| `/api/metrics` | GET | Returns all time-series data |
| `/api/metrics/save` | POST | Saves to `Outputs/metrics_*.json` |

---

## models.py -- YOLO Model Management

**File:** `iSpy/web/modules/models.py` (202 lines)

Manages `.pt` files in `YoloModels/pytorch/`.

### Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/models` | GET | Renders `models.html` |
| `/api/models` | GET | Lists all .pt files with metadata |
| `/api/models/upload` | POST | Uploads a .pt file |
| `/api/models/<name>` | GET | Model detail |
| `/api/models/<name>` | DELETE | Deletes model (if not active) |
| `/api/models/select` | POST | Sets active model for all model-backed cams |

### `_select()` (lines 142-179)

Sets the active model across all cameras. Updates both `source_pt` and
`file_path` (optimized artifact) for each model-backed camera config.

---

## datasets.py -- Image Dataset Management

**File:** `iSpy/web/modules/datasets.py` (line count varies)

Manages quantization datasets flat in `QuantizeDataset/<name>/` (images sit
directly in the dataset folder, e.g. `QuantizeDataset/robotics_calibration/img1.png`).

### Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/datasets` | GET | Renders `datasets.html` |
| `/api/datasets` | GET/POST | List/create datasets |
| `/api/datasets/<name>/images` | GET/POST | List/upload images |
| `/api/datasets/<name>/images/<filename>` | GET/DELETE | Get/delete image |
| `/api/fs/dirs` | GET | Browse filesystem directories |

---

## logs.py -- Log Viewer

**File:** `iSpy/web/modules/logs.py` (48 lines)

### `_efficient_tail(path, n)` (lines 30-48)

Reads the last N lines from a file using reverse block reads (8KB blocks).
This avoids loading the entire file into memory.

### Route

`GET /api/logs?n_lines=200` -- Returns `{lines: [...]}` (max 5000).

---

## onboarding.py -- First-Run Tour

**File:** `iSpy/web/modules/onboarding.py` (34 lines)

### Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/onboarding` | GET | Returns `{show_tour: bool}` |
| `/api/onboarding` | POST | Sets `onboarding.completed` in config |

---

## recommendations.py -- Config Health

**File:** `iSpy/web/modules/recommendations.py` (16 lines)

### Route

`GET /api/recommendations` -- Returns `{recommendations: [...], critical_count: N}`.

Delegates to `iSpy.validations.recommendations.get_structured_recommendations()`.

---

## Support Files

### save_store.py (23 lines)

JSON file persistence in `Save/*.json`. Thread-safe via `threading.Lock`.

```python
read(key, default=None) -> Any    # line 11
write(key, data) -> None          # line 21
```

### SetupWizard.py (75 lines)

First-boot wizard. Renders inline HTML with camera source, name, subsystem,
unit, and NetworkTables settings. On submit, normalizes config via
`ensure_camera_entries_ready()`.

### Status.py (deleted)

Legacy `StatusReporter` (UtilityBase) registered a duplicate `/health` route
that crashed Flask when enabled alongside `HealthModule`. It was removed:
`HealthModule` (above) is the single canonical health implementation.

---

## Request Lifecycle

### Vision Tick -> Web Update

```
Vision loop tick
  |-- frame_data = {frame, detections, fps, vision_s, ...}
  |-- web_app.update(frame_data)
        |-- cameras.update()    -> stores frames, pushes SSE
        |-- dashboard.update()  -> pushes metrics SSE
        |-- health.update()     -> updates counters
        |-- viewer3d.update()   -> rebuilds detection list
        |-- metrics.update()    -> records time-series
```

### Browser -> Camera Feed

```
Browser <img src="/video/camera_name">
  |-- Flask route: _video_feed(camera_name)
  |-- Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
  |-- _generate() loop:
        |-- Read self.frames[name] (thread-safe via self.lock)
        |-- cv2.imencode(".jpg", frame, [JPEG_QUALITY=70])
        |-- yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf
        |-- sleep to maintain ~20 FPS target
```

### Browser -> API Call

```
Browser fetch("/api/cameras/config", {method: "POST", body: {...}})
  |-- Flask route: _add_camera()
  |-- request.get_json(force=True)
  |-- Validate input
  |-- config.set("camera_configs", cams)
  |-- config.save()
  |-- return jsonify(success=True)
```

### Browser -> SSE Stream

```
Browser new EventSource("/api/events")
  |-- Flask route: _sse_stream()
  |-- Response(generate(), mimetype="text/event-stream")
  |-- generate():
        |-- Send immediate full payload
        |-- Loop: poll subscriber queue every 50ms
        |-- yield "data: {json}\n\n"
```
