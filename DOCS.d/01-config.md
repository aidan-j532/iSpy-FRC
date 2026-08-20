# 01 — Configuration System

> How iSpy loads, saves, validates, migrates, and provides access to its JSON
> configuration. Every class, method, data structure, and code path in detail.

---

## Config File Location

| Context | Path |
|---------|------|
| Development | `Config/config.json` (relative to repo root) |
| Deployed (Linux) | `/etc/iSpy/config.json` |
| Override | `ISPY_CONFIG` env variable |
| Default fallback | `Config/config.json` (constructed in `game_loop.py:53`) |

The path is resolved in `game_loop.py:53`:
```python
config_path = Path.cwd() / "Config" / "config.json"
```

---

## Constants and Module-Level Helpers

### `_CAMERA_CORE_KEYS` (iSpyConfig.py:10-17)

A `set` of keys that belong to the camera entry itself (not the pipeline settings). Used by `normalize_camera_entry()` to decide which keys stay on the camera dict vs. get moved into `pipeline.settings`:

```python
_CAMERA_CORE_KEYS = {
    "name", "source", "device_id", "subsystem", "grayscale",
    "x", "y", "z", "height", "yaw", "pitch", "calibration",
    "exposure_time", "gain", "fps_cap",
    "brightness", "contrast", "saturation", "gamma",
    "white_balance", "tint",
    "csi", "path",
}
```

### `_VISION_MODEL_SETTINGS_KEYS` (iSpyConfig.py:19-22)

Settings that live in the pipeline settings (not inside `vision_model` sub-dict):
```python
_VISION_MODEL_SETTINGS_KEYS = (
    "min_conf", "quantize", "quantization_dataset", "optimize", "target_format",
)
```

### `_LEGACY_SETTING_ALIASES` (iSpyConfig.py:24-27)

Maps old config key names to their current equivalents:
```python
_LEGACY_SETTING_ALIASES = {
    "auto_opt": "optimize",
    "quantized": "quantize",
}
```

### `_ADDON_TYPES` (iSpyConfig.py:29)

Valid plugin categories:
```python
_ADDON_TYPES = ("trackers", "utilities", "frame_processors", "pipelines")
```

---

## Unit Conversion System

### `_UNIT_TO_INCHES` (iSpyConfig.py:32-43)

Conversion factors: how many inches does 1 unit equal. All internal math uses inches.

```python
_UNIT_TO_INCHES = {
    "inch": 1.0,
    "inches": 1.0,
    "foot": 12.0,
    "feet": 12.0,
    "meter": 1.0 / 0.0254,       # ≈ 39.3701
    "meters": 1.0 / 0.0254,
    "centimeter": 1.0 / 2.54,    # ≈ 0.3937
    "centimeters": 1.0 / 2.54,
    "frc": 1.0,                   # FRC/WPILib: inputs in inches, outputs in meters
}
```

### `_UNIT_LABELS` (iSpyConfig.py:45-51)

Display labels for each unit system:
```python
_UNIT_LABELS = {
    "inch": "in", "inches": "in",
    "foot": "ft", "feet": "ft",
    "meter": "m", "meters": "m",
    "centimeter": "cm", "centimeters": "cm",
    "frc": "in",
}
```

### `unit_to_inches(value, unit)` (iSpyConfig.py:54-56)

Converts a value from any supported unit to inches:
```python
def unit_to_inches(value: float, unit: str) -> float:
    return value * _UNIT_TO_INCHES.get(unit.lower().strip(), 1.0)
```

### `unit_label(unit)` (iSpyConfig.py:59-61)

Returns a short display label:
```python
def unit_label(unit: str) -> str:
    return _UNIT_LABELS.get(unit.lower().strip(), unit)
```

---

## Helper Functions

### `default_vision_model()` (iSpyConfig.py:100-108)

Creates a default `vision_model` dict by finding the first `.pt` file in `YoloModels/pytorch/`:
1. Lists all `.pt` files sorted alphabetically
2. Filters out files starting with `_default`
3. If user models exist: uses the first one
4. Otherwise falls back to `_default_pose.pt`
5. Returns `{"file_path": rel, "source_pt": rel, "min_conf": 0.5}`

### `is_model_backed_pipeline(pipeline)` (iSpyConfig.py:111-112)

Returns `True` if the pipeline name is in `_MODEL_BACKED_PIPELINES` (currently just `"object_detection"`). Used to decide whether a camera needs a `vision_model` block.

### `get_pipeline_name(cam_entry)` (iSpyConfig.py:115-125)

Extracts the pipeline name from a camera entry dict. Handles three layouts:
1. `cam_entry["pipeline"]` is a string → return it
2. `cam_entry["pipeline"]` is a dict with `"name"` key → return that
3. Fallback → `"object_detection"`

### `get_pipeline_settings(cam_entry)` (iSpyConfig.py:128-135)

Extracts pipeline settings from a camera entry. Handles two layouts:
1. If `pipeline` is a dict with a `settings` sub-dict → return `pipeline["settings"]`
2. Otherwise: return all keys that are NOT in `_CAMERA_CORE_KEYS` and NOT `"pipeline"` (legacy flat layout)

### `normalize_camera_entry(cam_entry)` (iSpyConfig.py:138-169)

Normalizes a camera config entry into the canonical nested format. This is the key migration function. Three code paths:

**Path 1: `pipeline` is already a dict** (line 142-151)
- Ensures `pipeline["settings"]` exists as a dict
- Moves any non-core keys from the top level into `pipeline["settings"]`
- Keeps core keys (`name`, `source`, `yaw`, etc.) on the camera entry

**Path 2: `pipeline` is a string** (line 152-160)
- Collects all non-core keys as settings
- Wraps as `{"name": pipeline_string, "settings": settings_dict}`
- Removes settings keys from the top level

**Path 3: No `pipeline` key at all** (line 161-169)
- Same as Path 2 but defaults to `"object_detection"` pipeline

### `ensure_camera_entries_ready(camera_configs)` (iSpyConfig.py:172-183)

Iterates all camera configs and:
1. Normalizes each entry via `normalize_camera_entry()`
2. If it's a model-backed pipeline without a `vision_model` block → adds `default_vision_model()`
3. Cleans up vision model settings via `_normalize_vision_model_settings()`

### `_normalize_vision_model_settings(settings)` (iSpyConfig.py:84-97)

Prevents duplication between `vision_model` sub-dict and pipeline settings:
1. Removes `_VISION_MODEL_SETTINGS_KEYS` from `vision_model` if they also exist in the parent settings
2. Removes legacy aliases (`auto_opt`→`optimize`, `quantized`→`quantize`) when the canonical key is present

---

## `iSpyConfig` Class (iSpyConfig.py:186-636)

The main configuration manager. Created once at startup, shared everywhere.

### Constructor: `__init__(self, file_path, create=True)` (line 187-288)

**Step 1: Create default config** (lines 192-271)

```python
self.default_config = {
    "num_gpus": "auto",
    "device": 0,
    "unit": "frc",                  # "frc"|"meter"|"inch"|"foot"|"centimeter"
    "debug_mode": True,
    "frame_sync": False,
    "optimize": False,
    "log_level": "INFO",
    "log_file": "Outputs/log.txt",
    "metrics": True,
    "app_mode": True,
    "max_fps": 0,                   # 0 = unlimited
    "onboarding": {"completed": False},
    "camera_configs": {
        "default_cam": {            # Template for new cameras
            "name": "default_cam",
            "source": 0,
            "subsystem": "field",
            "grayscale": False,
            "calibration": {
                "distance": 0.0,
                "game_piece_size": 0.0,
                "size": 0,
                "fov": 0,
            },
            "yaw": 0,
            "pitch": 0,
            "height": 1.0,
            "x": 0,
            "y": 0,
            "pipeline": {
                "name": "object_detection",
                "settings": {
                    "vision_model": {
                        "file_path": "YoloModels/pytorch/_default_pose.pt",
                        "source_pt": "YoloModels/pytorch/_default_pose.pt",
                        "min_conf": 0.5,
                        "quantize": False,
                    },
                },
            },
        }
    },
    "plugins": {
        "trackers": {},
        "utilities": {},
        "frame_processors": {},
    },
}
self.config = json.loads(json.dumps(self.default_config))  # deep copy
```

**Step 2: Load from file** (lines 274-278)
- If `create=True` and file doesn't exist → calls `self.save()` to create it
- If `file_path` provided → calls `self.load_from_file(file_path)`

**Step 3: Post-load processing** (lines 280-288)
```python
self._check_config()              # Ensure required top-level keys exist
self._migrate_addons()            # Upgrade legacy addon format
self._migrate_camera_configs()    # Normalize all camera entries
self._rebuild_camera_configs()    # Wrap in iSpyCameraConfig instances
self._configure_logging()         # Apply log level from config
```

### `load_from_file(self, file_path)` (line 396-427)

1. Opens with `utf-8-sig` encoding (strips BOM from Windows Notepad saves)
2. Parses JSON via `json.load()`
3. Rejects legacy top-level `"vision_model"` layout with a clear error
4. Calls `_update_config(data)` to deep-merge into `self.config`
5. Error handling:
   - `JSONDecodeError` → RuntimeError with "Fix it or run 'boot -f'" message
   - `FileNotFoundError` → RuntimeError with "Run 'boot -f'" message
6. Finally block: re-applies `_configure_logging()`

### `save(self, quiet=False)` (line 428-441)

1. If no `file_path` set → defaults to `Config/config.json`
2. Creates parent directory if needed
3. Writes `json.dump(self.config, f, indent=4)`
4. Logs the save (unless `quiet=True`)

### `_update_config(self, data, current_dict=None)` (line 549-580)

Recursive deep-merge of incoming data into the config dict:
- If `data` is a dict and key exists in `current_dict` as a dict → recurse
- For `camera_configs`: deep-copy + `ensure_camera_entries_ready()` + rebuild wrappers
- Rejects top-level `"vision_model"` key (legacy layout)
- All other values: direct assignment (overwrites)

### `_check_config(self)` (line 300-306)

Ensures required top-level keys exist with `setdefault()`:
- `camera_configs` → `{}`
- `plugins` → `{}` with sub-keys `trackers`, `utilities`, `frame_processors`, `pipelines`

### `_configure_logging(self)` (line 582-615)

1. Reads `log_level` from config (default `"INFO"`)
2. Clears all existing root handlers
3. Creates a `StreamHandler` to stdout with an `_iSpyFilter` (only logs `iSpy.*` loggers)
4. If `log_file` is set → creates a `FileHandler` with the same filter
5. Both use format: `"%(asctime)s [iSpy] %(levelname)s:%(name)s: %(message)s"`

---

### CRUD Methods

#### `get(self, key, default=None)` (line 442-443)
Simple top-level key access: `self.config.get(key, default)`

#### `get_nested(self, *keys, default=None)` (line 445-452)
Chained key access: `get_nested("plugins", "trackers")` traverses `self.config["plugins"]["trackers"]`. Returns `default` on `KeyError` or `TypeError`.

#### `set(self, *keys_and_value)` (line 535-547)
Variadic setter: `set("plugins", "trackers", "my_plugin", {})` creates nested dicts as needed. Special case: if the last key is `"camera_configs"`, rebuilds the `iSpyCameraConfig` wrappers.

#### `__getitem__(self, args)` (line 617-620)
Supports `config["key"]` (top-level) and `config["key1", "key2"]` (nested via tuple).

#### `__call__(self, *keys)` (line 622-623)
Supports `config("plugins", "trackers")` as an alias for `get_nested`.

#### `__getattr__(self, item)` (line 625-636)
Attribute-style access: `config.unit` returns `config.get("unit")`. Raises `AttributeError` for private attrs or missing keys.

---

### Add-on CRUD Methods

#### `addon_entries(self, addon_type)` (line 461-467)
Returns the dict at `plugins.<addon_type>` (e.g., `plugins.trackers`). Returns `{}` if invalid type.

#### `get_addon_settings(self, addon_type, addon_name)` (line 469-474)
Returns the settings dict for a specific addon, or `None` if not present (not enabled).

#### `get_addon_setting(self, addon_type, addon_name, key, default)` (line 476-482)
Returns a single setting value from an addon's settings.

#### `is_addon_enabled(self, addon_type, addon_name)` (line 484-485)
Returns `True` if the addon has a settings entry (presence == enabled).

#### `enable_addon(self, addon_type, addon_name, settings=None, save=True)` (line 487-497)
Creates an entry for the addon with optional initial settings. Auto-saves.

#### `disable_addon(self, addon_type, addon_name, save=True)` (line 499-507)
Removes the addon entry. Auto-saves.

#### `set_addon_settings(self, addon_type, addon_name, settings, save=True)` (line 509-518)
Replaces the entire settings dict for an addon. Auto-saves.

#### `update_addon_settings(self, addon_type, addon_name, settings, save=True)` (line 520-533)
Merges new settings into an addon's existing settings. Auto-saves.

---

### Config Migration

#### `_migrate_addons(self)` (line 308-355)

Upgrades legacy addon configurations to the current format:

**Step 1: List → dict conversion** (lines 310-317)
Old configs stored plugins as lists: `["object_tracker", "path_planner"]`. Converted to dict format: `{"object_tracker": {}, "path_planner": {}}`.

**Step 2: Legacy fold-in** (lines 322-328)
The `dbscan` top-level key is folded into `trackers.path_planner` settings.

**Step 3: Legacy key folds** (lines 329-336)
Uses `_ADDON_LEGACY_FOLDS` to move top-level settings into their addon:
```python
_ADDON_LEGACY_FOLDS = (
    ("dbscan",             "trackers",    "object_tracker", None),
    ("distance_threshold", "trackers",    "object_tracker", "distance_threshold"),
    ("stale_threshold",    "trackers",    "object_tracker", "stale_threshold"),
    ("stale_threshold",    "utilities",   "health_reporter", "stale_threshold"),
)
```

**Step 4: Legacy flags** (lines 338-341)
Boolean flags like `use_network_tables: true` become addon presence:
```python
_ADDON_LEGACY_FLAGS = {
    "use_network_tables": ("utilities", "network_table_handler"),
    "record_mode": ("utilities", "video_recorder"),
}
```

**Step 5: Legacy settings** (lines 343-348)
Top-level settings like `network_tables_ip` are moved into the corresponding addon.

**Step 6: Cleanup** (lines 350-355)
Removes all legacy keys from the top-level config.

#### `_migrate_camera_configs(self)` (line 357-373)

For each camera:
1. `normalize_camera_entry()` — canonicalizes the pipeline structure
2. `_normalize_vision_model_settings()` — removes duplicate keys
3. If model-backed pipeline without `vision_model` → adds `default_vision_model()`

---

## `iSpyCameraConfig` Class (iSpyConfig.py:639-714)

Wraps a single camera's config dict with typed accessors and lazy migration.

### `DEFAULTS` (line 640-660)

```python
DEFAULTS = {
    "name": "default",
    "x": 0, "y": 0, "z": 0,
    "height": 0,
    "pitch": 0, "yaw": 0,
    "grayscale": False,
    "brightness": 0, "contrast": 0, "saturation": 0,
    "white_balance": 0, "tint": 0, "gamma": 1.0,
    "calibration": {"size": 0, "distance": 0, "game_piece_size": 0, "fov": 0},
    "source": "/dev/video0",
    "device_id": None,
    "subsystem": "field",
    "pipeline": {"name": "object_detection", "settings": {}},
}
```

### Constructor (line 662-665)

Deep-copies `DEFAULTS` then updates with the provided `config_dict`.

### `pipeline_entry()` (line 673-687)

Returns the `pipeline` dict. If the pipeline is still a string (legacy), lazily migrates it:
1. Collects non-core keys as settings
2. Creates `{"name": str, "settings": dict}`
3. Removes migrated keys from the top level

### `pipeline_name()` (line 689-690)
Delegates to `get_pipeline_name(self.data)`.

### `pipeline_settings()` (line 692-693)
Returns `self.pipeline_entry()["settings"]`.

### `get_pipeline_setting(key, default)` (line 695-702)
Checks the settings dict first, then falls back to the camera entry's top level (legacy flat layout).

### `set_pipeline_setting(key, value)` (line 704-705)
Sets a value in the pipeline settings dict.

### `__getitem__`, `get`, `__contains__` (line 707-714)
Standard dict-like access to `self.data`.

---

## `iSpyAddonConfig` Class (iSpyConfig.py:717-757)

Wraps an addon's settings dict with schema-merged defaults.

### Constructor (line 718-724)

```python
def __init__(self, settings=None, defaults=None):
    self.data = {}
    if isinstance(settings, dict):
        self.data.update(settings)          # User's saved settings
    if isinstance(defaults, dict):
        for key, value in defaults.items():
            self.data.setdefault(key, value) # Schema defaults (only fill missing)
```

This means an empty settings dict `{}` still gets all schema defaults applied.

### Methods (line 726-757)

- `get(key, default)` — standard dict access
- `get_nested(*keys, default)` — chained key access
- `set(key, value)` — direct assignment
- `setdefault(key, value)` — fill only if missing
- `items()`, `keys()` — dict iteration
- `to_dict()` — deep-copy to plain dict
- `__getitem__`, `__contains__` — standard operators

---

## Default Config Structure (Complete)

```json
{
    "num_gpus": "auto",
    "device": 0,
    "unit": "frc",
    "debug_mode": true,
    "frame_sync": false,
    "optimize": false,
    "log_level": "INFO",
    "log_file": "Outputs/log.txt",
    "metrics": true,
    "app_mode": true,
    "max_fps": 0,
    "onboarding": {
        "completed": false
    },
    "camera_configs": {
        "default_cam": {
            "name": "default_cam",
            "source": 0,
            "subsystem": "field",
            "grayscale": false,
            "calibration": {
                "distance": 0.0,
                "game_piece_size": 0.0,
                "size": 0,
                "fov": 0
            },
            "yaw": 0,
            "pitch": 0,
            "height": 1.0,
            "x": 0,
            "y": 0,
            "pipeline": {
                "name": "object_detection",
                "settings": {
                    "vision_model": {
                        "file_path": "YoloModels/pytorch/_default_pose.pt",
                        "source_pt": "YoloModels/pytorch/_default_pose.pt",
                        "min_conf": 0.5,
                        "quantize": false
                    }
                }
            }
        }
    },
    "plugins": {
        "trackers": {},
        "utilities": {},
        "frame_processors": {}
    }
}
```

---

## Hardware Auto-Optimization (`AutoOpt.py`)

### Module-Level Constants

```python
SUPPORTED_FORMATS = {"tflite", "openvino", "coreml", "onnx", "rknn", "engine", "tpu"}
```

### Helper Functions

#### `_run(cmd)` (line 12-23)
Runs a shell command with 2s timeout, returns stdout lowercased.

#### `_cmd_ok(cmd)` (line 26-39)
Returns `True` if the command exits with code 0.

#### `_lsusb_output()` (line 42-48)
Cached USB device listing. Uses `lsusb` on Linux, `wmic` on Windows.

### Hardware Detection Functions (all `@lru_cache()`)

#### `has_jetson()` (line 52-64)
Checks `/etc/nv_tegra_release`, `/proc/device-tree/model`, `/sys/firmware/devicetree/base/model` for "jetson" or "tegra".

#### `has_hailo_npu()` (line 67-73)
Checks `/dev/hailo*` glob, `lsusb` for "hailo", and `hailortcli fw-control identify`.

#### `has_nvidia()` (line 75-94)
Multi-check: Jetson, `/dev/nvidia0`, `nvidia-smi`, `torch.cuda.is_available()`, Windows `wmic`.

#### `has_tensorrt()` (line 97-106)
Tries `import tensorrt`, falls back to system Python check.

#### `has_tpu()` (line 109-118)
Tries `import torch_xla` and creates an XLA device.

#### `has_rockchip_npu()` (line 151-180)
Checks `/dev/rknpu*`, `lsmod | grep rknpu`, `/proc/cpuinfo` for RK3588/RK3576/etc., device tree model.

#### `has_apple_silicon()` (line 137-138)
Darwin + ARM architecture.

#### `has_edge_tpu()` (line 146-148)
USB IDs `18d1:9302` or `1ac1:089a`.

#### `has_arm()` (line 133-134)
Machine architecture contains "arm" or "aarch".

#### `has_intel_gpu()`, `has_amd_gpu()`, `has_intel_vpu()` (lines 121-143)
Various GPU detection methods.

### `resolve_openvino_device(requested_device)` (line 182-194)

Queries OpenVINO `Core().available_devices` and returns the best device string:
1. If user specified `intel:*` → use it as-is
2. If `GPU` available → `"intel:gpu"`
3. If `NPU` available → `"intel:npu"`
4. Fallback → `"intel:cpu"`

### `recommend_format(ignore_dependencies=False)` (line 197-249)

Priority order for hardware detection:
```
1. Rockchip NPU    → "rknn"
2. Hailo NPU       → "hef"
3. Edge TPU        → "tflite"
4. Apple Silicon   → "coreml"
5. Google TPU      → "tpu"
6. NVIDIA GPU      → "engine" (if TensorRT installed) or "onnx" (fallback)
7. Intel VPU       → "openvino"
8. Intel GPU       → "openvino"
9. AMD GPU         → "onnx"
10. ARM CPU        → "tflite"
11. CPU fallback   → "onnx"
```

---

## Config Access Patterns

### From the vision loop (`iSpy.py`):

```python
config = self.config
max_fps = config.get("max_fps", 0)         # line 420
unit = config.get("unit", "frc")            # used in ObjectDetectionCamera
debug_mode = config["debug_mode"]           # attribute-style access
```

### From plugins (via context dict):

```python
class MyPlugin(UtilityBase):
    def __init__(self, context):
        my_settings = context["config"]           # iSpyAddonConfig for THIS plugin
        global_config = context["global_config"]  # iSpyConfig for everything
        cameras = context["cameras"]              # list[VisionPipeline] instances
        flask_app = context["flask_app"]          # Flask app (for adding routes)
        vision = context["vision_instance"]       # iSpy instance
        viewer3d = context["viewer3d"]            # Viewer3DModule
```

### From the web API (via Settings module):

```python
# GET /api/settings  → returns full config dict as JSON
# POST /api/settings → updates config dict, saves to disk
```

### Pipeline settings extraction:

```python
from iSpy.config.iSpyConfig import get_pipeline_settings
settings = get_pipeline_settings(camera_config_dict)
# Returns the nested vision_model + calibration + tuning dict
```

---

## Camera Config Normalization Flow

When a camera config is loaded, it goes through this pipeline:

```
Raw JSON from config.json
  │
  ▼
ensure_camera_entries_ready()
  │
  ├── normalize_camera_entry(cam_cfg)
  │     ├── If pipeline is dict → move non-core keys into settings
  │     ├── If pipeline is string → wrap as {"name": str, "settings": {...}}
  │     └── If no pipeline → default to "object_detection"
  │
  ├── If model-backed + no vision_model → default_vision_model()
  │
  └── _normalize_vision_model_settings()
        ├── Remove duplicated keys between vision_model and parent settings
        └── Remove legacy aliases (auto_opt→optimize, quantized→quantize)

  │
  ▼
iSpyCameraConfig wrapper created
  │
  ├── Accessors: name(), source(), pipeline_name(), pipeline_settings()
  ├── Lazy migration: pipeline_entry() converts string→dict on first touch
  └── get_pipeline_setting(): checks settings dict, falls back to legacy flat
```
