# 11 — Test Suite

> Test structure, how to run tests, what each test file covers, known failures, and validation tools.

---

## Running Tests

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_calibration_web.py -v

# Run a specific test by class and method name
python -m pytest tests/test_calibration_web.py::CalibrationWebTests::test_charuco_status_reports_detection -v

# Skip slow/hanging tests
python -m pytest tests/ -v -k "not test_auto_capture_stores_frames_then_finish_saves"

# Run tests matching a pattern
python -m pytest tests/ -v -k "calibration"

# Run with short tracebacks
python -m pytest tests/ -v --tb=short

# Stop on first failure
python -m pytest tests/ -v -x
```

---

## Test Files Overview

| File | Lines | Tests | Coverage Area |
|------|-------|-------|---------------|
| `test_calibration.py` | 161 | 20 | Calibration math: focal length, intrinsics, ChArUco detection, scoring, overlay rendering |
| `test_calibration_web.py` | 408 | 24 | Calibration web API: ChArUco feed/capture/status, PnP, auto-capture |
| `test_addons.py` | 464 | 50 | Plugin discovery, base classes, object_tracker, path_planner, video_recorder, network_table_handler, HealthModule, examples |
| `test_addon_web.py` | 431 | 30 | Plugin status API: toggle, settings, source view, delete, coerce, upload |
| `test_addon_config.py` | 318 | 26 | Config migration: legacy list to addon dict format |
| `test_architecture.py` | 452 | 27 | Pipeline config, lifecycle, boot readiness, quantize dirs |
| `test_model_select.py` | 298 | 14 | Model selection, artifact resolution, optimized-active checks |
| `test_pipeline_schemas.py` | 148 | 12 | Vision pipeline config schemas, debug hooks, capture backend |
| `test_settings_page_keys.py` | 47 | 4 | Regression: removed legacy keys not in settings.html |
| `test_camera_editor_regressions.py` | 32 | 4 | Regression: camera editor HTML has expected JS functions |
| `test_camera_geometry.py` | - | 14 | Camera geometry math regression coverage |
| `test_control_channel.py` | - | 11 | Control channel messaging |
| `test_output_schema.py` | - | 8 | Vision output schema validation |
| `test_boot_camera_cleanup.py` | - | 10 | Boot-time stale camera cleanup |

**Total:** 254 tests across 14 files (3,258 lines).

---

## Test Infrastructure

### Framework

All tests use `unittest.TestCase` (not pytest fixtures). Tests are discovered
by pytest but follow the unittest pattern with `setUp()` and `tearDown()`.

### Common Setup Pattern (`_setup()`)

Most test classes define a `_setup()` method (note: not `setUp()`) that is
called manually in each test or in a shared helper:

```python
class SomeTests(unittest.TestCase):
    def _setup(self):
        # Create temp directory for config
        tmpdir = tempfile.mkdtemp()
        config = iSpyConfig(Path(tmpdir) / "config.json")

        # Create mock camera
        cam = MockCamera()

        # Create Flask app with routes
        app = flask.Flask(__name__)
        context = {"config": config, "cameras": [cam], "flask_app": app}
        module = CamerasModule(context)
        module.register_routes(app)

        # Get test client
        client = app.test_client()
        return config, cam, module, client
```

### MockCamera

A test double that simulates a camera for testing:

```python
class MockCamera:
    def __init__(self):
        self.name = "test_cam"
        self.config = type("Cfg", (), {"get": lambda self, k, d=None: ...})()
        self.calibration_active = False
        self.calibration_last_seen = 0.0
        self._frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def get_raw_frame(self):
        return self._frame.copy()

    def get_frame_age(self):
        return 0.0

    def in_calibration_mode(self):
        return bool(self.calibration_active and
                    time.monotonic() - self.calibration_last_seen < 10.0)
```

The `_FakeCamera` in `test_calibration_web.py` extends this with
`set_calibration()` and `calibration_heartbeat()` methods.

### Key Test Helpers

#### _first_jpeg() (in test_calibration_web.py)

Extracts the first JPEG frame from an MJPEG multipart stream:

```python
def _first_jpeg(generator):
    """Get first JPEG frame from an MJPEG generator."""
    for chunk in generator:
        if b"\xff\xd8" in chunk:  # JPEG SOI marker
            # extract JPEG bytes from multipart chunk
            ...
```

The JPEG SOI (Start of Image) marker `0xFF 0xD8` identifies the start of a
JPEG frame within the multipart MJPEG stream.

#### _charuco_b64() (in test_calibration_web.py)

Generates a base64-encoded ChArUco board image for test injection:

```python
def _charuco_b64(pattern=(7, 5), size=(640, 480), rotate=False):
    board = c.make_charuco_board(*pattern)
    img = board.generateImage(size, marginSize=20)
    if rotate:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
```

---

## Detailed Test Coverage

### test_calibration.py (248 lines)

Tests the pure math functions in `iSpy.vision.calibration`:

| Test Class | Tests | What It Covers |
|------------|-------|----------------|
| `FocalLengthTests` | 3 | `focal_from_object`, focal↔FOV roundtrip, FOV sanity |
| `IntrinsicsScalingTests` | 3 | `intrinsics_for_frame` with/without resolution, None handling |
| `ChArUcoDetectionTests` | 4 | `detect_charuco` with synthetic boards, shape normalization |
| `ChArUcoCalibrationTests` | 3 | `calibrate_charuco` with synthetic captures, degenerate detection |
| `OverlayTests` | 2 | `draw_corners`, `draw_markers`, `draw_charuco` |

**FocalLengthTests:**
- `test_focal_from_object` — verifies `f = (pixel_height * distance) / real_size` with known values (6.5in object, 24in away, 60px tall).
- `test_focal_from_fov_roundtrip` — verifies focal→FOV→focal produces the same value.
- `test_fov_sane` — verifies FOV formula against manual calculation.

**IntrinsicsScalingTests:**
- `test_returns_none_without_calibration` — empty dict returns None.
- `test_scales_to_live_resolution` — 2x resolution doubles fx, fy, cx, cy.
- `test_missing_resolution_assumes_same_frame` — no resolution means scale factor = 1.0.

**ChArUcoDetectionTests:**
- Create synthetic boards and verify detection finds corners.
- Test shape normalization for OpenCV version compatibility.

**ChArUcoCalibrationTests:**
- Verify calibration requires >= 3 captures.
- Test degenerate pose detection.

### test_calibration_web.py (491 lines, 26 tests)

Tests the calibration web API endpoints:

| Test Class | Tests | What It Covers |
|------------|-------|----------------|
| `CalibrationWebTests` | 24 | Full calibration web API |

**Key tests:**
- `test_charuco_feed_returns_jpeg` — verifies `/calibration/charuco/feed` returns MJPEG stream.
- `test_charuco_capture_stores_frame` — POST to `/calibration/charuco/capture` stores the frame.
- `test_charuco_status_reports_detection` — GET `/calibration/charuco/status` reports detection status.
- `test_charuco_intrinsics_after_captures` — after enough captures, intrinsics endpoint returns results.
- `test_pnp_calibration` — PnP calibration endpoint test.
- `test_auto_capture_pause_resume` — auto-capture mode pause/resume.

### test_addons.py (464 lines)

Tests the plugin system:

| Test Class | Tests | What It Covers |
|------------|-------|----------------|
| `AddonDiscoveryTests` | 3 | Plugin discovery, base class validation, schema validation |
| `ObjectTrackerTests` | 4 | object_tracker plugin: init, track, distance threshold |
| `PathPlannerTests` | 3 | path_planner plugin: init, cluster, plan |
| `VideoRecorderTests` | 2 | video_recorder plugin: init, record |
| `NetworkTableHandlerTests` | 3 | network_table_handler: init, publish, IP config |
| `HealthModuleTests` | 8 | core health module: stale threshold config, payload, cameras, NT wiring, plugin statuses |
| `ExamplePluginTests` | 3 | example_* plugins: load, basic functionality |

**AddonDiscoveryTests:**
- `test_all_builtin_addons_are_discovered` — verifies all expected plugin names are found.
- `test_every_addon_is_an_addonbase_subclass` — verifies class hierarchy.
- `test_addon_schemas_are_valid` — validates schema field types (text/number/toggle).

### test_addon_web.py (431 lines)

Tests the plugin status web API:

| Test Class | Tests | What It Covers |
|------------|-------|----------------|
| `PluginStatusModuleTests` | ~15 | Toggle, settings, source, delete, coerce, upload |

**Key tests:**
- `test_available_lists_all_addons_with_schemas` — verifies `_available()` returns all plugins with schemas.
- `test_toggle_addon_on_off` — toggle endpoint enables/disables plugins.
- `test_settings_update` — settings endpoint updates plugin config.
- `test_source_view` — source endpoint returns plugin source code.
- `test_delete_addon` — delete endpoint removes plugin configuration.
- `test_coerce_setting_value` — tests `_coerce_setting_value()` type coercion.

### test_addon_config.py (318 lines)

Tests configuration migration from legacy format:

| Test Class | Tests | What It Covers |
|------------|-------|----------------|
| `AddonDefaultConfigTests` | 3 | Default config structure validation |
| `AddonMigrationTests` | ~7 | Legacy→addon dict migration |

**AddonDefaultConfigTests:**
- `test_default_plugins_are_dicts_not_lists` — plugins sections are dicts, not lists.
- `test_default_config_has_no_legacy_global_keys` — legacy keys removed from global config.
- `test_default_config_still_has_shared_global_keys` — shared keys still present.

**AddonMigrationTests:**
- Tests migration from legacy list format (`"trackers": ["object_tracker", ...]`)
  to addon dict format (`"trackers": {"object_tracker": {"enabled": true, ...}}`).
- Verifies legacy global keys (`dbscan`, `distance_threshold`, etc.) are migrated
  into their respective addon configs.

### test_architecture.py (452 lines)

Tests pipeline architecture and configuration:

| Test Class | Tests | What It Covers |
|------------|-------|----------------|
| `PipelineConfigTests` | 4 | Default config, legacy migration, nested pipeline |
| `PipelineLifecycleTests` | 4 | Construct, run, destroy, is_ready |
| `BootReadinessTests` | 3 | Pipeline readiness, timeout, error states |
| `QuantizeDatasetTests` | 2 | Dataset directory structure |
| `PipelineDiscoveryTests` | 2 | get_pipeline_classes, all pipelines registered |

**PipelineConfigTests:**
- `test_default_config_is_object_detection_with_bundled_pose_model` — default pipeline is object_detection with `_default_pose.pt`.
- `test_legacy_flat_camera_config_migrates_to_nested_pipeline` — old flat configs are migrated to nested pipeline structure.

**PipelineLifecycleTests:**
- `test_concrete_pipeline_constructs` — VisionPipeline subclass can be instantiated.
- `test_pipeline_run_returns_objects` — run() returns (objects, frame).
- `test_pipeline_destroy_cleans_up` — destroy() completes without error.

### test_model_select.py (298 lines)

Tests model selection and artifact resolution:

| Test Class | Tests | What It Covers |
|------------|-------|----------------|
| `ExistingArtifactTests` | 4 | `existing_artifact_for()` resolution |
| `OptimizedActiveTests` | 3 | Optimized model activation |
| `ModelSelectionTests` | 3 | Model file path resolution |

**ExistingArtifactTests:**
- `test_none_when_nothing_built` — no artifact when only .pt exists.
- `test_prefers_requested_format` — prefers the requested format over others.
- `test_falls_back_to_any_built_format` — falls back to available formats.

### test_pipeline_schemas.py (148 lines)

Tests vision pipeline configuration schemas:

| Test | What It Covers |
|------|----------------|
| `test_base_vision_schema_is_empty_by_default` | VisionBase.config_schema() returns {} |
| `test_april_tag_schema_exposes_tag_size_inches` | AprilTag schema has tag_size_inches field |
| `test_object_detection_pipeline_is_exposed` | object_detection in pipeline payloads |
| `test_additional_pipeline_schemas_are_discovered` | qr_code, depth_anything discovered |
| `test_calibration_fields_hidden_when_pipeline_does_not_use_them` | Calibration visibility per pipeline |
| `test_vision_base_exposes_debug_contract` | get_debug_data, get_debug_frame, plot |
| `test_plugin_plot_hook_returns_annotated_frame` | plot() returns annotated frame |
| `test_windows_capture_backend_candidates_use_msmf_only` | Windows uses MSMF backend |
| `test_camera_demo_objects_are_emitted_for_placeholder_visualization` | Demo objects for placeholder |
| `test_april_tag_run_prefers_real_detection_over_demo_placeholder` | Real detection over demo |
| `test_builtin_plugins_emit_visible_objects_and_annotations` | QRCode emits objects |
| `test_depth_plugin_unloaded_returns_raw_frame` | DepthAnything without model returns raw frame |

### test_settings_page_keys.py (47 lines)

Regression test ensuring removed legacy keys aren't in settings.html:

```python
REMOVED_KEYS = [
    "distance_threshold", "stale_threshold",
    "dbscan.epsilon", "dbscan.min_samples",
    "record_mode", "record_dir",
    "use_network_tables", "network_tables_ip",
]
```

- `test_no_removed_keys_as_data_key` — no `data-key` attributes use removed keys.
- `test_all_settings_fields_are_still_valid_global_keys` — only valid keys remain.
- `test_bool_key_set_no_longer_contains_removed_keys` — removed keys not in HTML.
- `test_settings_page_points_to_addons_page` — `/addons` link present.

### test_camera_editor_regressions.py (32 lines)

Regression test for camera editor HTML:

- `test_model_picker_keeps_original_snapshot` — `openModelPicker` doesn't use `dataset.original`.
- `test_model_upload_keeps_original_snapshot` — `uploadModelFromPicker` doesn't use it.
- `test_dataset_import_keeps_original_snapshot` — `uploadDatasetFromPicker` doesn't use it.
- `test_change_detection_still_uses_original_snapshot` — `modelPayloadField` and
  `collectSchemaFields` still use `el.dataset.original` for change detection.

---

## Known Test Issues

None. The historical auto-detect failures (`detect_charuco` identifying a 7x5
print against wrong grids) and chessboard endpoint gaps were resolved by
implementing `detect_charuco_auto()` (layout/dictionary sweep ranked by corner
coverage) and removing chessboard support entirely.

### Historically Slow Tests

These timing-sensitive flows can be slow on busy machines but pass reliably:

- **`test_auto_capture_stores_frames_then_finish_saves`** — waits for
  auto-capture to collect frames and save.
- **`test_charuco_capture_finish_flow`** — capture completion sequence.

---


## Validation Tools (`iSpy/validations/`)

Separate from the pytest suite, these are runtime validation tools:

| Module | Lines | Purpose |
|--------|-------|---------|
| `validate_system.py` | ~150 | System validation: model paths, config, quantization dataset |
| `model_validator.py` | ~200 | Model file organization (YoloModels/ structure) |
| `benchmarking.py` | ~300 | Model inference benchmarking across backends |
| `validate_camera_frame.py` | ~100 | Camera frame quality (sharpness, exposure, contrast) |
| `recommendations.py` | ~150 | Config health recommendations for web UI |
| `unit_tests.py` | ~80 | Additional unit tests for validation |
| `ez.py` | ~50 | Simplified validation entry points |
| `tests/compare_models.py` | ~200 | Base vs optimized model detection comparison |

### Running Validation

```bash
# Full system validation
python -c "from iSpy.validations.validate_system import validate_system; validate_system()"

# Model organization check
python -m iSpy.validations.model_validator check-org

# Benchmarks
iSpy-bench

# Model comparison
iSpy-stats

# Camera frame quality
python -c "from iSpy.validations.validate_camera_frame import validate_camera_frame; validate_camera_frame()"

# Config recommendations
python -c "from iSpy.validations.recommendations import get_recommendations; print(get_recommendations())"
```

### validate_system.py

Checks:
- Required Python packages are installed
- Model files exist and are loadable
- Config is valid JSON with required keys
- Quantization dataset directory exists
- Output directory is writable

### model_validator.py

Validates the `YoloModels/` directory structure:
- Each format directory contains valid model files
- Metadata sidecars exist for each model
- No orphaned files
- Model naming conventions are followed

### benchmarking.py

Runs inference benchmarks across available backends:
- Measures FPS, latency, memory usage
- Compares PT vs ONNX vs RKNN vs TensorRT
- Reports per-frame timing statistics

### compare_models.py

Compares detection results between base and optimized models:
- Runs both models on the same input
- Computes IoU between detections
- Reports accuracy degradation from quantization

---

## Test Configuration

### tests/Config/config.json

Provides a test fixture config used by some tests:

```json
{
    "vision_model": {
        "file_path": "YoloModels/pytorch/test_model.pt",
        "input_size": [640, 640],
        "min_conf": 0.5,
        "margin": 10
    },
    "plugins": {
        "trackers": {},
        "utilities": {},
        "frame_processors": {}
    },
    "camera_configs": {}
}
```

### Test Data

- Synthetic images are generated in-test (no external test data files).
- ChArUco boards are generated via `c.make_charuco_board()` and
  `generateImage(marginSize=...)` (margins control apparent board size).
- Camera frames are numpy arrays filled with zeros or random data.
