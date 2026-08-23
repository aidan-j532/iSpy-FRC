# Pipeline-Agnostic Add-on Audit

Every built-in add-on was checked against the contract that add-ons must not
assume detections come from `object_detection` (or any other single pipeline).
All spatial data crosses add-on boundaries as `iSpy.vision.Object.Object`
instances inside `frame_data["detections"]`, regardless of which pipeline
produced it (`object_detection`, `april_tag`, `qr_code`, `optical_flow`,
`depth_anything`, `yolo_world`, custom).

**Audit date:** 2026-08-22 · **Branch:** Restructure

## Verdicts

| Add-on | Type | Verdict | Notes |
|---|---|---|---|
| `network_table_handler` (`NetworkHandler.py`) | utility | PASS | Publishes only generic accessors: `Object.get_position_normally()` / `.roll` / `.pitch` / `.yaw`. The `publish` config list resolves arbitrary `frame_data` keys, so any pipeline's metadata can be shipped without code changes. Covered by `test_update_is_pipeline_agnostic` (AprilTag/QR objects flow through untouched). |
| `object_tracker` (`ObjectTracker.py`) | tracker | PASS | Operates purely on `Object.get_position()` (x/y/z) and angle fields; no `vis_type` assumptions. Merges are gated on class/name identity so planar (`planar` vis_type AprilTag/QR) and generic objects never cross-merge. |
| `path_planner` (`PathPlanner.py`) | tracker | PASS | DBSCAN clustering over `Object.get_position()` only. Returns the same `Object` instances, filtered - no pipeline-specific fields touched. |
| `rollback` (`RollBack.py`) | utility | PASS | Only touches `frame_data["frame"]` (the rendered image). Completely independent of detection content. |
| `example_tracker` | tracker | PASS | Pass-through demo. |
| `example_frame_processor` | frame processor | PASS | Pixel-only transform of the frame. |
| `example_utility` | utility | PASS | Route registration demo, no detection access. |

### Frame processors

No built-in frame processor inspects detections; they transform frames
pixel-wise before/after plotting and are pipeline-agnostic by construction.

## Pipeline-side consolidation

Model-backed pipelines (`object_detection`, `yolo_world`, `depth_anything`)
share one base: `iSpy/vision/pipelines/optimizable.py::OptimizableModelPipeline`.
It owns the common `config_schema()` fields (`optimize`, `target_format`,
`quantize`, `quantization_dataset`, `input_size`), target-format resolution,
optimization-request handling, and the stale-model resync-on-boot protection
(filename-stem comparison). Pipelines override only genuinely
pipeline-specific behaviour (postprocess shape, backend loaders). Fixes to
optimization behaviour now land once, not three times.

## Opt-in compatibility declarations

Add-ons that genuinely only work with certain pipelines can declare:

```python
class MyObjectDetectionOnlyAddon(UtilityBase):
    plugin_name = "my_addon"
    supported_pipelines = ("object_detection",)
```

`None` (default) means "works with everything". When declared, the web UI
(`/addons`) shows a **pipeline mismatch** warning pill on cameras whose
pipeline isn't in the tuple, instead of letting the add-on silently misbehave.
Enforcement is advisory-by-design: nothing blocks execution, because many
"restriction" assumptions turn out to be soft.

## Universal output schema

The de-facto contract above is now formalized. `Object.to_dict()`
(`iSpy/vision/Object.py`) serializes any pipeline's detection to one JSON-safe
shape (flat position/rotation, `vis_type`, free-form `vis_meta`, optional
keypoints/rays); `VisionPipeline.serialize_detections()` /
`serialize_frame_data()` (`iSpy/vision/pipelines/base.py`) apply it in bulk and
stamp a `schema_version`. The schema contract is documented in full at the top
of `base.py` (`OUTPUT_SCHEMA_VERSION = 1`). `from_dict()` round-trips it.

Consumers using it:

- `NetworkTableHandler._send_detections` flattens every entry through the
  schema before publishing `FuelStruct[]` - no per-pipeline branches.
- `Viewer3DModule.update` builds `/api/detections/latest` payloads from
  `to_dict()` (keeping its deliberate optical-flow exclusion).

New consumers should serialize via these helpers instead of hand-rolled
`getattr` dicts; legacy plain-dict entries still pass through untouched.

## Rules for future add-ons

1. Consume detections exclusively via `Object` public accessors
   (`get_position()`, `get_rotation()`, `.name`, `.confidence`). Never import
   from a concrete pipeline module.
2. Read pipeline specifics from `frame_data["pipeline_settings"]` /
   `frame_data["camera_config"]` if configuration must drive behaviour -
   don't hardcode per-pipeline branches.
3. If you truly depend on one pipeline, declare `supported_pipelines` so the
   UI warns users on mismatched cameras.
4. Add a row to the table above when introducing a built-in add-on.
