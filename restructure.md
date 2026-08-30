# iSpy Restructure Summary

This document describes the **camera-source / vision-pipeline modular restructure**
and the **AGPL Ultralytics runtime removal** on branch `Restructure`.

The work landed in two commits so each concern is independently reviewable and
testable:

| Commit | Message | Scope |
|---|---|---|
| `9d55e8f` | `TESTING 1` | Camera sources separated from vision pipelines (runtime still uses Ultralytics). |
| `3c4b587` | `TESTING 2` | AGPL Ultralytics removed from the runtime (dependency-free `yolo_pt` loader). |

---

## 1. TESTING 1 — Camera Sources vs Vision Pipelines

### Problem

Before this change a single class served two jobs at once: the **camera source**
(reading frames from an OpenCV camera index, a Tello UDP stream, etc.) and the
**vision pipeline** (running detection, ArUco/tag decoding, calibration, ...) were
intertwined in one `Camera.py` / per-feature class. Adding a new camera type or a
new pipeline meant editing overlapping code, and the UI conflated the two concepts.

### What changed

- **New package `iSpy/vision/Cameras/`** now owns the real frame-acquisition
  machinery:
  - `base.py` — `CameraBase` (frame loop, grayscale, placeholders, synthetic frames).
  - `OpenCVCamera.py` — camera-index / file / network read via OpenCV.
  - `TelloCamera.py` — Tello UDP H.264 stream reader.
  - `_discovery.py` — enumerating available camera sources for the UI.
  - `__init__.py` — package exports/source registry.
- **`iSpy/vision/Camera.py`** is now a thin **facade** that delegates to the
  `Cameras/` package (kept importable for external callers).
- **`iSpy/vision/TelloEduCamera.py`** is a backward-compatible alias so existing
  configs / imports that referenced it keep working.
- **Pipeline classes renamed** `*Camera` → `*Pipeline` in
  `iSpy/vision/pipelines/`, with a backward-compatible alias kept
  (e.g. `ObjectDetectionCamera = ObjectDetectionPipeline`, and the same for
  `april_tag`, `depth_anything`, `object_detection`, `optical_flow`, `qr_code`,
  `yolo_world`). `PIPELINES` / `get_pipeline_classes()` now map to the `*Pipeline`
  classes.
- **`iSpy/vision/pipelines/base.py`** gained a hardware/compute-backend reporting
  surface (`hardware`, `hardware_options()`, `active_hardware()`).
- **`iSpy/vision/pipelines/optimizable.py`** gained the shared model optimization
  plumbing needed by the model-backed pipelines.
- **Web UI** (`iSpy/web/modules/cameras.py`, `dashboard.py`,
  `templates/{cameras,dashboard}.html`, `static/css/design.css`,
  `Backend/PluginStatus.py`) updated to model camera *sources* separately from
  pipelines, including `/api/cameras/sources` discovery.
- **Config** (`iSpy/config/iSpyConfig.py`) extended for the source concept.
- **Tests** — new `tests/test_cameras.py`; `tests/test_architecture.py` updated.

### Result

Camera acquisition and vision processing are now independent, separately
pluggable concerns. Pipelines consume frames from whatever source they are given.

---

## 2. TESTING 2 — Removing AGPL Ultralytics from the Runtime

### Goal

The project's target license is **PolyForm Noncommercial / CC BY-NC**. Ultralytics
is **AGPL-3.0** (strong copyleft) — for a Noncommercial license to be legally
enforceable, the runtime must not depend on AGPL code. The objective: **the runtime
no longer requires AGPL Ultralytics** while keeping every existing model working.

### Approach

Instead of swapping to a different network family, the existing Ultralytics-style
`.pt` checkpoints are loaded with a new **dependency-free loader** that
re-implements only the small YOLOv8 inference surface (MIT/BSD math — nothing
copied from the AGPL library).

- **New `iSpy/vision/yolo_pt.py`** — `load_yolo_pt()` and a namespace shim:
  - Re-implements the blocks the pickles reference (`Conv`, `C2f`, `SPPF`,
    `Concat`, `DFL`, `Detect`, `DetectionModel`) under the exact module names the
    checkpoint pickle expects (`ultralytics.nn.*`).
  - `register_shim()` installs those as fake `sys.modules` entries so
    `torch.load` can unpickle a checkpoint **without ever importing Ultralytics**.
  - `YoloPT` wrapper exposes the same surface iSpy relies on: `.task`, `.names`,
    `.nc`, `.model`, `.to()`, and `__call__(frames, ...)` returning results with
    `.boxes` (`.xyxy`/`.conf`/`.cls`) and (for pose) `.keypoints.data`.
  - Includes letterbox, forward, DFL/anchor decode, scale-back, and per-class
    non-max-suppression in pure torch/numpy.

### Runtime wiring (now Ultralytics-free)

- `iSpy/vision/genericYolo.py` loads `.pt` models (CPU/GPU/TPU) via
  `load_yolo_pt` instead of `from ultralytics import YOLO`.
  - Added `genericYolo.torch_load(path)` — registers the shim and calls
    `torch.load` — used by metadata extraction.
- `iSpy/vision/metadata.py` — `metadata_from_pt()` reads checkpoint metadata via
  `genericYolo.torch_load` (this closes the one latent bug where `torch_load`
  was referenced but undefined).
- `iSpy/vision/ModelInspector.py` — `_inspect_ultralytics()` reads metadata via
  `load_yolo_pt`.
- `iSpy/vision/pipelines/yolo_world.py` — both full-precision and quantized
  inference paths run through `GenericYolo` / `load_yolo_pt`. Ultralytics is kept
  **only** inside the optional build-time `_reparameterize_world()` step.
- `iSpy/vision/optimizer.py` — the model exporter imports Ultralytics lazily, and
  only as documented build-time tooling.

### Dependency / license changes

- `pyproject.toml`:
  - **Removed** `ultralytics` from core `dependencies` — the runtime no longer
    requires AGPL code.
  - **Added** optional `[optimizer]` extra for build-time model export /
    YOLO-World reparameterization (`pip install ".[optimizer]"`), documented as
    build-time only.
  - **License** set to `PolyForm Noncommercial License 1.0.0` to match the
    project's non-commercial goal.
- `iSpy/dataset/dataset.py` / `iSpy/validations/validate_system.py` — renamed the
  dataset readiness flag `ultralytics_ready` → `yolo_data_ready`.
- `iSpy/validations/unit_tests.py` — removed the fake-ultralytics test shim.

### What still references "ultralytics"

Run `git grep -nE "^\s*(from ultralytics|import ultralytics)"` — only **two**
matches remain, both explicitly marked **optional build-time tooling** (not
runtime):

- `iSpy/vision/optimizer.py` (`_export_ultralytics`, `[optimizer]` extra).
- `iSpy/vision/pipelines/yolo_world.py` (`_reparameterize_world`, build-time).

`yolo_pt.py` uses the string `"ultralytics.nn.*"` only as **fake module names** for
the pickle shim — it never imports the real package.

---

## 3. Verification

### Test status

`env\Scripts\python.exe -m pytest tests/ -q`

**347 passed, 9 failed** on **both** TESTING 1 and TESTING 2. The 9 failures are
the same **pre-existing** failures, unrelated to this work:

- 4 NetworkTable struct publishing (`tests/test_addons.py`,
  `tests/test_addon_output_system.py`).
- 3 camera_geometry triangulation precision (`tests/test_camera_geometry.py`).
- 2 settings-page key drift (`tests/test_settings_page_keys.py`).

No test regressions were introduced by either commit. The restructure tests
(`tests/test_cameras.py`, `tests/test_architecture.py`) and the model/pipeline
tests all pass.

### Runtime smoke checks (run with `env\Scripts\python.exe`)

The dependency-free loader was verified end-to-end against a real `yolov8n.pt`
checkpoint:

- `load_yolo_pt(...)` loads the checkpoint (task=`detect`, `nc=80`, COCO names)
  and runs inference, with **no Ultralytics import**.
- `metadata_from_pt(Path("...pt"))` reads task/nc/names/input_size (exercises the
  fixed `torch_load`).
- `GenericYolo({...pt config...}).predict(frame)` runs as `model_type="yolo"`.
- `ModelInspector._inspect_ultralytics(...)` returns task/nc/input_size correctly.

### How to verify the runtime needs no Ultralytics

The only place the runtime can touch Ultralytics is `load_yolo_pt`'s shim, and
that path creates *fake* modules rather than importing the package. A quick check:

```bash
env\Scripts\python.exe -c "import importlib.util; print(importlib.util.find_spec('ultralytics'))"
```

does not need to return a spec for iSpy's `.pt` runtime inference to work.

---

## 4. Running the app

```bash
env\Scripts\python.exe -m pytest tests/ -q          # full test suite
env\Scripts\python.exe -m iSpy.boot.boot            # boot iSpy normally
```

Build-time model export / YOLO-World reparameterization (optional, x86):

```bash
pip install ".[optimizer]"
```

---

_Last updated: 2026-08-30 · Branch: `Restructure`_
