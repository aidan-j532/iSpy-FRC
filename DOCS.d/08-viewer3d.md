# 08 — 3D Viewer

> Three.js-based 3D detection viewer with generic overlay API.

---

## Overview

The 3D viewer provides a real-time 3D visualization of:
1. **Detections** — objects detected by the vision pipeline, rendered as cubes, planar markers, or point clouds
2. **Overlays** — arbitrary 3D objects pushed by plugins (e.g., robot position)

The viewer polls two endpoints every 200ms (5 Hz):
- `/api/detections/latest` — returns detection objects
- `/api/overlays` — returns overlay objects

---

## Backend (iSpy/web/modules/viewer3d.py)

### Detection Storage

Every vision tick, `update(frame_data)` is called:
```python
def update(self, frame_data):
    detections = frame_data.get("detections", [])
    self._latest_objects = []
    for idx, obj in enumerate(detections):
        if getattr(obj, "depth_source", "") == "optical_flow":
            continue  # skip velocity-only signals
        obj_entry = {
            "id": idx,
            "x": getattr(obj, "x", 0),
            "y": getattr(obj, "y", 0),
            "z": getattr(obj, "z", 0),
            "roll": getattr(obj, "roll", 0),
            "pitch": getattr(obj, "pitch", 0),
            "yaw": getattr(obj, "yaw", 0),
            "confidence": getattr(obj, "confidence", 0),
            "class_name": getattr(obj, "class_name", ""),
            "vis_type": ...,  # resolved from keypoints, etc.
            "vis_meta": ...,  # tag_id, size, etc.
            "keypoints_3d": [...],
            "num_keypoints": ...,
        }
        self._latest_objects.append(obj_entry)
```

### Overlay API

Any plugin can push overlays to the viewer:

```python
# Get viewer reference from context
viewer = context["viewer3d"]

# Add/update an overlay
viewer.add_overlay("my_id", {
    "type": "box",          # "box"|"sphere"|"group"
    "x": 0, "y": 0, "z": 0.15,
    "roll": 0, "pitch": 0, "yaw": 0,
    "color": "#4c8bf5",
    "label": "My Object",
    "data": {"width": 0.5, "height": 0.3, "depth": 0.5},
})

# Remove an overlay
viewer.remove_overlay("my_id")
```

### Endpoints

| Endpoint | Response |
|----------|----------|
| GET /viewer3d | HTML page |
| GET /api/detections/latest | `{"objects": [...]}` |
| GET /api/overlays | `{"overlays": [...]}` |

---

## Frontend (iSpy/web/templates/viewer3d.html)

### Three.js Setup

```javascript
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
const controls = new OrbitControls(camera, renderer.domElement);

// Lighting
const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
const hemiLight = new THREE.HemisphereLight(0x4c8bf5, 0x111318, 0.3);

// Grid (0.25m cells, 20m span)
const gridHelper = new THREE.GridHelper(20, 80, 0x2a2e38, 0x1a1d24);
const axesHelper = new THREE.AxesHelper(8);
```

### Coordinate System

iSpy uses a Z-up coordinate system (X-forward, Y-left, Z-up). Three.js uses Y-up. The `toViewer()` function converts:
```javascript
function toViewer(px, py, pz) {
    return new THREE.Vector3(px, pz, py);  // swap Y and Z
}
```

### Detection Renderers

| Vis Type | Create | Update | Description |
|----------|--------|--------|-------------|
| generic | CubeGeometry(CUBE_SIZE) | Set position + rotation | Default: 0.25-unit colored cube |
| planar | PlaneGeometry(size) + edges + axes | Set position + rotation + scale | Flat rectangle for AprilTags, sized by vis_meta.size |
| points | Group of SphereGeometry dots | Add/remove dots to match keypoint count | 3D keypoints as red dots |

### Resolution Logic

```javascript
function resolveVisType(o) {
    const hasKp = o.keypoints_3d && o.keypoints_3d.length >= 3;
    if (hasKp) return "points";
    if (o.vis_type && VIS_RENDERERS[o.vis_type]) return o.vis_type;
    return "generic";
}
```

### Detection Sync

Maintains a `boxes` dict mapping detection ID to Three.js object:

```javascript
const boxes = {};

function syncDetections(objects) {
    const seen = new Set();
    for (const o of objects) {
        seen.add(o.id);
        if (!boxes[o.id]) {
            boxes[o.id] = createBoxForDetection(o);
        } else if (visType changed) {
            cleanupBox(o.id);
            boxes[o.id] = createBoxForDetection(o);
        } else {
            updateExistingBox(boxes[o.id], o);
        }
    }
    // remove stale boxes
    for (const k in boxes) {
        if (!seen.has(parseInt(k))) cleanupBox(k);
    }
}
```

### Overlay Renderers

| Type | Create | Update |
|------|--------|--------|
| box | BoxGeometry(w,h,d) with wireframe edges | Set position + rotation |
| sphere | SphereGeometry(r) | Set position + rotation |
| group | THREE.Group | Set position + rotation, sync children |

### Overlay Sync

```javascript
const overlayMeshes = {};

function syncOverlays(overlays) {
    const seen = new Set();
    for (const o of overlays) {
        seen.add(o.id);
        if (!overlayMeshes[o.id]) {
            const mesh = OVERLAY_RENDERERS[o.type].create(o);
            scene.add(mesh);
            overlayMeshes[o.id] = { mesh, type: o.type };
        } else {
            OVERLAY_RENDERERS[o.type].update(overlayMeshes[o.id].mesh, o);
        }
    }
    // remove stale overlays
    for (const id in overlayMeshes) {
        if (!seen.has(id)) {
            scene.remove(overlayMeshes[id].mesh);
            disposeMesh(overlayMeshes[id].mesh);
            delete overlayMeshes[id];
        }
    }
}
```

### Table Rendering

The bottom table shows both detections and overlays:

```javascript
function renderTable(objects, overlays) {
    // detection rows in default color
    // overlay rows in accent color (#4c8bf5)
}
```

### Controls

- **Orbit**: Left-click drag to rotate, scroll to zoom, right-click drag to pan
- **View Presets**: Top, front, side camera positions
- **Toggle Overlays**: Checkbox to show/hide overlay objects
- **Toggle White BG**: Switch between dark and light background

---

## Adding Custom Overlays from Plugins

Example: NetworkTable handler pushes robot position every tick:

```python
# In NetworkHandler.update():
pose = frame_data.get("robot_pose")
if pose:
    self._viewer.add_overlay("robot", {
        "type": "box",
        "x": pose["x"],
        "y": pose["y"],
        "z": 0.15,
        "yaw": pose["heading"],
        "color": "#4c8bf5",
        "label": "Robot",
        "data": {"width": 0.76, "height": 0.30, "depth": 0.69},
    })
```

This renders as a blue box at the robot's position on the 3D field.
