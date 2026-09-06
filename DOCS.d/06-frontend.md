# 06 -- Frontend

> Templates, static files, CSS design system, and JavaScript architecture for
> every page of the iSpy web UI. This document covers every HTML structure,
> JavaScript function, CSS variable, component class, and interaction pattern.

---

## Table of Contents

1. [Technology Stack](#technology-stack)
2. [Design System (CSS)](#design-system-css)
3. [base.html -- Layout Shell](#basehtml----layout-shell)
4. [cameras.html -- Camera Management](#camerashtml----camera-management)
5. [addons.html -- Plugin Management](#addonshtml----plugin-management)
6. [viewer3d.html -- 3D Viewer](#viewer3dhtml----3d-viewer)
7. [Other Templates](#other-templates)
8. [Static Assets](#static-assets)
9. [Common JavaScript Patterns](#common-javascript-patterns)

---

## Technology Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Templating | Jinja2 | Server-side rendering, `{% extends "base.html" %}` |
| JavaScript | Vanilla JS | No framework, no build step, inline `<script>` blocks |
| CSS | Custom design system | CSS variables, no preprocessor |
| 3D Rendering | Three.js (ES module) | Vendored at `static/vendor/three.module.js` |
| Charts | Chart.js | Vendored at `static/vendor/chart.umd.min.js` |
| Orbit Controls | Three.js addon | Vendored at `static/vendor/OrbitControls.js` |

There is no module bundler, no npm, and no shared JS library. Each page defines
its own inline `<script>` block with page-specific functions.

---

## Design System (CSS)

**File:** `iSpy/web/static/css/design.css` (565 lines)

### CSS Custom Properties (lines 2-27)

All colors, spacing, fonts, and effects are controlled by CSS variables in
`:root`:

```css
:root {
  /* Backgrounds */
  --bg: #0d1117;              /* Main page background */
  --bg-elevated: #161b22;     /* Sidebar, cards */
  --bg-surface: #1c2128;      /* Hover states, elevated surfaces */

  /* Borders */
  --border: #30363d;          /* Default borders */
  --border-light: #3d444d;    /* Lighter borders, hover borders */

  /* Text */
  --text: #e6edf3;            /* Primary text */
  --text-dim: #9198a1;        /* Secondary text, labels */
  --text-muted: #5b6472;      /* Muted text, placeholders */

  /* Accent */
  --accent: #2f81f7;          /* Primary action color (blue) */
  --accent-hover: #4a93f8;    /* Hover state */
  --accent-dim: rgba(47,129,247,0.14);  /* Background tint */

  /* Status */
  --ok: #3fb950;              /* Success green */
  --ok-dim: rgba(63,185,80,0.12);
  --bad: #f85149;             /* Error red */
  --bad-dim: rgba(248,81,73,0.12);
  --warn: #d29922;            /* Warning yellow */
  --warn-dim: rgba(210,153,34,0.12);

  /* Spacing */
  --radius: 6px;
  --radius-sm: 4px;

  /* Fonts */
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI Variable Display", ...;
  --mono: "SF Mono", "Cascadia Code", "Fira Code", Consolas, ...;

  /* Effects */
  --shadow: 0 1px 2px rgba(0,0,0,0.35);
  --shadow-lg: 0 8px 24px rgba(0,0,0,0.5);
  --transition: 0.15s ease;
}
```

### Component Classes

#### Shell Layout (lines 44-74)

| Class | Description |
|-------|-------------|
| `.app-shell` | Flex container for sidebar + content |
| `.sidebar` | Fixed left nav (224px, sticky) |
| `.content` | Main content area (flex: 1, padded) |

#### Cards (lines 80-98)

| Class | Description |
|-------|-------------|
| `.card` | Bordered container with padding, shadow, hover border transition |
| `.card h3` | Uppercase section header (0.72rem, dimmed) |

#### Buttons (lines 114-136)

| Class | Description |
|-------|-------------|
| `button, .btn` | Primary blue button |
| `.btn-ok` | Green success button |
| `.btn-bad` | Red danger button |
| `.btn-warn` | Yellow warning button |
| `.btn-sm` | Small button variant |
| `.btn-secondary` | Dark surface button |
| `.btn-loading` | Disabled/loading state |

#### Tables (lines 137-145)

| Class | Description |
|-------|-------------|
| `table` | Full-width, collapsed borders |
| `th` | Uppercase header (0.7rem, dimmed) |
| `tr:hover td` | Subtle row highlight |

#### Form Elements (lines 147-157)

| Class | Description |
|-------|-------------|
| `input, select, textarea` | Dark background, border, focus ring |
| `.form-row` | Flex row with label + input |
| `.form-label` | Fixed-width label (180px min) |
| `.toggle-input` | CSS-only toggle switch (44x22px) |

#### Camera Components (lines 166-211)

| Class | Description |
|-------|-------------|
| `.camera-card` | Camera tile in grid |
| `.camera-feed-wrap` | 16:9 aspect ratio container |
| `.camera-feed` | MJPEG img element |
| `.camera-lightbox` | Full-screen camera view overlay |
| `.camera-lightbox-tuning` | 280px sidebar with tuning sliders |

#### Status & Indicators (lines 108-112)

| Class | Description |
|-------|-------------|
| `.status-dot` | 8px colored circle |
| `.status-ok` | Green dot |
| `.status-bad` | Red dot |
| `.status-warn` | Yellow dot |

#### Toast Notifications (lines 322-333)

| Class | Description |
|-------|-------------|
| `.toast-container` | Fixed top-right container |
| `.toast` | Notification card with border-left color |
| `.toast-success` | Green left border |
| `.toast-error` | Red left border |
| `.toast-info` | Blue left border |

#### Modals (lines 335-351)

| Class | Description |
|-------|-------------|
| `.modal-overlay` | Full-screen backdrop with blur |
| `.modal` | Centered card (max 420px) |
| `.modal-wide` | Wider variant (max 780px) |

#### Add-ons Page (lines 444-538)

| Class | Description |
|-------|-------------|
| `.addon-row` | Plugin card row |
| `.addon-row.is-enabled` | Enabled state (accent name color) |
| `.addon-settings-panel` | Inline settings panel |
| `.switch` | Row-level toggle switch (40x22px) |
| `.list-table` | Editable table for list-type settings |
| `.builtin-pill` | Gray "built-in" badge |
| `.beta-pill` | Yellow "beta" badge |

### Responsive Breakpoints (lines 553-565)

```css
@media (max-width: 768px) {
  .sidebar { width: 64px; min-width: 64px; }  /* Icons only */
  .sidebar .brand span { display: none; }      /* Hide "iSpy" text */
  .sidebar a { font-size: 0; }                 /* Hide link text */
  .content { padding: 16px; }                  /* Tighter padding */
  .form-row { flex-direction: column; }        /* Stack label + input */
}
```

Additional breakpoints:
- Stats grid: 6 cols -> 3 cols at 1200px -> 2 cols at 700px (lines 254-257)
- Dashboard columns: 2 cols -> 1 col at 900px (line 261)
- Camera grid: 2 cols -> 1 col at 768px (line 96)

---

## base.html -- Layout Shell

**File:** `iSpy/web/templates/base.html` (354 lines)

### Structure (lines 1-70)

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{% block title %}iSpy{% endblock %}</title>
  <link rel="stylesheet" href="/static/css/design.css">
  <style>/* tour styles */</style>
  {% block head %}{% endblock %}
</head>
<body>
  <div class="app-shell">
    <nav class="sidebar">
      <div class="brand"><!-- iSpy logo + text --></div>
      <a href="/dashboard">Dashboard</a>
      <a href="/cameras">Cameras</a>
      <a href="/health-page">Health</a>
      <a href="/metrics">Metrics</a>
      <a href="/models">Models</a>
      <a href="/datasets">Datasets</a>
      <a href="/viewer3d">3D Viewer</a>
      <a href="/logs">Logs</a>
      <a href="/addons">Add-ons</a>
      <a href="/settings">Settings</a>
    </nav>
    <main class="content">
      {% block content %}{% endblock %}
    </main>
  </div>
  <div class="toast-container" id="toast-container"></div>
  {% block scripts %}{% endblock %}
  <script>/* sidebar active + toast + tour */</script>
</body>
</html>
```

### Sidebar Navigation (lines 49-64)

Each nav link contains an inline SVG icon (16x16, stroke-based) and text.
The active link is highlighted via JavaScript that matches `a.pathname ===
location.pathname` (line 72-74).

### Toast System (lines 76-88)

```javascript
function showToast(message, type) {
  // type: 'info' | 'success' | 'error'
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = 'toast toast-' + type;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}
```

### First-Run Tutorial (lines 90-352)

A complete guided tour system implemented as a self-contained IIFE:

**Data:** `TOUR` array (lines 96-168) with 18 steps across all pages. Each step
has: `page` (URL), optional `target` (CSS selector for highlight), `title`,
`text`.

**State:** Stored in `sessionStorage` under `ispy_tour_index`.

**Key functions:**
- `show(idx)` (line 218): Creates highlight overlay + card, positions relative
  to target element
- `position()` (line 195): Repositions highlight overlay on resize/scroll
- `next()` / `prev()` (lines 262, 272): Navigate steps, handle page transitions
- `finish()` / `skip()` (lines 286, 292): Clear state, call API dismiss
- `showBanner()` (line 297): Shows "New to iSpy?" welcome card on dashboard
- `init()` (line 331): On DOMContentLoaded, checks if tour is active or needed

**Keyboard:** Escape=skip, ArrowRight=next, ArrowLeft=prev (lines 342-347).

**API:** `POST /api/onboarding {completed: true}` to dismiss permanently.

---

## cameras.html -- Camera Management

**File:** `iSpy/web/templates/cameras.html` (2335 lines)

The largest and most complex template. Contains the camera grid, add/edit modal,
calibration wizard, lightbox with tuning sliders, folder picker, and model picker.

### HTML Structure (lines 1-176)

#### Camera Grid (lines 1-18)
```html
<h1>Cameras</h1>
<div id="camera-controls" class="card controls-bar">
  <label>Refresh interval
    <select id="refresh-rate" onchange="setRefreshRate()">
      <option value="500">500ms</option>
      <option value="1000" selected>1s</option>
      <option value="2000">2s</option>
      <option value="5000">5s</option>
    </select>
  </label>
  <button onclick="refreshCameras()" class="btn-refresh">Refresh</button>
  <button onclick="openCameraModal('add')" class="btn-ok" style="margin-left:auto;">+ Add Camera</button>
</div>
<div id="camera-grid" class="grid"></div>
```

The camera grid is dynamically populated via JavaScript. Each card contains:
- Camera feed (MJPEG img)
- Camera name and status indicator
- Pipeline badge
- Edit/Delete/Lightbox buttons

#### Camera Lightbox (lines 20-40)

Full-screen overlay with:
- Live MJPEG feed (left)
- Tuning sliders sidebar (right, 280px)
- Header with camera name, tuning status, close button

Tuning sliders cover (defined in `_TUNING_KEYS`):
- Brightness (-100 to 100)
- Contrast (-100 to 100)
- Saturation (-100 to 100)
- White Balance (-100 to 100)
- Tint (-100 to 100)
- Gamma (0.1 to 3.0)
- Exposure Time (1 to 10000)
- Gain (0 to 255)

#### Camera Add/Edit Modal (lines 42-55)

Shared modal for both add and edit modes. Content is schema-driven:
```html
<div id="camera-modal" class="modal-overlay">
  <div class="modal camera-modal">
    <div id="camera-modal-header">
      <h3 id="camera-modal-title">Camera</h3>
      <span id="camera-modal-note" class="restart-badge">Restart vision to apply</span>
    </div>
    <div id="camera-modal-body"></div>
    <div class="modal-actions">
      <button onclick="closeCameraModal()">Cancel</button>
      <button id="camera-modal-submit" onclick="handleCameraSubmit()" class="btn-ok">Save and Continue</button>
    </div>
  </div>
</div>
```

Form fields (generated dynamically):
- **Identity**: name, source (from device picker)
- **Mount & Location**: x, y, z, yaw, pitch, height, subsystem
- **Pipeline**: name selector, model picker, pipeline-specific settings
- **Calibration**: link to open calibration wizard

#### Calibration Wizard (lines 57-73)

Multi-step modal:
```html
<div id="calibration-modal" class="modal-overlay">
  <div class="modal calibration-modal">
    <div id="calibration-header">
      <h3 id="calibration-title">Calibrate Camera</h3>
      <span id="calibration-pipeline-badge"></span>
      <span id="calibration-note" class="restart-badge">Applies after vision restart</span>
      <button class="modal-close" onclick="closeCalibrationWizard()">&times;</button>
    </div>
    <div id="calibration-body"></div>
    <div class="modal-actions calib-footer">
      <button onclick="resetCalibration()" class="btn-sm btn-warn">Reset calibration</button>
      <span style="flex:1"></span>
      <button onclick="closeCalibrationWizard()" class="btn-secondary">Close</button>
    </div>
  </div>
</div>
```

Calibration stages (rendered dynamically in `#calibration-body`):

1. **Focal Length**: Input fields for real object size, distance, measured
   pixel height. Shows live feed with measurement canvas overlay.

2. **ChArUco Intrinsics**: Live feed with board detection overlay, auto-capture
   progress, rolling RMS display, capture count, and "Calibrate Intrinsics"
   button.

3. **PnP Pose**: Object point editor (3D keypoints table), solve button,
   results display.

#### Folder Picker (lines 75-120)

Two-tab modal (device / upload):
- **On Device**: File browser that can also browse a dataset folder
- **Upload**: Name input + file upload for images

#### Model Picker (lines 122-146)

Two-tab modal (on device / upload):
- **On Device**: List of .pt files from `YoloModels/pytorch/`
- **Upload**: File upload for new .pt files

### CSS (lines 148-300)

Inline `<style>` block with page-specific styles:

| Class | Purpose |
|-------|---------|
| `.camera-modal` | Max width 640px, 92% viewport |
| `.calibration-modal` | Max width 760px, 94% viewport |
| `.calib-stage` | Live feed container (16:9 aspect, black background) |
| `.calib-section` | Settings section with surface background |
| `.calib-auto-bar` | Auto-capture progress bar |
| `.calib-result-card` | Green/red result card |
| `.calib-footer` | Footer with border-top |
| `.form-help` | Help text below form fields (192px margin-left) |
| `.camera-lightbox-tuning` | 280px tuning sidebar |
| `.tune-slider-row` | Individual slider row |
| `.folder-picker-list` | Scrollable folder list |
| `.folder-picker-item` | Clickable folder item |

### Key JavaScript Functions

#### Camera Grid

- `refreshCameras()` (dynamic): Fetches `/api/cameras`, renders cards
- `setRefreshRate()` (dynamic): Updates polling interval
- Camera card rendering (dynamic): Creates grid cards with feed imgs

#### Lightbox

- `openCameraLightbox(camName)`: Opens fullscreen view, sets MJPEG src to
  `/video/<camName>`, loads tuning values via `/api/cameras/tuning/<camName>`
- `closeCameraLightbox()`: Closes overlay, clears feed
- `resetTuning()`: Resets all sliders to defaults
- Tuning sliders: Range inputs that POST to `/api/cameras/tuning/<camName>`

#### Camera Add/Edit

- `openCameraModal(mode, camName)`: Opens modal in 'add' or 'edit' mode
  - Add: renders empty form
  - Edit: fetches camera config, pre-fills form
- `handleCameraSubmit()`: Collects form data, POSTs to `/api/cameras/config`
  (add) or PUTs to `/api/cameras/config/<name>` (edit)
- `closeCameraModal()`: Closes overlay

#### Calibration Wizard

- `openCalibrationWizard(camName)` (referenced at line 152-155): Opens
  calibration modal, fetches current calibration state, renders appropriate
  stage
- `pauseCameraGridFeeds()` (referenced): Blanks all grid feed img srcs to
  free camera bandwidth during calibration
- `calibrationFeedUrl(camName)`: Returns
  `/api/cameras/calibration/<camName>/feed?overlay=charuco`
- `calibrationFeedError()`: Handles feed load failure, shows error message
- `_first_jpeg(generator)`: Helper that gets the first JPEG frame from an
  MJPEG generator (used in tests)

#### Auto-Capture Flow

1. Wizard opens -> fetches `/api/cameras/calibration/<cam>/charuco/status`
2. Starts calibration feed via `_calibration_feed` MJPEG stream
3. Background `detection_worker` thread detects ChArUco boards
4. `_auto_consider()` checks coverage, diversity, and auto-captures
5. Rolling solve provides live RMS readout
6. User clicks "Calibrate Intrinsics" -> `_charuco_finish` endpoint

---

## addons.html -- Plugin Management

**File:** `iSpy/web/templates/addons.html` (642 lines)

### HTML Structure (lines 1-97)

#### New Add-on Card (lines 10-57)

Two-tab interface (Upload File / Write Code):
- **Upload**: Kind selector (tracker/utility/frame_processor) + drag-drop zone
- **Write Code**: Kind selector + filename input + textarea + save button

#### Add-ons List (lines 59-75)

Search bar + type filter dropdown + count indicator + dynamic list container.

#### Modals

- **Source Viewer** (lines 77-85): Full-screen modal showing plugin source code
  in a `<pre>` block
- **Delete Confirmation** (lines 87-96): Modal asking to confirm deletion

### JavaScript (lines 98-642)

#### State Variables (lines 99-101)

```javascript
let allAddons = [];          // All available plugins from API
let loaded = {};             // Plugin name -> status mapping
let pendingDeleteAddon = null;  // Pending delete target
let lastSnapshot = null;     // Last API response (for change detection)
```

#### Utility Functions (lines 104-109)

```javascript
const esc = s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
const jsStr = s => String(s ?? '').replace(/\\/g, '\\\\').replace(/'/g, "\\'");
const TYPE_COLORS = { tracker: '#4c8bf5', utility: '#4caf50', ... };
const TYPE_LABELS = { frame_processor: 'frame processor', ... };
```

#### `loadAddons()` / `loadStatus(force)` (lines 384-400)

Fetches both `/api/plugins/status` and `/api/plugins/available` in parallel.
Compares snapshot to avoid unnecessary re-renders. Updates `allAddons` and
`loaded` state, then calls `renderList()`.

Auto-refreshes every 5 seconds (line 640).

#### `renderList()` (lines 372-382)

Filters `allAddons` via `matchesFilter()`, stashes open settings panels,
renders HTML, restores open panels.

#### `renderRow(p)` (lines 258-327)

Renders a single add-on card. For each plugin:
1. Toggle switch (unless vision_pipeline)
2. Name + type badge + built-in/beta pills
3. Doc text (truncated)
4. Status dot (running/idle/pending restart)
5. Action buttons: Settings, Code, Delete
6. Settings panel (if enabled + has schema)

#### `settingsFields(schema, current)` (lines 140-169)

Generates form fields from a config schema:
- `"toggle"` -> checkbox with `.toggle-input` class
- `"number"` -> number input with `step="any"`
- `"text"` -> text input
- `"list"` -> calls `renderListField()`

Each input gets `data-key`, `data-type` attributes for collection.

#### `renderListField(key, def, rows)` (lines 171-211)

Renders an editable table for list-type settings:
- Column headers from `def.fields` labels
- Rows with input/select/checkbox cells
- Remove button per row
- "Add" button below table
- Each cell has `data-key`, `data-row`, `data-field`, `data-field-type` attrs

#### `addListRow(key)` (lines 213-252)

Dynamically adds a row to a list-type settings table:
1. Finds the table by `data-list-key`
2. Looks up the addon's schema for default values
3. Creates `<tr>` with cells for each field
4. Appends to tbody

#### `removeListRow(btn)` (lines 254-256)

Removes the closest `<tr>` ancestor.

#### `stashOpenPanels()` / `restoreOpenPanels()` (lines 329-370)

Preserves form state across re-renders:
- Stash: Reads all form values from open panels into a dict, including
  list-type settings as JSON-serializable row arrays
- Restore: Re-opens panels and sets values (lists restored via re-render)

#### `saveSettings(name, type)` (lines 425-493)

Collects settings from the panel:
1. Iterates `[data-key]` elements
2. Handles toggle (checked), number (Number()), text (value), list (table rows)
3. POSTs to `/api/plugins/settings`
4. Shows success/error toast
5. Reloads addon list

#### `togglePlugin(name, type, enable)` (lines 496-513)

POSTs to `/api/plugins/toggle`, shows toast, reloads list.

#### `viewSource(type, filename)` (lines 517-537)

Opens source modal, fetches `/api/plugins/<type>/<name>/source`, displays in
`<pre>` block.

#### Upload Handling (lines 572-603)

Drag-drop zone + file input:
- `handleAddonUpload(file)`: Creates FormData with file + type, POSTs to
  `/api/plugins/upload`

#### Create from Paste (lines 549-570)

`submitPasted()`: Collects type, filename, code from form, POSTs to
`/api/plugins/create`.

#### Delete Flow (lines 607-634)

1. `openDeleteAddonModal()`: Shows confirmation modal
2. `confirmDeleteAddon()`: DELETEs `/api/plugins/<type>/<filename>`
3. `closeDeleteAddonModal()`: Hides modal

---

## viewer3d.html -- 3D Viewer

**File:** `iSpy/web/templates/viewer3d.html` (486 lines)

### HTML Structure (lines 1-48)

```html
<h1>3D Detection Viewer</h1>
<div class="viewer-toolbar">
  <label>Camera
    <select id="view-preset" onchange="setViewPreset()">
      <option value="default">Default</option>
      <option value="top">Top-down</option>
      <option value="front">Front</option>
      <option value="side">Side</option>
    </select>
  </label>
  <label><input type="checkbox" id="show-grid" checked onchange="toggleGrid()"> Grid</label>
  <label><input type="checkbox" id="show-axes" checked onchange="toggleAxes()"> Axes</label>
  <label><input type="checkbox" id="light-bg" onchange="toggleLightBg()"> White BG</label>
  <label><input type="checkbox" id="show-overlays" checked onchange="toggleOverlays()"> Overlays</label>
</div>
<div id="scene-container"></div>
<div class="card" style="margin-top:12px;">
  <h3>Detections &amp; Overlays</h3>
  <div id="object-list">
    <table>
      <thead><tr><th>ID</th><th>X</th><th>Y</th><th>Z</th><th>Pitch</th><th>Yaw</th><th>Roll</th><th>Vis</th></tr></thead>
      <tbody id="obj-tbody"></tbody>
    </table>
  </div>
</div>
```

### Three.js Setup (lines 58-96)

Uses ES module imports via import map:
```html
<script type="importmap">
{ "imports": { "three": "/static/vendor/three.module.js", "three/addons/": "/static/vendor/" } }
</script>
```

**Scene setup:**
```javascript
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);           // Near-black
scene.fog = new THREE.FogExp2(0x0a0a0f, 0.04);          // Exponential fog

const camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 200);
camera.position.set(0, 8, 14);                           // Elevated default view
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
```

**Lighting (lines 83-90):**
```javascript
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(5, 10, 5);
dirLight.castShadow = true;
const hemiLight = new THREE.HemisphereLight(0x4c8bf5, 0x111318, 0.3);
```

**Grid & Axes (lines 92-95):**
```javascript
const gridHelper = new THREE.GridHelper(20, 80, 0x2a2e38, 0x1a1d24);  // 20m, 80 cells = 0.25m
const axesHelper = new THREE.AxesHelper(8);                              // 8m colored axes
```

### Constants (lines 134-138)

```javascript
const GRID_UNIT = 0.25;      // Grid cell size in meters
const CUBE_SIZE = 0.25;      // Default detection cube size
const DOT_RADIUS = 0.05;     // Keypoint dot radius
const BOX_COLOR = 0x4c8bf5;  // Default blue
```

### Coordinate Conversion (lines 152-154)

```javascript
function toViewer(px, py, pz) {
  return new THREE.Vector3(px, pz, py);  // Swap Y and Z (Z-up -> Y-up)
}
```

iSpy uses Z-up (X-forward, Y-left, Z-up). Three.js uses Y-up. The conversion
swaps py and pz.

### Detection Renderers (lines 156-235)

#### `makeCubeMarker()` (lines 160-169)

Creates a 0.25-unit blue cube with white wireframe edges:
```javascript
const geo = new THREE.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE);
const mat = new THREE.MeshStandardMaterial({ color: BOX_COLOR, transparent: true, opacity: 0.85 });
const mesh = new THREE.Mesh(geo, mat);
mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), wireMat));
```

#### `makePlanarMarker(o)` (lines 176-191)

Creates a flat rectangle for AprilTags/QR codes:
- Size from `o.vis_meta.size` (default `GRID_UNIT`)
- Orange plane (`0xffaa00`) with white edges
- AxesHelper at 60% of tag size

#### `buildPointCloud(kpts3d)` (lines 200-212)

Creates red sphere dots for each 3D keypoint:
- `SphereGeometry(DOT_RADIUS, 12, 12)` per point
- Red material (`0xff3333`)

#### `updatePointCloud(g, kpts3d)` (lines 214-228)

Adds/removes dots to match keypoint count, updates positions.

### VIS_RENDERERS Registry (lines 230-235)

```javascript
const VIS_RENDERERS = {
  generic: { create: () => makeCubeMarker(),     update: updateCubeMarker },
  points:  { create: (o) => buildPointCloud(o.keypoints_3d),
             update: (g, o) => updatePointCloud(g, o.keypoints_3d) },
  planar:  { create: (o) => makePlanarMarker(o), update: updatePlanarMarker },
};
```

### `resolveVisType(o)` (lines 237-242)

Determines which renderer to use:
1. If `keypoints_3d` has >= 3 points -> `"points"`
2. If `o.vis_type` matches a known renderer -> that type
3. Fallback -> `"generic"`

### Detection Sync (lines 244-296)

**State:** `boxes` dict mapping detection ID -> Three.js object.

**`createBoxForDetection(o)`** (lines 247-259):
1. Resolves vis type
2. Checks if existing box has same type (reuse) or different (cleanup + recreate)
3. Creates via `VIS_RENDERERS[visType].create(o)`
4. Stores `visType` in `userData`
5. Adds to scene

**`syncDetections(objects)`** (lines 276-296):
1. Builds `seen` set from incoming objects
2. For each object: create new, recreate if type changed, or update existing
3. Removes stale boxes not in `seen`

### Overlay Renderers (lines 298-393)

#### `OVERLAY_RENDERERS.box` (lines 313-331)

```javascript
{
  create(o) {
    const d = o.data || {};
    const w = d.width || 0.5, h = d.height || 0.5, dep = d.depth || 0.5;
    const geo = new THREE.BoxGeometry(w, h, dep);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), wireMat));
    return mesh;
  },
  update(mesh, o) {
    mesh.position.copy(toViewer(o.x, o.y, o.z));
    mesh.rotation.set(o.roll, o.yaw, o.pitch);
  },
}
```

#### `OVERLAY_RENDERERS.sphere` (lines 333-348)

Uses `SphereGeometry(r, 20, 20)` with configurable radius from `o.data.radius`.

#### `OVERLAY_RENDERERS.group` (lines 350-363)

Creates a `THREE.Group`. On update, calls `syncOverlayChildren()` to
recursively create/update/remove child meshes.

#### `syncOverlayChildren(group, childrenDef)` (lines 365-393)

Manages child meshes within a group overlay:
- Creates missing children via their renderer
- Updates existing children if type matches
- Removes children not in the definition

### Overlay Sync (lines 395-429)

**State:** `overlayMeshes` dict mapping overlay ID -> `{mesh, type}`.

**`syncOverlays(overlays)`** (lines 398-422):
1. For each overlay: create if new, update if type matches
2. Remove stale overlays not in `seen` set
3. Dispose geometry and material of removed meshes

**`toggleOverlays()`** (lines 424-429): Sets `visible` on all overlay meshes.

### Table Rendering (lines 434-452)

**`renderTable(objects, overlays)`**:
- Detection rows: default text color, shows X/Y/Z/Pitch/Yaw/Roll/Vis type
- Overlay rows: accent color (`#4c8bf5`), shows X/Y/Z/Yaw/label
- Empty state: "No detections" message

### Poll Loop (lines 458-471)

```javascript
async function poll() {
  const [detRes, ovRes] = await Promise.all([
    fetch('/api/detections/latest').then(r => r.json()),
    fetch('/api/overlays').then(r => r.json()),
  ]);
  syncDetections(detRes.objects || []);
  syncOverlays(ovRes.overlays || []);
  renderTable(detRes.objects || [], ovRes.overlays || []);
}
setInterval(poll, 200);  // 5 Hz
```

### View Presets (lines 121-132)

```javascript
const PRESETS = {
  default: { pos: [0, 8, 14], target: [0, 0, 0] },
  top:     { pos: [0, 22, 0.01], target: [0, 0, 0] },
  front:   { pos: [0, 2, 18], target: [0, 0, 0] },
  side:    { pos: [18, 2, 0], target: [0, 0, 0] },
};
```

### White Background Toggle (lines 103-119)

Switches between:
- Dark: `background: 0x0a0a0f`, fog enabled, dark grid colors
- Light: `background: 0xf0f0f0`, no fog, gray grid colors

### Animate Loop (lines 479-484)

```javascript
function animate() {
  requestAnimationFrame(animate);
  controls.update();              // Damping
  renderer.render(scene, camera);
}
animate();
```

---

## Other Templates

### dashboard.html (~300 lines)

Main dashboard with:
- Service status bar (start/stop vision)
- Live telemetry: FPS, vision ms, loop time, detection count, camera lag
- System metrics: CPU%, memory%, temperature
- Camera list with status dots
- Model info card
- Detection class breakdown
- Plugin list

### models.html (~100 lines)

YOLO model library with:
- Model cards showing name, size, task, class count
- Upload button
- Active model badge
- Delete button (disabled for active models)

### datasets.html (~100 lines)

Image dataset management with:
- Dataset grid with image counts
- Image upload
- Image grid with delete overlay on hover
- Folder browser for selecting dataset location

### health.html (~100 lines)

System health dashboard with auto-refresh (250ms interval).

### logs.html (~50 lines)

Log viewer with configurable line count and auto-refresh.

### metrics.html (~100 lines)

Performance charts using Chart.js:
- FPS over time
- Loop time / vision time / camera lag
- Pipeline stage breakdown

### settings.html (~100 lines)

Global configuration form with:
- Tabbed interface
- Save button with restart badge
- "Start tutorial" button

---

## Static Assets

| Path | Purpose |
|------|---------|
| `static/css/design.css` | Complete design system (565 lines) |
| `static/vendor/three.module.js` | Three.js ES module |
| `static/vendor/OrbitControls.js` | Three.js orbit controls |
| `static/vendor/chart.umd.min.js` | Chart.js UMD bundle |

---

## Common JavaScript Patterns

### API Calls

```javascript
const response = await fetch('/api/cameras');
const data = await response.json();
```

### SSE (Server-Sent Events)

```javascript
const source = new EventSource('/api/events');
source.onmessage = function(e) {
  const data = JSON.parse(e.data);
  updateDashboard(data);
};
```

### Polling

```javascript
setInterval(async () => {
  const data = await fetch('/api/detections/latest').then(r => r.json());
  syncDetections(data.objects);
}, 200);
```

### Parallel Fetch

```javascript
const [detRes, ovRes] = await Promise.all([
  fetch('/api/detections/latest').then(r => r.json()),
  fetch('/api/overlays').then(r => r.json()),
]);
```

### Toast Notifications

```javascript
showToast('Settings saved', 'success');
showToast('Failed to load', 'error');
```

### Dynamic HTML Generation

```javascript
container.innerHTML = list.map(item => `
  <div class="card">
    <h3>${esc(item.name)}</h3>
    <p>${esc(item.doc)}</p>
  </div>
`).join('');
```

All dynamic HTML uses the `esc()` helper to prevent XSS.
