# 03 — Cameras Module

> The web backend for camera management, live feeds, calibration, and device discovery.

---

## Overview

`iSpy/web/modules/cameras.py` (1687 lines) is the largest module in the codebase. It handles:
- Camera CRUD (add/edit/remove via web UI)
- Live MJPEG video feeds for each camera
- ChArUco board calibration (intrinsics + extrinsics)
- Focal length calibration (known-object method)
- PnP pose calibration
- Camera device discovery (Linux v4l2, Windows registry)
- SSE events for hot-plug detection
- Image tuning sliders (exposure, brightness, etc.)
- Camera grid (multi-camera overview)

---

## Registered Routes

| Route | Method | Purpose |
|-------|--------|---------|
| /cameras | GET | Camera management page |
| /api/cameras | GET | List all cameras as JSON |
| /api/cameras/<name>/feed | GET | MJPEG stream for a camera |
| /api/cameras/<name>/settings | GET/POST | Camera settings read/write |
| /api/cameras/<name>/calibration/focal | GET | Focal length calibration data |
| /api/cameras/<name>/calibration/charuco/feed | GET | ChArUco calibration MJPEG stream |
| /api/cameras/<name>/calibration/charuco/capture | POST | Capture a ChArUco frame |
| /api/cameras/<name>/calibration/charuco/status | GET | ChArUco detection status |
| /api/cameras/<name>/calibration/charuco/intrinsics | GET | Computed intrinsics matrix |
| /api/cameras/<name>/calibration/pnp | GET/POST | PnP pose calibration |
| /api/cameras/discover | GET | Discover connected camera devices |
| /api/cameras/grid | GET | Camera grid overview |

---

## Camera Feed Generator (_generate)

The main video feed is an MJPEG stream. The generator:

1. Reads the raw frame from `cam.get_raw_frame()` (thread-safe, lock-protected)
2. Checks if `_FEED_MAX_DIM` is set; if frame is wider, resizes proportionally
3. Encodes to JPEG at quality 70
4. Yields as `--frame` multipart chunk
5. Sleeps 30ms between frames (~33 FPS max)

```python
def _generate(self, cam_name):
    while True:
        cam = self.live_cameras.get(cam_name)
        frame = cam.get_raw_frame() if cam else None
        if frame is not None:
            # resize if needed, encode JPEG, yield
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")
        time.sleep(0.03)
```

---

## Calibration Feed Generator (_generate_calibration)

Similar to _generate, but with overlay rendering for calibration boards.

### ChArUco mode (overlay="charuco")

1. Starts a detection worker thread that:
   - Reads frames from the camera
   - Runs ChArUco board detection (cv2.aruco)
   - Caches the latest detection result
2. The generator holds the opening MJPEG chunk until the detector's first tick
   (deadline: `_CALIB_FEED_WARMUP_S` = 1.5s) so the first served frame is
   already annotated rather than raw - prevents the "camera not loading" and
   "feed goes black" bugs without streaming un-annotated frames
3. Each frame gets the detection overlay drawn on it:
   - Green lines connecting detected corners
   - Blue dots for marker centers
4. If nothing matches the session's stored layout, `detect_charuco_auto()`
   sweeps common board patterns (5x7, 7x5, 8x6, 6x6, ...) and dictionaries
   (DICT_4X4/5X5/6X6_250) and ranks candidates by corner coverage; the matched
   layout is remembered in the session so later ticks use it directly

---

## Calibration Wizard

The calibration wizard in cameras.html is a multi-step process:

### Step 1: Focal Length Calibration

Uses the known-object method:
1. User enters: game piece size (inches), distance to game piece, pixel height of bounding box, camera FOV
2. API computes focal length: `focal_pixels = (pixel_height * distance) / game_piece_size`
3. Saves to camera config

### Step 2: ChArUco Intrinsics

1. User selects board size (rows x cols) and dictionary
2. Stream shows live ChArUco detection overlay
3. Auto-capture collects frames when board is detected
4. After enough frames, computes camera matrix + distortion coefficients
5. User can also manually capture frames

### Step 3: PnP Pose Calibration

1. Uses the intrinsics from Step 2
2. User places ChArUco board at a known position on the field
3. Captures a frame, detects board corners
4. Solves PnP to get camera extrinsics (rotation + translation relative to board)
5. Saves the pose model for the camera

---

## Camera CRUD

### Adding a camera

POST /api/cameras with JSON body:
```json
{
    "name": "front_cam",
    "source": "/dev/video0",
    "pipeline": "object_detection",
    "fps_cap": 30
}
```

This creates a camera config entry and returns the new camera data. The camera won't be live until the vision loop is restarted.

### Editing a camera

POST /api/cameras/<name>/settings with updated settings dict. Some settings (source, pipeline) require a restart.

### Removing a camera

DELETE /api/cameras/<name> removes the camera config entry.

---

## Device Discovery

### Linux (v4l2)

```bash
v4l2-ctl --list-devices
```
Parses output to find video devices and their names.

### Windows

Reads from Windows registry under HKLM\SYSTEM\CurrentControlSet\Control\Class\... to find connected USB cameras.

---

## Camera Grid

The camera grid page shows a tiled view of all camera feeds. Each tile shows:
- Camera name
- Live MJPEG feed
- Status indicators (FPS, detection count)

During calibration, the grid is paused (all feeds blanked) to free camera resources for the calibration wizard. The `pauseCameraGridFeeds()` JS function handles this.

---

## Image Tuning

The lightbox modal provides real-time image adjustment sliders:
- Brightness
- Contrast
- Saturation
- Exposure
- Gain
- White Balance

These map to OpenCV VideoCapture properties (cv2.CAP_PROP_*) and are applied directly to the camera device.

---

## Thread Safety

- Camera frames are protected by a Lock in Camera.py (`self._lock`)
- Calibration detection results use `result_lock` (threading.Lock)
- The detection worker thread runs separately from the feed generator
- `self.live_cameras` dict is accessed from both the main thread and Flask request threads

---

## Calibration Feed Bug Fix (Recent)

The calibration feed previously blocked for 3 seconds waiting for ChArUco detection before yielding any frames. This caused the img tag to fire onerror and show "Live feed not loading" while all other camera grid feeds were already blanked.

Fix: During the warmup period, frames are yielded immediately without overlay. The detection thread runs in parallel, and once it produces a result, the overlay starts being drawn on subsequent frames.
