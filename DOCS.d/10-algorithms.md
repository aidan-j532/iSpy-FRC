# 10 — Algorithms and Math

> Vision algorithms, camera calibration, triangulation, model conversion, quantization, and object detection post-processing.

---

## Calibration (`iSpy/vision/calibration.py`, 542 lines)

The calibration module implements camera intrinsics estimation, focal length
computation, ChArUco board detection, and visualization overlays. It supports
two calibration backends: a primary scipy-based joint optimizer and an OpenCV
fallback.

### OpenCV Version Compatibility

#### _pick_aruco() (line 13)

OpenCV 4.9+ moved several ArUco helpers from `cv2.aruco` to the main `cv2`
namespace (old aliases were removed). This function resolves bindings from
whichever namespace the installed build exposes:

```python
def _pick_aruco(*names: tuple) -> callable:
    for obj, name in names:
        fn = getattr(obj, name, None)
        if fn is not None:
            return fn
    raise AttributeError(f"no cv2 binding found for any of {[n for _, n in names]}")
```

**Usage (lines 21–26):**
```python
_draw_detected_markers = _pick_aruco(
    (cv2, "drawDetectedMarkers"), (cv2.aruco, "drawDetectedMarkers")
)
_draw_detected_corners_charuco = _pick_aruco(
    (cv2, "drawDetectedCornersCharuco"), (cv2.aruco, "drawDetectedCornersCharuco")
)
```

This ensures the same code runs on both old (<=4.8) and new (>=4.9) OpenCV
wheels without import errors.

### Board Constants (lines 221–226)

```python
DEFAULT_CHARUCO_PATTERN = (7, 9)  # squares wide x squares tall
DEFAULT_CHARUCO_DICT = cv2.aruco.DICT_4X4_50
DEFAULT_CHARUCO_SQUARE = 25.0     # mm
DEFAULT_CHARUCO_MARKER = 18       # mm
```

The square and marker lengths only need to be consistent (intrinsics are
scale-invariant). The marker must be smaller than the square it sits in.

### Focal Length Computation

#### focal_from_object() (line 229)

Simple pinhole camera model: given a known object size, distance, and pixel
height, compute focal length in pixels.

```python
def focal_from_object(real_size: float, distance: float, pixel_height: float) -> float:
    if real_size <= 0 or distance <= 0 or pixel_height <= 0:
        return 0.0
    return pixel_height * distance / real_size
```

**Math:** `f = (pixel_height * distance) / real_size`

- `real_size`: physical size of the object in the same units as `distance`
- `distance`: known distance from camera to object
- `pixel_height`: height of the object's bounding box in pixels

#### fov_from_focal() (line 235)

Converts focal length in pixels to horizontal field of view in degrees:

```python
def fov_from_focal(focal_px: float, width_px: float) -> float:
    if focal_px <= 0 or width_px <= 0:
        return 0.0
    return float(2.0 * np.degrees(np.arctan2(width_px / 2.0, focal_px)))
```

**Math:** `FOV = 2 * atan(width / (2 * focal))`

#### focal_from_fov() (line 241)

Inverse of `fov_from_focal` — converts horizontal FOV in degrees to focal
length in pixels:

```python
def focal_from_fov(fov_deg: float, width_px: float) -> float:
    if fov_deg <= 0 or width_px <= 0:
        return 0.0
    return (width_px / 2.0) / np.tan(np.radians(fov_deg / 2.0))
```

**Math:** `focal = (width / 2) / tan(FOV / 2)`

### ChArUco Board Operations

#### make_charuco_board() (line 364)

Creates an OpenCV `CharucoBoard` object:

```python
def make_charuco_board(cols, rows, square_length=DEFAULT_CHARUCO_SQUARE,
                       marker_length=DEFAULT_CHARUCO_MARKER,
                       dictionary_id=DEFAULT_CHARUCO_DICT):
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.CharucoBoard((cols, rows), square_length, marker_length, dictionary)
```

Parameters:
- `cols`, `rows`: number of board squares (not markers — markers are
  inside each square)
- `square_length`: side length of each square in mm
- `marker_length`: side length of each ArUco marker in mm (must be < square_length)
- `dictionary_id`: ArUco dictionary (default DICT_4X4_50)

#### detect_charuco() (line 375)

Full detection pipeline for a single frame:

```python
def detect_charuco(frame, cols, rows, square_length=..., marker_length=..., dictionary_id=...):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    board = make_charuco_board(cols, rows, square_length, marker_length, dictionary_id)
    detector = cv2.aruco.CharucoDetector(board)
    try:
        corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
    except cv2.error:
        return False, None, None, None, None, gray
    if corners is None or ids is None:
        return False, None, None, None, None, gray
    corners = _to_corners(corners)
    ids = _to_ids(ids)
    marker_corners = _to_marker_corners(marker_corners)
    marker_ids = _to_ids(marker_ids)
    found = len(corners) >= 4
    return found, corners, ids, marker_corners, marker_ids, gray
```

**Steps:**
1. Convert to grayscale if needed.
2. Create the board and a `CharucoDetector`.
3. Call `detector.detectBoard(gray)` to get ChArUco corners, IDs, marker
   corners, and marker IDs.
4. Normalize array shapes using `_to_corners()`, `_to_ids()`,
   `_to_marker_corners()` (handles OpenCV 5.x flat arrays vs classic shapes).
5. Return `found=True` if at least 4 corners are detected.

**Return tuple:** `(found, corners, ids, marker_corners, marker_ids, gray)`

#### detect_charuco_auto() (line 420)

Auto-detects a board without knowing its layout up front. Sweeps
`CHARUCO_COMMON_PATTERNS` (7x5, 5x7, 8x6, 6x8, 6x6 + default) x
`CHARUCO_COMMON_DICTS` (DICT_4X4_50/5X5_250/6X6_250) and returns the best
match:

```python
def detect_charuco_auto(frame, preferred_pattern=None, preferred_dict=None):
    # try preferred layout first, then sweep the common ones;
    # rank by (corner count, -board area) so a wrong-but-compatible grid
    # (e.g. a 7x5 print also "fits" a 7x9 grid) loses to the smallest
    # board that explains what is visible
    score = (len(corners), -(pattern[0] * pattern[1]))
```

**Behavior:**
1. Builds the candidate list with the preferred pattern/dict first (so a
   known-good session layout wins ties).
2. Runs `detect_charuco()` per candidate; exceptions are swallowed.
3. Returns the highest-scoring detection.

**Return tuple:** `(found, corners, ids, marker_corners, marker_ids, gray,
matched_pattern, matched_dict)` — last two are `None` when not found.

**Cost:** every miss is a full detector pass per candidate (~18 passes worst
case), so callers only fall back to the sweep after the session layout fails,
and remember the matched layout afterwards (`_remember_charuco_layout`).

#### _to_corners() (line 468), _to_ids() (line 475), _to_marker_corners() (line 482)

Shape normalization helpers for cross-version compatibility:

```python
def _to_corners(corners):
    a = np.asarray(corners, dtype=np.float32)
    if a.ndim == 2:
        return a.reshape(-1, 1, 2)  # (N,2) → (N,1,2)
    return a

def _to_ids(ids):
    a = np.asarray(ids, dtype=np.int32)
    if a.ndim == 1:
        return a.reshape(-1, 1)  # (N,) → (N,1)
    return a

def _to_marker_corners(marker_corners):
    out = []
    for marker in marker_corners:
        a = np.asarray(marker, dtype=np.float32).reshape(-1, 2)
        out.append(a.reshape(4, 1, 2))  # 4 corners per marker
    return out
```

### Camera Calibration (Intrinsics)

#### calibrate_charuco() (line 402)

The main calibration entry point. Requires at least 3 captures.

```python
def calibrate_charuco(captures, pattern=DEFAULT_CHARUCO_PATTERN, ...):
    if len(captures) < 3:
        return None
    cols, rows = pattern
    board = make_charuco_board(cols, rows, square_length, marker_length, dictionary_id)
    all_corners = []
    all_ids = []
    img_size = None
    for gray, corners, ids in captures:
        if corners is None or ids is None or len(corners) < 4:
            continue
        h, w = gray.shape[:2]
        img_size = (w, h)
        all_corners.append(np.asarray(corners, dtype=np.float32))
        all_ids.append(np.asarray(ids, dtype=np.int32))
    if len(all_corners) < 3:
        return None
    if _charuco_captures_degenerate(all_corners, all_ids):
        return None
    try:
        result = _calibrate_scipy(all_corners, all_ids, board, img_size)
    except Exception:
        result = None
    if result is not None:
        rms, cam_mat, dist, _, _ = result
    else:
        try:
            rms, cam_mat, dist, _, _ = _calibrate_camera_charuco(...)
        except cv2.error:
            return None
    if not np.isfinite(rms) or rms > 100.0:
        return None
    return {
        "rms": float(rms),
        "camera_matrix": cam_mat.tolist(),
        "dist_coeffs": dist.flatten().tolist(),
        "resolution": list(img_size),
        "count": len(captures),
    }
```

**Algorithm:**
1. Filter captures with < 4 corners.
2. Check for degenerate poses (all frames identical board position).
3. Try scipy-based calibration first.
4. Fall back to OpenCV's `calibrateCameraCharuco`.
5. Reject results with RMS > 100 or non-finite values.

#### _charuco_captures_degenerate() (line 350)

Detects when all captures show the board in the same position. Compares corner
locations by marker ID across frames — if all corners are within 1.0 pixel of
the first frame's positions, the captures are degenerate.

```python
def _charuco_captures_degenerate(all_corners, all_ids):
    if len(all_corners) < 2:
        return True
    ref = {}
    for id_val, corner in zip(all_ids[0].flat, all_corners[0]):
        ref[int(id_val)] = corner
    for corners, ids in zip(all_corners[1:], all_ids[1:]):
        for id_val, corner in zip(ids.flat, corners):
            r = ref.get(int(id_val))
            if r is not None and np.linalg.norm(np.asarray(corner) - np.asarray(r)) > 1.0:
                return False
    return True
```

### Scipy-Based Calibration

#### _calibrate_scipy() (line 185)

Joint intrinsic + extrinsic optimization using scipy's Levenberg-Marquardt:

```python
def _calibrate_scipy(all_corners, all_ids, board, img_size):
    pairs = _build_obj_img_pairs(all_corners, all_ids, board)
    if pairs is None:
        return None
    all_obj, all_img = pairs
    n_images = len(all_obj)
    if n_images < 2:
        return None

    K0 = _estimate_initial_intrinsics(all_obj, all_img, img_size)
    rvecs0, tvecs0 = _estimate_initial_extrinsics(all_obj, all_img, K0)
    x0 = _pack_params(K0, np.zeros(5), rvecs0, tvecs0)

    result = scipy.optimize.least_squares(
        _calibrate_scipy_residuals,
        x0,
        args=(all_obj, all_img, n_images),
        method="lm",
        max_nfev=200,
    )
    if not result.success and result.cost > 1e6:
        return None

    K, dist, rvecs, tvecs = _unpack_params(result.x, n_images)

    # Compute RMS
    total_err = 0.0
    total_pts = 0
    for i in range(n_images):
        proj, _ = cv2.projectPoints(all_obj[i], rvecs[i], tvecs[i], K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - all_img[i], axis=1)
        total_err += float((err ** 2).sum())
        total_pts += len(err)
    rms = float(np.sqrt(total_err / max(total_pts, 1)))
    return rms, K, dist, None, None
```

**Steps:**
1. Build object/image point pairs via `_build_obj_img_pairs()`.
2. Estimate initial intrinsics from per-image homographies.
3. Estimate initial extrinsics via `solvePnP` per image.
4. Pack all parameters into a flat vector.
5. Run `scipy.optimize.least_squares` with method `"lm"` (Levenberg-Marquardt).
6. Unpack optimized parameters and compute RMS reprojection error.

#### _build_obj_img_pairs() (line 80)

Maps detected ChArUco corners to 3D board coordinates for each image:

```python
def _build_obj_img_pairs(all_corners, all_ids, board):
    board_corners = np.asarray(board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
    all_obj = []
    all_img = []
    for corners, ids in zip(all_corners, all_ids):
        obj = []
        img = []
        for c, i in zip(corners.reshape(-1, 2), ids.reshape(-1)):
            pt = board_corners[int(i)]
            if pt is not None:
                obj.append(pt)
                img.append(c)
        if len(obj) < 4:
            return None  # not enough corners in this image
        all_obj.append(np.asarray(obj, dtype=np.float64))
        all_img.append(np.asarray(img, dtype=np.float64))
    return all_obj, all_img
```

#### _estimate_initial_intrinsics() (line 105)

Estimates rough camera matrix from per-image homographies:

```python
def _estimate_initial_intrinsics(all_obj, all_img, img_size):
    w, h = img_size
    Ks = []
    for obj, img in zip(all_obj, all_img):
        H, _ = cv2.findHomography(obj[:, :2], img)
        if H is None:
            continue
        fx = abs(H[0, 0])
        fy = abs(H[1, 1])
        cx = abs(H[0, 2])
        cy = abs(H[1, 2])
        if 0 < fx < w*4 and 0 < fy < h*4 and 0 < cx < w*2 and 0 < cy < h*2:
            Ks.append((fx, fy, cx, cy))
    if Ks:
        arr = np.array(Ks)
        fx, fy, cx, cy = arr.mean(axis=0)
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return np.array([[max(w,h), 0, w/2], [0, max(w,h), h/2], [0, 0, 1]], dtype=np.float64)
```

For each image, computes a homography and extracts focal length and principal
point. Averages across all images. Sanity-checks that values are within
reasonable bounds. Falls back to `max(w,h)` as focal length if no homographies
succeed.

#### _estimate_initial_extrinsics() (line 132)

Per-image `solvePnP` to seed rotation and translation vectors:

```python
def _estimate_initial_extrinsics(all_obj, all_img, K):
    rvecs = []
    tvecs = []
    for obj, img in zip(all_obj, all_img):
        obj3 = obj.astype(np.float64)
        img2 = img.astype(np.float64).reshape(-1, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(obj3, img2, K, np.zeros(5),
                                        flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            rvecs.append(rvec.flatten())
            tvecs.append(tvec.flatten())
        else:
            rvecs.append(np.zeros(3))
            tvecs.append(np.zeros(3))
    return np.array(rvecs), np.array(tvecs)
```

#### _pack_params() / _unpack_params() (lines 149, 161)

Flatten and unflatten the parameter vector for scipy optimization:

**Packed layout:** `[fx, fy, cx, cy, k1, k2, p1, p2, k3, rvec_0, tvec_0, rvec_1, tvec_1, ...]`

- 9 intrinsic/distortion parameters
- 6 parameters per image (3 rotation + 3 translation)
- Total: `9 + 6 * n_images` parameters

#### _calibrate_scipy_residuals() (line 172)

The residual function for `least_squares`:

```python
def _calibrate_scipy_residuals(params, all_obj, all_img, n_images):
    K, dist, rvecs, tvecs = _unpack_params(params, n_images)
    residuals = []
    for i in range(n_images):
        proj, _ = cv2.projectPoints(all_obj[i], rvecs[i], tvecs[i], K, dist)
        err = proj.reshape(-1, 2) - all_img[i]
        residuals.append(err.ravel())
    return np.concatenate(residuals)
```

For each image, projects 3D board points using current parameters and computes
the reprojection error as the residual vector.

### OpenCV Fallback Calibration

#### _calibrate_camera_charuco() (line 66)

Tries OpenCV's built-in `calibrateCameraCharuco` from either `cv2` or
`cv2.aruco`. If neither exists (older builds), falls back to
`_calibrate_charuco_via_calibrate_camera()` which manually maps detected ChArUco
corners to 3D object points and calls `cv2.calibrateCamera` directly:

```python
def _calibrate_charuco_via_calibrate_camera(all_corners, all_ids, board, img_size):
    obj_by_id = {}
    for idx, oc in enumerate(
        np.asarray(board.getChessboardCorners(), dtype=np.float32).reshape(-1, 3)
    ):
        obj_by_id[idx] = oc
    obj_pts = []
    img_pts = []
    for corners, ids in zip(all_corners, all_ids):
        objs = []
        imgs = []
        for c, i in zip(corners.reshape(-1, 2), ids.reshape(-1)):
            oc = obj_by_id.get(int(i))
            if oc is None:
                return None
            objs.append(oc)
            imgs.append(c)
        obj_pts.append(np.asarray(objs, dtype=np.float32).reshape(-1, 1, 3))
        img_pts.append(np.asarray(imgs, dtype=np.float32).reshape(-1, 1, 2))
    rms, cam_mat, dist, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts, img_size, None, None, flags=cv2.CALIB_FIX_K3
    )
    return rms, cam_mat, dist, None, None
```

### Intrinsics Scaling

#### intrinsics_for_frame() (line 490)

Scales calibration intrinsics to match a different frame resolution:

```python
def intrinsics_for_frame(calib: dict, frame_w: int, frame_h: int):
    m = np.array(calib["camera_matrix"], dtype=np.float64).reshape(3, 3)
    res = calib.get("resolution")
    if res:
        sx = frame_w / float(res[0])
        sy = frame_h / float(res[1])
    else:
        sx = sy = 1.0
    fx = float(m[0, 0]) * sx
    fy = float(m[1, 1]) * sy
    cx = float(m[0, 2]) * sx
    cy = float(m[1, 2]) * sy
    cam_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return cam_mat, np.asarray(d, dtype=np.float64).ravel()
```

If the calibration was performed at a different resolution than the current
frame, all intrinsic parameters are scaled proportionally.

### Visualization Helpers

#### draw_corners() / draw_corners_into() (lines 265, 255)

Draws filled circles at each detected ChArUco corner:

```python
def draw_corners_into(out, corners, color):
    for corner in corners:
        pts = np.asarray(corner, dtype=np.float64).reshape(-1, 2)
        for x, y in pts:
            cv2.circle(out, (int(round(float(x))), int(round(float(y)))), 5, color, -1)
    return out
```

#### draw_markers() / draw_markers_into() (lines 279, 270)

Draws unfilled polylines around each detected ArUco marker:

```python
def draw_markers_into(out, corners, color):
    for corner in corners:
        pts = np.asarray(corner, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(out, [pts], True, color, 2)
    return out
```

#### draw_charuco() / draw_charuco_into() (lines 298, 284)

High-level overlay that draws both markers and corners. If `color=None`, uses
OpenCV's built-in `_draw_detected_markers` and `_draw_detected_corners_charuco`.
If a custom color is provided, uses the `draw_*_into` helpers instead.

#### random_overlay_color() (line 247)

Generates a random bright color in BGR for visualization:
```python
def random_overlay_color():
    hue = int(np.random.randint(0, 180))
    sat = int(np.random.randint(180, 256))
    val = int(np.random.randint(180, 256))
    bgr = cv2.cvtColor(np.array([[[hue, sat, val]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
    return tuple(int(x) for x in bgr[0, 0])
```

### Utility Functions

#### expected_charuco_corners() (line 303)

Returns the expected number of ChArUco corners for a given board pattern:
`(cols - 1) * (rows - 1)`

#### capture_coverage() (line 311)

Returns what fraction of expected corners were detected: `min(1.0, n / expected)`

#### frame_diverse() (line 318)

Checks whether a new capture is sufficiently different from existing captures
to be worth keeping. Computes mean minimum distance between new and old corner
positions — if it's less than `min_shift_px` (default 12.0) for any stored
capture, returns `False`.

#### derive_fov_from_intrinsics() (line 336)

Extracts FOV and focal length from a calibration result dict using the `fov_from_focal()` function.

#### generate_calibration_board_pdf() (line 516)

Generates a printable PDF of the default ChArUco board at 300 DPI with a
0.5-inch margin. Uses PIL for PDF creation.

---

## Triangulation (`iSpy/vision/triangulation.py`, 133 lines)

Implements 3D coordinate recovery from 2D pixel detections using ray casting,
ground-plane intersection, and stereo triangulation.

### Ray Dataclass (line 7)

```python
@dataclass
class Ray:
    origin: np.ndarray     # (3,) robot-frame, inches
    direction: np.ndarray  # (3,) unit vector, robot-frame
```

Represents a 3D ray in robot frame coordinates. Origin is in inches.

### pixel_to_ray() (line 13)

Converts a 2D pixel coordinate to a 3D ray in robot frame:

```python
def pixel_to_ray(pixel_x, pixel_y, img_w, img_h, focal_length_px,
                 camera_x, camera_y, camera_z, yaw_deg, pitch_deg) -> Ray:
```

**Algorithm:**

1. **Normalized camera coordinates (lines 25–32):**
   ```
   cx, cy = img_w / 2.0, img_h / 2.0
   dx = (pixel_x - cx) / f
   dy = (pixel_y - cy) / f
   dz = 1.0
   normalize(dx, dy, dz)
   ```

2. **Apply pitch rotation (lines 34–38):**
   ```python
   pitch = math.radians(pitch_deg)
   cp, sp = math.cos(pitch), math.sin(pitch)
   dy2 = dy * cp + dz * sp
   dz2 = -dy * sp + dz * cp
   dx2 = dx
   ```

3. **Apply yaw rotation (lines 40–43):**
   ```python
   yaw = math.radians(yaw_deg)
   cy_, sy_ = math.cos(yaw), math.sin(yaw)
   x_rot = dx2 * cy_ + dz2 * sy_
   y_rot = dz2 * cy_ - dx2 * sy_
   ```

4. **Flip Y axis (line 44):**
   ```python
   z_rot = -dy2  # camera Y-down → world Y-up
   ```

5. **Normalize and create Ray (lines 46–50):**
   ```python
   direction = np.array([x_rot, y_rot, z_rot], dtype=np.float64)
   direction /= (np.linalg.norm(direction) or 1.0)
   origin = np.array([camera_x, camera_y, camera_z], dtype=np.float64)
   return Ray(origin=origin, direction=direction)
   ```

### camera_point_to_robot() (line 53)

Transforms a point from camera frame to robot frame:

```python
def camera_point_to_robot(camera_point, camera_x, camera_y, camera_z, yaw_deg, pitch_deg):
    fx, fy, fz = camera_point
    pitch = math.radians(pitch_deg)
    cp, sp = math.cos(pitch), math.sin(pitch)
    down = fy * cp + fz * sp
    forward = -fy * sp + fz * cp

    yaw = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    x_rot = fx * cos_y + forward * sin_y
    y_rot = forward * cos_y - fx * sin_y
    z_rot = -down  # down-positive → up-positive

    return np.array([x_rot + camera_x, y_rot + camera_y, z_rot + camera_z])
```

**Steps:**
1. Apply pitch rotation (camera tilt).
2. Apply yaw rotation (camera horizontal angle).
3. Flip Z axis (camera down-positive to world up-positive).
4. Translate by camera mount position.

### camera_rotation_to_robot() (line 78)

Composes a camera-space rotation matrix with the camera-to-robot rotation:

```python
def camera_rotation_to_robot(rotation_matrix, yaw_deg, pitch_deg):
    cam_to_robot = np.array([
        [cy, -sp*sy, cp*sy],
        [-sy, -sp*cy, cp*cy],
        [0.0, -cp,    -sp ],
    ])
    return cam_to_robot @ rotation_matrix
```

### ground_plane_intersection() (line 97)

Finds where a ray intersects the ground plane (z = `ground_z`):

```python
def ground_plane_intersection(ray, ground_z=0.0):
    dz = ray.direction[2]
    if abs(dz) < 1e-9:
        return None  # ray parallel to ground
    t = (ground_z - ray.origin[2]) / dz
    if t <= 0:
        return None  # intersection behind camera
    return ray.origin + t * ray.direction
```

**Math:** `P = origin + t * direction`, solve for `t` where `P.z = ground_z`.

### closest_point_between_rays() (line 107)

Stereo triangulation — finds the closest approach between two rays:

```python
def closest_point_between_rays(ray_a, ray_b, max_residual=0.5):
    o1, d1 = ray_a.origin, ray_a.direction
    o2, d2 = ray_b.origin, ray_b.direction

    d1d2 = float(np.dot(d1, d2))
    denom = 1.0 - d1d2 * d1d2
    if abs(denom) < 1e-9:
        return None  # rays nearly parallel

    w0 = o1 - o2
    a = float(np.dot(d1, w0))
    b = float(np.dot(d2, w0))
    t1 = (d1d2 * b - a) / denom
    t2 = (b - d1d2 * a) / denom

    if t1 <= 0 or t2 <= 0:
        return None  # intersection behind camera

    p1 = o1 + t1 * d1
    p2 = o2 + t2 * d2
    residual = float(np.linalg.norm(p1 - p2))
    if residual > max_residual:
        return None

    return (p1 + p2) / 2.0, residual
```

**Math:** Standard closest-point-between-two-lines formula. The two rays
typically don't intersect exactly, so we find the points on each ray that are
closest to each other and return their midpoint.

**Returns:** `(midpoint, residual)` or `None` if rays are parallel, intersect
behind cameras, or residual exceeds threshold.

---

## Model Conversion Pipeline (`iSpy/vision/optimizer.py`, 1422 lines)

The optimizer handles converting `.pt` (PyTorch) models to hardware-specific
inference formats.

### Conversion Flow

```
.pt (PyTorch)
  ├── RKNN detected → ONNX → rknn-toolkit2 → YoloModels/rknn/
  ├── NVIDIA GPU    → ONNX → trtexec      → YoloModels/engine/
  ├── Intel GPU     → ONNX → mo (Model Optimizer) → YoloModels/openvino/
  ├── ARM CPU       → ONNX → TFLite       → YoloModels/tflite/
  ├── Apple Silicon → ONNX → coremltools   → YoloModels/coreml/
  └── Hailo NPU     → ONNX → hailo compiler → YoloModels/hailo/
```

### Supported Conversion Targets

| Target | Tool | Notes |
|--------|------|-------|
| ONNX | ultralytics export | Universal intermediate format |
| RKNN | rknn-toolkit2 | Requires Rockchip NPU toolkit |
| TensorRT | trtexec | Requires CUDA + TensorRT |
| OpenVINO | mo (Model Optimizer) | Intel GPU/NPU/CPU |
| TFLite | tensorflow.lite | ARM CPU, Edge TPU |
| CoreML | coremltools | Apple Silicon |
| Hailo | hailo compiler | Hailo-8 NPU |

### Hardware Detection (line 186)

```python
@lru_cache()
def _detect_rknn_target_platform() -> str:
    for path in ("/proc/device-tree/compatible", "/proc/device-tree/model",
                 "/sys/firmware/devicetree/base/model"):
        try:
            content = open(path, "rb").read().decode(errors="ignore").lower()
            for chip in _RKNN_KNOWN_CHIPS:
                if chip in content:
                    return chip
        except Exception:
            continue
```

Reads device tree files to detect Rockchip SoC型号 (rk3588, rk3576, etc.).

### Output Management

- `_progress_spinner()` — context manager showing a spinning progress indicator.
- `_silence_third_party()` — context manager that redirects stdout/stderr to
  `/dev/null` and silences ultralytics/nncf loggers during conversion.

---

## Quantization (`iSpy/vision/QuantizedModel.py`, 275 lines)

### ensure_quantized_model() (line 11)

Main entry point for quantization:

```python
def ensure_quantized_model(source_pt, target_format="auto", input_size=(640,640),
                           quantize=True, force=False, dataset_path=None):
```

**Algorithm:**
1. If `quantize=False` or source doesn't exist, return the source path.
2. If target is `"auto"`, call `recommend_format()` to detect hardware.
3. If target is `"tpu"`, return source directly (TFLite uses .pt via torch_xla).
4. Call `optimizer.convert_model()` with `quantize=True`.
5. On failure, fall back to the original `.pt`.

### _CalibrationDataReader (line 76)

Provides calibration data for ONNX int8 static quantization:

```python
class _CalibrationDataReader:
    def __init__(self, image_paths, input_size, mean=None, std=None):
        self._images = list(image_paths)
        self._size = int(input_size)
        self._mean = mean or _IMAGENET_MEAN
        self._std = std or _IMAGENET_STD
        self._index = 0

    def _preprocess(self, path):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._size, self._size), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        return ((img.transpose(2, 0, 1) - self._mean) / self._std).astype(np.float32)

    def get_next(self):
        batch = []
        while self._index < len(self._images) and len(batch) < 1:
            sample = self._preprocess(self._images[self._index])
            self._index += 1
            if sample is not None:
                batch.append(sample)
        if not batch:
            return None
        return {"pixel_values": np.concatenate(batch, axis=0)[None]}
```

Uses ImageNet mean/std normalization by default. Each call to `get_next()`
returns one preprocessed image.

### _quantize_static_onnx() (line 140)

Performs ONNX int8 static quantization using onnxruntime:

```python
def _quantize_static_onnx(fp32_path, int8_path, dataset_path, input_size):
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    images = _calibration_image_paths(dataset_path)
    reader = _CalibrationDataReader(images, input_size)
    quantize_static(
        str(fp32_path), str(int8_path), reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
    )
```

### ensure_onnx_model() (line 176)

Converts a PyTorch model to ONNX format with optional quantization:

1. Checks for existing artifacts (fp32 and int8).
2. If `force=True` or no artifact exists, exports via ultralytics.
3. If `quantize=True`, runs `_quantize_static_onnx()`.

---

## Model Inspector (`iSpy/vision/ModelInspector.py`, 946 lines)

Introspects model files to auto-detect format, layout, and configuration.

### inspect_model() (line 10)

Routes to format-specific inspector:

```python
def inspect_model(model_path: str, task: str = "detect") -> dict:
    ext = Path(model_path).suffix.lower()
    if ext == ".onnx":
        return _inspect_onnx(model_path, task)
    elif ext == ".rknn":
        return _inspect_rknn(model_path, task)
    elif ext == ".tflite":
        return _inspect_tflite(model_path, task)
    elif ext in (".pt", ".engine") or "openvino_model" in suffix:
        return _inspect_ultralytics(model_path, task)
    else:
        raise ValueError(f"Unsupported model extension: {ext}")
```

### _inspect_onnx() (line 52)

Uses onnxruntime to inspect ONNX models:

1. Creates an `InferenceSession` and reads input/output metadata.
2. Reads optional `_metadata.yaml` sidecar for ground-truth values.
3. Parses input shape to determine layout (NCHW/NHWC), height, width, channels.
4. Maps input type to dtype (`tensor(float)` → `float32`, `tensor(uint8)` → `uint8`).
5. Parses output shape to determine format (raw, hardware_nms, etc.).
6. Detects number of classes from output width.
7. Returns a complete config dict with `certain_fields`, `detected_fields`,
   `manual_fields`, and `warnings` for UI display.

### _inspect_rknn() (line ~400)

Uses `rknn.query()` to inspect RKNN models:
- Input/output format
- Quantized flag
- Input shapes

### _inspect_tflite() (line ~500)

Uses `tf.lite.Interpreter` to inspect TFLite models:
- Input/output tensor shapes
- Quantization parameters

### _inspect_ultralytics() (line ~600)

Loads via ultralytics YOLO to inspect `.pt`, `.engine`, and OpenVINO models:
- Class names
- Input size
- Task type

---

## DBSCAN Clustering (`iSpy/algorithms/CustomDBScan.py`, 15 lines)

Thin wrapper around sklearn's DBSCAN:

```python
class CustomDBScan:
    def __init__(self, points: list, eps: int, samples: int):
        self.points = points
        self.eps = eps
        self.samples = samples
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.samples)

    def get_dbscan(self):
        if self.eps == 0:
            return [0] * len(self.points)  # no clustering
        clusters = self.dbscan.fit_predict(self.points)
        return clusters
```

When `eps=0`, clustering is disabled and all points get label 0 (not noise
label -1). Used by the path_planner tracker to group detections into piles.

---

## Object Detection Post-Processing

### Non-Max Suppression

Ultralytics handles NMS internally for `.pt` models. For raw backends (RKNN,
ONNX direct), the `genericYolo.py` backend applies NMS manually:

1. Filter by confidence threshold (`min_conf`).
2. Compute IoU between all bounding box pairs.
3. Suppress overlapping boxes (IoU > threshold).
4. Keep top-K detections per class.

### Ground-Plane Coordinate Conversion

In `object_detection.py`, each detection is converted to robot-frame 3D
coordinates:

```python
for det in detections:
    # 1. Pixel center of bounding box
    cx = (det.bbox[0] + det.bbox[2]) / 2
    cy = (det.bbox[1] + det.bbox[3]) / 2

    # 2. Convert to ray via pixel_to_ray()
    origin, direction = pixel_to_ray(cx, cy, camera_matrix)

    # 3. Intersect ground plane
    field_point = ground_plane_intersection(origin, direction, camera_height)

    # 4. Transform to robot frame
    robot_point = camera_to_robot(field_point, (cam_x, cam_y), cam_yaw)

    det.x, det.y, det.z = robot_point
```
