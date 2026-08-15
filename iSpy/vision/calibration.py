"""Camera calibration helpers.

Three complementary workflows, all fed from the web UI:

1. Known-object calibration - measure an object of known real size at a
   known distance and derive the camera's focal length (px) + horizontal
   FOV. This feeds every pipeline's depth estimation (object_detection etc).

2. Chessboard intrinsics - run OpenCV's calibrateCamera over a handful of
   chessboard frames to get a real camera matrix + distortion coefficients.
   AprilTag / QR PnP use these for accurate pose.

3. ChArUco intrinsics - same idea as the chessboard but the board is an
   ArUco marker grid (fully visible at any angle, no untextured corner
   ambiguity). AprilTag / QR PnP use these for accurate pose.

Results are stored on the camera's ``calibration`` config dict. The
known-object values follow iSpy's internal convention: calibration inputs
are inches (the final unit conversion to meters/feet happens in each
pipeline). Intrinsics are stored in pixels at their capture resolution and
scaled to the live frame size at runtime.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# OpenCV 4.9+ moved several aruco helpers out of the cv2.aruco namespace into
# the main cv2 namespace (the old aliases were deleted there). Resolve each
# from whichever namespace the installed build exposes so the same code runs
# on both old (<=4.8) and new (>=4.9) wheels.
def _pick_aruco(*names: tuple) -> callable:
    for obj, name in names:
        fn = getattr(obj, name, None)
        if fn is not None:
            return fn
    raise AttributeError(f"no cv2 binding found for any of {[n for _, n in names]}")


_draw_detected_markers = _pick_aruco(
    (cv2, "drawDetectedMarkers"), (cv2.aruco, "drawDetectedMarkers")
)
_draw_detected_corners_charuco = _pick_aruco(
    (cv2, "drawDetectedCornersCharuco"), (cv2.aruco, "drawDetectedCornersCharuco")
)


def _calibrate_charuco_via_calibrate_camera(
    all_corners: list, all_ids: list, board, img_size
):
    """manual ChArUco intrinsics solve for builds that dropped
    calibrateCameraCharuco (some aruco wheels ship the detector but not the
    calibrator): build 3D object points from the board's chessboard corners
    and run plain calibrateCamera. Returns (rms, cam_mat, dist) or None."""
    try:
        # detectBoard ids index straight into getChessboardCorners() (row-major
        # grid layout), so map each corner index to its 3D object point.
        obj_by_id = {}
        for idx, oc in enumerate(
            np.asarray(board.getChessboardCorners(), dtype=np.float32).reshape(-1, 3)
        ):
            obj_by_id[idx] = oc
    except (AttributeError, TypeError, ValueError):
        return None
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
    try:
        rms, cam_mat, dist, _, _ = cv2.calibrateCamera(
            obj_pts, img_pts, img_size, None, None, flags=cv2.CALIB_FIX_K3
        )
    except cv2.error as exc:
        logger.exception("cv2.calibrateCamera failed: %s", exc)
        return None
    # match calibrateCameraCharuco's 5-tuple shape
    return rms, cam_mat, dist, None, None


def _calibrate_camera_charuco(all_corners, all_ids, board, img_size, *extra):
    """run calibrateCameraCharuco when the build has it, else fall back to a
    manual object-point solve through plain calibrateCamera."""
    for obj, name in (
        (cv2, "calibrateCameraCharuco"),
        (cv2.aruco, "calibrateCameraCharuco"),
    ):
        fn = getattr(obj, name, None)
        if fn is not None:
            return fn(all_corners, all_ids, board, img_size, *extra)
    result = _calibrate_charuco_via_calibrate_camera(all_corners, all_ids, board, img_size)
    if result is None:
        raise cv2.error("ChArUco calibration unavailable in this OpenCV build")
    return result

DEFAULT_CHESSBOARD_PATTERN = (9, 6)

DEFAULT_CHARUCO_PATTERN = (7, 5)  # squares wide x squares tall
DEFAULT_CHARUCO_DICT = cv2.aruco.DICT_6X6_250
# square/marker lengths only need to be consistent (intrinsics are scale
# invariant) - marker must be smaller than the square it sits in
DEFAULT_CHARUCO_SQUARE = 1.0
DEFAULT_CHARUCO_MARKER = 0.8

# boards people actually print - the wizard auto-detects across these so
# any common layout/dictionary shows up without matching the default by hand
CHARUCO_PATTERNS = ((7, 5), (5, 7), (8, 6), (6, 4), (5, 4), (7, 6), (6, 5), (10, 8))
CHARUCO_DICTIONARY_IDS = (
    cv2.aruco.DICT_6X6_250,
    cv2.aruco.DICT_5X5_250,
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_6X6_50,
    cv2.aruco.DICT_5X5_50,
    cv2.aruco.DICT_4X4_250,
)


def focal_from_object(real_size: float, distance: float, pixel_height: float) -> float:
    """focal length in px from an object of known real size (in) seen at a
    known distance (in) with a measured pixel height."""
    if real_size <= 0 or distance <= 0 or pixel_height <= 0:
        return 0.0
    return pixel_height * distance / real_size


def fov_from_focal(focal_px: float, width_px: float) -> float:
    """horizontal FOV in degrees from a focal length in px at a given frame width."""
    if focal_px <= 0 or width_px <= 0:
        return 0.0
    return float(2.0 * np.degrees(np.arctan2(width_px / 2.0, focal_px)))


def focal_from_fov(fov_deg: float, width_px: float) -> float:
    """focal length in px from a horizontal FOV in degrees at a given frame width."""
    if fov_deg <= 0 or width_px <= 0:
        return 0.0
    return (width_px / 2.0) / np.tan(np.radians(fov_deg / 2.0))


def detect_chessboard(frame, cols: int, rows: int):
    """find a chessboard's inner corners; returns (found, corners, gray).
    corners is refined with cornerSubPix when found, else None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
    if found and corners is not None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return bool(found), (corners if found else None), gray


def random_overlay_color():
    """bright random BGR color so a captured detection stands out on any scene."""
    hue = int(np.random.randint(0, 180))
    sat = int(np.random.randint(180, 256))
    val = int(np.random.randint(180, 256))
    bgr = cv2.cvtColor(np.array([[[hue, sat, val]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
    return tuple(int(x) for x in bgr[0, 0])


def draw_corners(frame, corners, color):
    """copy of the frame with a filled circle at every corner point in the
    given color - used to overlay a captured detection on the live feed."""
    out = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if corners is None:
        return out
    for corner in corners:
        pts = np.asarray(corner, dtype=np.float64).reshape(-1, 2)
        for x, y in pts:
            cv2.circle(out, (int(round(float(x))), int(round(float(y)))), 5, color, -1)
    return out


def draw_markers(frame, corners, color):
    """copy of the frame with each marker's outline drawn in the given color."""
    out = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if corners is None:
        return out
    for corner in corners:
        pts = np.asarray(corner, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(out, [pts], True, color, 2)
    return out


def draw_chessboard(frame, corners, cols: int, rows: int, color=None):
    """copy of the frame with detected chessboard corners overlaid (for preview).
    Pass a color to draw the corners in that color instead of OpenCV's defaults."""
    out = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if corners is not None:
        if color is None:
            cv2.drawChessboardCorners(out, (cols, rows), corners, True)
        else:
            out = draw_corners(out, corners, color)
    return out


def _object_points(cols: int, rows: int) -> np.ndarray:
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return pts


def calibrate_chessboard(captures, pattern=DEFAULT_CHESSBOARD_PATTERN):
    """run calibrateCamera over [(gray_frame, corners), ...]; returns a dict
    with rms + intrinsics, or None when it cannot calibrate. pattern is the
    (cols, rows) inner-corner size the frames were detected with."""
    if len(captures) < 3:
        return None
    cols, rows = pattern
    obj_pts = []
    img_pts = []
    img_size = None
    for gray, corners in captures:
        h, w = gray.shape[:2]
        img_size = (w, h)
        obj_pts.append(_object_points(cols, rows))
        img_pts.append(corners.reshape(-1, 1, 2).astype(np.float32))
    try:
        rms, cam_mat, dist, _, _ = cv2.calibrateCamera(
            obj_pts,
            img_pts,
            img_size,
            None,
            None,
            flags=cv2.CALIB_FIX_K3,
        )
    except cv2.error as exc:
        logger.warning("Chessboard calibration failed: %s", exc)
        return None
    # degenerate captures (e.g. every frame the same board pose) produce a
    # garbage-but-successful solve - refuse it so the UI asks for variety
    if not np.isfinite(rms) or rms > 100.0:
        logger.warning(
            "Chessboard calibration gave an implausible RMS %.3f - frames were probably too similar",
            rms,
        )
        return None
    return {
        "rms": float(rms),
        "camera_matrix": cam_mat.tolist(),
        "dist_coeffs": dist.flatten().tolist(),
        "resolution": list(img_size),
        "count": len(captures),
    }


def _charuco_captures_degenerate(all_corners: list, all_ids: list) -> bool:
    """True when every capture shows the board at (essentially) the same pose -
    no corner moved more than 1px between the first capture and the rest."""
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


def make_charuco_board(
    cols: int,
    rows: int,
    square_length: float = DEFAULT_CHARUCO_SQUARE,
    marker_length: float = DEFAULT_CHARUCO_MARKER,
    dictionary_id: int = DEFAULT_CHARUCO_DICT,
):
    """build a ChArUco board for (cols x rows) squares. square_length and
    marker_length share a unit - the actual value is irrelevant to the
    intrinsic solve, only the ratio matters."""
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.CharucoBoard((cols, rows), square_length, marker_length, dictionary)


def detect_charuco(
    frame,
    cols: int,
    rows: int,
    square_length: float = DEFAULT_CHARUCO_SQUARE,
    marker_length: float = DEFAULT_CHARUCO_MARKER,
    dictionary_id: int = DEFAULT_CHARUCO_DICT,
):
    """find a ChArUco board's corners; returns (found, corners, ids,
    marker_corners, marker_ids, gray). corners/ids feed calibrate_charuco,
    the marker arrays are only used for the preview overlay."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    board = make_charuco_board(cols, rows, square_length, marker_length, dictionary_id)
    detector = cv2.aruco.CharucoDetector(board)
    try:
        corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
    except cv2.error:
        return False, None, None, None, None, gray
    if corners is None or ids is None:
        return False, None, None, None, None, gray
    # cv2 >= 5 returns flat (N, 2) / (N,) arrays - normalize to the classic
    # (N, 1, 2) / (N, 1) shapes the draw + calibrate helpers expect.
    corners = _to_corners(corners)
    ids = _to_ids(ids)
    marker_corners = _to_marker_corners(marker_corners)
    marker_ids = _to_ids(marker_ids)
    found = len(corners) >= 4
    return found, corners, ids, marker_corners, marker_ids, gray


def draw_charuco(frame, corners, ids, marker_corners, marker_ids, color=None):
    """copy of the frame with detected ChArUco markers + board corners
    overlaid (for preview). Pass a color to draw the detection in that color
    instead of OpenCV's defaults (used for captured-frame overlays)."""
    out = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if color is None:
        if marker_corners is not None and marker_ids is not None:
            _draw_detected_markers(out, marker_corners, marker_ids)
        if corners is not None and ids is not None:
            _draw_detected_corners_charuco(out, corners, ids)
    else:
        if marker_corners is not None:
            out = draw_markers(out, marker_corners, color)
        if corners is not None:
            out = draw_corners(out, corners, color)
    return out


def calibrate_charuco(
    captures,
    pattern=DEFAULT_CHARUCO_PATTERN,
    square_length: float = DEFAULT_CHARUCO_SQUARE,
    marker_length: float = DEFAULT_CHARUCO_MARKER,
    dictionary_id: int = DEFAULT_CHARUCO_DICT,
):
    """run calibrateCameraCharuco over [(gray_frame, corners, ids), ...];
    returns a dict with rms + intrinsics, or None when it cannot calibrate.
    pattern is the (cols, rows) square size the frames were detected with."""
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
    # degenerate captures (e.g. every frame the same board pose) solve fine
    # to a garbage-but-plausible intrinsics set - compare corner locations by
    # id and refuse when nothing moved
    if _charuco_captures_degenerate(all_corners, all_ids):
        logger.warning(
            "ChArUco calibration frames were all the same board pose - move the board between captures"
        )
        return None
    try:
        rms, cam_mat, dist, _, _ = _calibrate_camera_charuco(
            all_corners,
            all_ids,
            board,
            img_size,
            None,
            None,
        )
    except cv2.error as exc:
        logger.warning("ChArUco calibration failed: %s", exc)
        return None
    if not np.isfinite(rms) or rms > 100.0:
        logger.warning(
            "ChArUco calibration gave an implausible RMS %.3f - frames were probably too similar",
            rms,
        )
        return None
    return {
        "rms": float(rms),
        "camera_matrix": cam_mat.tolist(),
        "dist_coeffs": dist.flatten().tolist(),
        "resolution": list(img_size),
        "count": len(captures),
    }


def detect_charuco_auto(
    frame,
    preferred_pattern: tuple | list | None = None,
    preferred_dict: int | None = None,
):
    """scan the common ChArUco layouts/dictionaries for a board in the frame,
    so a printed board of any typical size is found without matching the
    default by hand. Returns (found, corners, ids, marker_corners, marker_ids,
    gray, pattern, dictionary_id); pattern/dictionary_id are None when not
    found. The preferred layout/dict are tried first."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    patterns: list[tuple] = []
    if preferred_pattern is not None:
        try:
            pp = tuple(int(x) for x in preferred_pattern[:2])
            if len(pp) == 2 and pp not in patterns:
                patterns.append(pp)
        except (TypeError, ValueError):
            pass
    for p in CHARUCO_PATTERNS:
        if p not in patterns:
            patterns.append(p)
    dicts: list[int] = []
    if preferred_dict is not None:
        try:
            pd = int(preferred_dict)
            if pd not in dicts:
                dicts.append(pd)
        except (TypeError, ValueError):
            pass
    for d in CHARUCO_DICTIONARY_IDS:
        if d not in dicts:
            dicts.append(d)
    for p in patterns:
        for d in dicts:
            try:
                found, corners, ids, mc, mi, _ = detect_charuco(
                    frame, p[0], p[1], dictionary_id=d
                )
            except Exception:
                continue
            if found:
                return True, corners, ids, mc, mi, gray, p, d
    return False, None, None, None, None, gray, None, None


def _to_corners(corners):
    """normalize a corners array to (N, 1, 2) regardless of the cv2 build."""
    a = np.asarray(corners, dtype=np.float32)
    if a.ndim == 2:
        return a.reshape(-1, 1, 2)
    return a


def _to_ids(ids):
    """normalize an ids array to (N, 1) regardless of the cv2 build."""
    a = np.asarray(ids, dtype=np.int32)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def _to_marker_corners(marker_corners):
    """normalize a marker corners list to [4x1x2, ...] regardless of cv2 build."""
    out = []
    for marker in marker_corners:
        a = np.asarray(marker, dtype=np.float32).reshape(-1, 2)
        out.append(a.reshape(4, 1, 2))
    return out


def intrinsics_for_frame(calib: dict, frame_w: int, frame_h: int):
    """(camera_matrix, dist_coeffs) scaled from the calibration's capture
    resolution to the given frame size, or None if not calibrated."""
    if not isinstance(calib, dict):
        return None
    m = calib.get("camera_matrix")
    d = calib.get("dist_coeffs")
    res = calib.get("resolution")
    if not m or not d:
        return None
    try:
        m = np.array(m, dtype=np.float64).reshape(3, 3)
        if res:
            sx = frame_w / float(res[0])
            sy = frame_h / float(res[1])
        else:
            # no stored resolution - assume intrinsics were captured at this frame size
            sx = sy = 1.0
        fx = float(m[0, 0]) * sx
        fy = float(m[1, 1]) * sy
        cx = float(m[0, 2]) * sx
        cy = float(m[1, 2]) * sy
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None
    cam_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return cam_mat, np.asarray(d, dtype=np.float64).ravel()
