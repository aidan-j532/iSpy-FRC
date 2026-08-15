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

DEFAULT_CHESSBOARD_PATTERN = (9, 6)

DEFAULT_CHARUCO_PATTERN = (7, 5)  # squares wide x squares tall
DEFAULT_CHARUCO_DICT = cv2.aruco.DICT_6X6_250
# square/marker lengths only need to be consistent (intrinsics are scale
# invariant) - marker must be smaller than the square it sits in
DEFAULT_CHARUCO_SQUARE = 1.0
DEFAULT_CHARUCO_MARKER = 0.8

DEFAULT_APRIL_DICT = cv2.aruco.DICT_APRILTAG_36h11
DEFAULT_APRIL_SIZE = 6.5  # inches

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


def draw_chessboard(frame, corners, cols: int, rows: int):
    """copy of the frame with detected chessboard corners overlaid (for preview)."""
    out = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if corners is not None:
        cv2.drawChessboardCorners(out, (cols, rows), corners, True)
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
    found = corners is not None and ids is not None and len(corners) >= 4
    return found, corners, ids, marker_corners, marker_ids, gray


def draw_charuco(frame, corners, ids, marker_corners, marker_ids):
    """copy of the frame with detected ChArUco markers + board corners
    overlaid (for preview)."""
    out = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if marker_corners is not None and marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(out, marker_corners, marker_ids)
    if corners is not None and ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(out, corners, ids)
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
        rms, cam_mat, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
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


def _april_detector(dictionary_id: int = DEFAULT_APRIL_DICT):
    params = cv2.aruco.DetectorParameters()
    params.useAruco3Detection = True
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(dictionary_id), params
    )


def detect_april(frame, dictionary_id: int = DEFAULT_APRIL_DICT):
    """find AprilTags in the frame; returns (found, corners, ids,
    marker_corners, marker_ids, gray). corners is the list of 4-corner tag
    detections (same layout as the charuco marker arrays) so the draw +
    calibrate helpers share one shape."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    try:
        corners, ids, _ = _april_detector(dictionary_id).detectMarkers(gray)
    except cv2.error:
        return False, None, None, None, None, gray
    found = corners is not None and ids is not None and len(ids) > 0
    if not found:
        return False, None, None, None, None, gray
    return True, corners, ids, corners, ids, gray


def draw_april(frame, corners, ids, marker_corners=None, marker_ids=None):
    """copy of the frame with detected AprilTags boxed + id-labelled."""
    out = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    tags = marker_corners if marker_corners is not None else corners
    tag_ids = marker_ids if marker_ids is not None else ids
    if tags is not None and tag_ids is not None:
        for corner, tag_id in zip(tags, tag_ids):
            pts = np.asarray(corner, dtype=np.int32).reshape(-1, 2)
            cv2.polylines(out, [pts], True, (0, 255, 0), 2)
            cv2.putText(out, f"ID: {int(tag_id[0])}", tuple(pts[0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def _april_object_points(tag_size_inches: float) -> np.ndarray:
    """object points for the 4 corners of a square tag centered at the origin."""
    half = tag_size_inches / 2.0
    return np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)


def calibrate_april(
    captures,
    tag_size_inches: float = DEFAULT_APRIL_SIZE,
    dictionary_id: int = DEFAULT_APRIL_DICT,
):
    """run calibrateCamera over [(gray_frame, tag_corners, tag_ids), ...]
    using each tag as a known-size square; returns an intrinsics dict or None
    when it cannot calibrate. Vary the tag's pose between captures - a static
    tag solves to garbage."""
    if len(captures) < 3 or tag_size_inches <= 0:
        return None
    obj_pts = []
    img_pts = []
    centers = []
    img_size = None
    for gray, corners, ids in captures:
        if corners is None or len(corners) == 0:
            continue
        h, w = gray.shape[:2]
        img_size = (w, h)
        for corner in corners:
            c = np.asarray(corner, dtype=np.float32).reshape(-1, 2)
            obj_pts.append(_april_object_points(tag_size_inches).reshape(-1, 1, 3))
            img_pts.append(c.reshape(-1, 1, 2))
            centers.append(c.mean(axis=0))
    if len(obj_pts) < 3:
        return None
    # degenerate captures (the tag at the same spot every frame) solve to
    # garbage - refuse it so the UI asks for variety
    spread = np.asarray(centers).max(axis=0) - np.asarray(centers).min(axis=0)
    if float(np.linalg.norm(spread)) < 1.0:
        logger.warning("AprilTag calibration frames were all the same tag pose - move the tag between captures")
        return None
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
        logger.warning("AprilTag calibration failed: %s", exc)
        return None
    if not np.isfinite(rms) or rms > 100.0:
        logger.warning(
            "AprilTag calibration gave an implausible RMS %.3f - frames were probably too similar",
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
