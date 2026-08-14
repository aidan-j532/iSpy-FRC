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
