import logging

import cv2
import numpy as np
import scipy.optimize

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


def _build_obj_img_pairs(all_corners, all_ids, board):
    """Map detected charuco corners to 3D board coordinates for each image."""
    try:
        board_corners = np.asarray(
            board.getChessboardCorners(), dtype=np.float64
        ).reshape(-1, 3)
    except (AttributeError, TypeError, ValueError):
        return None
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
            return None
        all_obj.append(np.asarray(obj, dtype=np.float64))
        all_img.append(np.asarray(img, dtype=np.float64))
    return all_obj, all_img


def _estimate_initial_intrinsics(all_obj, all_img, img_size):
    """Rough intrinsics from per-image homographies, averaged."""
    w, h = img_size
    Ks = []
    for obj, img in zip(all_obj, all_img):
        try:
            H, _ = cv2.findHomography(obj[:, :2], img)
            if H is None:
                continue
            fx = abs(H[0, 0])
            fy = abs(H[1, 1])
            cx = abs(H[0, 2])
            cy = abs(H[1, 2])
            if 0 < fx < w * 4 and 0 < fy < h * 4 and 0 < cx < w * 2 and 0 < cy < h * 2:
                Ks.append((fx, fy, cx, cy))
        except cv2.error:
            continue
    if Ks:
        arr = np.array(Ks)
        fx, fy, cx, cy = arr.mean(axis=0)
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return np.array(
        [[max(w, h), 0, w / 2.0], [0, max(w, h), h / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )


def _estimate_initial_extrinsics(all_obj, all_img, K):
    """Per-image solvePnP to seed rotation/translation vectors."""
    rvecs = []
    tvecs = []
    for obj, img in zip(all_obj, all_img):
        obj3 = obj.astype(np.float64)
        img2 = img.astype(np.float64).reshape(-1, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(obj3, img2, K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            rvecs.append(rvec.flatten())
            tvecs.append(tvec.flatten())
        else:
            rvecs.append(np.zeros(3))
            tvecs.append(np.zeros(3))
    return np.array(rvecs), np.array(tvecs)


def _pack_params(K, dist, rvecs, tvecs):
    """Flatten intrinsics + distortion + all extrinsics into a 1-D vector."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2 = dist[0], dist[1]
    p1, p2 = dist[2], dist[3]
    k3 = dist[4] if len(dist) > 4 else 0.0
    header = np.array([fx, fy, cx, cy, k1, k2, p1, p2, k3], dtype=np.float64)
    extr = np.column_stack([rvecs, tvecs]).ravel()
    return np.concatenate([header, extr])


def _unpack_params(params, n_images):
    """Unpack a flat parameter vector back into K, dist, rvecs, tvecs."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = params[:9]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    extr = params[9:].reshape(n_images, 6)
    rvecs = extr[:, :3]
    tvecs = extr[:, 3:]
    return K, dist, rvecs, tvecs


def _calibrate_scipy_residuals(params, all_obj, all_img, n_images):
    """Reprojection error vector for scipy.optimize.least_squares."""
    K, dist, rvecs, tvecs = _unpack_params(params, n_images)
    residuals = []
    for i in range(n_images):
        proj, _ = cv2.projectPoints(
            all_obj[i], rvecs[i], tvecs[i], K, dist
        )
        err = proj.reshape(-1, 2) - all_img[i]
        residuals.append(err.ravel())
    return np.concatenate(residuals)


def _calibrate_scipy(all_corners, all_ids, board, img_size):
    """Joint intrinsic + extrinsic calibration via scipy Levenberg-Marquardt."""
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

    total_err = 0.0
    total_pts = 0
    for i in range(n_images):
        proj, _ = cv2.projectPoints(all_obj[i], rvecs[i], tvecs[i], K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - all_img[i], axis=1)
        total_err += float((err ** 2).sum())
        total_pts += len(err)
    rms = float(np.sqrt(total_err / max(total_pts, 1)))
    return rms, K, dist, None, None

DEFAULT_CHARUCO_PATTERN = (7, 9)  # squares wide x squares tall
DEFAULT_CHARUCO_DICT = cv2.aruco.DICT_4X4_50
# square/marker lengths only need to be consistent (intrinsics are scale
# invariant) - marker must be smaller than the square it sits in
DEFAULT_CHARUCO_SQUARE = 25.0
DEFAULT_CHARUCO_MARKER = 18


def focal_from_object(real_size: float, distance: float, pixel_height: float) -> float:
    if real_size <= 0 or distance <= 0 or pixel_height <= 0:
        return 0.0
    return pixel_height * distance / real_size


def fov_from_focal(focal_px: float, width_px: float) -> float:
    if focal_px <= 0 or width_px <= 0:
        return 0.0
    return float(2.0 * np.degrees(np.arctan2(width_px / 2.0, focal_px)))


def focal_from_fov(fov_deg: float, width_px: float) -> float:
    if fov_deg <= 0 or width_px <= 0:
        return 0.0
    return (width_px / 2.0) / np.tan(np.radians(fov_deg / 2.0))


def random_overlay_color():
    hue = int(np.random.randint(0, 180))
    sat = int(np.random.randint(180, 256))
    val = int(np.random.randint(180, 256))
    bgr = cv2.cvtColor(np.array([[[hue, sat, val]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
    return tuple(int(x) for x in bgr[0, 0])


def draw_corners_into(out, corners, color):
    if corners is None:
        return out
    for corner in corners:
        pts = np.asarray(corner, dtype=np.float64).reshape(-1, 2)
        for x, y in pts:
            cv2.circle(out, (int(round(float(x))), int(round(float(y)))), 5, color, -1)
    return out


def draw_corners(frame, corners, color):
    out = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return draw_corners_into(out, corners, color)


def draw_markers_into(out, corners, color):
    if corners is None:
        return out
    for corner in corners:
        pts = np.asarray(corner, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(out, [pts], True, color, 2)
    return out


def draw_markers(frame, corners, color):
    out = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return draw_markers_into(out, corners, color)


def draw_charuco_into(out, corners, ids, marker_corners, marker_ids, color=None):
    if color is None:
        if marker_corners is not None and marker_ids is not None:
            _draw_detected_markers(out, marker_corners, marker_ids)
        if corners is not None and ids is not None:
            _draw_detected_corners_charuco(out, corners, ids)
    else:
        if marker_corners is not None:
            draw_markers_into(out, marker_corners, color)
        if corners is not None:
            draw_corners_into(out, corners, color)
    return out


def draw_charuco(frame, corners, ids, marker_corners, marker_ids, color=None):
    out = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return draw_charuco_into(out, corners, ids, marker_corners, marker_ids, color=color)


def expected_charuco_corners(pattern) -> int:
    try:
        cols, rows = int(pattern[0]), int(pattern[1])
    except (TypeError, ValueError, IndexError):
        return 0
    return max(0, (cols - 1) * (rows - 1))


def capture_coverage(corners, expected: int) -> float:
    if corners is None or expected <= 0:
        return 0.0
    n = len(np.asarray(corners).reshape(-1, 2))
    return float(min(1.0, n / expected))


def frame_diverse(corners, existing_captures, min_shift_px: float = 12.0) -> bool:
    new = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if len(new) == 0:
        return False
    for stored in existing_captures:
        try:
            old = np.asarray(stored[1], dtype=np.float64).reshape(-1, 2)
        except (TypeError, ValueError, IndexError):
            continue
        if len(old) == 0:
            continue
        diffs = new[:, None, :] - old[None, :, :]
        mean_min = float(np.sqrt((diffs**2).sum(-1)).min(axis=1).mean())
        if mean_min < min_shift_px:
            return False
    return True


def derive_fov_from_intrinsics(result: dict) -> dict:
    try:
        fx = float(result["camera_matrix"][0][0])
        width = float(result["resolution"][0])
    except (KeyError, TypeError, ValueError, IndexError):
        return {}
    if fx <= 0 or width <= 0:
        return {}
    return {
        "fov": round(fov_from_focal(fx, width), 3),
        "focal_length_pixels": round(fx, 2),
    }


def _charuco_captures_degenerate(all_corners: list, all_ids: list) -> bool:
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


# Layouts/dictionaries swept by detect_charuco_auto when nothing matches the
# session's stored layout. Kept short on purpose: every miss costs a full
# detector pass, and this only runs until a board has been matched once.
CHARUCO_COMMON_PATTERNS = (
    DEFAULT_CHARUCO_PATTERN,
    (7, 5),
    (5, 7),
    (8, 6),
    (6, 8),
    (6, 6),
)
CHARUCO_COMMON_DICTS = (
    DEFAULT_CHARUCO_DICT,
    cv2.aruco.DICT_5X5_250,
    cv2.aruco.DICT_6X6_250,
)


def detect_charuco_auto(frame, preferred_pattern=None, preferred_dict=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    patterns = []
    if preferred_pattern is not None:
        try:
            patterns.append((int(preferred_pattern[0]), int(preferred_pattern[1])))
        except (TypeError, ValueError, IndexError):
            pass
    for pat in CHARUCO_COMMON_PATTERNS:
        if tuple(pat) not in patterns:
            patterns.append(tuple(pat))
    dicts = []
    if preferred_dict is not None:
        dicts.append(int(preferred_dict))
    for dictionary_id in CHARUCO_COMMON_DICTS:
        if dictionary_id not in dicts:
            dicts.append(dictionary_id)
    best = None
    for pattern in patterns:
        for dictionary_id in dicts:
            try:
                found, corners, ids, mc, mi, _ = detect_charuco(
                    gray, pattern[0], pattern[1], dictionary_id=dictionary_id
                )
            except Exception:
                continue
            if not found:
                continue
            # a wrong-but-compatible grid can match the same corners, so rank
            # by coverage first, then prefer the smallest board that explains
            # what is visible (a 7x5 print also "fits" a 7x9 grid)
            score = (len(corners), -(pattern[0] * pattern[1]))
            if best is None or score > best[0]:
                best = (score, corners, ids, mc, mi, pattern, dictionary_id)
    if best is not None:
        _, corners, ids, mc, mi, pattern, dictionary_id = best
        return True, corners, ids, mc, mi, gray, pattern, dictionary_id
    return False, None, None, None, None, gray, None, None


def calibrate_charuco(
    captures,
    pattern=DEFAULT_CHARUCO_PATTERN,
    square_length: float = DEFAULT_CHARUCO_SQUARE,
    marker_length: float = DEFAULT_CHARUCO_MARKER,
    dictionary_id: int = DEFAULT_CHARUCO_DICT,
):
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
        result = _calibrate_scipy(all_corners, all_ids, board, img_size)
    except Exception as exc:
        logger.debug("scipy calibration failed, falling back to OpenCV: %s", exc)
        result = None
    if result is not None:
        rms, cam_mat, dist, _, _ = result
    else:
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


def _to_corners(corners):
    a = np.asarray(corners, dtype=np.float32)
    if a.ndim == 2:
        return a.reshape(-1, 1, 2)
    return a


def _to_ids(ids):
    a = np.asarray(ids, dtype=np.int32)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def _to_marker_corners(marker_corners):
    out = []
    for marker in marker_corners:
        a = np.asarray(marker, dtype=np.float32).reshape(-1, 2)
        out.append(a.reshape(4, 1, 2))
    return out


def intrinsics_for_frame(calib: dict, frame_w: int, frame_h: int):
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


def generate_calibration_board_pdf(path):
    try:
        from PIL import Image
    except ImportError:
        return False
    board = make_charuco_board(
        *DEFAULT_CHARUCO_PATTERN,
        square_length=DEFAULT_CHARUCO_SQUARE,
        marker_length=DEFAULT_CHARUCO_MARKER,
        dictionary_id=DEFAULT_CHARUCO_DICT,
    )
    dpi = 300
    sq_px = round(DEFAULT_CHARUCO_SQUARE / 25.4 * dpi)
    cols, rows = DEFAULT_CHARUCO_PATTERN
    board_w = cols * sq_px
    board_h = rows * sq_px
    # generateImage with marginSize is buggy in some OpenCV builds, so draw
    # the board at exact board-pixel size and composite onto a white canvas
    # that includes a half-inch print margin.
    # generateImage is picky about exact sizes in some OpenCV builds; add 1px
    # and let the white canvas margin absorb the difference.
    img = board.generateImage((board_w + 1, board_h + 1))
    margin = round(0.5 * dpi)  # 0.5 inch margin
    canvas = Image.new("L", (board_w + 2 * margin, board_h + 2 * margin), 255)
    canvas.paste(Image.fromarray(img), (margin, margin))
    canvas.save(path, "PDF", resolution=dpi)
    return True
