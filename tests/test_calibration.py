import math
import unittest

import cv2
import numpy as np

from iSpy.vision import calibration as c


def _make_board(pattern, img_size=(480, 640), square=60, shift_x=0, scale=1.0):
    """synthetic chessboard that cv2 reliably detects (white bg, black squares)."""
    cols, rows = pattern
    board = np.full((img_size[0], img_size[1], 3), 255, np.uint8)
    sq = int(square * scale)
    ox = (img_size[1] - (cols + 1) * sq) // 2 + shift_x
    oy = (img_size[0] - (rows + 1) * sq) // 2
    for y in range(rows + 1):
        for x in range(cols + 1):
            if (x + y) % 2 == 0:
                cv2.rectangle(board, (ox + x * sq, oy + y * sq),
                              (ox + (x + 1) * sq, oy + (y + 1) * sq), (0, 0, 0), -1)
    return board


class FocalLengthTests(unittest.TestCase):
    def test_focal_from_object(self):
        # 6.5in object, 24in away, 60px tall -> f = 60 * 24 / 6.5
        self.assertAlmostEqual(c.focal_from_object(6.5, 24.0, 60), 60 * 24 / 6.5, places=3)

    def test_focal_from_fov_roundtrip(self):
        f = c.focal_from_object(6.5, 24.0, 60)
        fov = c.fov_from_focal(f, 640)
        self.assertAlmostEqual(c.focal_from_fov(fov, 640), f, places=3)

    def test_fov_sane(self):
        fov = c.fov_from_focal(500, 640)
        self.assertAlmostEqual(fov, math.degrees(2 * math.atan(320 / 500)), places=3)


class IntrinsicsScalingTests(unittest.TestCase):
    def test_returns_none_without_calibration(self):
        self.assertIsNone(c.intrinsics_for_frame({}, 640, 480))

    def test_scales_to_live_resolution(self):
        calib = {
            "camera_matrix": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            "resolution": [640, 480],
        }
        m, d = c.intrinsics_for_frame(calib, 1280, 960)
        self.assertEqual(m[0][0], 1000.0)
        self.assertEqual(m[0][2], 640.0)
        self.assertEqual(len(d), 5)

    def test_missing_resolution_assumes_same_frame(self):
        calib = {"camera_matrix": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                 "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]}
        m, _ = c.intrinsics_for_frame(calib, 640, 480)
        self.assertEqual(m[0][0], 500.0)


class CharucoCalibrationTests(unittest.TestCase):
    def test_detects_synthetic_board(self):
        pattern = (7, 5)
        board = c.make_charuco_board(*pattern)
        img = board.generateImage((700, 500), marginSize=20)
        found, corners, ids, mc, mid, gray = c.detect_charuco(img, *pattern)
        self.assertTrue(found)
        self.assertGreaterEqual(len(corners), 4)
        self.assertIsNotNone(ids)
        self.assertIsNotNone(mc)

    def test_auto_detect_finds_non_default_pattern_and_dict(self):
        # an 8x6 board built with a non-default dictionary should be found by
        # the auto-scan and reported with its real layout/dictionary
        pattern = (8, 6)
        board = c.make_charuco_board(*pattern, dictionary_id=cv2.aruco.DICT_5X5_250)
        img = board.generateImage((900, 700), marginSize=20)
        found, corners, ids, _, _, _, found_pattern, found_dict = c.detect_charuco_auto(img)
        self.assertTrue(found)
        self.assertEqual(tuple(found_pattern), pattern)
        self.assertEqual(found_dict, cv2.aruco.DICT_5X5_250)
        self.assertGreaterEqual(len(corners), 4)

    def test_auto_detect_prefers_preferred_layout(self):
        board = c.make_charuco_board(7, 5)
        img = board.generateImage((700, 500), marginSize=20)
        _, _, _, _, _, _, found_pattern, _ = c.detect_charuco_auto(
            img, preferred_pattern=(7, 5)
        )
        self.assertEqual(tuple(found_pattern), (7, 5))

    def test_auto_detect_returns_none_on_blank(self):
        blank = np.full((480, 640, 3), 255, np.uint8)
        found, _, _, _, _, _, found_pattern, found_dict = c.detect_charuco_auto(blank)
        self.assertFalse(found)
        self.assertIsNone(found_pattern)
        self.assertIsNone(found_dict)

    def test_calibrates_varied_frames(self):
        pattern = (7, 5)
        board = c.make_charuco_board(*pattern)
        captures = []
        sizes = [(640, 480), (800, 600), (640, 480), (800, 600), (640, 480)]
        for i, size in enumerate(sizes):
            img = board.generateImage(size, marginSize=20)
            if i % 2 == 1:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif i % 4 == 3:
                img = cv2.resize(img, (int(img.shape[1] * 0.9), int(img.shape[0] * 0.9)))
            found, corners, ids, _, _, gray = c.detect_charuco(img, *pattern)
            if found:
                captures.append((gray, corners, ids))
        self.assertGreaterEqual(len(captures), 3)
        res = c.calibrate_charuco(captures, pattern)
        self.assertIsNotNone(res)
        self.assertEqual(len(res["camera_matrix"]), 3)
        self.assertEqual(len(res["dist_coeffs"]), 5)

    def test_rejects_degenerate_frames(self):
        pattern = (7, 5)
        board = c.make_charuco_board(*pattern)
        img = board.generateImage((640, 480), marginSize=20)
        found, corners, ids, _, _, gray = c.detect_charuco(img, *pattern)
        self.assertTrue(found)
        captures = [(gray, corners, ids) for _ in range(5)]
        self.assertIsNone(c.calibrate_charuco(captures, pattern))


def _render_april_pose(tag, rx, ry, t):
    """warp a tag into a frame via a projective homography so each capture is
    a genuinely different pose (focal 300 px)."""
    h, w = tag.shape[:2]
    R, _ = cv2.Rodrigues(np.array([rx, ry, 0.0], dtype=np.float64))
    K = np.array([[300.0, 0, 320.0], [0, 300.0, 240.0], [0, 0, 1.0]], dtype=np.float64)
    H = K @ np.hstack([R[:, :2], np.array([[0.0], [0.0], [t]])])
    canvas = np.full((480, 640, 3), 255, np.uint8)
    return cv2.warpPerspective(tag, H, (640, 480), canvas, borderValue=(255, 255, 255))


class AprilTagCalibrationTests(unittest.TestCase):
    def _tag_image(self):
        tag = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 0, 200
        )
        return np.stack([tag] * 3, axis=-1)

    def test_detects_synthetic_tag(self):
        frame = _render_april_pose(self._tag_image(), 0, 0, 500)
        found, corners, ids, mc, mi, gray = c.detect_april(frame)
        self.assertTrue(found)
        self.assertEqual(int(ids[0][0]), 0)
        self.assertEqual(len(corners), 1)
        drawn = c.draw_april(frame, corners, ids)
        self.assertEqual(drawn.shape, frame.shape)

    def test_returns_none_on_blank(self):
        found, corners, ids, _, _, _ = c.detect_april(np.full((480, 640, 3), 255, np.uint8))
        self.assertFalse(found)
        self.assertIsNone(corners)
        self.assertIsNone(ids)

    def test_calibrates_varied_frames(self):
        tag = self._tag_image()
        captures = []
        for rx, ry, t in [(0, 0, 500), (0.5, 0, 460), (-0.4, 0.6, 520),
                          (0.6, -0.5, 470), (-0.3, -0.4, 540), (0.2, 0.8, 500)]:
            found, corners, ids, _, _, gray = c.detect_april(_render_april_pose(tag, rx, ry, t))
            if found:
                captures.append((gray, corners, ids))
        self.assertGreaterEqual(len(captures), 3)
        res = c.calibrate_april(captures, tag_size_inches=6.5)
        self.assertIsNotNone(res)
        self.assertEqual(len(res["camera_matrix"]), 3)
        self.assertEqual(len(res["dist_coeffs"]), 5)
        self.assertAlmostEqual(res["camera_matrix"][0][0], 300.0, delta=40)

    def test_rejects_degenerate_frames(self):
        tag = self._tag_image()
        found, corners, ids, _, _, gray = c.detect_april(_render_april_pose(tag, 0, 0, 500))
        self.assertTrue(found)
        self.assertIsNone(c.calibrate_april([(gray, corners, ids)] * 5, tag_size_inches=6.5))


class ChessboardCalibrationTests(unittest.TestCase):
    def test_detects_synthetic_board(self):
        pattern = (9, 6)
        found, corners, gray = c.detect_chessboard(_make_board(pattern), *pattern)
        self.assertTrue(found)
        self.assertEqual(corners.shape[0], 9 * 6)

    def test_calibrates_varied_frames(self):
        pattern = (9, 6)
        captures = []
        for i, (sx, scale) in enumerate([(0, 1.0), (-40, 1.0), (60, 1.0),
                                         (0, 0.85), (-60, 1.0), (80, 0.9),
                                         (0, 1.0), (50, 0.88), (-20, 1.0)]):
            board = _make_board(pattern, shift_x=sx, scale=scale)
            found, corners, _ = c.detect_chessboard(board, *pattern)
            if found:
                captures.append((board, corners))
        self.assertGreaterEqual(len(captures), 3)
        res = c.calibrate_chessboard(captures, pattern)
        self.assertIsNotNone(res)
        self.assertEqual(res["resolution"], [640, 480])
        self.assertEqual(len(res["camera_matrix"]), 3)
        self.assertEqual(len(res["dist_coeffs"]), 5)

    def test_rejects_degenerate_frames(self):
        pattern = (9, 6)
        board = _make_board(pattern)
        found, corners, _ = c.detect_chessboard(board, *pattern)
        self.assertTrue(found)
        captures = [(board, corners) for _ in range(5)]
        self.assertIsNone(c.calibrate_chessboard(captures, pattern))


class OverlayColorTests(unittest.TestCase):
    def _flat_frame(self, color=(30, 30, 30)):
        return np.full((120, 160, 3), color, dtype=np.uint8)

    def test_random_overlay_color_is_bright_bgr(self):
        color = c.random_overlay_color()
        self.assertEqual(len(color), 3)
        self.assertTrue(all(0 <= v <= 255 for v in color))
        self.assertGreater(max(color), 150, "color should be bright enough to show on any scene")

    def test_draw_corners_paints_given_color(self):
        frame = self._flat_frame()
        corners = np.array([[[40.0, 60.0]], [[120.0, 60.0]]], dtype=np.float32)
        out = c.draw_corners(frame, corners, (10, 200, 250))
        b, g, r = out[:, :, 0].astype(int), out[:, :, 1].astype(int), out[:, :, 2].astype(int)
        painted = (r > 200) & (g > 150) & (b < 50)
        self.assertTrue(bool(painted.any()), "expected the given color to be drawn")

    def test_draw_chessboard_color_uses_given_color(self):
        frame = self._flat_frame()
        found, corners, _ = c.detect_chessboard(_make_board((9, 6)), * (9, 6))
        self.assertTrue(found)
        out = c.draw_chessboard(frame, corners, 9, 6, color=(10, 200, 250))
        b, g, r = out[:, :, 0].astype(int), out[:, :, 1].astype(int), out[:, :, 2].astype(int)
        painted = (r > 200) & (g > 150) & (b < 50)
        self.assertTrue(bool(painted.any()), "expected colored chessboard corners on the copy")

    def test_draw_charuco_color_uses_given_color(self):
        frame = self._flat_frame()
        board_img = c.make_charuco_board(7, 5).generateImage((640, 480), marginSize=20)
        found, corners, ids, mc, mi, _ = c.detect_charuco(board_img, 7, 5)
        self.assertTrue(found)
        out = c.draw_charuco(frame, corners, ids, mc, mi, color=(10, 200, 250))
        b, g, r = out[:, :, 0].astype(int), out[:, :, 1].astype(int), out[:, :, 2].astype(int)
        painted = (r > 200) & (g > 150) & (b < 50)
        self.assertTrue(bool(painted.any()), "expected colored charuco corners on the copy")

    def test_draw_april_color_uses_given_color(self):
        tag = np.stack(
            [cv2.aruco.generateImageMarker(
                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 0, 200)] * 3,
            axis=-1,
        )
        frame = _render_april_pose(tag, 0, 0, 500)
        found, corners, ids, _, _, _ = c.detect_april(frame)
        self.assertTrue(found)
        out = c.draw_april(frame, corners, ids, color=(10, 200, 250))
        b, g, r = out[:, :, 0].astype(int), out[:, :, 1].astype(int), out[:, :, 2].astype(int)
        painted = (r > 200) & (g > 150) & (b < 50)
        self.assertTrue(bool(painted.any()), "expected colored april box on the copy")


if __name__ == "__main__":
    unittest.main()
