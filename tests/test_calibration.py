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


if __name__ == "__main__":
    unittest.main()
