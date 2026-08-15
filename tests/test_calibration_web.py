"""web calibration wizard tests: ChArUco capture/finish flow, the raw
calibration feed, and detection pause/heartbeat endpoints."""

import base64
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import flask
import numpy as np

from iSpy.config.iSpyConfig import iSpyConfig
from iSpy.vision import calibration as c
from iSpy.web.modules.cameras import CamerasModule


class _FakeVision:
    pass


class _FakeCamera:
    """stand-in for a Camera/pipeline exposing the calibration API the web layer uses"""

    def __init__(self, name):
        self.name = name
        self.config = type("Cfg", (), {"get": lambda self, k, d=None: {"name": name}.get(k, d)})()
        self.calibration_active = False
        self.calibration_last_seen = 0.0
        self._frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def set_calibration(self, active):
        self.calibration_active = bool(active)
        self.calibration_last_seen = time.monotonic() if active else 0.0

    def calibration_heartbeat(self):
        self.calibration_last_seen = time.monotonic()

    def in_calibration_mode(self):
        return bool(self.calibration_active and time.monotonic() - self.calibration_last_seen < 10.0)

    def get_raw_frame(self):
        return self._frame.copy()


def _charuco_b64(pattern=(7, 5), size=(640, 480), rotate=False):
    board = c.make_charuco_board(*pattern)
    img = board.generateImage(size, marginSize=20)
    if rotate:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _chessboard_image(pattern=(9, 6), img_size=(480, 640), square=60):
    cols, rows = pattern
    board = np.full((img_size[0], img_size[1], 3), 255, np.uint8)
    sq = int(square)
    ox = (img_size[1] - (cols + 1) * sq) // 2
    oy = (img_size[0] - (rows + 1) * sq) // 2
    for y in range(rows + 1):
        for x in range(cols + 1):
            if (x + y) % 2 == 0:
                cv2.rectangle(board, (ox + x * sq, oy + y * sq),
                              (ox + (x + 1) * sq, oy + (y + 1) * sq), (0, 0, 0), -1)
    ok, buf = cv2.imencode(".jpg", board, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _pose_camera_setup(num_kpts=17, intrinsics=False):
    """config with a pose-model camera (fake .pt + YAML sidecar) and a live cam."""
    tmpdir = Path(tempfile.mkdtemp())
    model = tmpdir / "pose_model.pt"
    sidecar = tmpdir / "pose_model_metadata.yaml"
    sidecar.write_text(f"task: pose\nkpt_shape: [{num_kpts}, 3]\n", encoding="utf-8")
    cfg = iSpyConfig(file_path=str(tmpdir / "config.json"))
    cfg.config["app_mode"] = False
    calibration = (
        {
            "camera_matrix": [[300.0, 0, 320.0], [0, 300.0, 240.0], [0, 0, 1.0]],
            "dist_coeffs": [0, 0, 0, 0, 0],
        }
        if intrinsics
        else {"distance": 0, "game_piece_size": 0, "size": 0, "fov": 0}
    )
    cfg.config["camera_configs"] = {
        "cam_0": {
            "name": "cam_0",
            "source": 0,
            "pipeline": {
                "name": "object_detection",
                "settings": {"vision_model": {"file_path": str(model)}},
            },
            "calibration": calibration,
        }
    }
    cam = _FakeCamera("cam_0")
    mod = CamerasModule({"config": cfg, "vision_instance": _FakeVision()})
    mod.live_cameras = {"cam_0": cam}
    app = flask.Flask(__name__)
    mod.register_routes(app)
    return cfg, model, app.test_client()


class CalibrationWebTests(unittest.TestCase):
    def _setup(self):
        tmp = Path(tempfile.mkdtemp()) / "config.json"
        cfg = iSpyConfig(file_path=str(tmp))
        cfg.config["app_mode"] = False
        cfg.config["camera_configs"] = {
            "cam_0": {
                "name": "cam_0",
                "source": 0,
                "pipeline": "april_tag",
                "calibration": {"distance": 0, "game_piece_size": 0, "size": 0, "fov": 0},
            }
        }
        cam = _FakeCamera("cam_0")
        mod = CamerasModule({"config": cfg, "vision_instance": _FakeVision()})
        mod.live_cameras = {"cam_0": cam}
        app = flask.Flask(__name__)
        mod.register_routes(app)
        return cfg, cam, mod, app.test_client()

    def test_calibration_mode_pauses_and_resumes(self):
        cfg, cam, mod, client = self._setup()
        r = client.post("/api/cameras/calibration/cam_0/mode", json={"active": True})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(cam.calibration_active)
        self.assertTrue(cam.in_calibration_mode())
        r = client.post("/api/cameras/calibration/cam_0/mode", json={"active": False})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(cam.in_calibration_mode())

    def test_heartbeat_refreshes(self):
        cfg, cam, mod, client = self._setup()
        client.post("/api/cameras/calibration/cam_0/mode", json={"active": True})
        cam.calibration_last_seen = 0.0
        r = client.post("/api/cameras/calibration/cam_0/heartbeat", json={})
        self.assertEqual(r.status_code, 200)
        self.assertGreater(cam.calibration_last_seen, 0.0)
        self.assertTrue(cam.in_calibration_mode())

    def test_charuco_capture_finish_flow(self):
        cfg, cam, mod, client = self._setup()
        for i in range(4):
            r = client.post("/api/cameras/calibration/cam_0/charuco/capture", json={
                "image": _charuco_b64(rotate=(i % 2 == 1)),
                "cols": 7, "rows": 5,
            })
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.get_json()["board_found"])
        self.assertEqual(r.get_json()["captured"], 4)

        r = client.post("/api/cameras/calibration/cam_0/charuco/finish", json={})
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertTrue(j["success"])
        self.assertIn("camera_matrix", j["result"])
        self.assertEqual(len(j["result"]["camera_matrix"]), 3)
        self.assertIn("camera_matrix", j["calibration"])
        self.assertEqual(cfg.config["camera_configs"]["cam_0"]["calibration"]["rms"], j["result"]["rms"])

    def test_charuco_intrinsics_flow_for_object_detection(self):
        # the focal wizard's optional precision-intrinsics step reuses the
        # charuco endpoints - camera_matrix/dist_coeffs land in the same
        # calibration dict an object_detection camera reads at runtime
        tmp = Path(tempfile.mkdtemp()) / "config.json"
        cfg = iSpyConfig(file_path=str(tmp))
        cfg.config["app_mode"] = False
        cfg.config["camera_configs"] = {
            "cam_0": {
                "name": "cam_0",
                "source": 0,
                "pipeline": "object_detection",
                "calibration": {"distance": 0, "game_piece_size": 0, "size": 0, "fov": 0},
            }
        }
        cam = _FakeCamera("cam_0")
        mod = CamerasModule({"config": cfg, "vision_instance": _FakeVision()})
        mod.live_cameras = {"cam_0": cam}
        app = flask.Flask(__name__)
        mod.register_routes(app)
        client = app.test_client()

        for i in range(4):
            r = client.post("/api/cameras/calibration/cam_0/charuco/capture", json={
                "image": _charuco_b64(rotate=(i % 2 == 1)),
                "cols": 7, "rows": 5,
            })
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.get_json()["board_found"])
        r = client.post("/api/cameras/calibration/cam_0/charuco/finish", json={})
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertTrue(j["success"])
        cal = cfg.config["camera_configs"]["cam_0"]["calibration"]
        self.assertIn("camera_matrix", cal)
        self.assertEqual(cal["camera_matrix"], j["result"]["camera_matrix"])
        self.assertIn("dist_coeffs", cal)
        # known-object values survive the intrinsics write
        self.assertEqual(cal["fov"], 0)

    def test_charuco_capture_clear(self):
        cfg, cam, mod, client = self._setup()
        client.post("/api/cameras/calibration/cam_0/charuco/capture", json={
            "image": _charuco_b64(), "cols": 7, "rows": 5,
        })
        r = client.delete("/api/cameras/calibration/cam_0/charuco")
        self.assertEqual(r.status_code, 200)
        get = client.get("/api/cameras/calibration/cam_0")
        self.assertEqual(get.get_json()["charuco_captures"], 0)

    def test_raw_calibration_feed_serves_unannotated_frame(self):
        cfg, cam, mod, client = self._setup()
        cam._frame = np.full((100, 160, 3), 77, dtype=np.uint8)
        gen = mod._generate_calibration("cam_0")
        try:
            chunk = next(gen)
        finally:
            gen.close()
        self.assertIn(b"--frame\r\nContent-Type: image/jpeg", chunk)
        body = chunk.split(b"\r\n\r\n", 1)[1].split(b"\r\n")[0]
        img = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(int(img[50, 80, 0]), 77)

    def _first_jpeg(self, gen):
        try:
            chunk = next(gen)
        finally:
            gen.close()
        body = chunk.split(b"\r\n\r\n", 1)[1]
        if body.endswith(b"\r\n"):
            body = body[:-2]
        return cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)

    def test_charuco_overlay_feed_draws_detected_board(self):
        cfg, cam, mod, client = self._setup()
        board_img = c.make_charuco_board(7, 5).generateImage((640, 480), marginSize=20)
        cam._frame = board_img
        served = self._first_jpeg(mod._generate_calibration("cam_0", overlay="charuco", pattern=(7, 5)))
        b, g, r = served[:, :, 0].astype(int), served[:, :, 1].astype(int), served[:, :, 2].astype(int)
        green = (g > r + 20) & (g > b + 20)
        self.assertTrue(bool(green.any()), "expected green detection overlay on the served frame")

    def test_charuco_overlay_feed_leaves_non_board_plain(self):
        cfg, cam, mod, client = self._setup()
        cam._frame = np.full((100, 160, 3), 30, dtype=np.uint8)
        served = self._first_jpeg(mod._generate_calibration("cam_0", overlay="charuco", pattern=(7, 5)))
        b, g, r = served[:, :, 0].astype(int), served[:, :, 1].astype(int), served[:, :, 2].astype(int)
        green = (g > r + 20) & (g > b + 20)
        self.assertFalse(bool(green.any()), "no board should mean no overlay")

    def test_chessboard_overlay_feed_draws_detected_board(self):
        cfg, cam, mod, client = self._setup()
        cam._frame = cv2.imdecode(
            np.frombuffer(base64.b64decode(_chessboard_image().split(",", 1)[1]), np.uint8),
            cv2.IMREAD_COLOR,
        )
        served = self._first_jpeg(mod._generate_calibration("cam_0", overlay="chessboard", pattern=(9, 6)))
        b, g, r = served[:, :, 0].astype(int), served[:, :, 1].astype(int), served[:, :, 2].astype(int)
        green = (g > r + 20) & (g > b + 20)
        self.assertTrue(bool(green.any()), "expected green detection overlay on the served frame")

    def test_chessboard_capture_reports_color(self):
        cfg, cam, mod, client = self._setup()
        r = client.post("/api/cameras/calibration/cam_0/chessboard/capture", json={
            "image": _chessboard_image(), "cols": 9, "rows": 6,
        })
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertTrue(j["board_found"])
        self.assertEqual(j["captured"], 1)
        self.assertEqual(len(j["color"]), 3)
        self.assertTrue(all(isinstance(v, int) for v in j["color"]))

    def test_charuco_capture_reports_color(self):
        cfg, cam, mod, client = self._setup()
        r = client.post("/api/cameras/calibration/cam_0/charuco/capture", json={
            "image": _charuco_b64(), "cols": 7, "rows": 5,
        })
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertTrue(j["board_found"])
        self.assertEqual(len(j["color"]), 3)
        self.assertTrue(all(isinstance(v, int) for v in j["color"]))

    def test_charuco_captured_overlay_drawn_in_its_color(self):
        cfg, cam, mod, client = self._setup()
        cam._frame = c.make_charuco_board(7, 5).generateImage((640, 480), marginSize=20)
        ok, buf = cv2.imencode(".jpg", cam._frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
        r = client.post("/api/cameras/calibration/cam_0/charuco/capture", json={
            "image": b64, "cols": 7, "rows": 5,
        })
        self.assertEqual(r.status_code, 200)
        color = r.get_json()["color"]
        # blank live frame so only the captured overlay is drawn - look for its color
        cam._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        served = self._first_jpeg(mod._generate_calibration("cam_0", overlay="charuco", pattern=(7, 5)))
        b, g, r = served[:, :, 0].astype(int), served[:, :, 1].astype(int), served[:, :, 2].astype(int)
        painted = (r > color[2] - 60) & (g > color[1] - 60) & (b > color[0] - 60)
        self.assertTrue(bool(painted.any()), f"expected captured overlay in color {color} on the feed")

    def test_charuco_status_reports_detection(self):
        cfg, cam, mod, client = self._setup()
        cam._frame = c.make_charuco_board(7, 5).generateImage((640, 480), marginSize=20)
        r = client.get("/api/cameras/calibration/cam_0/charuco/status")
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertTrue(j["found"])
        self.assertGreater(j["corners"], 0)
        self.assertGreater(j["markers"], 0)
        self.assertEqual(j["pattern"], [7, 5])

    def test_charuco_status_not_found(self):
        cfg, cam, mod, client = self._setup()
        cam._frame = np.full((100, 160, 3), 30, dtype=np.uint8)
        r = client.get("/api/cameras/calibration/cam_0/charuco/status")
        self.assertFalse(r.get_json()["found"])

    def test_charuco_status_auto_detects_layout_and_dictionary(self):
        cfg, cam, mod, client = self._setup()
        board = c.make_charuco_board(8, 6, dictionary_id=cv2.aruco.DICT_5X5_250)
        cam._frame = board.generateImage((900, 700), marginSize=20)
        r = client.get("/api/cameras/calibration/cam_0/charuco/status")
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertTrue(j["found"])
        self.assertEqual(j["pattern"], [8, 6])
        self.assertEqual(j["dictionary"], cv2.aruco.DICT_5X5_250)

    def test_pnp_get_reports_pose_model(self):
        cfg, model, client = _pose_camera_setup(num_kpts=17, intrinsics=False)
        r = client.get("/api/cameras/calibration/cam_0/pnp")
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertIsNone(j["model_error"])
        self.assertEqual(j["num_keypoints"], 17)
        self.assertFalse(j["has_intrinsics"])
        self.assertIsNone(j["pnp"])

    def test_pnp_save_requires_intrinsics(self):
        cfg, model, client = _pose_camera_setup(num_kpts=17, intrinsics=False)
        r = client.post(
            "/api/cameras/calibration/cam_0/pnp",
            json={"object_points": [[0, 0, 0]] * 17, "min_keypoint_conf": 0.5, "mode": "flexible"},
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("intrinsics", r.get_json()["error"])

    def test_pnp_save_validates_point_count(self):
        cfg, model, client = _pose_camera_setup(num_kpts=17, intrinsics=True)
        r = client.post(
            "/api/cameras/calibration/cam_0/pnp",
            json={"object_points": [[0, 0, 0]] * 4, "min_keypoint_conf": 0.5, "mode": "flexible"},
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("exactly 17", r.get_json()["error"])

    def test_pnp_save_and_clear_persists(self):
        cfg, model, client = _pose_camera_setup(num_kpts=3, intrinsics=True)
        points = [[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0]]
        r = client.post(
            "/api/cameras/calibration/cam_0/pnp",
            json={"object_points": points, "min_keypoint_conf": 0.7, "mode": "rigid"},
        )
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertTrue(j["success"])
        self.assertEqual(j["pnp"]["object_points"], points)
        self.assertEqual(j["pnp"]["mode"], "rigid")
        self.assertEqual(j["pnp"]["min_keypoint_conf"], 0.7)
        self.assertEqual(
            j["pnp"]["camera_matrix"][0][0], 300.0,
        )
        stored = cfg.get("camera_configs", {}).get("cam_0", {})
        self.assertEqual(
            stored["pipeline"]["settings"]["vision_model"]["pnp"]["object_points"], points
        )
        r = client.delete("/api/cameras/calibration/cam_0/pnp")
        self.assertEqual(r.status_code, 200)
        stored = cfg.get("camera_configs", {}).get("cam_0", {})
        self.assertNotIn("pnp", stored["pipeline"]["settings"]["vision_model"])

    def test_pnp_get_non_pose_model_reports_error(self):
        cfg, model, client = _pose_camera_setup(num_kpts=17, intrinsics=True)
        sidecar = model.with_name(model.stem + "_metadata.yaml")
        sidecar.write_text("task: detect\n", encoding="utf-8")
        r = client.get("/api/cameras/calibration/cam_0/pnp")
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertIn("pose model", j["model_error"])


if __name__ == "__main__":
    unittest.main()
