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


if __name__ == "__main__":
    unittest.main()
