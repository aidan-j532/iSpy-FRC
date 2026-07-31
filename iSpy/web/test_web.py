# iSpy/web/test_web.py
"""Run the entire web stack with fabricated data - no cameras, no models,
no RKNN/ONNX/etc. required. Useful for iterating on frontend/pages without
booting real hardware.

Usage: python -m iSpy.web.test_web
"""
import logging
import math
import random
import threading
import time

import cv2
import numpy as np

from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.web.Backend.WebApp import create_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ispy-test-web")

CAM_NAMES = ["camera_1", "camera_2"]


class FakeObject:
    NAMES = {0: "fuel_cell", 1: "cone", 2: "cube"}

    def __init__(self, i: int):
        self.id = i
        self.x = random.uniform(-3, 3)
        self.y = random.uniform(-3, 3)
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = random.uniform(0, math.tau)
        self.name = self.NAMES.get(i % len(self.NAMES), "unknown")
        self.confidence = random.uniform(0.5, 0.99)

        mode = i % 3
        if mode == 0:
            self.vis_type = "generic"
            self.vis_meta = {}
            self.keypoints_3d = None
        elif mode == 1:
            self.vis_type = "planar"
            self.vis_meta = {"tag_id": i, "size": 0.15}
            self.keypoints_3d = None
        else:
            self.vis_type = "generic"  # forces auto-detect via keypoints_3d
            self.vis_meta = {}
            self.keypoints_3d = [
                [random.uniform(-0.5, 0.5), random.uniform(0, 1.8), random.uniform(-0.3, 0.3)]
                for _ in range(17)
            ]

class FakeCamera:
    """Stands in for ObjectDetectionCamera - only what the web layer touches
    (config.get, get_frame_age)."""
    def __init__(self, name: str):
        self.config = iSpyCameraConfig({"name": name})
        self._last_frame_time = time.perf_counter()

    def get_frame_age(self) -> float:
        return time.perf_counter() - self._last_frame_time

    def touch(self):
        self._last_frame_time = time.perf_counter()


def _make_frame(name: str, w=640, h=480) -> np.ndarray:
    frame = (np.random.rand(h, w, 3) * 40).astype(np.uint8)
    cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, time.strftime("%H:%M:%S"), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return frame


def main():
    config = iSpyConfig()
    cameras = [FakeCamera(n) for n in CAM_NAMES]

    web_app = create_app(cameras=cameras, config=config)
    threading.Thread(target=web_app.run, daemon=True).start()
    logger.info("ispy-test-web running at http://localhost:5000  (Ctrl+C to stop)")

    try:
        while True:
            loop_start = time.perf_counter()
            for cam in cameras:
                cam.touch()

            fuel_list = [FakeObject(i) for i in range(random.randint(0, 5))]
            camera_frames = {n: _make_frame(n) for n in CAM_NAMES}
            loop_s = random.uniform(0.012, 0.03)

            frame_data = {
                "fuel_list": fuel_list,
                "frame": camera_frames[CAM_NAMES[0]],
                "camera_frames": camera_frames,
                "fps": 1 / loop_s,
                "loop_s": loop_s,
                "vision_s": random.uniform(0.005, 0.02),
                "camera_lag_s": random.uniform(0, 0.05),
                "detections": len(fuel_list),
                "cameras": cameras,
                "code_times": {
                    "vision": random.uniform(0.005, 0.02),
                    "trackers": random.uniform(0.001, 0.004),
                    "pose": random.uniform(0.0002, 0.002),
                    "utilities": random.uniform(0.0005, 0.003),
                    "web": random.uniform(0.0005, 0.005),
                },
            }

            web_app.update(frame_data)
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, (1 / 30) - elapsed))
    except KeyboardInterrupt:
        logger.info("Stopping ispy-test-web.")
        web_app.stop()


if __name__ == "__main__":
    main()