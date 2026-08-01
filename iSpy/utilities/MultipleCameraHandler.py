import time
import math

from iSpy.vision.ObjectDetectionCamera import ObjectDetectionCamera
from iSpy.vision.Object import Object
from iSpy.vision import triangulation
import cv2
import numpy as np
import logging
import threading


class MultipleCameraHandler:
    def __init__(self, cameras: list[ObjectDetectionCamera], config=None):
        self.cameras = cameras
        self.logger = logging.getLogger(__name__)
        self._stopped = False
        self._max_residual = (config.get("triangulation_max_residual", 0.5) if config else 0.5)
        self._match_gate = (config.get("triangulation_match_distance", 2.0) if config else 2.0)

        self._objects: list[list[Object]] = [[] for _ in cameras]
        self._frames = [None] * len(cameras)
        self._locks = [threading.Lock() for _ in cameras]
        self._fresh = [threading.Event() for _ in cameras]

        for i, cam in enumerate(cameras):
            threading.Thread(
                target=self._camera_loop, args=(i, cam), daemon=True
            ).start()

    def _camera_loop(self, i: int, camera: ObjectDetectionCamera):
        while not self._stopped:
            try:
                objects, frame = camera.run()
                with self._locks[i]:
                    self._objects[i] = objects if objects is not None else []
                    self._frames[i] = frame
                self._fresh[i].set()
            except Exception as e:
                self.logger.warning(f"Camera {camera.source} error: {e}")
                time.sleep(0.05) # Dont starve CPU

    def predict(self) -> list[Object]:
        for event in self._fresh:
            if not event.wait(timeout=0.2):
                self.logger.debug("Camera timed out waiting for fresh frame")
            event.clear()

        per_camera: list[list[Object]] = []
        for i in range(len(self.cameras)):
            with self._locks[i]:
                per_camera.append(list(self._objects[i]))

        return self._merge_with_triangulation(per_camera)

    def _merge_with_triangulation(self, per_camera: list[list["Object"]]) -> list["Object"]:
        used: set[tuple[int, int]] = set()
        merged: list[Object] = []

        for cam_a in range(len(per_camera)):
            for idx_a, obj_a in enumerate(per_camera[cam_a]):
                if (cam_a, idx_a) in used:
                    continue
                best = None  # (residual, cam_b, idx_b, point)
                if obj_a.ray_origin is not None:
                    for cam_b in range(cam_a + 1, len(per_camera)):
                        for idx_b, obj_b in enumerate(per_camera[cam_b]):
                            if (cam_b, idx_b) in used:
                                continue
                            if obj_b.name != obj_a.name or obj_b.ray_origin is None:
                                continue
                            rough = math.hypot(obj_a.x - obj_b.x, obj_a.y - obj_b.y)
                            if rough > self._match_gate:
                                continue
                            ray_a = triangulation.Ray(obj_a.ray_origin, obj_a.ray_direction)
                            ray_b = triangulation.Ray(obj_b.ray_origin, obj_b.ray_direction)
                            result = triangulation.closest_point_between_rays(
                                ray_a, ray_b, max_residual=self._max_residual
                            )
                            if result is None:
                                continue
                            point, residual = result
                            if best is None or residual < best[0]:
                                best = (residual, cam_b, idx_b, point)

                if best is not None:
                    residual, cam_b, idx_b, point = best
                    obj_a.x, obj_a.y, obj_a.z = float(point[0]), float(point[1]), float(point[2])
                    obj_a.depth_source = "triangulated"
                    used.add((cam_b, idx_b))

                used.add((cam_a, idx_a))
                merged.append(obj_a)

        return merged

    def get_combined_frame(self, display_width=640):
        frames = []
        for i, cam in enumerate(self.cameras):
            with self._locks[i]:
                f = self._frames[i]
            if f is None:
                f = cam.get_frame()
            if f is not None:
                frames.append(f.copy())

        if not frames:
            return None
        if len(frames) == 1:
            f = frames[0]
        else:
            target_h = min(f.shape[0] for f in frames)
            resized = []
            for f in frames:
                h, w = f.shape[:2]
                if h != target_h:
                    new_w = int(w * (target_h / h))
                    f = cv2.resize(f, (new_w, target_h), interpolation=cv2.INTER_AREA)
                resized.append(f)
            f = np.hstack(resized)

        h, w = f.shape[:2]
        if w > display_width:
            scale = display_width / w
            f = cv2.resize(
                f, (display_width, int(h * scale)), interpolation=cv2.INTER_AREA
            )
        return f

    def get_camera_frames(self) -> dict[str, np.ndarray]:
        """Named per-camera frames for the web layer - lets CamerasModule
        serve individual feeds instead of only the stitched combined view."""
        result = {}
        for i, cam in enumerate(self.cameras):
            with self._locks[i]:
                f = self._frames[i]
            if f is None:
                continue
            name = cam.config.get("name", f"Camera {i+1}") if hasattr(cam, "config") else f"Camera {i+1}"
            result[name] = f.copy()
        return result

    def destroy(self):
        self._stopped = True
        for cam in self.cameras:
            cam.destroy()