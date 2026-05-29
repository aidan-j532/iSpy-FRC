from iSpy.vision.ObjectDetectionCamera import ObjectDetectionCamera
from iSpy.vision.Object import Object
import cv2
import numpy as np
import logging
import threading


class MultipleCameraHandler:
    def __init__(self, cameras: list[ObjectDetectionCamera]):
        self.cameras = cameras
        self.logger = logging.getLogger(__name__)
        self._stopped = False

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

    def predict(self) -> list[Object]:
        for event in self._fresh:
            if not event.wait(timeout=0.2):
                self.logger.debug("Camera timed out waiting for fresh frame")
            event.clear()

        all_objects: list[Object] = []
        for i in range(len(self.cameras)):
            with self._locks[i]:
                all_objects.extend(self._objects[i])
        return all_objects

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

    def destroy(self):
        self._stopped = True
        for cam in self.cameras:
            cam.destroy()
