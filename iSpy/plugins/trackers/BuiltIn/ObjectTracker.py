import numpy as np
import math
import logging

from iSpy.plugins.bases import TrackerBase
from iSpy.vision.Object import Object

_EMA_ALPHA = 0.3

class ObjectTracker(TrackerBase):
    plugin_name = "object_tracker"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "distance_threshold": {
                "type": "number",
                "label": "Merge Distance (m)",
                "hint": "Detections closer than this to an existing tracked "
                        "object (m) are merged into it instead of spawning a "
                        "new one.",
                "default": 0.5,
            },
            "stale_threshold": {
                "type": "number",
                "label": "Stale Threshold (s)",
                "hint": "Tracked objects not seen for this many seconds are "
                        "dropped.",
                "default": 1.0,
            },
        }

    def __init__(self, context: dict):
        super().__init__(context)
        self.logger = logging.getLogger(__name__)

        self.tracked_objects: list[Object] = []

        raw_threshold = self.config.get("distance_threshold", 0.5)
        if raw_threshold is None or raw_threshold < 0:
            self.distance_threshold = 0.5
            self.logger.warning(
                "distance_threshold invalid or missing, defaulting to 0.5"
            )
        else:
            self.distance_threshold = float(raw_threshold)

        self.stale_threshold = float(self.config.get("stale_threshold", 1.0))

    def update(
        self,
        new_detections: list[Object],
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        robot_z: float = 0.0,
    ) -> list[Object]:

        # age + cleanup
        for obj in self.tracked_objects:
            obj.update()

        self.tracked_objects = [o for o in self.tracked_objects if not o.destroyed]

        # convert detections into robot frame
        for det in new_detections:
            det.relative_to(robot_x, robot_y, robot_z, robot_yaw=robot_yaw)

        # merge
        self._merge(new_detections)

        return self.tracked_objects
    
    def _merge(self, detections: list[Object]):
        for det in detections:
            if not self._exists_and_update(det):
                det.alive_time = self.stale_threshold
                self.tracked_objects.append(det)

    def _exists_and_update(self, new_det: Object) -> bool:
        if not self.tracked_objects:
            return False

        new_pos = np.array(new_det.get_position())

        for existing in self.tracked_objects:
            existing_pos = np.array(existing.get_position())

            if np.linalg.norm(new_pos - existing_pos) < self.distance_threshold:
                # distance alone is not enough - a cone 30cm from a robot
                # must never merge into the robot. only merge detections of
                # the same class/name (empty names on both sides are treated
                # as equal for backwards compatibility)
                if not self._same_identity(new_det, existing):
                    continue

                # reset timer
                existing.reset_time()

                # EMA smoothing on position
                existing.x += _EMA_ALPHA * (new_det.x - existing.x)
                existing.y += _EMA_ALPHA * (new_det.y - existing.y)
                existing.z += _EMA_ALPHA * (new_det.z - existing.z)

                # EMA smoothing on angles (shortest-path wrapping)
                d_roll = (new_det.roll - existing.roll + math.pi) % (2 * math.pi) - math.pi
                existing.roll = (existing.roll + _EMA_ALPHA * d_roll) % (2 * math.pi)

                d_pitch = (new_det.pitch - existing.pitch + math.pi) % (2 * math.pi) - math.pi
                existing.pitch = (existing.pitch + _EMA_ALPHA * d_pitch) % (2 * math.pi)

                d_yaw = (new_det.yaw - existing.yaw + math.pi) % (2 * math.pi) - math.pi
                existing.yaw = (existing.yaw + _EMA_ALPHA * d_yaw) % (2 * math.pi)

                return True

        return False

    @staticmethod
    def _same_identity(a: Object, b: Object) -> bool:
        name_a = getattr(a, "name", "") or ""
        name_b = getattr(b, "name", "") or ""
        if not name_a and not name_b:
            return True
        return bool(name_a) and bool(name_b) and name_a == name_b

    def get_tracked_objects(self) -> list[Object]:
        return self.tracked_objects

    def run(self):
        return self.tracked_objects

    def stop(self):
        self.tracked_objects.clear()