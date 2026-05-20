import numpy as np
import logging

from VisionCore.plugins.bases import TrackerBase
from VisionCore.config.VisionCoreConfig import VisionCoreConfig
from VisionCore.vision.Object import Object

_EMA_ALPHA = 0.3

class ObjectTracker(TrackerBase):
    plugin_name = "object_tracker"

    def __init__(self, config: VisionCoreConfig):
        self.logger = logging.getLogger(__name__)

        self.fuel_list: list[Object] = []

        raw_threshold = config.get("distance_threshold", 0.5)
        if raw_threshold is None or raw_threshold < 0:
            self.distance_threshold = 0.5
            self.logger.warning(
                "distance_threshold invalid or missing, defaulting to 0.5"
            )
        else:
            self.distance_threshold = float(raw_threshold)

        self.stale_threshold = float(config.get("stale_threshold", 1.0))

    def update(
        self,
        new_fuel_list: list[Object],
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> list[Object]:

        # age + cleanup
        for fuel in self.fuel_list:
            fuel.update()

        self.fuel_list = [f for f in self.fuel_list if not f.destroyed]

        # convert detections into robot frame
        for fuel in new_fuel_list:
            fuel.relative_to(robot_x, robot_y, robot_yaw)

        # merge
        self._merge(new_fuel_list)

        return self.fuel_list
    
    def _merge(self, fuels: list[Object]):
        for fuel in fuels:
            if not self._exists_and_update(fuel):
                fuel.alive_time = self.stale_threshold
                self.fuel_list.append(fuel)

    def _exists_and_update(self, new_fuel: Object) -> bool:
        if not self.fuel_list:
            return False

        new_pos = np.array(new_fuel.get_position())

        for existing in self.fuel_list:
            existing_pos = np.array(existing.get_position())

            if np.linalg.norm(new_pos - existing_pos) < self.distance_threshold:

                # reset timer
                existing.reset_time()

                # EMA smoothing
                existing.x += _EMA_ALPHA * (new_fuel.x - existing.x)
                existing.y += _EMA_ALPHA * (new_fuel.y - existing.y)

                return True

        return False

    def get_fuel_list(self) -> list[Object]:
        return self.fuel_list

    def run(self):
        return self.fuel_list

    def stop(self):
        self.fuel_list.clear()