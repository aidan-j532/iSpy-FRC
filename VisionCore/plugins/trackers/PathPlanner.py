import numpy as np
from VisionCore.plugins.bases import TrackerBase
from VisionCore.config.VisionCoreConfig import VisionCoreConfig
from VisionCore.algorithms.CustomDBScan import CustomDBScan

class PathPlanner(TrackerBase):
    plugin_name = "path_planner"

    def __init__(self, config: VisionCoreConfig):
        self.epsilon = config["dbscan"]["elipson"]
        self.min_samples = config["dbscan"]["min_samples"]

        self.fuel_positions = []
        self.noise_positions = []

    def get_noise_positions(self):
        return self.noise_positions

    def get_fuel_positions(self):
        return self.fuel_positions

    def update(self, fuel_list, robot_x, robot_y, robot_yaw):
        self.fuel_positions, self.noise_positions = self._dbscan(fuel_list)
        return self.noise_positions, self.fuel_positions

    def _dbscan(self, fuels):
        if len(fuels) == 0:
            return [], []

        points = np.array([f.get_position() for f in fuels])

        dbscan = CustomDBScan(points, eps=self.epsilon, samples=self.min_samples)
        labels = dbscan.get_dbscan()

        cleaned = [f for f, label in zip(fuels, labels) if label != -1]
        noise = [f for f, label in zip(fuels, labels) if label == -1]

        return cleaned, noise

    def run(self):
        return self.fuel_positions

    def stop(self):
        pass
