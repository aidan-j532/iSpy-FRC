import numpy as np
from iSpy.plugins.bases import TrackerBase
from iSpy.algorithms.CustomDBScan import CustomDBScan

class PathPlanner(TrackerBase):
    plugin_name = "path_planner"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "epsilon": {
                "type": "number",
                "label": "Cluster Radius (m)",
                "hint": "DBSCAN epsilon - detections within this distance (m) "
                        "of each other form a cluster.",
                "default": 0.3,
            },
            "min_samples": {
                "type": "number",
                "label": "Min Cluster Size",
                "hint": "DBSCAN min_samples - clusters smaller than this are "
                        "treated as noise.",
                "default": 3,
            },
        }

    def __init__(self, context: dict):
        super().__init__(context)
        self.epsilon = float(self.config.get("epsilon", 0.3))
        self.min_samples = int(self.config.get("min_samples", 3))

        self.fuel_positions = []
        self.noise_positions = []

    def get_noise_positions(self):
        return self.noise_positions

    def get_fuel_positions(self):
        return self.fuel_positions

    def update(self, fuel_list, robot_x, robot_y, robot_yaw, robot_z: float = 0.0):
        self.fuel_positions, self.noise_positions = self._dbscan(fuel_list)
        return self.fuel_positions

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
