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

        self.cluster_positions = []
        self.noise_positions = []

    def get_noise_positions(self):
        return self.noise_positions

    def get_cluster_positions(self):
        return self.cluster_positions

    def update(self, detections, robot_x, robot_y, robot_yaw, robot_z: float = 0.0):
        self.cluster_positions, self.noise_positions = self._dbscan(detections)
        return self.cluster_positions

    def _dbscan(self, detections):
        if len(detections) == 0:
            return [], []

        points = np.array([d.get_position() for d in detections])

        dbscan = CustomDBScan(points, eps=self.epsilon, samples=self.min_samples)
        labels = dbscan.get_dbscan()

        cleaned = [d for d, label in zip(detections, labels) if label != -1]
        noise = [d for d, label in zip(detections, labels) if label == -1]

        return cleaned, noise

    def run(self):
        return self.cluster_positions

    def stop(self):
        pass
