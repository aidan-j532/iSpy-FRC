try:
    from sklearn.cluster import DBSCAN

    HAS_SKLEARN = True
except Exception:  # pragma: no cover - sklearn may not be installed
    HAS_SKLEARN = False

class CustomDBScan:
    def __init__(self, points: list, eps: int, samples: int):
        self.points = points
        self.eps = eps
        self.samples = samples
        if HAS_SKLEARN:
            self.dbscan = DBSCAN(eps=self.eps, min_samples=self.samples)
        else:
            self.dbscan = None

    def get_dbscan(self):
        if not HAS_SKLEARN:
            return [0] * len(self.points)
        if self.eps == 0:
            return [0] * len(self.points)
        clusters = self.dbscan.fit_predict(self.points)
        return clusters
