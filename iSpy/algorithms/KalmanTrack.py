import numpy as np


class KalmanTrack:
    # fixed measurement model: we only observe the 3 position components
    _H = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )

    def __init__(
        self,
        initial_position,
        process_noise: float = 0.5,
        measurement_noise: float = 0.1,
    ):
        pos = np.asarray(initial_position, dtype=float).reshape(3)
        self.x = np.concatenate([pos, np.zeros(3)])  # [x, y, z, vx, vy, vz]
        self.P = np.eye(6) * 1.0
        self.Q = np.eye(6) * max(process_noise, 0.0)
        # velocity elements of process noise get a little extra slop
        self.Q[3:, 3:] = self.Q[3:, 3:] * 2.0
        self.R = np.eye(3) * max(measurement_noise, 0.0)

    def predict(self, dt: float) -> np.ndarray:
        dt = max(float(dt), 0.0)
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x

    def update(self, measurement) -> np.ndarray:
        z = np.asarray(measurement, dtype=float).reshape(3)
        H = self._H
        z_hat = H @ self.x

        innovation = z - z_hat
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        self.P = (np.eye(6) - K @ H) @ self.P
        return self.x

    def predict_measure(self, dt: float, measurement) -> np.ndarray:
        self.predict(dt)
        self.update(measurement)
        return self.get_position()

    def get_position(self) -> np.ndarray:
        return self.x[:3]

    def get_velocity(self) -> np.ndarray:
        return self.x[3:]

    def extrapolate(self, dt: float) -> np.ndarray:
        dt = max(float(dt), 0.0)
        return self.get_position() + self.get_velocity() * dt