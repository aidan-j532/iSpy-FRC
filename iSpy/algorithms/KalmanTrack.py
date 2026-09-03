"""Constant-velocity Kalman filter for a single tracked object.

This is the math behind ``EKFTracker`` (see
``iSpy/plugins/trackers/BuiltIn/EKFTracker.py``), kept in its own thin class
the same way ``CustomDBScan`` is consumed by ``PathPlanner``. The motion model
is a plain linear constant-velocity filter - in the FRC/DJI sense "EKF" just
means "a Kalman filter with a motion model" - so no nonlinear observation
step or linearization is needed.

State ordering is ``[x, y, z, vx, vy, vz]``; the measurement is ``[x, y, z]``
as produced by ``Object.get_position()``.
"""

import numpy as np


class KalmanTrack:
    """6-DOF (position + velocity) linear Kalman filter in robot space."""

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
        """Advance the state estimate using a constant-velocity model.

        Returns the predicted state vector ``[x, y, z, vx, vy, vz]``.
        """
        dt = max(float(dt), 0.0)
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x

    def update(self, measurement) -> np.ndarray:
        """Correct the estimate with a measured ``[x, y, z]`` position.

        Returns the filtered state vector after the update.
        """
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
        """Convenience: predict to ``dt`` seconds then correct with a
        measurement. Returns the filtered position ``[x, y, z]``.
        """
        self.predict(dt)
        self.update(measurement)
        return self.get_position()

    def get_position(self) -> np.ndarray:
        return self.x[:3]

    def get_velocity(self) -> np.ndarray:
        return self.x[3:]

    def extrapolate(self, dt: float) -> np.ndarray:
        """Position-only extrapolation for when a track missed a detection
        this tick but is still within the stale threshold. Does not mutate the
        estimator (the caller decides whether to adopt the extrapolation)."""
        dt = max(float(dt), 0.0)
        return self.get_position() + self.get_velocity() * dt