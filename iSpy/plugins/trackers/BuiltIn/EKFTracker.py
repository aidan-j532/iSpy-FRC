"""Kalman-filtered object tracker add-on.

A drop-in alternative to ``ObjectTracker``: it merges detections into an
existing track the same way (distance + same-class gating), but smooths the
track's position with a constant-velocity Kalman filter instead of an EMA.
Only *one* tracker add-on is enabled per camera, so this never conflicts with
``ObjectTracker``. The filter math lives in ``KalmanTrack``; this class is the
plugin / lifecycle wrapper.
"""

import logging
import time

import numpy as np

from iSpy.plugins.bases import TrackerBase
from iSpy.vision.Object import Object
from iSpy.algorithms.KalmanTrack import KalmanTrack

_EMA_ALPHA = 0.3


class EKFTracker(TrackerBase):
    plugin_name = "ekf_tracker"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "process_noise": {
                "type": "number",
                "label": "Process Noise",
                "hint": "Trust the motion model less (higher = track follows "
                        "measurements faster, more responsive but noisier).",
                "default": 0.5,
            },
            "measurement_noise": {
                "type": "number",
                "label": "Measurement Noise",
                "hint": "Trust each detection less (higher = smoother, more "
                        "laggy; lower = closer to raw detections).",
                "default": 0.1,
            },
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
        # track id -> (KalmanTrack, last update monotonic time)
        self._filters: dict[int, tuple[KalmanTrack, float]] = {}

        raw_threshold = self.config.get("distance_threshold", 0.5)
        if raw_threshold is None or raw_threshold < 0:
            self.distance_threshold = 0.5
            self.logger.warning(
                "distance_threshold invalid or missing, defaulting to 0.5"
            )
        else:
            self.distance_threshold = float(raw_threshold)

        raw_process = self.config.get("process_noise", 0.5)
        raw_measurement = self.config.get("measurement_noise", 0.1)
        self.process_noise = float(raw_process if raw_process is not None else 0.5)
        self.measurement_noise = float(
            raw_measurement if raw_measurement is not None else 0.1
        )
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

        # merge new detections into existing tracks (or spawn new ones)
        self._merge(new_detections)

        # velocity-extrapolate any track that missed a detection this tick
        self._extrapolate_missing()

        return self.tracked_objects

    def _merge(self, detections: list[Object]):
        for det in detections:
            if not self._exists_and_update(det):
                det.alive_time = self.stale_threshold
                self.tracked_objects.append(det)
                now = time.monotonic()
                self._filters[det.id] = (
                    KalmanTrack(
                        det.get_position(),
                        process_noise=self.process_noise,
                        measurement_noise=self.measurement_noise,
                    ),
                    now,
                )

    def _exists_and_update(self, new_det: Object) -> bool:
        if not self.tracked_objects:
            return False

        new_pos = np.array(new_det.get_position())

        for existing in self.tracked_objects:
            existing_pos = np.array(existing.get_position())

            if np.isnan(new_pos).any() or np.isnan(existing_pos).any():
                continue

            if np.linalg.norm(new_pos - existing_pos) < self.distance_threshold:
                # distance alone is not enough - a cone 30cm from a robot
                # must never merge into the robot. only merge detections of
                # the same class/name (empty names on both sides are treated
                # as equal for backwards compatibility)
                if not self._same_identity(new_det, existing):
                    continue

                # reset the Object's stale timer
                existing.reset_time()

                # Kalman predict+update on the merged track's position using
                # the elapsed wall-clock time since its last observation
                now = time.monotonic()
                filt, last_t = self._filters.get(existing.id, (None, None))
                if filt is None:
                    filt = KalmanTrack(
                        existing.get_position(),
                        process_noise=self.process_noise,
                        measurement_noise=self.measurement_noise,
                    )
                dt = (now - last_t) if last_t is not None else 0.0
                smoothed = filt.predict_measure(dt, new_pos)
                self._filters[existing.id] = (filt, now)

                existing.x = float(smoothed[0])
                existing.y = float(smoothed[1])
                existing.z = float(smoothed[2])

                # EMA smoothing on angles (shortest-path wrapping) - angles
                # are not the point of this feature, keep ObjectTracker's cue
                d_roll = (new_det.roll - existing.roll + np.pi) % (2 * np.pi) - np.pi
                existing.roll = (existing.roll + _EMA_ALPHA * d_roll) % (2 * np.pi)

                d_pitch = (new_det.pitch - existing.pitch + np.pi) % (2 * np.pi) - np.pi
                existing.pitch = (existing.pitch + _EMA_ALPHA * d_pitch) % (2 * np.pi)

                d_yaw = (new_det.yaw - existing.yaw + np.pi) % (2 * np.pi) - np.pi
                existing.yaw = (existing.yaw + _EMA_ALPHA * d_yaw) % (2 * np.pi)

                return True

        return False

    def _extrapolate_missing(self):
        """Velocity-extrapolate tracks that saw no measurement this tick.

        Only nudges the Object's position while it is still within the stale
        threshold; the Kalman filter itself is NOT advanced here so a later
        re-observation re-syncs with the measured position cleanly.
        """
        now = time.monotonic()
        for obj in self.tracked_objects:
            entry = self._filters.get(obj.id)
            if entry is None:
                continue
            filt, last_t = entry
            dt = now - last_t if last_t is not None else 0.0
            if dt <= 0.0:
                continue
            forecast = filt.extrapolate(dt)
            obj.x = float(forecast[0])
            obj.y = float(forecast[1])
            obj.z = float(forecast[2])

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
        self._filters.clear()