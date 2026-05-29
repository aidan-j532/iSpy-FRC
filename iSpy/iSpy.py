from pathlib import Path
from iSpy.utilities.MultipleCameraHandler import MultipleCameraHandler
import time
from iSpy.web.CameraApp import CameraApp
import threading
import logging
import os
import numpy as np
from iSpy.web.Metrics import Metrics
from iSpy.config.iSpyConfig import iSpyConfig
import signal
from iSpy.vision.ObjectDetectionCamera import ObjectDetectionCamera
from iSpy.vision.Object import Object
from iSpy.validations.model_validator import (
    enforce_model_organization,
    validate_model_organization,
)
from iSpy.plugins._loader import load_plugins
from iSpy.plugins.bases import TrackerBase, UtilityBase
from wpimath.geometry import Pose2d

try:
    from rknnlite.api import RKNNLite

    RKNN_FOUND = True
except ImportError:
    RKNN_FOUND = False

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()

while not (PROJECT_ROOT / "plugins").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

_PLUGIN_ROOT = PROJECT_ROOT / "plugins"

class iSpy:
    def __init__(self, cameras: list[ObjectDetectionCamera], config: iSpyConfig):
        self.cameras = cameras
        self.config = config

        self.shutdown_event = threading.Event()
        os.makedirs("Outputs", exist_ok=True)
        self.logger = logging.getLogger(__name__)

        signal.signal(signal.SIGINT, lambda *_: self._handle_shutdown())
        signal.signal(signal.SIGTERM, lambda *_: self._handle_shutdown())

        self.metrics = Metrics() if config["metrics"] else None

        self.camera_app = (
            CameraApp(cameras=cameras, config=config) if config["app_mode"] else None
        )

        if len(cameras) == 0:
            self.logger.warning("No cameras provided - vision will not run.")
            self.camera_handler = None
        elif len(cameras) == 1:
            self.logger.info("Single camera mode.")
            self.camera_handler = None
        else:
            self.logger.info("%d cameras - multi mode.", len(cameras))
            self.camera_handler = MultipleCameraHandler(cameras)

        tracker_classes = load_plugins(_PLUGIN_ROOT / "trackers", TrackerBase)
        self.trackers = {}
        for name in config.get_nested("plugins", "trackers", default=[]):
            if name in tracker_classes:
                self.trackers[name] = tracker_classes[name](config)
            else:
                self.logger.warning("Unknown tracker: %s", name)

        # Grab the two built-in trackers by name for use in the loop
        self._fuel_tracker = self.trackers.get("fuel")
        self._detection_cleanup = self.trackers.get("path_planner")

        context = {
            "config": config,
            "camera_app": self.camera_app,
            "cameras": self.cameras,
            "flask_app": self.camera_app.app if self.camera_app else None,
        }

        utility_classes = load_plugins(_PLUGIN_ROOT / "utilities", UtilityBase)
        self.utilities = {}
        for name in config.get_nested("plugins", "utilities", default=[]):
            if name in utility_classes:
                try:
                    self.utilities[name] = utility_classes[name](context)
                except Exception:
                    self.logger.exception("Failed to initialize utility: %s", name)
            else:
                self.logger.warning("Unknown utility: %s", name)

        # Wire health reporter to network handler if both exist
        health = self.utilities.get("health_reporter")
        nt = self.utilities.get("network_table")
        if health and nt:
            health.set_network_handler(nt)

        logging.getLogger("werkzeug").setLevel(logging.WARNING)

        if config["app_mode"]:
            threading.Thread(target=self.camera_app.run, daemon=True).start()

        self._silence_external_loggers()

    def _silence_external_loggers(self):
        for name in logging.root.manager.loggerDict:
            if not name.startswith("iSpy"):
                logging.getLogger(name).setLevel(logging.WARNING)

    def _handle_shutdown(self):
        if self.shutdown_event.is_set():
            return
        self.logger.info("Shutdown signal received.")
        self.shutdown_event.set()

    def _stop_all_plugins(self):
        for name, plugin in {**self.trackers, **self.utilities}.items():
            if plugin is None:
                continue
            if hasattr(plugin, "stop"):
                try:
                    plugin.stop()
                except Exception:
                    self.logger.exception("Error stopping plugin '%s'", name)

    def _record_metrics(self, **kwargs):
        if self.metrics:
            self.metrics.record(**kwargs)

    def _tick_metrics(self):
        if self.metrics:
            self.metrics.tick()

    def _destroy_metrics(self):
        if self.metrics:
            self.metrics.destroy()

    def _get_pose(self) -> Pose2d:
        for util in self.utilities.values():
            pose = util.get_robot_pose()
            if pose is not None:
                return pose
        return Pose2d()

    def _update_utilities(self, frame_data: dict):
        for util in self.utilities.values():
            try:
                util.update(frame_data)
            except Exception:
                self.logger.exception("Utility update failed")

    def _update_camera_app(self, frame, camera=None, handler=None):
        if not self.camera_app or frame is None:
            return
        self.camera_app.set_frame(frame)
        if camera:
            cam_name = (
                camera.config.get("name", "Camera 1")
                if hasattr(camera, "config")
                else "Camera 1"
            )
            self.camera_app.set_frame(frame, camera_name=cam_name)
        if handler:
            for i, cam in enumerate(handler.cameras):
                cam_name = (
                    cam.config.get("name", f"Camera {i+1}")
                    if hasattr(cam, "config")
                    else f"Camera {i+1}"
                )
                with handler._locks[i]:
                    cached = handler._frames[i]
                if cached is not None:
                    self.camera_app.set_frame(cached.copy(), camera_name=cam_name)

    def run_multi_vision(self, handler):
        try:
            objects = handler.predict()
            return objects, handler.get_combined_frame()
        except Exception:
            self.logger.exception("Multi-vision exception")
            return [], None

    def run_solo_vision(self, camera):
        try:
            objects, frame = camera.run()
            return objects, frame
        except Exception:
            self.logger.exception("Solo-vision exception")
            return [], None

    def validate_vision_model(self, repo_root: Path | None = None):
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[1]
        validation_result = validate_model_organization(repo_root)
        if validation_result.orphan_models:
            for p, r in validation_result.orphan_models.items():
                self.logger.warning("Orphan model %s: %s", p, r)
        return enforce_model_organization(repo_root, self.config.config)

    def get_default_config(self):
        return self.config.get_default_config()

    def run(self, duration_s: float | None = None):
        if not self.cameras:
            self.logger.error("No cameras provided.")
            return
        if duration_s is not None:

            def _stop():
                time.sleep(duration_s)
                self._handle_shutdown()

            threading.Thread(target=_stop, daemon=True).start()

        if len(self.cameras) == 1:
            self.run_solo_mode()
        else:
            self.run_multi_mode()

    def _run_loop_body_solo(self, camera) -> dict:
        t0 = time.perf_counter()
        camera_lag_s = camera.get_frame_age()

        t_vis = time.perf_counter()
        fuel_list, frame = self.run_solo_vision(camera)
        vision_s = time.perf_counter() - t_vis

        pose = self._get_pose()
        fuel_list = (
            self._fuel_tracker.update(
                fuel_list, pose.X(), pose.Y(), pose.rotation().radians()
            )
            if self._fuel_tracker
            else fuel_list
        )

        self._update_camera_app(frame, camera=camera)

        if self._detection_cleanup and fuel_list:
            _, fuel_list = self._detection_cleanup.update(fuel_list, pose.X(), pose.Y(), pose.rotation().radians())

        loop_s = time.perf_counter() - t0

        return {
            "fuel_list": fuel_list,
            "frame": frame,
            "fps": 1 / loop_s if loop_s > 0 else 0,
            "loop_s": loop_s,
            "vision_s": vision_s,
            "camera_lag_s": camera_lag_s,
            "detections": len(fuel_list),
            "cameras": self.cameras,
        }

    def _run_loop_body_multi(self, handler) -> dict:
        t0 = time.perf_counter()
        ages = [cam.get_frame_age() for cam in handler.cameras]
        camera_lag_s = sum(ages) / len(ages) if ages else 0.0

        t_vis = time.perf_counter()
        fuel_list, frame = self.run_multi_vision(handler)
        vision_s = time.perf_counter() - t_vis

        pose = self._get_pose()
        fuel_list = (
            self._fuel_tracker.update(
                fuel_list, pose.X(), pose.Y(), pose.rotation().radians()
            )
            if self._fuel_tracker
            else fuel_list
        )

        self._update_camera_app(frame, handler=handler)

        if self._detection_cleanup and fuel_list:
            _, fuel_list = self._detection_cleanup.update(fuel_list, pose.X(), pose.Y(), pose.rotation().radians())

        loop_s = time.perf_counter() - t0

        return {
            "fuel_list": fuel_list,
            "frame": frame,
            "fps": 1 / loop_s if loop_s > 0 else 0,
            "loop_s": loop_s,
            "vision_s": vision_s,
            "camera_lag_s": camera_lag_s,
            "detections": len(fuel_list),
            "cameras": self.cameras,
        }

    def run_solo_mode(self):
        # Tell them where to lood for web stuff
        self.logger.info("Check out the web interface at http://localhost:5000/")
        camera = self.cameras[0]
        try:
            self.logger.info("Solo mode - warming up...")
            self.run_solo_vision(camera)
            self.logger.info("Warm-up complete.")

            while not self.shutdown_event.is_set():
                frame_data = self._run_loop_body_solo(camera)
                self._update_utilities(frame_data)
                self._record_metrics(
                    loop_s=frame_data["loop_s"],
                    vision_s=frame_data["vision_s"],
                    camera_lag_s=frame_data["camera_lag_s"],
                )
                self._tick_metrics()
                print(f"\rFPS: {frame_data['fps']:.1f}   ", end="")

        finally:
            print()
            self._stop_all_plugins()
            camera.destroy()
            self._destroy_metrics()

    def run_multi_mode(self):
        handler = self.camera_handler
        if handler is None:
            self.logger.error("Multi-camera handler not initialized.")
            return
        try:
            self.logger.info("Multi mode - warming up...")
            self.run_multi_vision(handler)
            self.logger.info("Warm-up complete.")

            while not self.shutdown_event.is_set():
                frame_data = self._run_loop_body_multi(handler)
                self._update_utilities(frame_data)
                self._record_metrics(
                    loop_s=frame_data["loop_s"],
                    vision_s=frame_data["vision_s"],
                    camera_lag_s=frame_data["camera_lag_s"],
                )
                self._tick_metrics()
                print(f"\rFPS: {frame_data['fps']:.1f}   ", end="")

        finally:
            print()
            self._stop_all_plugins()
            handler.destroy()
            self._destroy_metrics()
