from pathlib import Path
from iSpy.utilities.MultipleCameraHandler import MultipleCameraHandler
import time
import threading
import logging
import os
from iSpy.config.iSpyConfig import iSpyConfig
import signal
from iSpy.vision.ObjectDetectionCamera import ObjectDetectionCamera
from iSpy.validations.model_validator import (
    enforce_model_organization,
    validate_model_organization,
)
from iSpy.plugins._loader import load_plugins
from iSpy.plugins.bases import TrackerBase, UtilityBase
from wpimath.geometry import Pose2d
from iSpy.web.Backend.WebApp import create_app
from iSpy.web.Backend.Status import StatusReporter
from iSpy.web.Backend.YOLOHandler import YOLOHandler
from iSpy.web.Backend.DatasetManager import DatasetManager

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
        self.trackers = {} # No default trackers
        for name in config.get_nested("plugins", "trackers", default=[]):
            if name in tracker_classes:
                self.trackers[name] = tracker_classes[name](config)
            else:
                self.logger.warning("Unknown tracker: %s", name)

        self.web_app = None if os.environ.get("ISPY_MANAGED") else (
            create_app(cameras=cameras, config=config) if config["app_mode"] else None
        )
        context = {
            "config": config,
            "cameras": self.cameras,
            "flask_app": self.web_app.flask_app if self.web_app else None,
        }

        utility_classes = load_plugins(_PLUGIN_ROOT / "utilities", UtilityBase)
        self.utilities = {}

        for name, cls in (
            ("status_reporter", StatusReporter),
            ("yolo_handler", YOLOHandler),
            ("dataset_manager", DatasetManager),
        ):
            try:
                self.utilities[name] = cls(context)
            except Exception:
                self.logger.exception("Failed to initialize built-in utility: %s", name)

        frame_processor_classes = load_plugins(_PLUGIN_ROOT / "frame_processors", UtilityBase)
        self.frame_processors = {}
        for name in config.get_nested("plugins", "frame_processors", default=[]):
            if name in frame_processor_classes:
                try:
                    self.frame_processors[name] = frame_processor_classes[name](context)
                    

                except Exception:
                    self.logger.exception("Failed to initialize frame processor: %s", name)
            else:
                self.logger.warning("Unknown frame processor: %s", name)


        # Wire health reporter to network handler if both exist
        health = self.utilities.get("health_reporter")
        nt = self.utilities.get("network_table_handler")
        if health and nt:
            health.set_network_handler(nt)

        logging.getLogger("werkzeug").setLevel(logging.WARNING)

        if self.web_app:
            threading.Thread(target=self.web_app.run, daemon=True).start()

        self._silence_external_loggers()
        
        # Make sure cameras get frame processors if any are configured
        if self.frame_processors:
            for camera in self.cameras:
                for name, processor in self.frame_processors.items():
                    camera.add_frame_processor(processor)

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

    def _get_pose(self) -> Pose2d:
        for util in self.utilities.values():
            if hasattr(util, "get_robot_pose"):
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

    def _update_web(self, frame_data: dict):
        if self.web_app:
            self.web_app.update(frame_data)

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

        for tracker in self.trackers.values():
            fuel_list = tracker.update(fuel_list, pose.X(), pose.Y(), pose.rotation().radians(), 0.0)
        loop_s = time.perf_counter() - t0
        frame_data = {
            "fuel_list": fuel_list, "frame": frame,
            "fps": 1 / loop_s if loop_s > 0 else 0,
            "loop_s": loop_s, "vision_s": vision_s, "camera_lag_s": camera_lag_s,
            "detections": len(fuel_list), "cameras": self.cameras,
        }
        return frame_data


    def _run_loop_body_multi(self, handler) -> dict:
        t0 = time.perf_counter()
        ages = [cam.get_frame_age() for cam in handler.cameras]
        camera_lag_s = sum(ages) / len(ages) if ages else 0.0

        t_vis = time.perf_counter()
        fuel_list, frame = self.run_multi_vision(handler)
        vision_s = time.perf_counter() - t_vis

        pose = self._get_pose()

        for tracker in self.trackers.values():
            fuel_list = tracker.update(
                fuel_list, pose.X(), pose.Y(), pose.rotation().radians(), pose.Z()
            )

        loop_s = time.perf_counter() - t0

        frame_data = {
            "fuel_list": fuel_list,
            "frame": frame,
            "fps": 1 / loop_s if loop_s > 0 else 0,
            "loop_s": loop_s,
            "vision_s": vision_s,
            "camera_lag_s": camera_lag_s,
            "detections": len(fuel_list),
            "cameras": self.cameras,
            "camera_frames": handler.get_camera_frames(),
        }

        return frame_data

    def run_solo_mode(self):
        # Tell them where to look for web stuff
        self.logger.info("Check out the web interface at http://localhost:5000/")
        camera = self.cameras[0]
        try:
            self.logger.info("Solo mode - warming up...")
            self.run_solo_vision(camera)
            self.logger.info("Warm-up complete.")

            while not self.shutdown_event.is_set():
                frame_data = self._run_loop_body_solo(camera)
                self._update_utilities(frame_data)
                self._update_web(frame_data)
                print(f"\rFPS: {frame_data['fps']:.1f}   ", end="")

        finally:
            print()
            self._stop_all_plugins()
            if self.web_app:
                self.web_app.stop()
            camera.destroy()

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
                self._update_web(frame_data)
                print(f"\rFPS: {frame_data['fps']:.1f}   ", end="")

        finally:
            print()
            self._stop_all_plugins()
            if self.web_app:
                self.web_app.stop()
            handler.destroy()
