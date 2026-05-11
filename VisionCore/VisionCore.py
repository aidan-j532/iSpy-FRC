import importlib.metadata
from pathlib import Path
from VisionCore.utilities.MultipleCameraHandler import MultipleCameraHandler
import time
from VisionCore.web.CameraApp import CameraApp
import threading
import logging
import os
import numpy as np
from VisionCore.web.Metrics import Metrics
from VisionCore.web.healthReporter import HealthReporter
from VisionCore.config.VisionCoreConfig import VisionCoreConfig
import signal
from VisionCore.vision.ObjectDetectionCamera import ObjectDetectionCamera
from VisionCore.trackers.Fuel import Fuel
from VisionCore.validations.model_validator import (
    enforce_model_organization,
    validate_model_organization,
)

try:
    from rknnlite.api import RKNNLite
    RKNN_FOUND = True
except ImportError:
    RKNN_FOUND = False

class VisionCore:
    def __init__(self, cameras: list[ObjectDetectionCamera], config: VisionCoreConfig):
        self.cameras = cameras
        self.config  = config
        self.shutdown_event = threading.Event()

        os.makedirs("Outputs", exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, config.get("log_level") or "INFO", logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filemode="w",
            filename=config.get("log_file") or "Outputs/log.txt",
        )
        self.logger = logging.getLogger(__name__)

        signal.signal(signal.SIGINT,  lambda *_: self.shutdown_event.set())
        signal.signal(signal.SIGTERM, lambda *_: self.shutdown_event.set())

        self.metrics = Metrics() if config["metrics"] else None

        self.camera_app = CameraApp(cameras=cameras, config=config) if config["app_mode"] else None
        self.health     = HealthReporter(self.camera_app.app, config) if config["app_mode"] else None

        if len(cameras) == 0:
            self.logger.warning("No cameras provided — vision will not run.")
            self.camera_handler = None
        elif len(cameras) == 1:
            self.logger.info("Single camera mode.")
            self.camera_handler = None
        else:
            self.logger.info("%d cameras — multi mode.", len(cameras))
            self.camera_handler = MultipleCameraHandler(cameras)

        self.trackers = {}

        # Load trackers
        tracker_entries = importlib.metadata.entry_points(group='visioncore_trackers')
        ep_map = {ep.name: ep for ep in tracker_entries}  # build once
        for tracker_name in config.get('trackers', []):
            if tracker_name in ep_map:
                self.trackers[tracker_name] = ep_map[tracker_name].load()(config)

        self.utilities = {}

        # Load utilities
        utility_entries = importlib.metadata.entry_points(group='visioncore_utilities')
        util_ep_map = {ep.name: ep for ep in utility_entries}
        for util_name in config.get('utilities', []):
            if util_name in util_ep_map:
                util_class = util_ep_map[util_name].load()
                if util_name == 'network_table':
                    self.utilities[util_name] = util_class(config["network_tables_ip"]) if config["use_network_tables"] else None
                elif util_name == 'video_recorder':
                    self.utilities[util_name] = util_class(output_dir="VideoRecordings") if config["record_mode"] else None
                else:
                    self.utilities[util_name] = util_class(config)
            else:
                self.logger.warning(f"Unknown utility: {util_name}")

        # Assign common ones
        self.planner = self.trackers.get('path_planner')
        self.fuel_tracker = self.trackers.get('fuel')
        self.recorder = self.utilities.get('video_recorder')
        self.network_handler = self.utilities.get('network_table')

        if config["app_mode"]:
            threading.Thread(target=self.camera_app.run, daemon=True).start()
            if self.health and cameras:
                self.health.set_camera(cameras[0])
            if self.network_handler and self.health:
                self.health.set_network_handler(self.network_handler)

    def get_default_config(self):
        return self.config.get_default_config()

    def validate_vision_model(self, repo_root: Path | None = None) -> tuple[bool, str | None]:
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[1]

        vision_model = self.config.config.get("vision_model", {})
        configured_path = vision_model.get("file_path", "")

        validation_result = validate_model_organization(repo_root)
        if validation_result.orphan_models:
            self.logger.warning("Found orphan vision model files not in organized structure:")
            for orphan_path, orphan_reason in validation_result.orphan_models.items():
                self.logger.warning("  %s: %s", orphan_path, orphan_reason)

        is_valid, corrected_model_path = enforce_model_organization(repo_root, self.config.config)
        
        if not is_valid:
            self.logger.warning(
                "Vision model validation failed. Relying on user-provided settings."
            )
            self.logger.warning("configured: %s", configured_path)

        return is_valid, corrected_model_path

    def _record_metrics(self, **kwargs):
        if self.metrics:
            self.metrics.record(**kwargs)

    def _tick_metrics(self):
        if self.metrics:
            self.metrics.tick()

    def _destroy_metrics(self):
        if self.metrics:
            self.metrics.destroy()

    def numpy_to_fuel_list(self, positions: np.ndarray) -> list[Fuel]:
        return [Fuel(float(p[0]), float(p[1])) for p in positions]

    def run_multi_vision(self, handler: MultipleCameraHandler):
        try:
            raw = handler.predict()
            return self.numpy_to_fuel_list(raw), handler.get_combined_frame()
        except Exception:
            self.logger.exception("Multi-vision exception")
            return [], None

    def run_solo_vision(self, camera: ObjectDetectionCamera):
        try:
            raw, frame = camera.run()
            return self.numpy_to_fuel_list(raw), frame
        except Exception:
            self.logger.exception("Solo-vision exception")
            return [], None

    def run(self, duration_s: float | None = None):
        if not self.cameras:
            self.logger.error("No cameras provided.")
            return

        if duration_s is not None:
            def _stop():
                time.sleep(duration_s)
                self.logger.info("Duration %.1fs reached — stopping.", duration_s)
                self.shutdown_event.set()
            threading.Thread(target=_stop, daemon=True).start()

        if len(self.cameras) == 1:
            self.run_solo_mode()
        else:
            self.run_multi_mode()

    def run_solo_mode(self):
        camera = self.cameras[0]
        try:
            if self.recorder:
                self.recorder.start(camera.input_size[0], camera.input_size[1])

            self.logger.info("Solo mode — warming up…")
            self.run_solo_vision(camera)
            self.logger.info("Warm-up complete.")

            while not self.shutdown_event.is_set():
                t0 = time.perf_counter()

                camera_lag_s = camera.get_frame_age()

                t_vis = time.perf_counter()
                fuel_list, annotated_frame = self.run_solo_vision(camera)
                vision_s = time.perf_counter() - t_vis

                if self.network_handler:
                    pose = self.network_handler.get_robot_pose()
                    fuel_list = self.fuel_tracker.update(
                        fuel_list, pose.X(), pose.Y(), pose.rotation().radians()
                    )
                else:
                    fuel_list = self.fuel_tracker.update(fuel_list, 0, 0, 0)

                flask_s = None
                if self.camera_app and annotated_frame is not None:
                    t_f = time.perf_counter()
                    cam_name = (camera.config.get("name", "Camera 1")
                                if hasattr(camera, "config") else "Camera 1")
                    self.camera_app.set_frame(annotated_frame, camera_name=cam_name)
                    self.camera_app.set_frame(annotated_frame)
                    flask_s = time.perf_counter() - t_f

                if self.recorder and annotated_frame is not None:
                    self.recorder.write(annotated_frame)

                loop_s = time.perf_counter() - t0

                if not fuel_list:
                    self._record_metrics(loop_s=loop_s, vision_s=vision_s,
                                         camera_lag_s=camera_lag_s, flask_s=flask_s)
                    self._tick_metrics()
                    print(f"\rFPS: {1/loop_s:.1f} (no detections)   ", end="")
                    continue

                _, fuel_list = self.planner.update_fuel_positions(fuel_list)

                # Process with custom trackers
                for tracker_name, tracker in self.trackers.items():
                    if hasattr(tracker, 'process_detections'):
                        try:
                            tracker.process_detections(fuel_list)
                        except Exception as e:
                            self.logger.exception(f"Error in tracker {tracker_name}: {e}")

                network_s = None
                if self.network_handler:
                    t_n = time.perf_counter()
                    self.network_handler.send_fuel_list(fuel_list, "vision_data", "VisionData")
                    self.network_handler.send_data(1 / loop_s if loop_s > 0 else 0, "fps", "VisionData")
                    self.network_handler.send_data(len(fuel_list), "num_detections", "VisionData")
                    self.network_handler.send_data(camera_lag_s, "camera_lag", "VisionData")

                    hopper = camera.get_data_for_subsystem("hopper")
                    if hopper is not None:
                        self.network_handler.send_boolean(hopper, "hopper_sees_object", "VisionData")
                    network_s = time.perf_counter() - t_n

                loop_s = time.perf_counter() - t0

                health_s = None
                if self.health:
                    t_h = time.perf_counter()
                    self.health.tick(fps=1 / loop_s if loop_s > 0 else 0,
                                     vision_s=vision_s, detections=len(fuel_list))
                    health_s = time.perf_counter() - t_h

                self._record_metrics(loop_s=loop_s, vision_s=vision_s,
                                     camera_lag_s=camera_lag_s, flask_s=flask_s,
                                     network_s=network_s, health_s=health_s)
                self._tick_metrics()
                self.logger.debug("FPS: %.1f", 1 / loop_s)
                print(f"\rFPS: {1/loop_s:.1f}   ", end="")

        finally:
            camera.destroy()
            self._destroy_metrics()

    def run_multi_mode(self):
        handler = self.camera_handler
        if handler is None:
            self.logger.error("Multi-camera mode requested but camera handler failed to initialize.")
            return
        try:
            if self.recorder:
                h, w = handler.cameras[0].input_size[1], handler.cameras[0].input_size[0]
                self.recorder.start(w, h)

            self.logger.info("Multi mode. warming up…")
            self.run_multi_vision(handler)
            self.logger.info("Warm-up complete.")

            while not self.shutdown_event.is_set():
                t0 = time.perf_counter()

                ages = [cam.get_frame_age() for cam in handler.cameras]
                camera_lag_s = sum(ages) / len(ages) if ages else 0.0

                t_vis = time.perf_counter()
                fuel_list, combined_frame = self.run_multi_vision(handler)
                vision_s = time.perf_counter() - t_vis

                if self.network_handler:
                    pose = self.network_handler.get_robot_pose()
                    fuel_list = self.fuel_tracker.update(
                        fuel_list, pose.X(), pose.Y(), pose.rotation().radians()
                    )
                else:
                    fuel_list = self.fuel_tracker.update(fuel_list, 0, 0, 0)

                flask_s = None
                if self.camera_app and combined_frame is not None:
                    t_f = time.perf_counter()
                    # Set the combined frame (default feed) …
                    self.camera_app.set_frame(combined_frame)
                    # and set per-camera frames from the already-computed cache
                    # (MultipleCameraHandler stores the last frame per camera).
                    for i, cam in enumerate(handler.cameras):
                        cam_name = (cam.config.get("name", f"Camera {i+1}")
                                    if hasattr(cam, "config") else f"Camera {i+1}")
                        with handler._locks[i]:
                            cached_frame = handler._frames[i]
                        if cached_frame is not None:
                            self.camera_app.set_frame(cached_frame.copy(), camera_name=cam_name)
                    flask_s = time.perf_counter() - t_f

                loop_s = time.perf_counter() - t0

                if not fuel_list:
                    self._record_metrics(loop_s=loop_s, vision_s=vision_s,
                                         camera_lag_s=camera_lag_s, flask_s=flask_s)
                    self._tick_metrics()
                    print(f"\rFPS: {1/loop_s:.1f} (no detections)   ", end="")
                    continue

                _, fuel_list = self.planner.update_fuel_positions(fuel_list)

                # Process with custom trackers
                for tracker_name, tracker in self.trackers.items():
                    if hasattr(tracker, 'process_detections'):
                        try:
                            tracker.process_detections(fuel_list)
                        except Exception as e:
                            self.logger.exception(f"Error in tracker {tracker_name}: {e}")

                network_s = None
                if self.network_handler:
                    t_n = time.perf_counter()
                    self.network_handler.send_fuel_list(fuel_list, "vision_data", "VisionData")
                    self.network_handler.send_data(1 / loop_s if loop_s > 0 else 0, "fps", "VisionData")
                    self.network_handler.send_data(len(fuel_list), "num_detections", "VisionData")
                    self.network_handler.send_data(camera_lag_s, "camera_lag", "VisionData")

                    for cam in handler.cameras:
                        hopper = cam.get_data_for_subsystem("hopper")
                        if hopper is not None:
                            self.network_handler.send_boolean(hopper, "hopper_sees_object", "VisionData")
                    network_s = time.perf_counter() - t_n

                loop_s = time.perf_counter() - t0

                health_s = None
                if self.health:
                    t_h = time.perf_counter()
                    self.health.tick(fps=1 / loop_s if loop_s > 0 else 0,
                                     vision_s=vision_s, detections=len(fuel_list))
                    health_s = time.perf_counter() - t_h

                self._record_metrics(loop_s=loop_s, vision_s=vision_s,
                                     camera_lag_s=camera_lag_s, flask_s=flask_s,
                                     network_s=network_s, health_s=health_s)
                self._tick_metrics()
                self.logger.debug("FPS: %.1f", 1 / loop_s)
                print(f"\rFPS: {1/loop_s:.1f}   ", end="")
        finally:
            handler.destroy()
            self._destroy_metrics()
