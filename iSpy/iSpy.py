from pathlib import Path
from iSpy.utilities.MultipleCameraHandler import MultipleCameraHandler
import time
import threading
import logging
import os
from iSpy.config.iSpyConfig import iSpyConfig
import signal
from iSpy.vision.pipelines.base import VisionPipeline
from iSpy.validations.model_validator import (
    enforce_model_organization,
    validate_model_organization,
)
from iSpy.plugins._loader import load_plugins
from iSpy.plugins.bases import TrackerBase, UtilityBase, FrameProcessorBase
from iSpy.config.iSpyConfig import iSpyAddonConfig
from wpimath.geometry import Pose2d
from iSpy.web.Backend.WebApp import create_app


PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "plugins").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Could not locate 'plugins' directory above iSpy.py")
    PROJECT_ROOT = PROJECT_ROOT.parent

_PLUGIN_ROOT = PROJECT_ROOT / "plugins"

class iSpy:
    def __init__(self, cameras: list[VisionPipeline], config: iSpyConfig, web_app=None):
        self.cameras = cameras
        self.config = config

        self.shutdown_event = threading.Event()
        self.pause_event = threading.Event()
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
            self.camera_handler = MultipleCameraHandler(cameras, config)
        self.web_app = web_app
        if self.web_app is None and config["app_mode"]:
            self.web_app = create_app(cameras=cameras, config=config)
        if self.web_app is not None:
            # The dashboard may have booted before the camera pipelines
            # existed (game_loop boots it first so it is reachable during
            # the slow camera/model initialization); hand them over now.
            self.web_app.set_cameras(cameras)

        # Shared context handed to every add-on; each add-on's constructor
        # receives its OWN view of its settings via iSpy._addon_context.
        self._base_context = {
            "config": config,
            "global_config": config,
            "cameras": self.cameras,
            "flask_app": self.web_app.flask_app if self.web_app else None,
            # HealthModule/PluginStatusModule read this lazily per-request,
            # so it's fine that it's set for real a few lines down.
            "vision_instance": self,
        }

        tracker_classes = load_plugins(_PLUGIN_ROOT / "trackers", TrackerBase)
        self.trackers = {}  # No default trackers
        for name, settings in self._enabled_addons("trackers"):
            if name in tracker_classes:
                self.trackers[name] = tracker_classes[name](
                    self._addon_context(tracker_classes[name], settings)
                )
            else:
                self.logger.warning("Unknown tracker: %s", name)

        utility_classes = load_plugins(_PLUGIN_ROOT / "utilities", UtilityBase)
        self.utilities = {}

        for name, settings in self._enabled_addons("utilities"):
            if name in utility_classes:
                try:
                    self.utilities[name] = utility_classes[name](
                        self._addon_context(utility_classes[name], settings)
                    )
                except Exception:
                    self.logger.exception("Failed to initialize utility plugin: %s", name)
            else:
                self.logger.warning("Unknown utility plugin: %s", name)

        frame_processor_classes = load_plugins(_PLUGIN_ROOT / "frame_processors", FrameProcessorBase)
        self.frame_processors = {}
        for name, settings in self._enabled_addons("frame_processors"):
            if name in frame_processor_classes:
                try:
                    self.frame_processors[name] = frame_processor_classes[name](
                        self._addon_context(frame_processor_classes[name], settings)
                    )
                except Exception:
                    self.logger.exception("Failed to initialize frame processor: %s", name)
            else:
                self.logger.warning("Unknown frame processor: %s", name)

        # Wire the NetworkTables handler (if the user enabled that plugin)
        # into the merged HealthModule so /api/health can report NT status.
        nt = self.utilities.get("network_table_handler")
        health_mod = self.web_app.modules.get("health") if self.web_app else None
        if health_mod and nt:
            health_mod.set_network_handler(nt)

        logging.getLogger("werkzeug").setLevel(logging.WARNING)

        if self.web_app:
            if web_app is None:
                # Only start the server when we created it here; a pre-booted
                # app was already serving before the cameras finished loading.
                threading.Thread(target=self.web_app.run, daemon=True).start()
            dash = self.web_app.modules.get("dashboard")
            if dash and hasattr(dash, "set_plugins"):
                dash.set_plugins(self.trackers, self.utilities, self.frame_processors)
            self.web_app.set_vision_instance(self)

        self._silence_external_loggers()

        # Make sure cameras get frame processors if any are configured
        if self.frame_processors:
            for camera in self.cameras:
                for name, processor in self.frame_processors.items():
                    camera.add_frame_processor(processor)

        # I think has to be at very bottom
        if self.web_app:
            self.web_app.set_vision_instance(self)

    def _enabled_addons(self, addon_type: str) -> list[tuple[str, dict]]:
        """(name, settings) pairs from plugins.<type>. Presence == enabled, so
        only dict entries are returned; entries with non-dict values are
        treated as enabled with no settings."""
        raw = self.config.get_nested("plugins", addon_type, default={})
        if not isinstance(raw, dict):
            return []
        out = []
        for name, settings in raw.items():
            if not isinstance(name, str):
                continue
            out.append((name, settings if isinstance(settings, dict) else {}))
        return out

    def _addon_context(self, addon_cls, settings: dict) -> dict:
        """Context for one add-on instance: the shared context plus an
        iSpyAddonConfig view of the add-on's own settings (schema defaults
        merged in)."""
        ctx = dict(self._base_context)
        ctx["config"] = iSpyAddonConfig(settings, defaults=addon_cls.default_settings())
        return ctx

    def _silence_external_loggers(self):
        for name in logging.root.manager.loggerDict:
            if not name.startswith("iSpy"):
                logging.getLogger(name).setLevel(logging.WARNING)

    def reload_camera(self, cam_key: str, new_config: dict):
        """No longer supported: camera settings are only applied on the next
        vision start (see CamerasModule._update_camera). Kept as a no-op to
        fail loudly if any stale caller still invokes it."""
        raise RuntimeError(
            "reload_camera was removed - camera settings require a vision "
            "restart to take effect."
        )

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
            # Never kill the loop because one pipeline hiccuped - fall back
            # to the raw camera frame so the feed keeps flowing.
            return [], camera.get_frame() if hasattr(camera, "get_frame") else None

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
        code_times = {}
        camera_lag_s = camera.get_frame_age()

        t_vis = time.perf_counter()
        fuel_list, frame = self.run_solo_vision(camera)
        vision_s = time.perf_counter() - t_vis
        code_times["vision"] = vision_s

        t_pose = time.perf_counter()
        pose = self._get_pose()
        code_times["pose"] = time.perf_counter() - t_pose

        t_track = time.perf_counter()
        for tracker in self.trackers.values():
            # WPILib pose yaw is CCW-positive; Object.relative_to uses the
            # codebase's convention (positive yaw = turned RIGHT), so negate.
            fuel_list = tracker.update(
                fuel_list, pose.X(), pose.Y(), -pose.rotation().radians(), 0.0
            )
        code_times["trackers"] = time.perf_counter() - t_track

        loop_s = time.perf_counter() - t0
        frame_data = {
            "fuel_list": fuel_list, "frame": frame,
            "fps": 1 / loop_s if loop_s > 0 else 0,
            "loop_s": loop_s, "vision_s": vision_s, "camera_lag_s": camera_lag_s,
            "detections": len(fuel_list), "cameras": self.cameras,
            "code_times": code_times,
            "debug_data": {},
            "objects": fuel_list,
        }
        if hasattr(camera, "get_debug_data"):
            frame_data["debug_data"] = camera.get_debug_data() or {}
        if hasattr(camera, "get_debug_frame"):
            debug_frame = camera.get_debug_frame(frame)
            if debug_frame is not None:
                frame_data["debug_frame"] = debug_frame
        if hasattr(camera, "plot"):
            plotted_frame = camera.plot(frame)
            if plotted_frame is not None:
                frame_data["frame"] = plotted_frame
                frame_data["debug_frame"] = plotted_frame
        return frame_data

    def _run_loop_body_multi(self, handler) -> dict:
        t0 = time.perf_counter()
        code_times = {}
        ages = [cam.get_frame_age() for cam in handler.cameras]
        camera_lag_s = sum(ages) / len(ages) if ages else 0.0

        t_vis = time.perf_counter()
        fuel_list, frame = self.run_multi_vision(handler)
        vision_s = time.perf_counter() - t_vis
        code_times["vision"] = vision_s

        t_pose = time.perf_counter()
        pose = self._get_pose()
        code_times["pose"] = time.perf_counter() - t_pose

        t_track = time.perf_counter()
        for tracker in self.trackers.values():
            # WPILib pose yaw is CCW-positive; Object.relative_to uses the
            # codebase's convention (positive yaw = turned RIGHT), so negate.
            fuel_list = tracker.update(
                fuel_list, pose.X(), pose.Y(), -pose.rotation().radians(), 0.0
            )
        code_times["trackers"] = time.perf_counter() - t_track

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
            "code_times": code_times,
            "debug_data": {},
            "objects": fuel_list,
        }
        if hasattr(handler, "get_camera_debug_data"):
            frame_data["debug_data"] = handler.get_camera_debug_data() or {}
        if hasattr(handler, "get_camera_debug_frame"):
            debug_frame = handler.get_camera_debug_frame(frame)
            if debug_frame is not None:
                frame_data["debug_frame"] = debug_frame

        return frame_data

    def run_solo_mode(self):
        max_fps = self.config.get("max_fps", 0)
        last_frame_data = None
        try:
            camera = self.cameras[0]
            self.run_solo_vision(camera)
            while not self.shutdown_event.is_set():
                camera = self.cameras[0]
                if self.pause_event.is_set():
                    if last_frame_data is not None:
                        frozen = {**last_frame_data, "fps": 0}
                        self._update_utilities(frozen)
                        self._update_web(frozen)
                    time.sleep(0.05)
                    continue

                t_start = time.perf_counter()
                frame_data = self._run_loop_body_solo(camera)
                last_frame_data = frame_data

                t_util = time.perf_counter()
                self._update_utilities(frame_data)
                frame_data["code_times"]["utilities"] = time.perf_counter() - t_util

                t_web = time.perf_counter()
                self._update_web(frame_data)
                frame_data["code_times"]["web"] = time.perf_counter() - t_web

                if max_fps > 0:
                    elapsed = time.perf_counter() - t_start
                    sleep_time = max(0, 1.0 / max_fps - elapsed)
                    time.sleep(sleep_time)

                actual_fps = 1.0 / max(time.perf_counter() - t_start, 1e-6)
                print(f"\rFPS: {actual_fps:.1f}   ", end="")

        finally:
            print()
            self._stop_all_plugins()
            if self.web_app:
                self.web_app.stop()
            if self.cameras:
                self.cameras[0].destroy()

    def run_multi_mode(self):
        handler = self.camera_handler
        if handler is None:
            self.logger.error("Multi-camera handler not initialized.")
            return
        max_fps = self.config.get("max_fps", 0)
        try:
            self.logger.info("Multi mode - warming up...")
            self.run_multi_vision(handler)
            self.logger.info("Warm-up complete.")

            last_frame_data = None
            while not self.shutdown_event.is_set():
                if self.pause_event.is_set():
                    if last_frame_data is not None:
                        frozen = {**last_frame_data, "fps": 0}
                        self._update_utilities(frozen)
                        self._update_web(frozen)
                    time.sleep(0.05)
                    continue

                t_start = time.perf_counter()
                frame_data = self._run_loop_body_multi(handler)
                last_frame_data = frame_data

                t_util = time.perf_counter()
                self._update_utilities(frame_data)
                frame_data["code_times"]["utilities"] = time.perf_counter() - t_util

                t_web = time.perf_counter()
                self._update_web(frame_data)
                frame_data["code_times"]["web"] = time.perf_counter() - t_web

                if max_fps > 0:
                    elapsed = time.perf_counter() - t_start
                    sleep_time = max(0, 1.0 / max_fps - elapsed)
                    time.sleep(sleep_time)

                actual_fps = 1.0 / max(time.perf_counter() - t_start, 1e-6)
                print(f"\rFPS: {actual_fps:.1f}   ", end="")

        finally:
            print()
            self._stop_all_plugins()
            if self.web_app:
                self.web_app.stop()
            handler.destroy()