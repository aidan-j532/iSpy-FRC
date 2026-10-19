import logging
import time
from pathlib import Path
from flask import Flask, jsonify, render_template

from iSpy import __version__
from iSpy.core import performance
from iSpy.web.modules.dashboard import DashboardModule
from iSpy.web.modules.cameras import CamerasModule
from iSpy.web.modules.models import ModelsModule
from iSpy.web.modules.datasets import DatasetsModule
from iSpy.web.modules.viewer3d import Viewer3DModule
from iSpy.web.modules.logs import LogsModule
from iSpy.web.modules.metrics import MetricsModule
from iSpy.web.modules.onboarding import OnboardingModule
from iSpy.web.modules.rollback import RollbackModule
from iSpy.web.Backend.WebModule import WebModule
from iSpy.web.Backend.Settings import SettingsModule
from iSpy.web.Backend.SetupWizard import SetupWizardModule
from iSpy.web.modules.recommendations import RecommendationsModule
from iSpy.web.Backend.PluginStatus import PluginStatusModule
from iSpy.web.modules.health import HealthModule

_WEB_ROOT = Path(__file__).resolve().parent.parent

#: modules that must run on every vision tick regardless of the performance
#: profile: health's staleness check is how /health knows the loop is alive,
#: and the rollback module does no per-tick work at all.
_ALWAYS_UPDATE = {"health", "rollback"}


def _get_version() -> str:
    # The version actually running on this board. Deliberately offline: boards
    # sit on an isolated field LAN, so reaching out to PyPI here would just
    # stall the request until it times out - and it answered the wrong question
    # anyway (latest published release, not what this process is running).
    return __version__


class iSpyWebApp:
    def __init__(self, cameras, config):
        self.logger = logging.getLogger(__name__)
        self.config = config

        self.flask_app = Flask(
            __name__,
            template_folder=str(_WEB_ROOT / "templates"),
            static_folder=str(_WEB_ROOT / "static"),
        )
        # Bound upload/request bodies so an oversized payload (model, dataset
        # images, plugin) can't be buffered unboundedly into process memory.
        # Model weights (.pt) can approach 1 GB, hence the generous ceiling.
        self.flask_app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
        context = {
            "config": config,
            "cameras": cameras or [],
            "flask_app": self.flask_app,
            "vision_instance": None,  # set later via set_vision_instance()
        }

        self.context = context

        self.modules: dict[str, WebModule] = {
            "cameras": CamerasModule(context),
            "models": ModelsModule(context),
            "datasets": DatasetsModule(context),
            "viewer3d": Viewer3DModule(context),
            "dashboard": DashboardModule(context),
            "health": HealthModule(context),
            "logs": LogsModule(context),
            "metrics": MetricsModule(context),
            "rollback": RollbackModule(context),
            "settings": SettingsModule(context),
            "onboarding": OnboardingModule(context),
            "setup_wizard": SetupWizardModule(context),
            "recommendations": RecommendationsModule(context),
            "plugin_status": PluginStatusModule(context),
        }

        for name, mod in self.modules.items():
            try:
                mod.register_routes(self.flask_app)
            except Exception:
                self.logger.exception(
                    "Failed to register routes for web module '%s'", name
                )

        self.context["dashboard_module"] = self.modules.get("dashboard")
        self._last_update: dict[str, float] = {}
        self.flask_app.add_url_rule(
            "/", "root", lambda: render_template("dashboard.html")
        )
        self.flask_app.add_url_rule(
            "/api/version", "api_version", lambda: jsonify(version=_get_version())
        )

    def update(self, frame_data: dict):
        profile = performance.get_profile()
        interval = (1.0 / profile.web_update_hz) if profile.web_update_hz > 0 else 0.0
        now = time.perf_counter()
        for name, mod in self.modules.items():
            # the profile caps how often the expensive modules rebuild their
            # payloads; health and rollback stay tick-rate on purpose
            if interval and name not in _ALWAYS_UPDATE:
                last = self._last_update.get(name)
                if last is not None and (now - last) < interval:
                    continue
            self._last_update[name] = now
            try:
                mod.update(frame_data)
            except Exception:
                self.logger.exception("Web module '%s' update failed", name)

    def anyone_watching(self) -> bool:
        """True when some browser page currently consumes annotated frames.

        Used by the vision loop to decide whether plot()/debug-frame drawing is
        worth doing at the cheaper performance levels. Any module that exposes
        has_viewers() participates; rollback recording is not a watcher because
        it consumes the raw frames, not the annotated ones.
        """
        now = time.monotonic()
        for mod in self.modules.values():
            checker = getattr(mod, "has_viewers", None)
            if not callable(checker):
                continue
            try:
                if checker(now):
                    return True
            except Exception:
                # a broken watcher must not blank the feeds
                return True
        return False

    def set_vision_instance(self, vision):
        self.context["vision_instance"] = vision

    def set_cameras(self, cameras):
        self.context["cameras"] = cameras
        for name, mod in self.modules.items():
            if hasattr(mod, "set_cameras"):
                try:
                    mod.set_cameras(cameras)
                except Exception:
                    self.logger.exception("Web module '%s' set_cameras failed", name)

    def run(self, host="0.0.0.0", port=5000):
        self.start()
        self.flask_app.run(host=host, port=port, threaded=True)

    def start(self):
        for name, mod in self.modules.items():
            if hasattr(mod, "start"):
                try:
                    mod.start()
                except Exception:
                    self.logger.exception("Failed to start web module '%s'", name)

    def stop(self):
        for mod in self.modules.values():
            try:
                mod.stop()
            except Exception:
                self.logger.exception("Error stopping web module")


def create_app(cameras, config) -> iSpyWebApp:
    return iSpyWebApp(cameras, config)
