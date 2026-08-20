import logging
from pathlib import Path
from flask import Flask, render_template

from iSpy.web.modules.dashboard import DashboardModule
from iSpy.web.modules.cameras import CamerasModule
from iSpy.web.modules.models import ModelsModule
from iSpy.web.modules.datasets import DatasetsModule
from iSpy.web.modules.viewer3d import Viewer3DModule
from iSpy.web.modules.logs import LogsModule
from iSpy.web.modules.metrics import MetricsModule
from iSpy.web.modules.onboarding import OnboardingModule
from iSpy.web.Backend.WebModule import WebModule
from iSpy.web.Backend.Settings import SettingsModule
from iSpy.web.Backend.SetupWizard import SetupWizardModule
from iSpy.web.modules.recommendations import RecommendationsModule
from iSpy.web.Backend.PluginStatus import PluginStatusModule
from iSpy.web.modules.health import HealthModule

_WEB_ROOT = Path(__file__).resolve().parent.parent


class iSpyWebApp:
    def __init__(self, cameras, config):
        self.logger = logging.getLogger(__name__)
        self.config = config

        self.flask_app = Flask(
            __name__,
            template_folder=str(_WEB_ROOT / "templates"),
            static_folder=str(_WEB_ROOT / "static"),
        )
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
                self.logger.exception("Failed to register routes for web module '%s'", name)

        self.context["dashboard_module"] = self.modules.get("dashboard")
        self.flask_app.add_url_rule("/", "root", lambda: render_template("dashboard.html"))

    def update(self, frame_data: dict):
        for name, mod in self.modules.items():
            try:
                mod.update(frame_data)
            except Exception:
                self.logger.exception("Web module '%s' update failed", name)

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
