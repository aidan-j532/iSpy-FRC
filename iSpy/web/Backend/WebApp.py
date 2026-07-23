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
from iSpy.web.Backend.WebModule import WebModule

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
        print("Template folder:", self.flask_app.template_folder)
        print("Jinja search path:", self.flask_app.jinja_loader.searchpath)
        context = {"config": config, "cameras": cameras, "flask_app": self.flask_app}

        # Add new pages here - this is the only place a new module needs
        # to be registered to show up everywhere (nav, routes, updates).
        self.modules: dict[str, WebModule] = {
            "cameras": CamerasModule(context),
            "models": ModelsModule(context),
            "datasets": DatasetsModule(context),
            "viewer3d": Viewer3DModule(context),
            "dashboard": DashboardModule(context, other_modules_ref=self),
            "logs": LogsModule(context),
            "metrics": MetricsModule(context),
        }

        for name, mod in self.modules.items():
            try:
                mod.register_routes(self.flask_app)
            except Exception:
                self.logger.exception("Failed to register routes for web module '%s'", name)

        self.flask_app.add_url_rule("/", "root", lambda: render_template("dashboard.html"))

        try:
            import werkzeug.serving
            werkzeug.serving.show_server_banner = lambda *a, **kw: None
        except Exception:
            pass

    def update(self, frame_data: dict):
        """The ONE call the game loop makes. Every web module gets the
        same frame_data every tick - nobody has to remember to call
        camera_app.set_frame() or yolo_app.update() separately again."""
        for name, mod in self.modules.items():
            try:
                mod.update(frame_data)
            except Exception:
                self.logger.exception("Web module '%s' update failed", name)

    def run(self, host="0.0.0.0", port=5000):
        self.flask_app.run(host=host, port=port, threaded=True)

    def stop(self):
        for mod in self.modules.values():
            try:
                mod.stop()
            except Exception:
                self.logger.exception("Error stopping web module")


def create_app(cameras, config) -> iSpyWebApp:
    return iSpyWebApp(cameras, config)