import logging
from pathlib import Path
from flask import Flask, render_template, jsonify, request

from iSpy.web.modules.dashboard import DashboardModule
from iSpy.web.modules.cameras import CamerasModule
from iSpy.web.modules.models import ModelsModule
from iSpy.web.modules.datasets import DatasetsModule
from iSpy.web.modules.viewer3d import Viewer3DModule
from iSpy.web.modules.logs import LogsModule
from iSpy.web.modules.metrics import MetricsModule
from iSpy.web.Backend.WebModule import WebModule
from iSpy.web.Backend.Settings import SettingsModule
from iSpy.web.Backend.SetupWizard import SetupWizardModule

_WEB_ROOT = Path(__file__).resolve().parent.parent


class iSpyWebApp:
    def __init__(self, cameras, config, standalone=False, process_manager=None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.standalone = standalone
        self.process_manager = process_manager

        self.flask_app = Flask(
            __name__,
            template_folder=str(_WEB_ROOT / "templates"),
            static_folder=str(_WEB_ROOT / "static"),
        )
        context = {
            "config": config,
            "cameras": cameras,
            "flask_app": self.flask_app,
            "standalone": standalone,
            "process_manager": process_manager,
        }

        self.modules: dict[str, WebModule] = {
            "cameras": CamerasModule(context),
            "models": ModelsModule(context),
            "datasets": DatasetsModule(context),
            "viewer3d": Viewer3DModule(context),
            "dashboard": DashboardModule(context),
            "logs": LogsModule(context),
            "metrics": MetricsModule(context),
            "settings": SettingsModule(context),
            "setup_wizard": SetupWizardModule(context),
        }

        for name, mod in self.modules.items():
            try:
                mod.register_routes(self.flask_app)
            except Exception:
                self.logger.exception("Failed to register routes for web module '%s'", name)

        self.flask_app.add_url_rule("/", "root", lambda: render_template("dashboard.html"))

        if standalone and process_manager:
            self._register_service_routes()

        try:
            import werkzeug.serving
            werkzeug.serving.show_server_banner = lambda *a, **kw: None
        except Exception:
            pass

    def _register_service_routes(self):
        pm = self.process_manager

        def _service_status():
            return jsonify(pm.status())

        def _service_start():
            return jsonify(pm.start())

        def _service_stop():
            return jsonify(pm.stop())

        def _service_restart():
            return jsonify(pm.restart())

        def _service_pause():
            return jsonify(pm.pause())

        def _service_resume():
            return jsonify(pm.resume())

        self.flask_app.add_url_rule(
            "/api/service/status", "service_status", _service_status, methods=["GET"]
        )
        self.flask_app.add_url_rule(
            "/api/service/start", "service_start", _service_start, methods=["POST"]
        )
        self.flask_app.add_url_rule(
            "/api/service/stop", "service_stop", _service_stop, methods=["POST"]
        )
        self.flask_app.add_url_rule(
            "/api/service/restart", "service_restart", _service_restart, methods=["POST"]
        )
        self.flask_app.add_url_rule(
            "/api/service/pause", "service_pause", _service_pause, methods=["POST"]
        )
        self.flask_app.add_url_rule(
            "/api/service/resume", "service_resume", _service_resume, methods=["POST"]
        )

    def update(self, frame_data: dict):
        """Called every tick when running inside the vision loop."""
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
