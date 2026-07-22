# iSpy/web/WebApp.py
import logging
from flask import Flask
from iSpy.web.CameraApp import CameraApp
from iSpy.web.Backend.YOLOHandler import YOLOHandler
from iSpy.web.DatasetManager import DatasetManager
from iSpy.web.Backend.Settings import Settings
from iSpy.web.Backend.Status import Status
from iSpy.web.Backend.SetupWizard import SetupWizard

class iSpyWebApp:
    def __init__(self, cameras, config):
        self.logger = logging.getLogger(__name__)
        self.flask_app = Flask(__name__)
        self.config = config

        # Every module below implements WebModule (register_routes/update/stop)
        self.modules = {
            "cameras": CameraApp(cameras=cameras, config=config),
            "models": YOLOHandler(config=config),
            "datasets": DatasetManager(config=config),
            "settings": Settings(config=config),
            "status": Status(config=config, cameras=cameras),
            "setup": SetupWizard(config=config),
        }

        for name, mod in self.modules.items():
            try:
                mod.register_routes(self.flask_app)
            except Exception:
                self.logger.exception("Failed to register routes for web module '%s'", name)

    def update(self, frame_data: dict):
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