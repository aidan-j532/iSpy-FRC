"""
The ONE Flask app for iSpy. Every page (status, settings, setup wizard,
model manager, cameras) registers a blueprint here instead of spinning up
its own Flask() instance. This replaces the old split between CameraApp's
own app and the flask_app passed through the utility context - there is
now exactly one server, one static/ dir, one templates/ dir.

Usage from iSpy.py / boot:

    from iSpy.web.app import create_app
    app = create_app(cameras=cameras, config=config)
    threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 5000}, daemon=True).start()

Then http://ispy.local:5000/ (or /status, /settings, /setup, /models, /cameras)
all come from this one app, sharing base.html / base.css / api.js.
"""

import logging
from flask import Flask, render_template

logger = logging.getLogger(__name__)


def create_app(cameras=None, config=None) -> Flask:
    app = Flask(__name__)

    # Shared state every blueprint can reach via current_app.config[...]
    # (Flask's own app.config dict, not iSpy's iSpyConfig - different thing,
    # just reusing the name Flask already gives us.)
    app.config["ISPY_CAMERAS"] = cameras or []
    app.config["ISPY_CONFIG"] = config

    # --- register blueprints -------------------------------------------------
    # Each of these files defines `bp = Blueprint("name", __name__)` and its
    # own routes. Import here (not at module load time) so a missing/broken
    # page can't take the whole app down - log and skip instead.
    _register(app, "iSpy.web.Backend.Status")
    _register(app, "iSpy.web.Backend.Settings")
    _register(app, "iSpy.web.Backend.SetupWizard")
    _register(app, "iSpy.web.Backend.YOLOHandler")
    _register(app, "iSpy.web.Backend.Camera")

    @app.route("/")
    def index():
        return render_template("index.html")

    try:
        import werkzeug.serving
        werkzeug.serving.show_server_banner = lambda *a, **kw: None
    except Exception:
        pass

    return app


def _register(app: Flask, module_path: str) -> None:
    # Convention: every iSpy/web/*.py blueprint module exports its
    # blueprint as `bp`. Keep it that way - one name, no per-module lookup.
    try:
        module = __import__(module_path, fromlist=["bp"])
        app.register_blueprint(module.bp)
        logger.info("Registered blueprint: %s", module_path)
    except Exception:
        logger.exception("Failed to register blueprint from %s - skipping.", module_path)