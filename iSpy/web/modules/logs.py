# iSpy/web/modules/logs.py
from pathlib import Path
from flask import jsonify, render_template
from iSpy.web.WebModule import WebModule


class LogsModule(WebModule):
    plugin_name = "logs"

    def __init__(self, context: dict):
        super().__init__(context)
        self.log_path = Path.cwd() / "Outputs" / "log.txt"

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/logs", "logs_page", lambda: render_template("logs.html"))
        flask_app.add_url_rule("/api/logs", "api_logs", self._tail)

    def _tail(self, n_lines: int = 200):
        if not self.log_path.exists():
            return jsonify(lines=[])
        with open(self.log_path, "r", errors="ignore") as f:
            lines = f.readlines()[-n_lines:]
        return jsonify(lines=lines)