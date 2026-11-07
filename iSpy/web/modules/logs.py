import logging
import os
from pathlib import Path
from flask import jsonify, render_template, request
from iSpy.web.Backend.WebModule import WebModule

logger = logging.getLogger(__name__)

class LogsModule(WebModule):
    plugin_name = "logs"

    def __init__(self, context: dict):
        super().__init__(context)
        self.log_path = Path.cwd() / "Outputs" / "log.txt"

    def register_routes(self, flask_app):
        flask_app.add_url_rule(
            "/logs", "logs_page", lambda: render_template("logs.html")
        )
        flask_app.add_url_rule("/api/logs", "api_logs", self._tail)

    def _tail(self):
        n_lines = request.args.get("n_lines", 200, type=int)
        n_lines = max(1, min(n_lines, 5000))
        if not self.log_path.exists():
            return jsonify(lines=[])
        try:
            lines = self._efficient_tail(self.log_path, n_lines)
            return jsonify(lines=lines)
        except Exception as e:
            logger.error(f"Error tailing logs: {e}")
            return jsonify(lines=[])

    @staticmethod
    def _efficient_tail(path: Path, n: int) -> list[str]:
        if n <= 0:
            return []
        try:
            chunk_size = 64 * 1024
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                pos = f.tell()
                data = b""
                while pos > 0 and data.count(b"\n") <= n:
                    size = min(chunk_size, pos)
                    pos -= size
                    f.seek(pos)
                    data = f.read(size) + data
                if pos > 0:
                    newline = data.find(b"\n")
                    data = data[newline + 1 :] if newline != -1 else b""
            text = data.decode("utf-8", errors="replace")
            return text.splitlines()[-n:]
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            return []
