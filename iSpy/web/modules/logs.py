import io
from pathlib import Path
from flask import jsonify, render_template, request
from iSpy.web.Backend.WebModule import WebModule


class LogsModule(WebModule):
    plugin_name = "logs"

    def __init__(self, context: dict):
        super().__init__(context)
        self.log_path = Path.cwd() / "Outputs" / "log.txt"

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/logs", "logs_page", lambda: render_template("logs.html"))
        flask_app.add_url_rule("/api/logs", "api_logs", self._tail)

    def _tail(self):
        n_lines = request.args.get("n_lines", 200, type=int)
        n_lines = max(1, min(n_lines, 5000))
        if not self.log_path.exists():
            return jsonify(lines=[])
        try:
            lines = self._efficient_tail(self.log_path, n_lines)
            return jsonify(lines=lines)
        except Exception:
            return jsonify(lines=[])

    @staticmethod
    def _efficient_tail(path: Path, n: int) -> list[str]:
        block_size = 8192
        lines_found: list[str] = []
        with open(path, "rb") as f:
            f.seek(0, io.SEEK_END)
            file_size = f.tell()
            if file_size == 0:
                return []
            block_start = max(0, file_size - block_size)
            while block_start >= 0 and len(lines_found) <= n:
                f.seek(block_start)
                block = f.read(min(block_size, file_size - block_start))
                lines_found = block.decode("utf-8", errors="replace").splitlines(keepends=True) + lines_found
                if len(lines_found) > n:
                    break
                block_start -= block_size
        if len(lines_found) > n:
            lines_found = lines_found[-n:]
        return [line.rstrip("\n\r") for line in lines_found]
