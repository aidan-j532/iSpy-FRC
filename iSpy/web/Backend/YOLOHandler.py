import logging
from pathlib import Path
from iSpy.plugins.bases import UtilityBase
from iSpy.vision.metadata import metadata_from_pt, write_metadata, metadata_path_for

try:
    from flask import jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

_PROJECT_ROOT = Path.cwd()
_YOLO_DIR = _PROJECT_ROOT / "YoloModels"


class YOLOHandler(UtilityBase):
    plugin_name = "yolo_handler"

    def __init__(self, context: dict):
        self.config = context["config"]
        self.logger = logging.getLogger(__name__)
        flask_app = context.get("flask_app")

        if flask_app and FLASK_AVAILABLE:
            flask_app.add_url_rule("/api/models/select", "yolo_select", self._select, methods=["POST"])
        elif not FLASK_AVAILABLE:
            self.logger.warning("Flask not available - model endpoints disabled.")

    def _select(self):
        data = request.get_json(force=True) or {}
        file_path = data.get("file_path")
        if not file_path:
            return jsonify(error="file_path required"), 400

        p = Path(file_path)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        if not p.exists():
            return jsonify(error=f"Model not found: {p}"), 404

        self.config.set("vision_model", "source_pt", str(p))
        self.config.set("vision_model", "file_path", str(p))
        self.config.save()
        return jsonify(success=True, note="Restart iSpy (or re-run boot.py) to convert and load this model.")

    def stop(self):
        pass
