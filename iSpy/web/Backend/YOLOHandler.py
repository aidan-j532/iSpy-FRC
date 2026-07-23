import logging
from pathlib import Path
from iSpy.plugins.bases import UtilityBase
from iSpy.vision.metadata import metadata_from_pt, write_metadata, metadata_path_for, read_metadata

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
            flask_app.add_url_rule("/api/models", "yolo_list", self._list, methods=["GET"])
            flask_app.add_url_rule("/api/models/select", "yolo_select", self._select, methods=["POST"])
            flask_app.add_url_rule("/api/models/upload", "yolo_upload", self._upload, methods=["POST"])
        elif not FLASK_AVAILABLE:
            self.logger.warning("Flask not available - model endpoints disabled.")

    def _list(self):
        models = []
        pytorch_dir = _YOLO_DIR / "pytorch"
        if pytorch_dir.exists():
            for pt in sorted(pytorch_dir.glob("*.pt")):
                meta = read_metadata(pt) or {}
                models.append({
                    "file_path": str(pt.relative_to(_PROJECT_ROOT)),
                    "name": pt.name,
                    "task": meta.get("task", "unknown"),
                    "nc": meta.get("nc"),
                    "size_mb": round(pt.stat().st_size / (1024 * 1024), 2),
                })
        current = self.config.get("vision_model", {}).get("source_pt") or self.config.get("vision_model", {}).get("file_path")
        return jsonify(models=models, current=current)

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
        # Point file_path at the .pt too - boot.py's auto-opt conversion will
        # replace it with the converted artifact next boot.
        self.config.set("vision_model", "file_path", str(p))
        self.config.save()
        return jsonify(success=True, note="Restart iSpy (or re-run boot.py) to convert and load this model.")

    def _upload(self):
        if "file" not in request.files:
            return jsonify(error="No file part"), 400
        f = request.files["file"]
        if not f.filename.endswith(".pt"):
            return jsonify(error="Only .pt files are accepted"), 400

        pytorch_dir = _YOLO_DIR / "pytorch"
        pytorch_dir.mkdir(parents=True, exist_ok=True)
        dest = pytorch_dir / f.filename
        f.save(str(dest))

        try:
            meta = metadata_from_pt(dest)
            write_metadata(metadata_path_for(dest), meta)
        except Exception as e:
            self.logger.warning("Could not generate metadata for uploaded model: %s", e)

        return jsonify(success=True, file_path=str(dest.relative_to(_PROJECT_ROOT)))

    def stop(self):
        pass