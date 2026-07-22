from pathlib import Path
from flask import jsonify, render_template, request
from werkzeug.utils import secure_filename
from iSpy.web.WebModule import WebModule
from iSpy.vision.metadata import read_metadata, metadata_from_pt, write_metadata, metadata_path_for


class ModelsModule(WebModule):
    plugin_name = "models"

    def __init__(self, context: dict):
        super().__init__(context)
        self.pytorch_dir = Path.cwd() / "YoloModels" / "pytorch"
        self.pytorch_dir.mkdir(parents=True, exist_ok=True)

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/models", "models_page", lambda: render_template("models.html"))
        flask_app.add_url_rule("/api/models", "api_models_list", self._list, methods=["GET"])
        flask_app.add_url_rule("/api/models/upload", "api_models_upload", self._upload, methods=["POST"])
        flask_app.add_url_rule("/api/models/<name>", "api_models_delete", self._delete, methods=["DELETE"])

    def _list(self):
        out = []
        for pt in sorted(self.pytorch_dir.glob("*.pt")):
            meta = read_metadata(pt) or {}
            out.append({
                "name": pt.name,
                "size_mb": round(pt.stat().st_size / (1024 * 1024), 2),
                "task": meta.get("task", "unknown"),
                "nc": meta.get("nc"),
                "names": meta.get("names"),
                "input_size": meta.get("input_size"),
            })
        return jsonify(models=out)

    def _upload(self):
        f = request.files.get("file")
        if not f or not f.filename.endswith(".pt"):
            return jsonify(error="Upload a .pt file"), 400
        name = secure_filename(f.filename)
        dest = self.pytorch_dir / name
        f.save(str(dest))
        try:
            meta = metadata_from_pt(dest)
            write_metadata(metadata_path_for(dest), meta)
        except Exception as e:
            return jsonify(error=f"Saved but metadata generation failed: {e}"), 207
        return jsonify(success=True, name=name)

    def _delete(self, name):
        target = self.pytorch_dir / secure_filename(name)
        if not target.exists():
            return jsonify(error="Not found"), 404
        target.unlink()
        meta = metadata_path_for(target)
        if meta.exists():
            meta.unlink()
        return jsonify(success=True)