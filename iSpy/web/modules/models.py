import logging
from pathlib import Path
from flask import jsonify, render_template, request
from werkzeug.utils import secure_filename
from iSpy.web.Backend.WebModule import WebModule
from iSpy.vision.metadata import read_metadata, metadata_from_pt, write_metadata, metadata_path_for

logger = logging.getLogger(__name__)


def _is_safe_path(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


class ModelsModule(WebModule):
    plugin_name = "models"

    def __init__(self, context: dict):
        super().__init__(context)
        self.pytorch_dir = Path.cwd() / "YoloModels" / "pytorch"
        self.pytorch_dir.mkdir(parents=True, exist_ok=True)

    def _get_protected_model_names(self) -> set[str]:
        config = self.context.get("config")
        if not config:
            return set()
        model_cfg = config.get("vision_model", {})
        names = set()
        for key in ("file_path", "source_pt"):
            fp = model_cfg.get(key)
            if fp:
                names.add(Path(fp).name)
        return names

    def _get_current_model(self) -> str | None:
        # kept for the "active" badge in the list/detail views - still just
        # the primary file_path's basename.
        config = self.context.get("config")
        if not config:
            return None
        model_cfg = config.get("vision_model", {})
        file_path = model_cfg.get("file_path") or model_cfg.get("source_pt")
        return Path(file_path).name if file_path else None

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/models", "models_page", lambda: render_template("models.html"))
        flask_app.add_url_rule("/api/models", "api_models_list", self._list, methods=["GET"])
        flask_app.add_url_rule("/api/models/upload", "api_models_upload", self._upload, methods=["POST"])
        flask_app.add_url_rule("/api/models/<name>", "api_models_detail", self._detail, methods=["GET"])
        flask_app.add_url_rule("/api/models/<name>", "api_models_delete", self._delete, methods=["DELETE"])
        flask_app.add_url_rule("/api/models/select", "api_models_select", self._select, methods=["POST"])

    def _list(self):
        current = self._get_current_model()
        out = []
        for pt in sorted(self.pytorch_dir.glob("*.pt")):
            meta = read_metadata(pt) or {}
            # is_active = pt.name == current
            protected = self._get_protected_model_names()
            out.append({
                "name": pt.name,
                "size_mb": round(pt.stat().st_size / (1024 * 1024), 2),
                "task": meta.get("task", "unknown"),
                "nc": meta.get("nc"),
                "names": meta.get("names"),
                "input_size": meta.get("input_size"),
                "active": pt.name in protected,
            })
        return jsonify(models=out, current=current)

    def _detail(self, name):
        safe_name = secure_filename(name)
        pt = self.pytorch_dir / safe_name
        if not _is_safe_path(self.pytorch_dir, pt) or not pt.exists():
            return jsonify(error="Model not found"), 404
        meta = read_metadata(pt) or {}
        # current = self._get_current_model()
        protected = self._get_protected_model_names()
        return jsonify(
            name=pt.name,
            size_mb=round(pt.stat().st_size / (1024 * 1024), 2),
            task=meta.get("task", "unknown"),
            nc=meta.get("nc"),
            names=meta.get("names"),
            input_size=meta.get("input_size"),
            active=pt.name in protected,
        )

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

    def _select(self):
        data = request.get_json(force=True) or {}
        file_path = data.get("file_path")
        if not file_path:
            return jsonify(error="file_path required"), 400
        p = Path(file_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            return jsonify(error=f"Model not found: {p}"), 404
        if not _is_safe_path(self.pytorch_dir, p) and not _is_safe_path(Path.cwd() / "YoloModels", p):
            return jsonify(error="Invalid model path"), 403
        config = self.context.get("config")
        if config:
            config.set("vision_model", "source_pt", str(p))
            config.set("vision_model", "file_path", str(p))
            config.save()
        return jsonify(success=True, note="Restart iSpy to load this model.")
    
    def _delete(self, name):
        safe_name = secure_filename(name)
        target = self.pytorch_dir / safe_name
        if not _is_safe_path(self.pytorch_dir, target) or not target.exists():
            return jsonify(error="Not found"), 404

        protected = self._get_protected_model_names()
        if safe_name in protected:
            return jsonify(error="Cannot delete a model currently referenced by the active config (file_path or source_pt). Select a different model first."), 400

        target.unlink()
        meta = metadata_path_for(target)
        if meta.exists():
            meta.unlink()
        return jsonify(success=True)