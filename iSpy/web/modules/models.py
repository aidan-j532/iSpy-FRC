import logging
from pathlib import Path
from flask import jsonify, render_template, request
from werkzeug.utils import secure_filename
from iSpy.web.Backend.WebModule import WebModule
from iSpy.config.iSpyConfig import get_pipeline_settings
from iSpy.vision.metadata import read_metadata, metadata_from_pt, write_metadata, metadata_path_for

logger = logging.getLogger(__name__)


def _camera_vision_model(cam) -> dict | None:
    if not isinstance(cam, dict):
        return None
    settings = get_pipeline_settings(cam) or {}
    model_cfg = settings.get("vision_model")
    if isinstance(model_cfg, dict):
        return model_cfg
    return None


def _is_safe_path(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _model_rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def _target_format(settings: dict) -> str:
    fmt = str(settings.get("target_format") or "auto").strip().lower()
    if fmt and fmt != "auto":
        return fmt
    try:
        from iSpy.vision.pipelines.object_detection import ObjectDetectionCamera
        return ObjectDetectionCamera.recommended_format()
    except Exception:
        return "onnx"


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
        names = set()
        cams = config.get_nested("camera_configs", default={})
        for cam in cams.values():
            model_cfg = _camera_vision_model(cam)
            if not model_cfg:
                continue
            for key in ("file_path", "source_pt"):
                fp = model_cfg.get(key)
                if fp:
                    names.add(Path(fp).name)
        return names

    def _get_current_model(self) -> str | None:
        # for the "active" badge - basename of the first model-backed cam's file_path
        config = self.context.get("config")
        if not config:
            return None
        cams = config.get_nested("camera_configs", default={})
        for cam in cams.values():
            model_cfg = _camera_vision_model(cam)
            if not model_cfg:
                continue
            file_path = model_cfg.get("file_path") or model_cfg.get("source_pt")
            if file_path:
                return Path(file_path).name
        return None

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
        pt = self._resolve_model_path(name)
        if pt is None or not pt.exists():
            return jsonify(error="Model not found"), 404
        meta = read_metadata(pt) or {}
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
        tmp = dest.with_suffix(".pt.uploading")
        try:
            f.save(str(tmp))
            meta = metadata_from_pt(tmp)
            write_metadata(metadata_path_for(dest), meta)
            tmp.rename(dest)
        except Exception as e:
            if tmp.exists():
                tmp.unlink()
            return jsonify(error=f"Metadata generation failed: {e}"), 400
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
        if not config:
            return jsonify(error="No config available"), 500
        # model selection = the per-camera vision_model block of every model-backed cam.
        # source_pt always points at the .pt; file_path points at an already-built
        # optimized artifact for it so the camera keeps running the picked model
        # (a stale artifact from a previous pick must not win).
        from iSpy.vision.optimizer import existing_artifact_for

        updated = []
        cams = config.get_nested("camera_configs", default={})
        for cam_name, cam in cams.items():
            settings = get_pipeline_settings(cam) or {}
            model_cfg = settings.get("vision_model")
            if not isinstance(model_cfg, dict):
                continue
            src = _model_rel_path(p)
            artifact = existing_artifact_for(p, _target_format(settings))
            settings["vision_model"] = {
                **model_cfg, "source_pt": src, "file_path": artifact or src,
            }
            updated.append(cam_name)
        if not updated:
            return jsonify(error="No camera uses a user-selectable model"), 400
        config.save()
        return jsonify(success=True, note="Restart iSpy to load this model.", cameras=updated)
    
    def _resolve_model_path(self, name: str) -> Path | None:
        target = (self.pytorch_dir / name).resolve()
        try:
            target.relative_to(self.pytorch_dir.resolve())
        except ValueError:
            return None
        return target
    
    def _delete(self, name):
        target = self._resolve_model_path(name)
        if target is None or not target.exists():
            return jsonify(error="Not found"), 404

        protected = self._get_protected_model_names()
        if target.name in protected:
            return jsonify(error="Cannot delete a model referenced by the active config. Select a different model first."), 400

        target.unlink()
        meta = metadata_path_for(target)
        if meta.exists():
            meta.unlink()
        return jsonify(success=True)