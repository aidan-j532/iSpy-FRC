import json
import logging
from pathlib import Path
from flask import jsonify, render_template, request, send_from_directory
from iSpy.web.Backend.WebModule import WebModule
from iSpy.dataset.dataset import add_image_to_dataset_txt, remove_image_from_dataset_txt

logger = logging.getLogger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_ACTIVE_DATASET_FILE = Path.cwd() / "Config" / "active_dataset.json"


def _is_safe_path(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _validate_filename(name: str) -> str:
    cleaned = "".join(c for c in name if c.isalnum() or c in "._- ")
    return cleaned.strip(". ") or "unnamed"


class DatasetsModule(WebModule):
    plugin_name = "datasets"

    def __init__(self, context: dict):
        super().__init__(context)
        self.dataset_root = Path.cwd() / "QuantizeDataset"
        self._active: str = self._load_active()

    def _load_active(self) -> str:
        try:
            if _ACTIVE_DATASET_FILE.exists():
                data = json.loads(_ACTIVE_DATASET_FILE.read_text())
                return data.get("active", "")
        except Exception:
            pass
        return ""

    def _save_active(self, name: str):
        _ACTIVE_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
        _ACTIVE_DATASET_FILE.write_text(json.dumps({"active": name}))

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/datasets", "datasets_page", lambda: render_template("datasets.html"))
        flask_app.add_url_rule("/api/datasets", "api_datasets_list", self._list, methods=["GET"])
        flask_app.add_url_rule("/api/datasets/active", "api_datasets_active_get", self._get_active, methods=["GET"])
        flask_app.add_url_rule("/api/datasets/active", "api_datasets_active_set", self._set_active, methods=["POST"])
        flask_app.add_url_rule("/api/datasets/<name>/images", "api_ds_images", self._list_images, methods=["GET"])
        flask_app.add_url_rule("/api/datasets/<name>/images", "api_ds_upload", self._upload_image, methods=["POST"])
        flask_app.add_url_rule("/api/datasets/<name>/images/<filename>", "api_ds_image_get", self._get_image, methods=["GET"])
        flask_app.add_url_rule("/api/datasets/<name>/images/<filename>", "api_ds_image_delete", self._delete_image, methods=["DELETE"])

    def _images_dir(self, name: str) -> Path:
        return self.dataset_root / name / "images"
    
    def _list(self):
        out = []
        if self.dataset_root.exists():
            for d in sorted(self.dataset_root.iterdir()):
                if not d.is_dir():
                    continue
                images_dir = d / "images"
                count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
                out.append({"name": d.name, "image_count": count, "active": d.name == self._active})
        return jsonify(datasets=out, active=self._active)

    def _get_active(self):
        return jsonify(active=self._active)

    def _set_active(self):
        data = request.get_json(force=True) or {}
        name = data.get("name", "")
        if name and not (self.dataset_root / name).exists():
            return jsonify(error=f"Dataset '{name}' not found"), 404
        self._active = name
        self._save_active(name)
        return jsonify(success=True, active=name)

    def _list_images(self, name):
        d = self._images_dir(name)
        if not d.exists():
            return jsonify(images=[], count=0)
        files = sorted(p.name for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS)
        return jsonify(images=files, count=len(files))

    def _get_image(self, name, filename):
        d = self._images_dir(name)
        if not _is_safe_path(self.dataset_root, d / filename):
            return jsonify(error="Invalid path"), 400
        return send_from_directory(str(d), _validate_filename(filename))

    def _upload_image(self, name):
        if "file" not in request.files:
            return jsonify(error="No file part"), 400
        f = request.files["file"]
        ext = Path(f.filename).suffix.lower()
        if ext not in _IMG_EXTS:
            return jsonify(error=f"Unsupported extension {ext}"), 400

        safe_name = _validate_filename(f.filename)
        d = self._images_dir(name)
        if not _is_safe_path(self.dataset_root, d):
            return jsonify(error="Invalid path"), 400
        d.mkdir(parents=True, exist_ok=True)
        dest = d / safe_name
        f.save(str(dest))

        ds_root = dest.parent.parent
        add_image_to_dataset_txt(ds_root, f"images/{dest.name}")

        return jsonify(success=True, filename=dest.name)

    def _delete_image(self, name, filename):
        d = self._images_dir(name)
        safe_filename = _validate_filename(filename)
        target = d / safe_filename
        if not _is_safe_path(self.dataset_root, target) or not target.exists():
            return jsonify(error="Not found"), 404
        target.unlink()

        ds_root = d.parent
        remove_image_from_dataset_txt(ds_root, f"images/{safe_filename}")

        return jsonify(success=True)
