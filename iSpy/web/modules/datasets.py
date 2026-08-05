import logging
from pathlib import Path
from flask import jsonify, render_template, request, send_from_directory
from iSpy.web.Backend.WebModule import WebModule
from iSpy.dataset.dataset import add_image_to_dataset_txt, remove_image_from_dataset_txt

logger = logging.getLogger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


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
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        # A quantization dataset is a reusable folder of calibration images.
        # `default` is the built-in dataset every pipeline falls back to when
        # none is configured.
        (self.dataset_root / "default" / "images").mkdir(parents=True, exist_ok=True)

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/datasets", "datasets_page", lambda: render_template("datasets.html"))
        flask_app.add_url_rule("/api/datasets", "api_datasets_list", self._list, methods=["GET"])
        flask_app.add_url_rule("/api/datasets", "api_datasets_create", self._create, methods=["POST"])
        flask_app.add_url_rule("/api/datasets/<name>/images", "api_ds_images", self._list_images, methods=["GET"])
        flask_app.add_url_rule("/api/datasets/<name>/images", "api_ds_upload", self._upload_image, methods=["POST"])
        flask_app.add_url_rule("/api/datasets/<name>/images/<filename>", "api_ds_image_get", self._get_image, methods=["GET"])
        flask_app.add_url_rule("/api/datasets/<name>/images/<filename>", "api_ds_image_delete", self._delete_image, methods=["DELETE"])
        flask_app.add_url_rule("/api/fs/dirs", "api_fs_dirs", self._browse_dirs, methods=["GET"])

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
                out.append({"name": d.name, "image_count": count})
        return jsonify(datasets=out)

    def _create(self):
        data = request.get_json(force=True) or {}
        name = _validate_filename(str(data.get("name", "")))
        if not name or name in (".", ".."):
            return jsonify(error="Invalid dataset name"), 400
        target = self.dataset_root / name
        if not _is_safe_path(self.dataset_root, target):
            return jsonify(error="Invalid path"), 400
        if target.exists():
            return jsonify(error=f"Dataset '{name}' already exists"), 409
        (target / "images").mkdir(parents=True, exist_ok=True)
        return jsonify(success=True, name=name)

    def _browse_dirs(self):
        """List the subdirectories of a folder for the interactive dataset
        picker in the camera settings UI."""
        raw = request.args.get("path", "").strip()
        try:
            base = Path(raw) if raw else Path.cwd()
            if not base.is_absolute():
                base = (Path.cwd() / base).resolve()
            base = base.resolve()
        except Exception:
            return jsonify(error="Invalid path"), 400

        if not base.is_dir():
            return jsonify(error="Folder not found"), 404

        dirs = []
        try:
            for d in sorted(base.iterdir()):
                if d.is_dir() and not d.name.startswith("."):
                    dirs.append({"name": d.name, "path": str(d)})
        except PermissionError:
            pass

        parent = str(base.parent) if base.parent != base else None
        return jsonify(
            path=str(base),
            name=base.name or str(base),
            parent=parent,
            dirs=dirs,
        )

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
