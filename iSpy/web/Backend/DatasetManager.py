import logging
from pathlib import Path
from iSpy.plugins.bases import UtilityBase

try:
    from flask import jsonify, request, send_from_directory
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

_PROJECT_ROOT = Path.cwd()
_DATASET_ROOT = _PROJECT_ROOT / "QuantizeDataset"
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class DatasetManager(UtilityBase):
    plugin_name = "dataset_manager"

    def __init__(self, context: dict):
        self.logger = logging.getLogger(__name__)
        flask_app = context.get("flask_app")

        if flask_app and FLASK_AVAILABLE:
            flask_app.add_url_rule("/api/dataset/<model_stem>/images", "ds_list", self._list, methods=["GET"])
            flask_app.add_url_rule("/api/dataset/<model_stem>/images", "ds_upload", self._upload, methods=["POST"])
            flask_app.add_url_rule("/api/dataset/<model_stem>/images/<filename>", "ds_delete", self._delete, methods=["DELETE"])
            flask_app.add_url_rule("/api/dataset/<model_stem>/images/<filename>", "ds_get", self._get_image, methods=["GET"])
        elif not FLASK_AVAILABLE:
            self.logger.warning("Flask not available - dataset endpoints disabled.")

    def _images_dir(self, model_stem: str) -> Path:
        # per-model dataset if it exists, else flat root
        per_model = _DATASET_ROOT / model_stem / "images"
        if per_model.exists():
            return per_model
        return _DATASET_ROOT / "images"

    def _list(self, model_stem):
        d = self._images_dir(model_stem)
        if not d.exists():
            return jsonify(images=[])
        files = sorted(p.name for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS)
        return jsonify(images=files, count=len(files))

    def _get_image(self, model_stem, filename):
        d = self._images_dir(model_stem)
        return send_from_directory(str(d), filename)

    def _upload(self, model_stem):
        if "file" not in request.files:
            return jsonify(error="No file part"), 400
        f = request.files["file"]
        ext = Path(f.filename).suffix.lower()
        if ext not in _IMG_EXTS:
            return jsonify(error=f"Unsupported extension {ext}"), 400

        d = self._images_dir(model_stem)
        d.mkdir(parents=True, exist_ok=True)
        dest = d / Path(f.filename).name
        f.save(str(dest))

        # keep dataset.txt in sync
        ds_root = dest.parent.parent
        txt = ds_root / "dataset.txt"
        rel = f"images/{dest.name}"
        existing = txt.read_text().splitlines() if txt.exists() else []
        if rel not in existing:
            existing.append(rel)
            txt.write_text("\n".join(existing) + "\n")

        return jsonify(success=True, filename=dest.name)

    def _delete(self, model_stem, filename):
        d = self._images_dir(model_stem)
        target = d / filename
        if not target.exists():
            return jsonify(error="Not found"), 404
        target.unlink()

        ds_root = d.parent
        txt = ds_root / "dataset.txt"
        if txt.exists():
            rel = f"images/{filename}"
            lines = [l for l in txt.read_text().splitlines() if l.strip() != rel]
            txt.write_text("\n".join(lines) + ("\n" if lines else ""))

        return jsonify(success=True)

    def stop(self):
        pass