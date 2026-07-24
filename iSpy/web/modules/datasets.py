import logging
from pathlib import Path
from flask import jsonify, render_template, request, send_from_directory, url_for
from iSpy.web.Backend.WebModule import WebModule

logger = logging.getLogger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class DatasetsModule(WebModule):
    plugin_name = "datasets"

    def __init__(self, context: dict):
        super().__init__(context)
        self.dataset_root = Path.cwd() / "QuantizeDataset"

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/datasets", "datasets_page", lambda: render_template("datasets.html"))
        flask_app.add_url_rule("/api/datasets", "api_datasets_list", self._list, methods=["GET"])
        flask_app.add_url_rule("/api/datasets/<name>/images", "api_ds_images", self._list_images, methods=["GET"])
        flask_app.add_url_rule("/api/datasets/<name>/images", "api_ds_upload", self._upload_image, methods=["POST"])
        flask_app.add_url_rule("/api/datasets/<name>/images/<filename>", "api_ds_image_get", self._get_image, methods=["GET"])
        flask_app.add_url_rule("/api/datasets/<name>/images/<filename>", "api_ds_image_delete", self._delete_image, methods=["DELETE"])

    def _images_dir(self, name: str) -> Path:
        per_model = self.dataset_root / name / "images"
        if per_model.exists():
            return per_model
        flat = self.dataset_root / "images"
        if flat.exists():
            return flat
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

    def _list_images(self, name):
        d = self._images_dir(name)
        if not d.exists():
            return jsonify(images=[], count=0)
        files = sorted(p.name for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS)
        return jsonify(images=files, count=len(files))

    def _get_image(self, name, filename):
        d = self._images_dir(name)
        return send_from_directory(str(d), filename)

    def _upload_image(self, name):
        if "file" not in request.files:
            return jsonify(error="No file part"), 400
        f = request.files["file"]
        ext = Path(f.filename).suffix.lower()
        if ext not in _IMG_EXTS:
            return jsonify(error=f"Unsupported extension {ext}"), 400

        d = self._images_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        dest = d / Path(f.filename).name
        f.save(str(dest))

        ds_root = dest.parent.parent
        txt = ds_root / "dataset.txt"
        rel = f"images/{dest.name}"
        existing = txt.read_text().splitlines() if txt.exists() else []
        if rel not in existing:
            existing.append(rel)
            txt.write_text("\n".join(existing) + "\n")

        return jsonify(success=True, filename=dest.name)

    def _delete_image(self, name, filename):
        d = self._images_dir(name)
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
