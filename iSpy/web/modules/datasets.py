from pathlib import Path
from flask import jsonify, render_template
from iSpy.web.Backend.WebModule import WebModule



class DatasetsModule(WebModule):
    plugin_name = "datasets"

    def __init__(self, context: dict):
        super().__init__(context)
        self.dataset_root = Path.cwd() / "QuantizeDataset"

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/datasets", "datasets_page", lambda: render_template("datasets.html"))
        flask_app.add_url_rule("/api/datasets", "api_datasets_list", self._list, methods=["GET"])

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