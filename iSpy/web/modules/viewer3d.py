import os
import json
from pathlib import Path
from flask import jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename
from iSpy.web.Backend.WebModule import WebModule

class Viewer3DModule(WebModule):
    plugin_name = "viewer3d"

    def __init__(self, context: dict):
        super().__init__(context)
        self._latest_objects = []
        self.models_dir = Path.cwd() / "Outputs" / "3d_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._active_pose_model = ""
        self._cached_num_keypoints = None

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/viewer3d", "viewer3d_page", lambda: render_template("viewer3d.html"))
        flask_app.add_url_rule("/api/detections/latest", "api_detections_latest", self._latest)
        flask_app.add_url_rule("/api/upload_glb", "api_upload_glb", self._upload_glb, methods=["POST"])
        flask_app.add_url_rule("/api/pose_models/list", "api_pose_models_list", self._list_pose_models)
        flask_app.add_url_rule("/api/pose_models/active", "api_pose_models_active", self._get_active_model)
        flask_app.add_url_rule("/api/pose_models/select", "api_pose_models_select", self._select_model, methods=["POST"])

        def serve_pose_model_file(filename):
            safe = secure_filename(filename)
            filepath = self.models_dir / safe
            if not filepath.exists():
                return jsonify(error="File not found"), 404
            return send_file(str(filepath))
        serve_pose_model_file.__name__ = "api_pose_models_file"
        flask_app.add_url_rule("/api/pose_models/file/<path:filename>", "api_pose_models_file", serve_pose_model_file)

    def update(self, frame_data: dict):
        fuel_list = frame_data.get("fuel_list", [])
        if self._cached_num_keypoints is None:
            config = self.context.get("config", {})
            vm = config.get("vision_model", {}) if config else {}
            self._cached_num_keypoints = self._get_num_keypoints(vm)
        num_kpts = self._cached_num_keypoints
        self._latest_objects = []
        for idx, obj in enumerate(fuel_list):
            obj_entry = {
                "id": idx,
                "x": getattr(obj, "x", 0),
                "y": getattr(obj, "y", 0),
                "z": getattr(obj, "z", 0),
                "roll": getattr(obj, "roll", 0),
                "yaw": getattr(obj, "yaw", 0),
                "pitch": getattr(obj, "pitch", 0),
                "name": getattr(obj, "name", "unknown"),
                "confidence": getattr(obj, "confidence", 0),
                "num_keypoints": num_kpts,
            }
            kpts = getattr(obj, "keypoints_3d", None)
            if kpts is not None:
                obj_entry["keypoints_3d"] = kpts
            self._latest_objects.append(obj_entry)

    def _get_num_keypoints(self, vm: dict) -> int:
        if not vm:
            return 17
        src = vm.get("source_pt", "")
        if not src:
            return 17
        meta_path = Path(str(src).replace(".pt", "_metadata.yaml"))
        if not meta_path.exists():
            meta_path = Path(str(src).replace(".pt", ".metadata.yaml"))
            if not meta_path.exists():
                return 17
        try:
            import yaml
            with open(meta_path) as f:
                meta = yaml.safe_load(f) or {}
            ks = meta.get("kpt_shape")
            if ks and len(ks) == 2:
                return int(ks[0])
        except Exception:
            pass
        return 17

    def _latest(self):
        return jsonify(objects=self._latest_objects)

    def _upload_glb(self):
        if "file" not in request.files:
            return jsonify(error="No file provided"), 400
        file = request.files["file"]
        if file.filename == '':
            return jsonify(error="Empty filename"), 400
        safe_name = secure_filename(file.filename)
        save_path = self.models_dir / safe_name
        file.save(str(save_path))
        return jsonify(success=True, message=f"Model saved to {save_path}")

    def _list_pose_models(self):
        models = []
        if self.models_dir.exists():
            for f in sorted(self.models_dir.iterdir()):
                if f.suffix.lower() in (".glb", ".gltf"):
                    models.append(f.name)
        return jsonify(models=models, active=self._active_pose_model)

    def _get_active_model(self):
        return jsonify(active=self._active_pose_model)

    def _select_model(self):
        data = request.get_json(silent=True) or {}
        name = data.get("name", "")
        if not name:
            self._active_pose_model = ""
            return jsonify(success=True, active="")
        model_path = self.models_dir / name
        if not model_path.exists():
            return jsonify(error="Model not found"), 404
        self._active_pose_model = name
        return jsonify(success=True, active=name)
