import os
from pathlib import Path
from flask import jsonify, render_template, request
from werkzeug.utils import secure_filename
from iSpy.web.Backend.WebModule import WebModule

class Viewer3DModule(WebModule):
    plugin_name = "viewer3d"

    def __init__(self, context: dict):
        super().__init__(context)
        self._latest_objects = []
        # Create a directory to store the uploaded models
        self.models_dir = Path.cwd() / "Outputs" / "3d_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/viewer3d", "viewer3d_page", lambda: render_template("viewer3d.html"))
        flask_app.add_url_rule("/api/detections/latest", "api_detections_latest", self._latest)
        
        # ADD THIS: The new route for uploading
        flask_app.add_url_rule("/api/upload_glb", "api_upload_glb", self._upload_glb, methods=["POST"])

    def _latest(self):
        return jsonify(objects=self._latest_objects)

    def _upload_glb(self):
        if "file" not in request.files:
            return jsonify(error="No file provided"), 400
            
        file = request.files["file"]
        if file.filename == '':
            return jsonify(error="Empty filename"), 400
            
        safe_name = secure_filename(file.filename + self.context.config.get("vision_model", {}).get("source_pt", ""))
        save_path = self.models_dir / safe_name
        file.save(str(save_path))
        
        return jsonify(success=True, message=f"Model saved to {save_path}")