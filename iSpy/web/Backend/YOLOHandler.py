import logging
from pathlib import Path
from iSpy.plugins.bases import UtilityBase
from iSpy.config.iSpyConfig import get_pipeline_settings
from iSpy.vision.metadata import metadata_from_pt, write_metadata, metadata_path_for

try:
    from flask import jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

_PROJECT_ROOT = Path.cwd()
_YOLO_DIR = _PROJECT_ROOT / "YoloModels"


class YOLOHandler(UtilityBase):
    plugin_name = "yolo_handler"

    def __init__(self, context: dict):
        self.config = context["config"]
        self.logger = logging.getLogger(__name__)
        # flask_app = context.get("flask_app")

        # if flask_app and FLASK_AVAILABLE:
        #     flask_app.add_url_rule("/api/models/select", "yolo_select", self._select, methods=["POST"])
        # elif not FLASK_AVAILABLE:
        #     self.logger.warning("Flask not available - model endpoints disabled.")

    def _select(self):
        data = request.get_json(force=True) or {}
        file_path = data.get("file_path")
        if not file_path:
            return jsonify(error="file_path required"), 400

        p = Path(file_path)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        if not p.exists():
            return jsonify(error=f"Model not found: {p}"), 404

        # model selection lives in each model-backed camera's pipeline
        # settings (per-camera vision_model blocks) - never at the config
        # root, where it gets ignored and trips the next boot
        updated = []
        cams = self.config.get("camera_configs", {})
        for cam in cams.values():
            if not isinstance(cam, dict):
                continue
            settings = get_pipeline_settings(cam) or {}
            model_cfg = settings.get("vision_model")
            if not isinstance(model_cfg, dict):
                continue
            settings["vision_model"] = {**model_cfg, "source_pt": str(p), "file_path": str(p)}
            updated.append(cam.get("name", "?"))
        if not updated:
            return jsonify(error="No camera uses a user-selectable model"), 400
        self.config.save()
        return jsonify(success=True, note="Restart iSpy (or re-run boot.py) to convert and load this model.", cameras=updated)

    def stop(self):
        pass
