from pathlib import Path
from flask import jsonify, render_template
from iSpy.web.Backend.WebModule import WebModule

class Viewer3DModule(WebModule):
    plugin_name = "viewer3d"

    def __init__(self, context: dict):
        super().__init__(context)
        self._latest_objects = []
        self._cached_num_keypoints = None

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/viewer3d", "viewer3d_page", lambda: render_template("viewer3d.html"))
        flask_app.add_url_rule("/api/detections/latest", "api_detections_latest", self._latest)

    def update(self, frame_data: dict):
        fuel_list = frame_data.get("fuel_list", [])
        if self._cached_num_keypoints is None:
            config = self.context.get("config", None)
            # The model is configured per camera under pipeline settings;
            # fall back to the first model-backed camera found.
            vm = {}
            if config:
                from iSpy.config.iSpyConfig import get_pipeline_settings
                for cam in config.get("camera_configs", {}).values():
                    if not isinstance(cam, dict):
                        continue
                    settings = get_pipeline_settings(cam) or {}
                    candidate = settings.get("vision_model")
                    if isinstance(candidate, dict) and candidate.get("source_pt"):
                        vm = candidate
                        break
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
                "vis_type": getattr(obj, "vis_type", "generic"),
                "vis_meta": getattr(obj, "vis_meta", {}) or {},
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