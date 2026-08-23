from pathlib import Path
from flask import jsonify, render_template
from iSpy.web.Backend.WebModule import WebModule


class Viewer3DModule(WebModule):
    """3D viewer backend.

    Provides a generic overlay system so any add-on can contribute 3D objects
    to the viewer without the viewer knowing about specific types.

    Overlay format (JSON)::

        {
            "id": "unique_id",
            "type": "box",           # renderer type
            "x": 0, "y": 0, "z": 0, # position (field coords)
            "roll": 0, "pitch": 0, "yaw": 0,
            "label": "optional",
            "color": "#4c8bf5",
            "data": { ... }          # type-specific payload
        }

    Built-in renderer types:
        box      -- data: {width, height, depth}
        sphere   -- data: {radius}
        group    -- data: {children: [...overlays...]}
    """

    plugin_name = "viewer3d"

    def __init__(self, context: dict):
        super().__init__(context)
        self._latest_objects = []
        self._cached_num_keypoints = None
        self._overlays: dict[str, dict] = {}

    # -- overlay API (called by add-ons) ----------------------------------

    def add_overlay(self, overlay_id: str, overlay: dict) -> None:
        """Register or update an overlay.  Call from any add-on that has a
        reference to this module (via ``context["vision_instance"].web_app``)."""
        overlay["id"] = overlay_id
        self._overlays[overlay_id] = overlay

    def remove_overlay(self, overlay_id: str) -> None:
        """Remove a previously registered overlay."""
        self._overlays.pop(overlay_id, None)

    # -- routes ------------------------------------------------------------

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/viewer3d", "viewer3d_page",
                               lambda: render_template("viewer3d.html"))
        flask_app.add_url_rule("/api/detections/latest", "api_detections_latest",
                               self._latest)
        flask_app.add_url_rule("/api/overlays", "api_overlays",
                               self._overlays_endpoint)

    # -- update (called every vision tick) ---------------------------------

    def update(self, frame_data: dict):
        detections = frame_data.get("detections", [])
        if self._cached_num_keypoints is None:
            config = self.context.get("config", None)
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
        for idx, obj in enumerate(detections):
            if getattr(obj, "depth_source", "") == "optical_flow":
                continue
            # universal pipeline-output schema (iSpy/vision/pipelines/base.py)
            if hasattr(obj, "to_dict"):
                obj_entry = obj.to_dict()
            else:  # legacy plain-object fallback
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
                    "vis_type": getattr(obj, "vis_type", "generic"),
                    "vis_meta": getattr(obj, "vis_meta", {}) or {},
                }
            obj_entry["id"] = idx
            obj_entry["num_keypoints"] = num_kpts
            kpts = obj_entry.get("keypoints_3d")
            if kpts is None:
                kpts = getattr(obj, "keypoints_3d", None)
                if kpts is not None:
                    obj_entry["keypoints_3d"] = kpts
            self._latest_objects.append(obj_entry)

    # -- internals ---------------------------------------------------------

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

    def _overlays_endpoint(self):
        return jsonify(overlays=list(self._overlays.values()))
