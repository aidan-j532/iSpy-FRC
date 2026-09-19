import math
from pathlib import Path

from flask import jsonify, render_template

from iSpy.config.iSpyConfig import unit_to_inches
from iSpy.web.Backend.WebModule import WebModule

# inches -> config output unit scale, same literal duplicated across every
# vision pipeline (see iSpy/vision/pipelines/object_detection.py self.conversions).
_INCHES_TO_OUTPUT_UNIT = {
    "meter": 0.0254,
    "meters": 0.0254,
    "inch": 1.0,
    "inches": 1.0,
    "foot": 1 / 12,
    "feet": 1 / 12,
    "centimeter": 2.54,
    "centimeters": 2.54,
    # FRC/WPILib convention: meters out (robot code), calibration in inches
    "frc": 0.0254,
}


class Viewer3DModule(WebModule):
    plugin_name = "viewer3d"

    def __init__(self, context: dict):
        super().__init__(context)
        self._latest_objects = []
        self._cached_num_keypoints = None
        self._cached_camera_names = None
        self._overlays: dict[str, dict] = {}

    # overlay API (called by add-ons)

    def add_overlay(self, overlay_id: str, overlay: dict) -> None:
        overlay["id"] = overlay_id
        self._overlays[overlay_id] = overlay

    def remove_overlay(self, overlay_id: str) -> None:
        self._overlays.pop(overlay_id, None)

    # routes

    def register_routes(self, flask_app):
        flask_app.add_url_rule(
            "/viewer3d", "viewer3d_page", lambda: render_template("viewer3d.html")
        )
        flask_app.add_url_rule(
            "/api/detections/latest", "api_detections_latest", self._latest
        )
        flask_app.add_url_rule("/api/overlays", "api_overlays", self._overlays_endpoint)

    # update (called every vision tick)

    def update(self, frame_data: dict):
        camera_names = {
            self._camera_display_name(cam) for cam in self.context.get("cameras", [])
        }
        if camera_names != self._cached_camera_names:
            self._cached_camera_names = camera_names
            self._refresh_camera_overlays()
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
            # keep the Object's OWN stable id (to_dict's "id"). Overwriting it
            # with this frame's list index churns the id every tick for fresh
            # detections and - worse - splits persistent objects like the voxel
            # world, whose centroid jumps enough to fail a tracker's distance
            # gate, into ghost render groups that flicker until their
            # stale_threshold expires. The legacy plain-object fallback above
            # already tags those with idx.
            obj_entry["num_keypoints"] = num_kpts
            kpts = obj_entry.get("keypoints_3d")
            if kpts is None:
                kpts = getattr(obj, "keypoints_3d", None)
                if kpts is not None:
                    obj_entry["keypoints_3d"] = kpts
            self._latest_objects.append(obj_entry)

    # -- internals ---------------------------------------------------------

    def _camera_display_name(self, cam) -> str:
        # same as CamerasModule: config name if present, else source.
        if hasattr(cam, "config") and cam.config is not None:
            name = cam.config.get("name")
            if name:
                return str(name)
        return str(getattr(cam, "source", "camera"))

    def _refresh_camera_overlays(self):
        # static per-camera overlays - only rebuilt when camera set changes,
        # not every tick. config yaw/pitch are degrees; renderers want radians.
        config = self.context.get("config")
        unit = config.get("unit", "frc") if config else "frc"
        scale = _INCHES_TO_OUTPUT_UNIT.get(unit, _INCHES_TO_OUTPUT_UNIT["frc"])
        current_names = set()
        for cam in self.context.get("cameras", []):
            cfg = getattr(cam, "config", None)
            if not hasattr(cfg, "get"):
                continue
            name = self._camera_display_name(cam)
            current_names.add(name)
            fov = cfg.get("calibration", {}).get("fov", 0)
            if fov <= 0:
                fov = 60
            self.add_overlay(
                f"camera:{name}",
                {
                    "type": "camera",
                    "x": unit_to_inches(cfg.get("x", 0) or 0, unit) * scale,
                    "y": unit_to_inches(cfg.get("y", 0) or 0, unit) * scale,
                    "z": unit_to_inches(cfg.get("height", 0) or 0, unit) * scale,
                    "roll": 0,
                    "yaw": math.radians(cfg.get("yaw", 0) or 0),
                    "pitch": math.radians(cfg.get("pitch", 0) or 0),
                    "label": name,
                    "data": {"fov": fov},
                },
            )
        stale = [
            oid
            for oid in self._overlays
            if oid.startswith("camera:") and oid.split(":", 1)[1] not in current_names
        ]
        for oid in stale:
            self.remove_overlay(oid)

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
