import math
import time
from pathlib import Path

import numpy as np
from flask import jsonify, render_template

from iSpy import units as unit_packs
from iSpy.core import performance
from iSpy.vision import triangulation
from iSpy.web.Backend.WebModule import WebModule

_VIEWER_GRACE_S = 5.0

class Viewer3DModule(WebModule):
    plugin_name = "viewer3d"

    def __init__(self, context: dict):
        super().__init__(context)
        self._latest_objects = []
        self._cached_num_keypoints = None
        self._cached_camera_names = None
        self._overlays: dict[str, dict] = {}
        self._last_poll = 0.0

    def has_viewers(self, now: float = 0.0) -> bool:
        now = now or time.monotonic()
        return (now - self._last_poll) < _VIEWER_GRACE_S

    def add_overlay(self, overlay_id: str, overlay: dict) -> None:
        overlay["id"] = overlay_id
        self._overlays[overlay_id] = overlay

    def remove_overlay(self, overlay_id: str) -> None:
        self._overlays.pop(overlay_id, None)

    def _unit(self) -> str:
        config = self.context.get("config")
        return config.get("unit", "frc") if config else "frc"

    def _viewer_config(self) -> dict:
        return unit_packs.viewer_settings(self._unit())

    def register_routes(self, flask_app):
        flask_app.add_url_rule(
            "/viewer3d", "viewer3d_page", self._page
        )
        flask_app.add_url_rule(
            "/api/viewer3d/config", "api_viewer3d_config", self._config
        )
        flask_app.add_url_rule(
            "/api/detections/latest", "api_detections_latest", self._latest
        )
        flask_app.add_url_rule("/api/overlays", "api_overlays", self._overlays_endpoint)

    def _page(self):
        return render_template("viewer3d.html", viewer_config=self._viewer_config())

    def _config(self):
        self._last_poll = time.monotonic()
        return jsonify(self._viewer_config())

    def update(self, frame_data: dict):
        if (
            performance.get_profile().viewer3d_open_only
            and (time.monotonic() - self._last_poll) >= _VIEWER_GRACE_S
        ):
            return
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
            obj_entry["num_keypoints"] = num_kpts
            kpts = obj_entry.get("keypoints_3d")
            if kpts is None:
                kpts = getattr(obj, "keypoints_3d", None)
                if kpts is not None:
                    obj_entry["keypoints_3d"] = kpts
            self._latest_objects.append(obj_entry)

    def _camera_display_name(self, cam) -> str:
        if hasattr(cam, "config") and cam.config is not None:
            name = cam.config.get("name")
            if name:
                return str(name)
        return str(getattr(cam, "source", "camera"))

    def _camera_orientation(self, yaw_deg: float, pitch_deg: float, viewer: dict):
        """Camera euler in the robot frame, including optional mount trim."""
        rotation = triangulation.camera_rotation_to_world(
            np.eye(3, dtype=float), yaw_deg, pitch_deg
        )
        roll_off = math.radians(viewer.get("camera_roll_offset_deg", 0.0) or 0.0)
        pitch_off = math.radians(viewer.get("camera_pitch_offset_deg", 0.0) or 0.0)
        yaw_off = math.radians(viewer.get("camera_yaw_offset_deg", 0.0) or 0.0)
        if roll_off or pitch_off or yaw_off:
            rotation = rotation @ triangulation.euler_to_matrix(
                roll_off, pitch_off, yaw_off
            )
        return triangulation.matrix_to_euler(rotation)

    def _refresh_camera_overlays(self):
        unit = self._unit()
        viewer = self._viewer_config()
        current_names = set()
        for cam in self.context.get("cameras", []):
            cfg = getattr(cam, "config", None)
            if not hasattr(cfg, "get"):
                continue
            name = self._camera_display_name(cam)
            current_names.add(name)
            fov = cfg.get("calibration", {}).get("fov", 0)
            if fov <= 0:
                fov = viewer["camera"]["fov"] if viewer else 60
            x, y, z = unit_packs.input_point_to_output(
                (
                    cfg.get("x", 0) or 0,
                    cfg.get("y", 0) or 0,
                    cfg.get("height", 0) or 0,
                ),
                unit,
            )
            roll, pitch, yaw = self._camera_orientation(
                cfg.get("yaw", 0) or 0,
                cfg.get("pitch", 0) or 0,
                viewer,
            )
            self.add_overlay(
                f"camera:{name}",
                {
                    "type": "camera",
                    "x": x,
                    "y": y,
                    "z": z,
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
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
        self._last_poll = time.monotonic()
        return jsonify(objects=self._latest_objects)

    def _overlays_endpoint(self):
        self._last_poll = time.monotonic()
        return jsonify(overlays=list(self._overlays.values()))
