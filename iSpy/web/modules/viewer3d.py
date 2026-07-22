from flask import jsonify, render_template
from iSpy.web.WebModule import WebModule


class Viewer3DModule(WebModule):
    plugin_name = "viewer3d"

    def __init__(self, context: dict):
        super().__init__(context)
        self._latest_objects: list[dict] = []

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/viewer3d", "viewer3d_page", lambda: render_template("viewer3d.html"))
        flask_app.add_url_rule("/api/detections/latest", "api_detections_latest", self._latest)
        # TODO: /api/models/<name>/asset  (upload/serve .glb per model, once
        # ModelsModule stores an asset path in the metadata sidecar)

    def update(self, frame_data: dict):
        fuel_list = frame_data.get("fuel_list") or []
        self._latest_objects = [
            {"id": getattr(o, "id", i), "x": o.x, "y": o.y, "z": o.z,
             "roll": o.roll, "pitch": o.pitch, "yaw": o.yaw}
            for i, o in enumerate(fuel_list)
        ]

    def _latest(self):
        return jsonify(objects=self._latest_objects)