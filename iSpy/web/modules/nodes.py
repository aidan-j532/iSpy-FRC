"""node editor page. just a page + one aggregate GET, writes go through the normal camera/plugin endpoints."""

from flask import jsonify, render_template

from iSpy.web.Backend.WebModule import WebModule
from iSpy.web.Backend.PluginStatus import _build_vision_pipeline_payloads
from iSpy.config.iSpyConfig import (
    get_pipeline_name,
    get_pipeline_settings,
)


class NodesModule(WebModule):
    plugin_name = "nodes"

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/nodes", "nodes_page", self._page)
        flask_app.add_url_rule("/api/nodes/graph", "api_nodes_graph", self._graph)

    def _page(self):
        return render_template("nodes.html")

    def _graph(self):
        # cameras straight from config (source of truth), schema from the same helpers the other pages use.
        config = self.context.get("config")

        from iSpy.vision.Cameras import get_camera_classes

        camera_schemas = {}
        for cam_type, cls in get_camera_classes().items():
            try:
                camera_schemas[cam_type] = cls.config_schema() or {}
            except Exception:
                camera_schemas[cam_type] = {}

        cameras = []
        for key, entry in (config.get("camera_configs", {}) if config else {}).items():
            if not isinstance(entry, dict):
                continue
            cameras.append(
                {
                    "key": key,
                    "name": entry.get("name", key),
                    "camera_type": entry.get("camera_type", "opencv"),
                    "source": entry.get("source"),
                    "pipeline": get_pipeline_name(entry),
                    "pipeline_settings": dict(get_pipeline_settings(entry) or {}),
                    # full config entry so the flow-graph inspector can edit
                    # every real key on the camera, not just the short list
                    "entry": entry,
                }
            )

        return jsonify(
            cameras=cameras,
            camera_schemas=camera_schemas,
            pipelines=_build_vision_pipeline_payloads(),
        )