from flask import jsonify, render_template
from iSpy.web.Backend.WebModule import WebModule


class PluginStatusModule(WebModule):
    plugin_name = "plugin_status"

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/plugins", "plugins_page", lambda: render_template("plugins.html"))
        flask_app.add_url_rule("/api/plugins/status", "api_plugins_status", self._status)

    def _status(self):
        vision = self.context.get("vision_instance")
        if not vision:
            return jsonify(plugins=[])
        out = []
        for group, items in (
            ("tracker", vision.trackers),
            ("utility", vision.utilities),
            ("frame_processor", vision.frame_processors),
        ):
            for name, inst in items.items():
                out.append({
                    "name": name,
                    "type": group,
                    "status": inst.get_status() if hasattr(inst, "get_status") else "unknown",
                })
        return jsonify(plugins=out)