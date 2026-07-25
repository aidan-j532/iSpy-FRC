import jsonify
from flask import render_template


def register_routes(self, flask_app):
    flask_app.add_url_rule("/plugins", "plugins_page", lambda: render_template("plugins.html"))
    flask_app.add_url_rule("/api/plugins/status", "api_plugins_status", self._status)

def _status(self):
    vision = self.context.get("vision_instance")  # needs to be threaded into context
    if not vision:
        return jsonify(plugins=[])
    out = []
    for group, items in (("tracker", vision.trackers), ("utility", vision.utilities), ("frame_processor", vision.frame_processors)):
        for name, inst in items.items():
            out.append({"name": name, "type": group, "status": inst.get_status() if hasattr(inst, "get_status") else "unknown"})
    return jsonify(plugins=out)