from flask import jsonify, render_template, request
from iSpy.web.Backend.WebModule import WebModule


class SettingsModule(WebModule):
    plugin_name = "settings"

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/settings", "settings_page", lambda: render_template("settings.html"))
        flask_app.add_url_rule("/api/settings", "api_settings_get", self._get, methods=["GET"])
        flask_app.add_url_rule("/api/settings", "api_settings_post", self._post, methods=["POST"])

    def _get(self):
        return jsonify(self.context["config"].config)

    def _post(self):
        try:
            data = request.get_json(force=True)
            required = ("unit", "vision_model", "camera_configs")
            missing = [k for k in required if k not in data]
            if missing:
                return jsonify(error=f"Missing required key(s): {missing}"), 400
            self.context["config"]._update_config(data)
            self.context["config"].save()
            return jsonify(success=True)
        except Exception as e:
            return jsonify(error=str(e)), 500