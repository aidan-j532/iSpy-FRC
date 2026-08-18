from flask import jsonify, request

from iSpy.web.Backend.WebModule import WebModule


class OnboardingModule(WebModule):
    plugin_name = "onboarding"

    def register_routes(self, flask_app):
        flask_app.add_url_rule(
            "/api/onboarding", "api_onboarding_get", self._get, methods=["GET"]
        )
        flask_app.add_url_rule(
            "/api/onboarding", "api_onboarding_post", self._post, methods=["POST"]
        )

    def _completed(self) -> bool:
        config = self.context.get("config")
        if config is None:
            return True
        return bool(config.get_nested("onboarding", "completed", default=False))

    def _get(self):
        return jsonify(show_tour=not self._completed())

    def _post(self):
        config = self.context.get("config")
        if config is None:
            return jsonify(error="no config"), 500
        data = request.get_json(force=True) or {}
        completed = bool(data.get("completed", True))
        config.set("onboarding", {"completed": completed})
        config.save()
        return jsonify(success=True)
