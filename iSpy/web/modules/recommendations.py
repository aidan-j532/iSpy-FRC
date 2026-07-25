from flask import jsonify, render_template
from iSpy.web.Backend.WebModule import WebModule
from iSpy.validations.recommendations import get_structured_recommendations

class RecommendationsModule(WebModule):
    plugin_name = "recommendations"

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/recommendations", "recommendations_page", lambda: render_template("recommendations.html"))
        flask_app.add_url_rule("/api/recommendations", "api_recommendations", self._get)

    def _get(self):
        config = self.context["config"]
        recs = get_structured_recommendations(config.config)
        return jsonify(recommendations=recs,
                        critical_count=sum(1 for r in recs if r["severity"] == "critical"))