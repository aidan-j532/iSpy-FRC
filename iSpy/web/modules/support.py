import logging
from flask import jsonify, render_template, request
from iSpy.web.Backend.WebModule import WebModule

logger = logging.getLogger(__name__)

class SupportModule(WebModule):
    plugin_name = "support"

    def __init__(self, context: dict):
        super().__init__(context)
        self.logger = logging.getLogger(__name__)

    def register_routes(self, flask_app):
        flask_app.add_url_rule(
            "/support", "support_page", lambda: render_template("support.html")
        )
        flask_app.add_url_rule(
            "/api/support/donate", "api_support_donate", self._api_donate,
            methods=["POST"]
        )

    def _api_donate(self):
        try:
            data = request.get_json()
            amount = data.get("amount", 0) if data else 0
            return jsonify({
                "ok": True,
                "message": f"Donation of ${amount} recorded (no real processing yet)",
                "amount": amount
            })
        except Exception as e:
            logger.exception("Donation API error")
            return jsonify({
                "ok": False,
                "error": str(e)
            }), 400

    def stop(self):
        pass