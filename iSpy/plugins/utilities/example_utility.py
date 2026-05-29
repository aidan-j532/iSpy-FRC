from iSpy.plugins.bases import UtilityBase

class YourUtility(UtilityBase):
    plugin_name = "example_utility"
    def __init__(self, context: dict):
        self.config = context["config"]
        self.flask_app = context["flask_app"]  # grab what you need
        
        if self.flask_app:
            self.flask_app.add_url_rule("/dashboard", "dashboard", self._route)

    def _route(self):
        return "hello from my dashboard :)"

    def stop(self):
        pass