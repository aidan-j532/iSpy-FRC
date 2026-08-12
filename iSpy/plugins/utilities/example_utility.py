from iSpy.plugins.bases import UtilityBase

class YourUtility(UtilityBase):
    plugin_name = "example_utility"

    @classmethod
    def config_schema(cls) -> dict:
        # Declare your add-on's settings so the web UI can render an editor
        # and the loader can apply defaults. e.g.:
        #   {"greeting": {"type": "text", "label": "Greeting",
        #                 "default": "hello from my dashboard :)"}}
        return {}

    def __init__(self, context: dict):
        super().__init__(context)
        # self.config is an iSpyAddonConfig view of YOUR add-on's settings
        # (schema defaults already merged in). Presence in the config ==
        # enabled - there is no "enabled" flag.
        self.flask_app = context.get("flask_app")  # grab what you need

        if self.flask_app:
            self.flask_app.add_url_rule("/dashboard", "dashboard", self._route)

    def _route(self):
        return "hello from my dashboard :)"

    def stop(self):
        pass