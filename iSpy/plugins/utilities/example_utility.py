import logging

from iSpy.plugins.bases import UtilityBase

class YourUtility(UtilityBase):
    """Example utility - copy this file to build your own.

    Declaring "output_key" in config_schema makes this utility's runtime
    value available to other systems (e.g. NetworkTables publishing) as:
        frame_data["addon_data"][<output_key>]
    selectable in the NetworkTables source dropdown as: addon_data.<output_key>
    """

    plugin_name = "example_utility"
    template = True

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "greeting": {
                "type": "text",
                "label": "Greeting",
                "default": "hello from my dashboard :)",
            },
            "output_key": {
                "type": "text",
                "label": "Output Key",
                "description": "The key used to expose this utility's "
                               "runtime output.",
                "default": "example_output",
            },
        }

    def __init__(self, context: dict):
        super().__init__(context)
        # self.config = YOUR settings view (defaults merged in); presence == enabled, no flag
        self.logger = logging.getLogger(__name__)
        self.flask_app = context.get("flask_app")  # grab what you need
        self._ticks = 0

        if self.flask_app:
            self.flask_app.add_url_rule("/ispy-example", "ispy_example", self._route)

    def _route(self):
        return self.config.get("greeting", "hello from my dashboard :)")

    def update(self, frame_data: dict):
        # expose a runtime value every tick under addon_data[<output_key>]
        self._ticks += 1
        self.publish_output(frame_data, self._ticks)

    def stop(self):
        pass
