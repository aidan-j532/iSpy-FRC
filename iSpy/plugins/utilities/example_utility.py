import logging

from iSpy.plugins.bases import UtilityBase


class YourUtility(UtilityBase):
    plugin_name = "example/example_utility"
    template = True

    # internet needs: while offline mode is on, the loader skips an add-on
    # with an unmet critical requirement and keeps one that only loses an
    # optional feature ("partially offline" in the UI). an add-on that never
    # touches the internet (like this one) declares nothing.
    #
    # everything needs the internet? one flag:
    # needs_internet = True
    #
    # only some settings do? override the classmethod instead:
    # @classmethod
    # def network_requirements(cls, settings):
    #     from iSpy.core.network_policy import NetworkRequirement
    #     reqs = []
    #     if settings.get("web_sync"):
    #         reqs.append(NetworkRequirement(
    #             name="web sync", reason="needs internet", critical=False))
    #     return reqs

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
                "description": "The key used to expose this utility's runtime output.",
                "default": "example_output",
            },
        }

    def __init__(self, context: dict):
        super().__init__(context)
        # self.config is this plugin's settings view, defaults already merged
        self.logger = logging.getLogger(__name__)
        self.flask_app = context.get("flask_app")
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
