from iSpy.plugins.bases import FrameProcessorBase


class YourTracker(FrameProcessorBase):
    plugin_name = "example/example_frame_processor"
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
        # e.g. {"darken": {"type": "toggle", "label": "Darken", "default": True}}
        return {}

    def __init__(self, context: dict):
        super().__init__(context)
        # self.config is this plugin's settings view, defaults already merged
        self.count = 0

    def process(self, frame):
        frame = frame * 0
        return frame

    def stop(self):
        pass
