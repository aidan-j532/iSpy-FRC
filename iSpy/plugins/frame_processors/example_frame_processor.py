from iSpy.plugins.bases import FrameProcessorBase


class YourTracker(FrameProcessorBase):
    plugin_name = "example/example_frame_processor"
    template = True

    # internet needs for offline mode: skip when critical + unmet, "partially
    # offline" when non-critical. this add-on needs none, so declares nothing.
    # all-in-one flag:  needs_internet = True
    # or per-setting, override network_requirements(cls, settings) and return
    # NetworkRequirement(name=..., reason=..., critical=False) entries:
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
