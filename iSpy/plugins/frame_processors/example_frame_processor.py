from iSpy.plugins.bases import FrameProcessorBase


class YourTracker(FrameProcessorBase):
    plugin_name = "example/example_frame_processor"
    template = True

    # set needs_internet = True to skip the add-on while offline, or override
    # network_requirements() when only some settings need the internet

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
