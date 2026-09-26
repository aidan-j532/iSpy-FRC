from iSpy.plugins.bases import FrameProcessorBase


class YourTracker(FrameProcessorBase):
    plugin_name = "example/example_frame_processor"
    template = True

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
