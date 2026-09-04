from iSpy.plugins.bases import FrameProcessorBase

class YourTracker(FrameProcessorBase):
    plugin_name = "example_frame_processor"
    template = True

    @classmethod
    def config_schema(cls) -> dict:
        # e.g. {"darken": {"type": "toggle", "label": "Darken Frame",
        #                  "default": True}}
        return {}

    def __init__(self, context: dict):
        super().__init__(context)
        # self.config = iSpyAddonConfig view of YOUR settings; presence == enabled, no flag
        self.count = 0

    def process(self, frame):
        # Just make black
        frame = frame * 0
        return frame

    def stop(self):
        pass