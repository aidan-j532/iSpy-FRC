from iSpy.plugins.bases import FrameProcessorBase

class YourTracker(FrameProcessorBase):
    plugin_name = "example_frame_processor"

    @classmethod
    def config_schema(cls) -> dict:
        # e.g. {"darken": {"type": "toggle", "label": "Darken Frame",
        #                  "default": True}}
        return {}

    def __init__(self, context: dict):
        super().__init__(context)
        # self.config is an iSpyAddonConfig view of YOUR add-on's settings.
        # Presence in the config == enabled - there is no "enabled" flag.
        self.count = 0

    def process(self, frame):
        # Just make black
        frame = frame * 0
        return frame

    def stop(self):
        pass