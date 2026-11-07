
from iSpy.plugins.bases import FrameProcessorBase

class Passthrough(FrameProcessorBase):
    plugin_name = "passthrough"

    @classmethod
    def config_schema(cls) -> dict:
        return {}

    def process(self, frame):
        return frame

    def stop(self):
        pass
