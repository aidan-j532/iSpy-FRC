"""Built-in no-op frame processor.

Every camera lane needs an explicit frame processor.  The example processor
intentionally blacks frames out, so use this pass-through when a lane should
feed the pipeline without changing the image.
"""

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
