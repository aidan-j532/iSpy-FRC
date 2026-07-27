from iSpy.plugins.bases import FrameProcessorBase

class YourTracker(FrameProcessorBase):
    plugin_name = "example_frame_processor"
    def __init__(self, config):
        super().__init__(config)
        self.count = 0

    def process(self, frame):
        # Just make black
        frame = frame * 0
        return frame
    
    def stop(self):
        pass