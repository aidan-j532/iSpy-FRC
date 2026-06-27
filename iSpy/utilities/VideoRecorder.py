import logging

from iSpy.plugins.bases import UtilityBase
from iSpy.utilities.VideoRecorder import VideoRecorder as _VideoRecorder


class VideoRecorder(UtilityBase):
    plugin_name = "video_recorder"

    def __init__(self, context: dict):
        config = context.get("config", {})
        self.logger = logging.getLogger(__name__)

        self._enabled = config.get("record_mode", False)
        self._started = False
        self._recorder = None

        if self._enabled:
            self._recorder = _VideoRecorder(
                output_dir=config.get("record_dir", "VideoRecordings")
            )

    def update(self, frame_data: dict):
        if not self._enabled or self._recorder is None:
            return

        frame = frame_data.get("frame")
        if frame is None:
            return

        if not self._started:
            h, w = frame.shape[:2]
            self._recorder.start(w, h)
            self._started = True
            self.logger.info("Video recording started")

        self._recorder.write(frame)

    def stop(self):
        if self._recorder:
            self._recorder.stop()