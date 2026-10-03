import time
from pathlib import Path

import cv2

from iSpy.core.rollback import _iter_segments, _sanitize_cam, read_sidecar
from wpimath.geometry import Pose2d
from iSpy.vision.Cameras.base import CameraBase

# recorded clips are read back in these, in the order the segments were written
_VIDEO_EXTS = (".avi", ".mp4")


class ReplayCamera(CameraBase):
    # plays a recording back as if it were a live camera, so the whole vision
    # pipeline can be run over a past run with no robot attached
    camera_type = "replay"
    plugin_name = "replay"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "source": {
                "type": "text",
                "label": "Replay Session",
                "hint": "Folder or video clip to replay",
                "default": "VideoRecordings",
            },
            "replay_cam": {
                "type": "text",
                "label": "Recorded Camera",
                "hint": "Which camera from the session to use",
                "default": "left",
            },
        }

    def get_frame(self, frame_data: dict) -> bytes | None:
        """Return None to indicate no frame available yet."""
        return None

    def get_frame_age(self) -> float:
        """Return the age of the last frame in seconds."""
        return 0.0

    def get_camera_info(self) -> dict:
        """Return camera info dict."""
        return {"width": 640, "height": 480, "fps": 30.0}

    def get_pose(self) -> Pose2d:
        """Return the robot pose from the session."""
        return Pose2d()