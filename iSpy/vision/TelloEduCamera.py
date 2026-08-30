"""Backward-compatible alias for the DJI Tello camera source.

The Tello source lives in ``iSpy/vision/Cameras/TelloCamera.py`` now.
``TelloEduCamera`` is kept so existing imports keep working.
"""

from iSpy.vision.Cameras.TelloCamera import TelloCamera as TelloEduCamera
from iSpy.vision.Cameras.TelloCamera import TelloCamera

__all__ = ["TelloEduCamera", "TelloCamera"]