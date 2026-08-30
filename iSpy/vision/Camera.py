"""Compatibility facade over the camera-source classes in ``Cameras/``.

The real frame machinery now lives in ``iSpy/vision/Cameras/`` (see
``CameraBase``, ``OpenCVCamera`` and ``TelloCamera``). ``Camera`` is kept as a
self-delegating proxy so pipelines, the vision loop, the web module and all
existing tests keep working without renames:

- constructing ``Camera(...)`` builds the right camera source (picked from the
  config's ``camera_type``) and stores it as ``self._delegate``,
- every attribute read falls through to the delegate via ``__getattr__``
  (``frame``, ``cap``, ``get_frame()``, ``calibration_*``, ...),
- writes forward to the delegate when the source owns that attribute, so the
  source's own state (``stopped``, ``cap``, ...) is shared - pipeline-private
  attributes that the source never defines stay local to the pipeline instance.
"""

import time

from iSpy.vision.Cameras import create_camera
from iSpy.vision.Cameras.base import CameraBase, CameraOpenTimeout
from iSpy.vision.Cameras.base import cv2
from iSpy.vision.Object import Object


class Camera:
    """A camera *source* facade: each instance wraps a real ``Cameras/`` source.

    Kept source-agnostic so pipelines branch by ``camera_type`` only when they
    care, and never need to know which class actually produces the frames.
    """

    _CALIBRATION_HEARTBEAT_TIMEOUT = 10.0

    # Real class-level methods - needed by schema/discovery tooling and by
    # tests that use Camera.__new__(Camera) without a delegate.
    _get_capture_backend_candidates = CameraBase._get_capture_backend_candidates
    config_schema = CameraBase.config_schema
    discover = CameraBase.discover

    def __init__(
        self,
        camera_config,
        input_size: tuple | None = None,
        grayscale: bool = False,
        **kwargs,
    ):
        object.__setattr__(
            self,
            "_delegate",
            create_camera(
                camera_config,
                input_size=input_size,
                grayscale=grayscale,
            ),
        )

    def __getattr__(self, name):
        delegate = self.__dict__.get("_delegate")
        if delegate is None:
            # No source was ever created (e.g. a __new__-only instance used by
            # unit tests). AttributeError keeps introspection honest.
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        return getattr(delegate, name)

    def __setattr__(self, name, value):
        if name == "_delegate":
            object.__setattr__(self, name, value)
            return
        delegate = self.__dict__.get("_delegate")
        if delegate is not None and (name in delegate.__dict__ or hasattr(delegate, name)):
            setattr(delegate, name, value)
            return
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        delegate = self.__dict__.get("_delegate")
        if delegate is not None and (name in delegate.__dict__ or hasattr(delegate, name)):
            delattr(delegate, name)
            return
        object.__delattr__(self, name)

    # ------------------------------------------------------------------
    # Real methods (no delegate required) - mirror CameraBase so the
    # plugin/demo contract holds even for __new__-only instances.
    # ------------------------------------------------------------------

    def get_demo_objects(self, frame):
        if frame is None:
            return []
        h, w = frame.shape[:2]
        # local: resolve the *pipeline's* plugin name (the yolo/qr/april tag
        # family), not the underlying camera source's.
        plugin_name = getattr(self, "plugin_name", "demo_object")
        return [
            Object(
                x=0.0,
                y=0.0,
                z=0.0,
                name=f"demo_{plugin_name}",
                confidence=0.5,
                vis_type="planar",
                vis_meta={"size": max(0.2, min(w, h) / 200.0)},
            )
        ]

    def get_debug_frame(self, frame):
        return None

    def get_debug_data(self) -> dict:
        return {}

    def plot(self, frame):
        return frame

    def set_calibration(self, active: bool):
        self.calibration_active = bool(active)
        self.calibration_last_seen = time.monotonic() if active else 0.0

    def calibration_heartbeat(self):
        self.calibration_last_seen = time.monotonic()

    def in_calibration_mode(self) -> bool:
        if not getattr(self, "calibration_active", False):
            return False
        timeout = getattr(
            self, "_CALIBRATION_HEARTBEAT_TIMEOUT", self._CALIBRATION_HEARTBEAT_TIMEOUT
        )
        return (
            time.monotonic() - getattr(self, "calibration_last_seen", 0.0)
        ) < timeout

    def destroy(self):
        # The facade must provide this as a real method: it sits before
        # VisionBase in the MRO, and super().destroy() from a pipeline would
        # otherwise land on VisionBase.destroy and never stop the source.
        delegate = self.__dict__.get("_delegate")
        if delegate is not None:
            delegate.destroy()
        else:
            self.stopped = True
            cv2.destroyAllWindows()

    def release(self):
        self.destroy()