"""Camera sources: anything that produces video frames.

Mirrors the pipeline registry. Each camera source is a class in
``iSpy/vision/Cameras/`` that implements the :class:`CameraBase` contract and
declares a unique ``camera_type``. :func:`create_camera` picks the class by
``camera_type`` and instantiates it - exactly like ``create_pipeline`` picks a
pipeline by name. Adding a new source is just a new class in this folder.
"""

from copy import deepcopy

from iSpy.config.iSpyConfig import iSpyCameraConfig
from iSpy.vision.Cameras.base import CameraBase, CameraOpenTimeout
from iSpy.vision.Cameras.OpenCVCamera import OpenCVCamera
from iSpy.vision.Cameras.TelloCamera import TelloCamera

BUILTIN_CAMERAS = {
    OpenCVCamera.camera_type: OpenCVCamera,
    TelloCamera.camera_type: TelloCamera,
}

CAMERA_TYPE_LABELS = {
    "opencv": "OpenCV (USB / stream / index)",
    "tello": "DJI Tello / Tello Edu",
}


def get_camera_classes() -> dict[str, type[CameraBase]]:
    """All registered camera-source classes, keyed by ``camera_type``."""
    return dict(BUILTIN_CAMERAS)


def get_camera_class(camera_type: str | None = None) -> type[CameraBase]:
    """Return the camera-source class for ``camera_type`` (default opencv)."""
    if camera_type is None:
        camera_type = "opencv"
    cls = BUILTIN_CAMERAS.get(camera_type)
    if cls is None:
        raise ValueError(
            f"Unknown camera type '{camera_type}' - available: "
            f"{', '.join(sorted(BUILTIN_CAMERAS))}"
        )
    return cls


def create_camera(
    camera_config,
    input_size: tuple | None = None,
    grayscale: bool = False,
    camera_type: str | None = None,
    camera_source=None,
):
    """Instantiate the camera source for ``camera_config``.

    Source selection (first match wins):

    - explicit ``camera_type`` argument,
    - ``camera_type`` key in the camera config,
    - otherwise the default ``opencv`` source.

    ``camera_source`` overrides the ``source`` key when given (used by the web
    UI when a discovered camera is picked).
    """
    if isinstance(camera_config, dict):
        config = iSpyCameraConfig(deepcopy(camera_config))
    else:
        config = camera_config

    resolved_type = str(camera_type or config.get("camera_type") or "opencv")
    config.data["camera_type"] = resolved_type

    if camera_source is not None:
        config.data["source"] = camera_source

    cls = get_camera_class(resolved_type)
    return cls(config, input_size=input_size, grayscale=grayscale)