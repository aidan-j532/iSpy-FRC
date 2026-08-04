"""Core vision pipelines.

Pipelines are first-class parts of iSpy, not plugins: they are imported
directly and registered here explicitly, so config pipeline names always
resolve to real code (no directory scanning, no plugin loader). Config
selects one per camera via the ``pipeline`` key.
"""

from iSpy.vision.pipelines.april_tag import AprilTagCamera
from iSpy.vision.pipelines.depth_anything import DepthAnythingCamera
from iSpy.vision.pipelines.line_tracking import LineTrackingCamera
from iSpy.vision.pipelines.object_detection import ObjectDetectionCamera
from iSpy.vision.pipelines.qr_code import QRCodeCamera
from iSpy.vision.pipelines.yolo_world import YoloWorldCamera

# Static, explicit registry: every pipeline is a direct part of the code.
PIPELINES: dict[str, type] = {
    "april_tag": AprilTagCamera,
    "depth_anything": DepthAnythingCamera,
    "line_tracking": LineTrackingCamera,
    "object_detection": ObjectDetectionCamera,
    "qr_code": QRCodeCamera,
    "yolo_world": YoloWorldCamera,
}


def get_pipeline_classes() -> dict[str, type]:
    """Return all registered camera pipelines keyed by config pipeline name."""
    return dict(PIPELINES)