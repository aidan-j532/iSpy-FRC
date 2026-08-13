"""Core vision pipelines.

Pipelines are first-class parts of iSpy, not plugins: config pipeline names
resolve to real code here, no directory scanning. Config selects one per
camera via the ``pipeline`` key.
"""

from iSpy.vision.pipelines.april_tag import AprilTagCamera
from iSpy.vision.pipelines.depth_anything import DepthAnythingCamera
from iSpy.vision.pipelines.object_detection import ObjectDetectionCamera
from iSpy.vision.pipelines.optical_flow import OpticalFlowCamera
from iSpy.vision.pipelines.qr_code import QRCodeCamera
from iSpy.vision.pipelines.yolo_world import YoloWorldCamera

PIPELINES: dict[str, type] = {
    "april_tag": AprilTagCamera,
    "depth_anything": DepthAnythingCamera,
    "object_detection": ObjectDetectionCamera,
    "optical_flow": OpticalFlowCamera,
    "qr_code": QRCodeCamera,
    "yolo_world": YoloWorldCamera,
}


def get_pipeline_classes() -> dict[str, type]:
    return dict(PIPELINES)