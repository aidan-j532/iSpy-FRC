from iSpy.vision.pipelines.april_tag import AprilTagPipeline, AprilTagCamera
from iSpy.vision.pipelines.depth_anything import DepthAnythingPipeline, DepthAnythingCamera
from iSpy.vision.pipelines.object_detection import ObjectDetectionPipeline, ObjectDetectionCamera
from iSpy.vision.pipelines.optical_flow import OpticalFlowPipeline, OpticalFlowCamera
from iSpy.vision.pipelines.qr_code import QRCodePipeline, QRCodeCamera
from iSpy.vision.pipelines.yolo_world import YoloWorldPipeline, YoloWorldCamera

PIPELINES: dict[str, type] = {
    "april_tag": AprilTagPipeline,
    "depth_anything": DepthAnythingPipeline,
    "object_detection": ObjectDetectionPipeline,
    "optical_flow": OpticalFlowPipeline,
    "qr_code": QRCodePipeline,
    "yolo_world": YoloWorldPipeline,
}


def get_pipeline_classes() -> dict[str, type]:
    return dict(PIPELINES)