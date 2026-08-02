import logging
from pathlib import Path

import cv2
import numpy as np

from iSpy.vision.Camera import Camera
from iSpy.plugins.bases import VisionBase
from iSpy.config.iSpyConfig import iSpyConfig, iSpyCameraConfig
from iSpy.vision.Object import Object


class YoloWorldCamera(Camera, VisionBase):
    plugin_name = "yolo_world"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "prompt": {
                "type": "text",
                "label": "Prompt",
                "default": "A dog.",
            },
        }

    def __init__(self, camera_config: iSpyCameraConfig, config: iSpyConfig, core_mask=None):
        self.logger = logging.getLogger(__name__)
        self.config = camera_config
        self.core_mask = core_mask
        self.prompt = str(camera_config.get("prompt") or "A dog.")
        self.model = None
        self._model_path = None
        super().__init__(camera_config, (640, 480), camera_config.get("grayscale", False))

        self._load_model()

    def _load_model(self):
        asset_path = Path(__file__).resolve().parents[3] / "assets" / "yolo-world.pt"
        self._model_path = asset_path
        if not asset_path.exists():
            self.logger.error("YOLO World weights not found at %s", asset_path)
            self.model = None
            return

        try:
            from ultralytics import YOLOWorld
        except Exception as exc:  # pragma: no cover - runtime dependency fallback
            self.logger.error("Ultralytics is required for YOLO World inference: %s", exc)
            self.model = None
            return

        try:
            self.model = YOLOWorld(str(asset_path), verbose=False)
            self.model.set_classes([self.prompt])
            self.logger.info("Loaded YOLO World model from %s", asset_path)
        except Exception as exc:  # pragma: no cover - runtime dependency fallback
            self.logger.exception("Failed to load YOLO World model: %s", exc)
            self.model = None

    def run(self):
        frame = self.get_frame()
        if frame is None:
            return [], None

        if self.model is None:
            return [], frame

        try:
            results = self.model(frame, stream=False, conf=0.25, imgsz=640)
            objects: list[Object] = []
            annotated = frame.copy()

            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name = getattr(result.names, str(cls_id), str(cls_id)) if hasattr(result, "names") else str(cls_id)
                    objects.append(
                        Object(
                            x=float((x1 + x2) / 2.0),
                            y=float((y1 + y2) / 2.0),
                            z=0.0,
                            name=name,
                            confidence=conf,
                            vis_type="generic",
                            vis_meta={"prompt": self.prompt, "kind": "detection"},
                        )
                    )

            if isinstance(results, list) and results:
                annotated = results[0].plot() if hasattr(results[0], "plot") else annotated
            elif hasattr(results, "plot"):
                annotated = results.plot()

            if annotated is None:
                annotated = frame
            return objects, annotated
        except Exception as exc:  # pragma: no cover - runtime dependency fallback
            self.logger.exception("YOLO World inference failed: %s", exc)
            return [], frame

    def plot(self, frame):
        if frame is None:
            return None
        try:
            overlay = frame.copy()
            cv2.putText(overlay, "YOLO World", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            return overlay
        except Exception:
            return frame

    def destroy(self):
        self.stopped = True
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
