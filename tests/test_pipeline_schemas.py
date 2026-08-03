import unittest

import cv2

from iSpy.plugins.bases import VisionBase
from iSpy.plugins.vision.BuiltIn.AprilTag import AprilTagCamera
from iSpy.plugins.vision.BuiltIn.QRCode import QRCodeCamera
from iSpy.plugins.vision.BuiltIn.DepthAnything import DepthAnythingCamera
from iSpy.plugins.vision.BuiltIn.LineTracking import LineTrackingCamera
from iSpy.web.Backend.PluginStatus import _build_vision_pipeline_payloads
from iSpy.vision.Camera import Camera


class VisionPipelineSchemaTests(unittest.TestCase):
    def test_base_vision_schema_is_empty_by_default(self):
        self.assertEqual(VisionBase.config_schema(), {})

    def test_april_tag_schema_exposes_tag_size_inches(self):
        schema = AprilTagCamera.config_schema()
        self.assertIn("tag_size_inches", schema)
        self.assertEqual(schema["tag_size_inches"]["type"], "number")
        self.assertEqual(schema["tag_size_inches"]["label"], "Tag Size (in)")
        self.assertEqual(schema["tag_size_inches"]["default"], 6.5)

    def test_object_detection_pipeline_is_exposed(self):
        pipelines = _build_vision_pipeline_payloads()
        object_detection = next((p for p in pipelines if p["name"] == "object_detection"), None)
        self.assertIsNotNone(object_detection)
        self.assertTrue(object_detection["show_common_fields"])

        april_tag = next((p for p in pipelines if p["name"] == "april_tag"), None)
        self.assertIsNotNone(april_tag)
        self.assertFalse(april_tag["show_common_fields"])

    def test_additional_pipeline_schemas_are_discovered(self):
        pipelines = _build_vision_pipeline_payloads()
        pipeline_names = {p["name"] for p in pipelines}
        self.assertIn("qr_code", pipeline_names)
        self.assertIn("depth_anything", pipeline_names)
        self.assertIn("line_tracking", pipeline_names)

        qr_schema = QRCodeCamera.config_schema()
        self.assertIn("decode_mode", qr_schema)

        depth_schema = DepthAnythingCamera.config_schema()
        self.assertIn("estimate_depth", depth_schema)

        line_schema = LineTrackingCamera.config_schema()
        self.assertIn("line_color", line_schema)

    def test_vision_base_exposes_debug_contract(self):
        class DummyVision(VisionBase):
            def run(self):
                return [], None

            def destroy(self):
                pass

        base = DummyVision.__new__(DummyVision)
        self.assertEqual(base.get_debug_data(), {})
        self.assertIsNone(base.get_debug_frame(None))
        self.assertIsNone(base.plot(None))

    def test_plugin_plot_hook_returns_annotated_frame(self):
        import numpy as np

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        annotated = AprilTagCamera.__new__(AprilTagCamera).plot(frame)
        self.assertIsNotNone(annotated)
        self.assertTrue(np.array_equal(annotated, frame) or annotated.shape == frame.shape)

    def test_windows_capture_backend_candidates_use_msmf_only(self):
        candidates = Camera._get_capture_backend_candidates("Windows")
        self.assertEqual(candidates, [cv2.CAP_MSMF])

    def test_camera_demo_objects_are_emitted_for_placeholder_visualization(self):
        import numpy as np

        camera = Camera.__new__(Camera)
        camera.plugin_name = "april_tag"
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        objects = camera.get_demo_objects(frame)
        self.assertTrue(objects)
        self.assertEqual(objects[0].vis_type, "planar")
        self.assertEqual(objects[0].name, "demo_april_tag")

    def test_april_tag_run_prefers_real_detection_over_demo_placeholder(self):
        import numpy as np

        from iSpy.vision.Object import Object

        class DummyDetector:
            def detectMarkers(self, gray):
                return [], None, []

        camera = AprilTagCamera.__new__(AprilTagCamera)
        camera.get_frame = lambda: np.zeros((20, 20, 3), dtype=np.uint8)
        camera.detector = DummyDetector()
        camera.get_demo_objects = lambda frame: [Object(0.0, 0.0, 0.0, name="demo", vis_type="planar")]

        objects, frame = camera.run()
        self.assertEqual(objects, [])
        self.assertIsNotNone(frame)

    def test_builtin_plugins_emit_visible_objects_and_annotations(self):
        import numpy as np

        for plugin_cls in (QRCodeCamera, DepthAnythingCamera, LineTrackingCamera):
            camera = plugin_cls.__new__(plugin_cls)
            camera.get_frame = lambda: np.zeros((80, 80, 3), dtype=np.uint8)
            camera.logger = None
            camera._last_frame = None
            objects, frame = camera.run()
            self.assertTrue(objects, f"{plugin_cls.__name__} should emit at least one object")
            self.assertIsNotNone(frame)
            annotated = camera.plot(frame)
            self.assertIsNotNone(annotated)
            self.assertTrue(np.any(annotated != frame))

    def test_depth_plugin_emits_heatmap_metadata(self):
        import numpy as np

        camera = DepthAnythingCamera.__new__(DepthAnythingCamera)
        camera.get_frame = lambda: np.zeros((80, 80, 3), dtype=np.uint8)
        camera.logger = None
        camera._last_frame = None

        objects, frame = camera.run()
        self.assertTrue(objects)
        self.assertIsNotNone(frame)
        self.assertIn("depth_estimate", objects[0].vis_meta)
        self.assertTrue(objects[0].vis_meta["heatmap"])


if __name__ == "__main__":
    unittest.main()
