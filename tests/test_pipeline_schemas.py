import unittest

from iSpy.plugins.bases import VisionBase
from iSpy.plugins.vision.BuiltIn.AprilTag import AprilTagCamera
from iSpy.plugins.vision.BuiltIn.QRCode import QRCodeCamera
from iSpy.plugins.vision.BuiltIn.DepthAnything import DepthAnythingCamera
from iSpy.plugins.vision.BuiltIn.LineTracking import LineTrackingCamera
from iSpy.web.Backend.PluginStatus import _build_vision_pipeline_payloads


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


if __name__ == "__main__":
    unittest.main()
