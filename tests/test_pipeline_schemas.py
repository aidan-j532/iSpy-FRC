import unittest

from iSpy.plugins.bases import VisionBase
from iSpy.plugins.vision.BuiltIn.AprilTag import AprilTagCamera
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
        self.assertTrue(any(p["name"] == "object_detection" for p in pipelines))


if __name__ == "__main__":
    unittest.main()
