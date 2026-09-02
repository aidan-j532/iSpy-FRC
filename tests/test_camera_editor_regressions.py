import re
import unittest
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "iSpy" / "web" / "templates" / "cameras.html"
HTML = TEMPLATE.read_text()


def func_body(name: str) -> str:
    match = re.search(rf"function {name}\((?:[^)]*)\) \{{(.*?)\n\}}", HTML, re.S)
    if not match:
        raise AssertionError(f"function {name} not found in {TEMPLATE.name}")
    return match.group(1)


class TestCameraEditorPickers(unittest.TestCase):
    def test_model_picker_keeps_original_snapshot(self):
        self.assertNotIn("dataset.original", func_body("openModelPicker"))

    def test_model_upload_keeps_original_snapshot(self):
        self.assertNotIn("dataset.original", func_body("uploadModelFromPicker"))

    def test_dataset_import_keeps_original_snapshot(self):
        self.assertNotIn("dataset.original", func_body("uploadDatasetFromPicker"))

    def test_change_detection_still_uses_original_snapshot(self):
        self.assertIn("el.value === el.dataset.original", func_body("modelPayloadField"))
        self.assertIn("el.dataset.original", func_body("collectSchemaFields"))


class TestSectionFlowValidation(unittest.TestCase):
    """Save and Continue must be able to advance past every section."""

    def test_list_fields_never_block_validation(self):
        # list-type schema fields (e.g. object_heights) render a table div
        # with no .value - they must not read as permanently empty
        body = func_body("validateCurrentSection")
        self.assertIn("'list'", body)

    def test_nullable_fields_of_any_type_can_be_empty(self):
        body = func_body("validateCurrentSection")
        self.assertIn("def.nullable || String(el.value || '').trim() !== ''", body)


class TestTelloSchemaWiring(unittest.TestCase):
    """BUG 5: Tello connection fields are rendered, captured, and schema-driven."""

    def test_render_source_section_renders_tello_fields(self):
        body = func_body("renderSourceSection")
        # the three Tello fields must be built/rendered (not dead data)
        for key in ("tello_ip", "tello_command_port", "tello_video_port"):
            self.assertIn(key, body)

    def test_tello_fields_prefer_backend_schema(self):
        body = func_body("renderSourceSection")
        self.assertIn("cameraFieldDef(key)", body)

    def test_submit_captures_tello_fields(self):
        body = func_body("submitCameraModal")
        for key in ("tello_ip", "tello_command_port", "tello_video_port"):
            self.assertIn(f"payload.{key} = readField('{key}')", body)

    def test_load_camera_schemas_hits_backend_endpoint(self):
        body = func_body("loadCameraSchemas")
        self.assertIn("'/api/camera_schemas'", body)

    def test_camera_field_def_falls_back_to_common_fields(self):
        body = func_body("cameraFieldDef")
        self.assertIn("COMMON_FIELDS[key]", body)


if __name__ == "__main__":
    unittest.main()