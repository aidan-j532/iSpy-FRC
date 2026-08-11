"""Regression tests for the camera editor model/dataset pickers.

The pickers must update only the input value. In edit mode the save flow
compares ``el.value`` against ``el.dataset.original`` to decide whether a
field changed; if a picker overwrites ``dataset.original`` the change is
silently dropped and the config keeps the old model/dataset.
"""

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


if __name__ == "__main__":
    unittest.main()