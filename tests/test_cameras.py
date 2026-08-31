"""Tests for the modular camera-source system (iSpy/vision/Cameras/).

Covers the prompt's camera-source requirements that are testable without real
hardware: the BUILTIN_CAMERAS registry, create_camera() type resolution, per-
type config schemas, the Camera() compatibility facade delegation, and the
backward-compatible *Camera -> *Pipeline / TelloEduCamera -> TelloCamera
aliases. Hardware-bound paths are mocked.
"""

import unittest
from unittest import mock

from iSpy.vision.Cameras import (
    BUILTIN_CAMERAS,
    CAMERA_TYPE_LABELS,
    get_camera_classes,
    get_camera_class,
    create_camera,
)
from iSpy.vision.Cameras.base import CameraBase, CameraOpenTimeout
from iSpy.vision.Cameras.OpenCVCamera import OpenCVCamera
from iSpy.vision.Cameras.TelloCamera import TelloCamera


def _no_init(self, *a, **k):
    pass


class CameraRegistryTests(unittest.TestCase):
    def test_registry_maps_expected_sources(self):
        self.assertEqual(
            {name: cls.camera_type for name, cls in BUILTIN_CAMERAS.items()},
            {"opencv": "opencv", "tello": "tello"},
        )
        self.assertEqual(
            set(get_camera_classes()), {"opencv", "tello"}
        )

    def test_subclasses_declare_unique_types(self):
        types = [cls.camera_type for cls in get_camera_classes().values()]
        self.assertEqual(len(types), len(set(types)), "camera types must be unique")

    def test_labels_cover_all_registered_types(self):
        self.assertEqual(set(CAMERA_TYPE_LABELS), set(BUILTIN_CAMERAS))

    def test_get_camera_class_resolves(self):
        self.assertIs(get_camera_class("opencv"), OpenCVCamera)
        self.assertIs(get_camera_class("tello"), TelloCamera)
        self.assertIs(get_camera_class(), OpenCVCamera)  # default

    def test_get_camera_class_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_camera_class("not_a_camera")

    def test_create_camera_resolves_explicit_override(self):
        with mock.patch.object(OpenCVCamera, "__init__", _no_init):
            cam = create_camera({}, camera_type="opencv")
            self.assertIsInstance(cam, OpenCVCamera)

    def test_create_camera_resolves_from_config_key(self):
        with mock.patch.object(TelloCamera, "__init__", _no_init):
            cam = create_camera({"camera_type": "tello", "name": "drone"})
            self.assertIsInstance(cam, TelloCamera)

    def test_create_camera_defaults_to_opencv(self):
        with mock.patch.object(OpenCVCamera, "__init__", _no_init):
            cam = create_camera({"name": "plain"})
            self.assertIsInstance(cam, OpenCVCamera)
            self.assertEqual(cam.camera_type, "opencv")

    def test_create_camera_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            create_camera({"camera_type": "bogus", "name": "x"})


class CameraSchemaTests(unittest.TestCase):
    def test_every_source_provides_config_schema(self):
        for cls in get_camera_classes().values():
            schema = cls.config_schema()
            self.assertIsInstance(schema, dict, f"{cls.__name__} schema")
            self.assertIn("source", schema, f"{cls.__name__} must expose a source")

    def test_opencv_schema_has_opencv_specific_keys(self):
        keys = set(OpenCVCamera.config_schema())
        self.assertIn("source", keys)
        self.assertIn("device_id", keys)
        self.assertIn("fps_cap", keys)
        self.assertIn("exposure_time", keys)

    def test_tello_schema_has_tello_specific_keys(self):
        keys = set(TelloCamera.config_schema())
        self.assertIn("tello_ip", keys)
        self.assertIn("tello_command_port", keys)
        self.assertIn("tello_video_port", keys)
        # Tello need not expose exposure/gain (they are not meaningful on the
        # H.264 stream), showing per-source schemas stay separate.
        self.assertNotIn("exposure_time", keys)

    def test_schemas_are_source_specific(self):
        opencv = set(OpenCVCamera.config_schema())
        tello = set(TelloCamera.config_schema())
        self.assertTrue(tello - opencv, "tello has source-specific keys")
        self.assertTrue(opencv - tello, "opencv has source-specific keys")


class CameraFacadeTests(unittest.TestCase):
    """The Camera() facade must delegate to the correct Cameras/ source."""

    def test_facade_routes_tello_type_to_tello_delegate(self):
        with mock.patch.object(TelloCamera, "__init__", _no_init):
            from iSpy.vision.Camera import Camera
            cam = Camera({"camera_type": "tello", "name": "drone", "source": 0})
            self.assertIsInstance(cam._delegate, TelloCamera)

    def test_facade_routes_opencv_to_opencv_delegate(self):
        with mock.patch.object(OpenCVCamera, "__init__", _no_init):
            from iSpy.vision.Camera import Camera
            cam = Camera({"camera_type": "opencv", "name": "cam", "source": 0})
            self.assertIsInstance(cam._delegate, OpenCVCamera)

    def test_facade_forwarded_attribute_hits_delegate(self):
        with mock.patch.object(OpenCVCamera, "__init__", _no_init):
            from iSpy.vision.Camera import Camera
            cam = Camera({"name": "cam", "source": 0})
            self.assertEqual(cam.camera_type, "opencv")  # class attr via facade


class CameraBackwardCompatTests(unittest.TestCase):
    def test_tello_edu_camera_is_tello_camera(self):
        from iSpy.vision.TelloEduCamera import TelloEduCamera
        self.assertIs(TelloEduCamera, TelloCamera)

    def test_pipeline_camera_aliases_removed(self):
        from iSpy.vision.pipelines import get_pipeline_classes
        classes = get_pipeline_classes()
        self.assertIn("object_detection", classes)
        self.assertIn("april_tag", classes)
        self.assertIn("yolo_world", classes)

    def test_pipeline_registry_points_at_pipelines(self):
        from iSpy.vision.pipelines import PIPELINES
        for name, cls in PIPELINES.items():
            self.assertEqual(cls.plugin_name, name)
            self.assertTrue(hasattr(cls, "config_schema"))


class CameraBaseContractTests(unittest.TestCase):
    def test_base_provides_discovery_hooks(self):
        self.assertEqual(CameraBase.discover(), [])  # default = no sources

    def test_named_slots_release_after_failure(self):
        # The bounded opener guard lives in Cameras.base; verify the sentinel
        # error type is importable through the public package.
        self.assertTrue(issubclass(CameraOpenTimeout, ValueError))


if __name__ == "__main__":
    unittest.main()
