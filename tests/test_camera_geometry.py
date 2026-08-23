"""Regression tests for PROMPT 4 fixes:

Bug 1 - 'height' is the single canonical mount-height field feeding
        triangulation (the old separate 'z' config key was dead/uneditable
        from the UI and silently shadowed 'height').
Bug 2 - per-class 'object heights above ground' support in object_detection:
        rays intersect a class-specific horizontal plane instead of always
        z=0, so objects off the ground get correct distance AND correct z.
"""

import math
import unittest

import numpy as np

from iSpy.config.iSpyConfig import iSpyCameraConfig, iSpyConfig, unit_to_inches
from iSpy.vision import triangulation
from iSpy.vision.genericYolo import Box
from iSpy.vision.pipelines.object_detection import ObjectDetectionCamera


IMG_W, IMG_H = 640, 480

# bottom-center of a synthetic detection box (lower half of frame so the
# pitched-down camera's ray actually hits the ground plane)
BOX_XYXY = [200.0, 300.0, 440.0, 470.0]


def make_camera(height_in: float = 48.0, object_heights=None):
    cam_cfg = iSpyCameraConfig({
        "name": "geom_test",
        # nonexistent image -> placeholder frames, no capture device opened
        "source": "definitely_missing_frame.png",
        "x": 0.0,
        "y": 0.0,
        "height": height_in,
        "pitch": 20.0,    # positive pitch tilts down toward the field
        "yaw": 0.0,
        "subsystem": "field",
        "calibration": {"size": 100, "distance": 60, "game_piece_size": 10, "fov": 68},
        "pipeline": {
            "name": "object_detection",
            "settings": {
                # missing file -> GenericYolo raises ModelFileError ->
                # pipeline runs model-less (exactly what these geometry
                # tests need, without loading weights)
                "vision_model": {"file_path": "does_not_exist.pt"},
                **({"object_heights": object_heights} if object_heights is not None else {}),
            },
        },
    })
    return ObjectDetectionCamera(cam_cfg, iSpyConfig())


class MountHeightTriangulationTests(unittest.TestCase):
    """Bug 1 regression: UI 'height' must drive the triangulation ray origin."""

    def test_camera_height_feeds_ray_origin(self):
        cam = make_camera(height_in=48.0)
        self.assertAlmostEqual(cam.camera_height, 48.0)
        ray = cam._pixel_ray(320.0, 400.0, IMG_W, IMG_H)
        self.assertAlmostEqual(float(ray.origin[2]), 48.0)

    def test_no_separate_camera_z_attribute(self):
        cam = make_camera(height_in=48.0)
        self.assertFalse(hasattr(cam, "camera_z"))

    def test_ground_point_matches_manual_triangulation_with_height(self):
        cam = make_camera(height_in=48.0)
        box = Box(BOX_XYXY, conf=0.9, cls_id=0)
        pt = cam._box_to_robot_point(box, IMG_W, IMG_H)

        scale = cam.conversions[cam.unit]
        ray = cam._pixel_ray(
            (BOX_XYXY[0] + BOX_XYXY[2]) / 2.0, BOX_XYXY[3], IMG_W, IMG_H
        )
        self.assertAlmostEqual(float(ray.origin[2]), 48.0)
        expected = triangulation.ground_plane_intersection(ray, ground_z=0.0) * scale
        np.testing.assert_allclose(pt, expected, rtol=1e-9)

    def test_result_actually_depends_on_height(self):
        # the whole point of the fix: height=X must NOT produce what z=0 did
        low = make_camera(height_in=5.0)
        high = make_camera(height_in=48.0)
        box = Box(BOX_XYXY, conf=0.9, cls_id=0)
        pt_low = low._box_to_robot_point(box, IMG_W, IMG_H)
        pt_high = high._box_to_robot_point(box, IMG_W, IMG_H)
        self.assertGreater(
            float(np.linalg.norm(pt_low[:2] - pt_high[:2])), 0.05,
            "mount height had no effect on triangulated ground position",
        )


class ElevatedObjectTests(unittest.TestCase):
    """Bug 2 regression: per-class plane heights instead of hardcoded z=0."""

    def test_configured_class_intersects_plane_at_height(self):
        cam = make_camera(height_in=48.0)
        cam._class_names = {0: "cone"}
        cam.object_heights = {"cone": 24.0}

        box = Box(BOX_XYXY, conf=0.9, cls_id=0)
        obj = cam._box_to_object(box, IMG_W, IMG_H)
        self.assertIsNotNone(obj)

        scale = cam.conversions[cam.unit]
        self.assertAlmostEqual(obj.z, 24.0 * scale, places=9)

        # x/y must match an explicit intersection with the z=24in plane
        ray = cam._pixel_ray(
            (BOX_XYXY[0] + BOX_XYXY[2]) / 2.0, BOX_XYXY[3], IMG_W, IMG_H
        )
        expected = triangulation.ground_plane_intersection(ray, ground_z=24.0) * scale
        np.testing.assert_allclose(obj.get_position()[:2], expected[:2], rtol=1e-9)

    def test_unconfigured_classes_keep_ground_plane_behaviour(self):
        cam = make_camera(height_in=48.0)
        cam._class_names = {0: "cone", 1: "ball"}
        cam.object_heights = {"cone": 24.0}

        cone = cam._box_to_object(Box(BOX_XYXY, 0.9, cls_id=0), IMG_W, IMG_H)
        ball = cam._box_to_object(Box(BOX_XYXY, 0.9, cls_id=1), IMG_W, IMG_H)
        self.assertAlmostEqual(cone.z, 24.0 * cam.conversions[cam.unit], places=9)
        self.assertEqual(ball.z, 0.0)

    def test_default_construction_is_pure_ground_plane(self):
        cam = make_camera(height_in=48.0)
        self.assertEqual(cam.object_heights, {})
        obj = cam._box_to_object(Box(BOX_XYXY, 0.9, cls_id=0), IMG_W, IMG_H)
        self.assertEqual(obj.z, 0.0)

    def test_size_based_fallback_respects_plane_height(self):
        # parallel-ray fallback: vertical triangle leg shrinks by the plane
        # height, and the returned point carries the plane's z
        cam = make_camera(height_in=48.0)
        cam.object_heights = {"cone": 40.0}
        cam._class_names = {0: "cone"}

        # force the fallback by making the ray parallel to the plane
        original_pixel_ray = cam._pixel_ray

        def flat_ray(px, py, w, h):
            ray = original_pixel_ray(px, py, w, h)
            return triangulation.Ray(ray.origin, np.array([0.0, 1.0, 0.0]))

        cam._pixel_ray = flat_ray
        obj = cam._box_to_object(Box(BOX_XYXY, 0.9, cls_id=0), IMG_W, IMG_H)
        self.assertIsNotNone(obj)
        self.assertAlmostEqual(obj.z, 40.0 * cam.conversions[cam.unit], places=9)
        self.assertTrue(np.all(np.isfinite(obj.get_position()[:2])))

    def test_object_heights_setting_parsed_from_ui_rows(self):
        cam = make_camera(
            height_in=48.0,
            object_heights=[
                {"class_name": "cone", "height_in": 24},
                {"class_name": "note", "height_in": 12.5},
                {"class_name": "", "height_in": 30},      # no class -> dropped
                {"class_name": "zero", "height_in": 0},   # zero -> ground plane
                {"class_name": "bad", "height_in": "abc"} # junk -> dropped
                ,
            ],
        )
        self.assertEqual(cam.object_heights, {"cone": 24.0, "note": 12.5})

    def test_object_heights_setting_accepts_dict_and_json(self):
        from_dict = make_camera(height_in=48.0, object_heights={"cone": 24})
        self.assertEqual(from_dict.object_heights, {"cone": 24.0})

        from_json = make_camera(
            height_in=48.0, object_heights='{"cone": 24}'
        )
        self.assertEqual(from_json.object_heights, {"cone": 24.0})

        from_junk = make_camera(height_in=48.0, object_heights="{not json")
        self.assertEqual(from_junk.object_heights, {})

    def test_schema_exposes_object_heights_list_field(self):
        schema = ObjectDetectionCamera.config_schema()
        field = schema["object_heights"]
        self.assertEqual(field["type"], "list")
        self.assertIn("class_name", field["fields"])
        self.assertIn("height_in", field["fields"])
        self.assertEqual(field["default"], [])


class OtherPipelinesUseMountHeightTests(unittest.TestCase):
    """april_tag / qr_code / optical_flow must read 'height' too."""

    def _cam_cfg(self, **extra):
        cfg = {
            "name": "geom_test",
            "source": "definitely_missing_frame.png",
            "height": 36.0,
            "pitch": 20.0,   # positive pitch tilts down toward the field
            "yaw": 0.0,
            "subsystem": "field",
            "calibration": {"size": 100, "distance": 60, "game_piece_size": 10, "fov": 68},
        }
        cfg.update(extra)
        return iSpyCameraConfig(cfg)

    def test_april_tag_uses_height_and_ignores_legacy_z(self):
        from iSpy.vision.pipelines.april_tag import AprilTagCamera

        cam = AprilTagCamera(self._cam_cfg(z=99.0), iSpyConfig())
        self.assertFalse(hasattr(cam, "camera_z"))
        self.assertAlmostEqual(cam.camera_height, 36.0)
        # a camera-frame point at the optical centre sits 36in up in robot frame
        pt = cam._camera_point_to_robot((0.0, 0.0, 0.0))
        self.assertAlmostEqual(float(pt[2]), 36.0 * cam.conversions[cam.unit], places=9)

    def test_qr_code_uses_height_and_ignores_legacy_z(self):
        from iSpy.vision.pipelines.qr_code import QRCodeCamera

        cam = QRCodeCamera(self._cam_cfg(z=99.0), iSpyConfig())
        self.assertFalse(hasattr(cam, "camera_z"))
        pt = cam._camera_point_to_robot((0.0, 0.0, 0.0))
        self.assertAlmostEqual(float(pt[2]), 36.0 * cam.conversions[cam.unit], places=9)

    def test_optical_flow_range_uses_mount_height(self):
        from iSpy.vision.pipelines.optical_flow import OpticalFlowCamera

        cam = OpticalFlowCamera(self._cam_cfg(), iSpyConfig())
        self.assertFalse(hasattr(cam, "camera_z"))
        self.assertAlmostEqual(
            cam.camera_height, unit_to_inches(36.0, "frc"), places=9
        )
        # downward pitch: assumed range = height / tan(pitch)
        expected = max(cam.camera_height / math.tan(math.radians(20.0)), 1.0)
        self.assertAlmostEqual(cam._range_inches(), expected, places=6)


if __name__ == "__main__":
    unittest.main()
