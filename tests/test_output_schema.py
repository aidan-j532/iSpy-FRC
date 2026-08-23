import json
import math
import unittest

import numpy as np

from iSpy.vision.Object import Object
from iSpy.vision.pipelines.base import (
    OUTPUT_SCHEMA_VERSION,
    VisionPipeline,
)

REQUIRED_KEYS = {
    "id", "name", "confidence",
    "x", "y", "z", "roll", "pitch", "yaw",
    "depth_source", "vis_type", "vis_meta",
    "keypoints_3d", "ray_origin", "ray_direction",
}


def _detection_object() -> Object:
    obj = Object(1.25, -0.5, 0.0, confidence=0.87, name="note")
    return obj


def _planar_object() -> Object:
    return Object(
        2.0, 3.0, 0.4, roll=0.05, pitch=math.pi / 2, yaw=-0.5,
        name="april_tag_7", confidence=0.99,
        ray_origin=np.array([0.1, 0.2, 0.9]),
        ray_direction=np.array([0.0, 0.98, -0.2]),
        depth_source="pnp", vis_type="planar",
        vis_meta={"tag_id": 7, "size": 0.15},
    )


def _flow_object() -> Object:
    return Object(
        0.0, 0.0, 0.0, name="optical_flow", depth_source="optical_flow",
        vis_type="generic",
        vis_meta={"kind": "velocity", "vx": 0.12, "vy": -0.4,
                  "speed": 0.42, "heading_deg": 16.7},
    )


def _depth_object() -> Object:
    return Object(
        0.0, 1.0, 2.35, name="depth_center", confidence=0.8,
        depth_source="depth_model", vis_type="generic",
        vis_meta={"kind": "depth", "heatmap": True, "depth_estimate": 2.35,
                  "max_depth": 10.0},
    )


class ObjectSchemaTests(unittest.TestCase):
    def test_to_dict_has_full_schema_for_every_pipeline_flavor(self):
        for obj in (_detection_object(), _planar_object(),
                    _flow_object(), _depth_object()):
            data = obj.to_dict()
            self.assertTrue(REQUIRED_KEYS <= set(data), data.keys())

    def test_to_dict_is_json_safe(self):
        for obj in (_planar_object(), _flow_object()):
            data = obj.to_dict()
            # must not raise even with numpy rays / float vis_meta
            encoded = json.dumps(data)
            decoded = json.loads(encoded)
            self.assertEqual(decoded["name"], obj.name)

    def test_to_dict_values_round_trip(self):
        obj = _planar_object()
        data = obj.to_dict()
        clone = Object.from_dict(data)
        self.assertEqual(clone.id, obj.id)
        self.assertAlmostEqual(clone.x, obj.x)
        self.assertAlmostEqual(clone.yaw, obj.yaw)
        self.assertEqual(clone.name, obj.name)
        self.assertEqual(clone.vis_type, obj.vis_type)
        self.assertEqual(clone.vis_meta, obj.vis_meta)
        np.testing.assert_allclose(clone.ray_origin, obj.ray_origin)
        np.testing.assert_allclose(clone.ray_direction, obj.ray_direction)

    def test_from_dict_tolerates_missing_and_unknown_keys(self):
        clone = Object.from_dict({"x": "1.5", "name": "cone",
                                  "bogus_future_field": {"a": 1}})
        self.assertAlmostEqual(clone.x, 1.5)
        self.assertEqual(clone.y, 0.0)
        self.assertEqual(clone.name, "cone")
        self.assertEqual(clone.vis_type, "generic")

    def test_from_dict_without_id_allocates_fresh_track_id(self):
        clone = Object.from_dict({"x": 1.0})
        self.assertIsNotNone(clone.get_id())


class PipelineSerializationTests(unittest.TestCase):
    def test_serialize_detections_handles_empty_and_none(self):
        self.assertEqual(VisionPipeline.serialize_detections([]), [])
        self.assertEqual(VisionPipeline.serialize_detections(None), [])

    def test_serialize_detections_flattens_any_pipeline(self):
        out = VisionPipeline.serialize_detections(
            [_detection_object(), _planar_object(), _flow_object()])
        self.assertEqual(len(out), 3)
        for entry in out:
            self.assertTrue(REQUIRED_KEYS <= set(entry))
        # pass-through for non-Object legacy entries
        mixed = VisionPipeline.serialize_detections([{"legacy": True}])
        self.assertEqual(mixed, [{"legacy": True}])

    def test_serialize_frame_data_shape(self):
        frame_data = {
            "detections": [_detection_object()],
            "detection_count": 1,
            "fps": 90.0,
            "pipeline_name": "object_detection",
            "frame": np.zeros((4, 4, 3), dtype=np.uint8),
            "cameras": ["not-json-safe"],
        }
        out = VisionPipeline.serialize_frame_data(frame_data)
        self.assertEqual(out["schema_version"], OUTPUT_SCHEMA_VERSION)
        self.assertEqual(out["detection_count"], 1)
        self.assertEqual(out["fps"], 90.0)
        self.assertEqual(out["pipeline_name"], "object_detection")
        self.assertEqual(len(out["detections"]), 1)
        self.assertTrue(REQUIRED_KEYS <= set(out["detections"][0]))
        # non-scalar payloads are dropped, detections serialized in place
        self.assertNotIn("frame", out)
        self.assertNotIn("cameras", out)


if __name__ == "__main__":
    unittest.main()
