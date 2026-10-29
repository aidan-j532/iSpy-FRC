"""Tests for the RollBack web module (iSpy/web/modules/rollback.py) and the
player template it serves.

The recorder itself is covered in test_addons.py; this file is about the things
that only bite in the player: cache eviction order, the cost of listing a
session, and the two units the sidecar and the camera calibration are stored in.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from iSpy.config.iSpyConfig import unit_to_output
from iSpy.web.modules import rollback as rb


def write_sidecar(path: Path, records, **header):
    body = {
        "stem": path.stem,
        "session": path.parent.name,
        "cams": header.pop("cams", ["Left"]),
        "started": header.pop("started", 1000.0),
        "fps": header.pop("fps", 30.0),
        "records": records,
    }
    body.update(header)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


class SegmentFrameCountCacheTests(unittest.TestCase):
    def setUp(self):
        rb._SEGMENT_FRAMES.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.session = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def _seg(self, stem):
        sidecar = write_sidecar(
            self.session / f"{stem}.json", [{"i": i} for i in range(3)]
        )
        return {"stem": stem, "cams": ["Left"], "files": [sidecar]}

    def test_counts_records_without_a_sidecar(self):
        seg = {"stem": "x", "cams": ["Left"], "files": [self.session / "x_Left.avi"]}
        self.assertEqual(rb._segment_frames(seg), 0)

    def test_the_count_is_not_reparsed_while_the_file_is_unchanged(self):
        seg = self._seg("a")
        self.assertEqual(rb._segment_frames(seg), 3)
        self.assertEqual(rb._segment_frames(seg), 3)

        with mock.patch.object(
            rb, "read_sidecar", wraps=rb.read_sidecar
        ) as reader:
            self.assertEqual(rb._segment_frames(seg), 3)
        reader.assert_not_called()

    def test_a_growing_segment_is_recounted(self):
        seg = self._seg("b")
        self.assertEqual(rb._segment_frames(seg), 3)
        # the open segment's sidecar is appended to, so the cached count is
        # only valid while the file's identity is unchanged
        write_sidecar(
            self.session / "b.json", [{"i": i} for i in range(6)]
        )
        self.assertEqual(rb._segment_frames(seg), 6)

    def test_a_truncated_sidecar_does_not_raise(self):
        # a power cut mid-write: the header is whole, the records array is not
        sidecar = self.session / "c.json"
        sidecar.write_text('{"stem": "c", "session": "s", "cams": ["Left"]', encoding="utf-8")
        self.assertEqual(rb._segment_frames({"stem": "c", "cams": [], "files": [sidecar]}), 0)


class SidecarHeaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.session = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_header_is_read_without_the_records(self):
        sidecar = write_sidecar(self.session / "a.json", [{"i": i} for i in range(200)])
        with mock.patch.object(rb, "read_sidecar", wraps=rb.read_sidecar) as reader:
            header = rb._segment_header({"stem": "a", "cams": [], "files": [sidecar]})
        reader.assert_not_called()
        self.assertEqual(header["cams"], ["Left"])
        self.assertEqual(header["started"], 1000.0)
        self.assertEqual(header["fps"], 30.0)
        self.assertEqual(header["stem"], "a")

    def test_a_truncated_header_salvages_whatever_parses(self):
        sidecar = self.session / "b.json"
        sidecar.write_text('{"stem": "b", "cams": ["Left"], "fps": 30', encoding="utf-8")
        header = rb.read_sidecar_header(sidecar)
        self.assertEqual(header["cams"], ["Left"])
        self.assertEqual(header["stem"], "b")
        self.assertNotIn("started", header)

    def test_an_empty_or_missing_sidecar_is_not_an_error(self):
        empty = self.session / "c.json"
        empty.write_text("", encoding="utf-8")
        self.assertEqual(rb.read_sidecar_header(empty), {})
        self.assertEqual(rb.read_sidecar_header(self.session / "gone.json"), {})


class CaptureCacheEvictionTests(unittest.TestCase):
    def setUp(self):
        rb._CAPS.clear()
        rb._FRAMES.clear()
        rb._FRAME_ORDER.clear()
        self.addCleanup(rb._CAPS.clear)

    def test_the_oldest_capture_is_evicted_not_the_newest(self):
        # _CAPS.popitem() with no argument is LIFO, which dropped the clip the
        # player had just opened and kept the stale one - scrubbing a segment
        # then re-opened the file on every frame
        class FakeCap:
            def __init__(self):
                self.released = False

            def release(self):
                self.released = True

            def isOpened(self):
                return False

        caps = {f"seg{n}.avi": FakeCap() for n in range(rb._MAX_CAPS)}
        with mock.patch.dict(rb._CAPS, caps, clear=True):
            newest = Path("newest.avi")
            with mock.patch.object(rb.cv2, "VideoCapture", return_value=FakeCap()):
                opened = rb._capture_for(newest)
            self.assertTrue(caps["seg0.avi"].released, "oldest was not freed")
            self.assertFalse(
                caps[f"seg{rb._MAX_CAPS - 1}.avi"].released,
                "LIFO evicted the capture just opened",
            )
            self.assertIn(str(newest), rb._CAPS)
            self.assertEqual(len(rb._CAPS), rb._MAX_CAPS)
            self.assertIs(rb._CAPS[str(newest)], opened)

    def test_an_already_open_capture_is_reused(self):
        with mock.patch.object(rb.cv2, "VideoCapture") as capture:
            first = rb._capture_for(Path("x.avi"))
            second = rb._capture_for(Path("x.avi"))
        self.assertIs(first, second)
        self.assertEqual(capture.call_count, 1)


class CameraMetaUnitTests(unittest.TestCase):
    class FakeModule:
        """Just enough of RollBackWeb for _camera_meta."""

        def __init__(self, meta):
            self._meta = meta

        def _session_meta(self, _session):
            return self._meta

        _camera_meta = rb.RollbackModule._camera_meta

    def test_frc_offsets_are_converted_to_the_output_unit(self):
        # FRC geometry is entered in inches but detections are reported in
        # metres, so an unconverted camera offset is 39x too large and every
        # projection lands in the wrong place
        mod = self.FakeModule(
            {
                "unit": "frc",
                "cameras": {"Left": {"x": 100.0, "y": -50.0, "height": 30.0, "yaw": 0}},
            }
        )
        left = mod._camera_meta(None)["Left"]
        self.assertAlmostEqual(left["x"], 2.54)
        self.assertAlmostEqual(left["y"], -1.27)
        self.assertAlmostEqual(left["height"], 0.762)
        self.assertEqual(left["yaw"], 0, "non-length fields are left alone")

    def test_units_that_already_match_are_unchanged(self):
        for unit in ("meter", "meters", "inch", "inches", "foot", "feet", "centimeter"):
            with self.subTest(unit=unit):
                mod = self.FakeModule(
                    {"unit": unit, "cameras": {"Left": {"x": 2.0, "y": 0.0, "height": 1.0}}}
                )
                self.assertAlmostEqual(mod._camera_meta(None)["Left"]["x"], 2.0)
                self.assertAlmostEqual(unit_to_output(2.0, unit), 2.0)

    def test_missing_or_malformed_fields_do_not_break_the_player(self):
        mod = self.FakeModule(
            {
                "unit": "frc",
                "cameras": {"Left": {}, "Right": {"x": "10"}, "Bad": "nope"},
            }
        )
        meta = mod._camera_meta(None)
        self.assertEqual(meta["Left"], {"x": 0.0, "y": 0.0, "height": 0.0})
        self.assertAlmostEqual(meta["Right"]["x"], 0.254)
        self.assertNotIn("Bad", meta)

    def test_a_session_without_cameras_yields_no_meta(self):
        self.assertEqual(self.FakeModule({"unit": "frc"})._camera_meta(None), {})


class PlayerTemplateTests(unittest.TestCase):
    HTML = (Path(rb.__file__).resolve().parents[1] / "templates" / "rollback.html").read_text(
        encoding="utf-8"
    )

    def test_projection_undoes_the_tracker_frame_first(self):
        # sidecar detections are post-tracker: relative_to() has already applied
        # the robot pose. projecting them as robot-space puts every target in
        # the wrong place as soon as the robot moves
        self.assertIn("function rbToRobot(det, pose)", self.HTML)
        self.assertIn("const robot = rbToRobot(det, rec.p);", self.HTML)
        self.assertIn("const yaw = -(pose.heading || 0)", self.HTML)
        # ... and the projection must be fed the transformed point, never det
        project = self.HTML.split("function rbProject", 1)[1].split("function rbDrawOverlay", 1)[0]
        self.assertIn("robot.x - (meta.x || 0)", project)
        self.assertIn("robot.y - (meta.y || 0)", project)
        self.assertIn("(robot.z || 0) - (meta.height || 0)", project)
        self.assertNotIn("det.x - (meta.x || 0)", project)
        self.assertNotIn("(det.z || 0) - (meta.height || 0)", project)

    def test_the_template_does_not_own_the_unit_conversion(self):
        # the API converts camera offsets into the detection unit; a second copy
        # of the table here is what got foot/cm wrong
        self.assertNotIn("0.0254", self.HTML)
        self.assertIn("the API converts them", self.HTML)

    def test_the_roster_is_labelled_as_the_field_frame(self):
        self.assertIn("'field frame (' + RB.unit + ')'", self.HTML)

    def test_reloading_the_list_keeps_the_player_where_it_was(self):
        # pin/delete/refresh reload the session list, which re-enters
        # rbLoadSegment; without this the player jumped back to frame 0
        self.assertIn("const prevSession = RB.session, prevStem = RB.stem, prevIdx = RB.idx;", self.HTML)
        self.assertIn("const sameClip = prevSession === RB.session && prevStem === RB.stem;", self.HTML)
        self.assertIn("const start = sameClip ? Math.min(prevIdx, last) : 0;", self.HTML)
        self.assertIn("RB.idx = start;", self.HTML)

    def test_the_session_reload_actions_stop_playback(self):
        # the timer used to keep running across a reload onto a different clip
        prune = self.HTML.split(
            "document.getElementById('rb-prune').addEventListener", 1
        )[1].split("});", 1)[0]
        refresh = self.HTML.split(
            "document.getElementById('rb-refresh').addEventListener", 1
        )[1].split("});", 1)[0]
        rows = self.HTML.split(
            "document.getElementById('rb-sessions').addEventListener", 1
        )[1].split("\ndocument.getElementById('rb-scrub')", 1)[0]
        for name, block in (("prune", prune), ("refresh", refresh), ("session rows", rows)):
            with self.subTest(handler=name):
                self.assertIn("rbStop();", block)

    def test_the_chosen_camera_survives_a_reload(self):
        self.assertIn("RB.cam = cams.includes(prevCam) ? prevCam", self.HTML)

    def test_multi_cam_runs_say_that_detections_are_not_attributed(self):
        self.assertIn("function rbMultiCamNote()", self.HTML)
        self.assertIn("detections are not attributed per camera", self.HTML)
        self.assertIn("rbMultiCamNote()", self.HTML.split("function rbDrawOverlay", 1)[1])


class TrackerFrameInverseTests(unittest.TestCase):
    """The overlay's rbToRobot() has to undo exactly what Object.relative_to()
    did to the detection before it was written to the sidecar - including the
    sign flip iSpy.py applies, because WPILib's pose yaw is CCW-positive while
    relative_to's is right-positive."""

    CASES = [
        (1.0, 2.0, 0.7, 3.0, -4.0, 0.5),
        (0.1, -0.2, 1.1, 0.0, 0.0, 0.0),
        (2.0, 3.0, 0.5, -1.5, 2.0, -1.2),
        (1.0, 0.0, 0.9, 5.0, 5.0, 3.0),
    ]

    @staticmethod
    def _rb_to_robot(det, pose):
        """the js, in python"""
        import math

        if not pose:
            return det
        yaw = -(pose["heading"] or 0)
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = det["x"] - (pose["x"] or 0), det["y"] - (pose["y"] or 0)
        return {"x": dx * c - dy * s, "y": dx * s + dy * c, "z": det["z"]}

    def test_rb_to_robot_undoes_relative_to(self):
        from iSpy.vision.Object import Object

        for x, y, z, px, py, heading in self.CASES:
            with self.subTest(x=x, y=y, heading=heading):
                obj = Object(x, y, z)
                obj.relative_to(px, py, robot_yaw=-heading)

                back = self._rb_to_robot(
                    {"x": obj.x, "y": obj.y, "z": obj.z},
                    {"x": px, "y": py, "heading": heading},
                )
                self.assertAlmostEqual(back["x"], x)
                self.assertAlmostEqual(back["y"], y)
                self.assertAlmostEqual(back["z"], z, msg="z is not rotated")

    def test_the_template_uses_that_algebra(self):
        html = (
            Path(rb.__file__).resolve().parents[1] / "templates" / "rollback.html"
        ).read_text(encoding="utf-8")
        body = html.split("function rbToRobot", 1)[1].split("}", 1)[0]
        self.assertIn("const yaw = -(pose.heading || 0)", body)
        self.assertIn("const dx = det.x - (pose.x || 0), dy = det.y - (pose.y || 0);", body)
        self.assertIn("x: dx * c - dy * s, y: dx * s + dy * c", body)

    def test_a_missing_pose_is_passed_through_unchanged(self):
        det = {"x": 1.0, "y": 2.0, "z": 3.0}
        self.assertIs(self._rb_to_robot(det, None), det)


class SegmentRouteTests(unittest.TestCase):
    def test_the_segment_route_does_not_rescan_the_whole_session(self):
        # _segment is hit on every segment change; _session_row walks and sizes
        # every segment in the session to produce numbers the player ignores
        source = Path(rb.__file__).resolve().read_text(encoding="utf-8")
        body = source.split("def _segment(self, session_name, stem)", 1)[1]
        body = body.split("\n    def ", 1)[0]
        code = [ln.split("#", 1)[0] for ln in body.splitlines()]
        self.assertNotIn("self._session_row(session)", "\n".join(code))
        self.assertIn("meta = self._session_meta(session)", "\n".join(code))


if __name__ == "__main__":
    unittest.main()