"""End-to-end orientation check for the 3D viewer.

Synthesizes a camera + AprilTag scene, runs corners through the SAME pose math
the pipelines ship (solvePnP -> camera_rotation_to_robot -> _matrix_to_euler),
then feeds the resulting robot-frame eulers into the real robotpose.js module
that viewer3d.html imports. The viewer quaternion must reproduce
M * R_seed where M is the robot -> viewer proper rotation (x,y,z)->(x,z,-y),
and physically obvious tilts (tag top away/toward the camera) must keep their
sign. The object_detection PnP eulers must land on the same robot frame.

The camera is LEVEL (yaw=pitch=0, looking +Y), the tag sits ahead at y=3 with
its printed face toward the camera, so tag local +Z == robot -Y and printed
top (local +Y) == robot +Z. Poses are chosen with a real tilt: a square's four
corners cannot observe a pure spin about the tag normal or an exactly
fronto-parallel tag (both are symmetric), and solvePnP/IPPE picks a mirrored
solution there - that ambiguity is inherent to monocular PnP, not the viewer.
"""

import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from iSpy.vision import triangulation
from iSpy.vision.genericYolo import GenericYolo
from iSpy.vision.pipelines.april_tag import AprilTagPipeline
from iSpy.vision.pipelines.object_detection import ObjectDetectionPipeline

ROOT = Path(__file__).resolve().parents[1]
ROBOTPOSE = ROOT / "iSpy" / "web" / "static" / "js" / "robotpose.js"

# robot -> viewer frame rotation M = Rx(-90): (x, y, z) -> (x, z, -y)
M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)

CAM_YAW, CAM_PITCH = 0.0, 0.0  # level camera looking +Y
CAM_POS = np.array([0.0, 0.25, 1.05], dtype=float)
TAG_SIZE_M = 0.165  # 6.5in tag
FOCAL_PX = 900.0
IMG_W, IMG_H = 1280, 720

# tag with its face toward the camera: local X=right(+X), Y=printed-up(+Z),
# Z=face normal toward the camera(-Y)
R_FRONT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
TAG_POS = np.array([0.0, 3.0, 1.05], dtype=float)

# a tilt that stays observable to a monocular square (not a pure normal spin)
R_TILT = None  # built in setUp via rot_x/rot_z composition


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def quat_to_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def camera_point_to_robot_linear(yaw_deg, pitch_deg) -> np.ndarray:
    """Linear part of triangulation.camera_point_to_robot, as a 3x3 matrix."""
    basis = np.eye(3)
    return np.column_stack(
        [
            triangulation.camera_point_to_robot(
                basis[:, i], 0, 0, 0, yaw_deg, pitch_deg
            )
            for i in range(3)
        ]
    )


def project_tag(R_tag_robot, t_tag_robot):
    """Projects the AprilTag's four corners to pixels given a robot pose."""
    C = camera_point_to_robot_linear(CAM_YAW, CAM_PITCH)
    half = TAG_SIZE_M / 2.0
    obj_pts = np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
        dtype=np.float64,
    )
    corners = []
    for pt in obj_pts:
        r = R_tag_robot @ pt + t_tag_robot
        c_cam = C.T @ (r - CAM_POS)  # C maps camera->robot, transpose inverts
        u = FOCAL_PX * c_cam[0] / c_cam[2] + IMG_W / 2.0
        v = FOCAL_PX * c_cam[1] / c_cam[2] + IMG_H / 2.0
        corners.append([u, v])
    return np.array(corners, dtype=np.float32), obj_pts


def solve_pnp_robot_euler(corners, obj_pts):
    """Replicates april_tag.run()'s pose chain; returns robot-frame euler."""
    cam_mat = np.array(
        [[FOCAL_PX, 0, IMG_W / 2.0], [0, FOCAL_PX, IMG_H / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )
    dist = np.zeros(5, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, corners, cam_mat, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    assert ok, "solvePnP failed on synthetic corners"
    R_cam, _ = cv2.Rodrigues(rvec)
    R_robot = triangulation.camera_rotation_to_robot(R_cam, CAM_YAW, CAM_PITCH)
    return AprilTagPipeline.__new__(AprilTagPipeline)._matrix_to_euler(R_robot)


def robotpose_quat(roll, pitch, yaw):
    """Runs the real robotpose.js module in node; returns [x, y, z, w]."""
    with tempfile.TemporaryDirectory() as tmp:
        mjs = Path(tmp) / "robotpose.mjs"
        mjs.write_bytes(ROBOTPOSE.read_bytes())
        script = (
            "import { robotToViewerQuat } from "
            f"'file:///{mjs.as_posix()}';"
            " const [r,p,y] = JSON.parse(process.argv[1]);"
            " const q = robotToViewerQuat(r, p, y);"
            " console.log(JSON.stringify(q));"
        )
        out = subprocess.run(
            ["node", "--input-type=module", "--eval", script,
             json.dumps([roll, pitch, yaw])],
            capture_output=True,
            text=True,
            check=True,
        )
        return np.array(json.loads(out.stdout), dtype=float)


def viewer_matrix_for_tag(r_truth):
    corners, obj_pts = project_tag(r_truth, TAG_POS)
    roll, pitch, yaw = solve_pnp_robot_euler(corners, obj_pts)
    return quat_to_matrix(robotpose_quat(roll, pitch, yaw))


class Viewer3dOrientationTests(unittest.TestCase):
    def setUp(self):
        if not ROBOTPOSE.exists():
            self.skipTest("robotpose.js missing")
        global R_TILT
        # kept tilted enough for a monocular square to observe (no pure normal spin)
        R_TILT = rot_x(0.24) @ rot_z(0.9) @ R_FRONT

    def test_frame_change_is_a_proper_rotation(self):
        q = robotpose_quat(0.0, 0.0, 0.0)
        R = quat_to_matrix(q)
        self.assertAlmostEqual(round(float(np.linalg.det(R)), 6), 1.0)
        np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(R @ [0, 0, 1], [0, -1, 0], atol=1e-6)

    def test_voxel_group_parented_to_robot_poses_points_in_field(self):
        # voxel points arrive robot-relative (+X right, +Y forward, +Z up).
        # the viewer parents the voxel group to the live NetworkHandler "robot"
        # overlay (position t, heading yaw) with the same transform updateCubeMarker /
        # the overlay box apply, and instances stay in raw robot-frame coords.
        # A point p must then render at M * (Rz(yaw) @ p) + M @ t - its true
        # field-absolute spot - not at M @ p (the old, robot-frame-at-origin bug).
        t = np.array([2.0, -1.0, 0.35], dtype=float)  # robot field pose
        yaw = 1.1
        p = np.array([0.7, 1.3, 0.2], dtype=float)     # robot-relative voxel
        R_group = quat_to_matrix(robotpose_quat(0.0, 0.0, yaw))
        rendered = R_group @ p + M @ t
        expected = M @ (rot_z(yaw) @ p) + M @ t
        np.testing.assert_allclose(rendered, expected, atol=1e-6)

    def test_tilted_tag_matches_true_orientation(self):
        np.testing.assert_allclose(
            viewer_matrix_for_tag(R_TILT), M @ R_TILT, atol=2e-2
        )

    def test_tag_top_away_from_camera_renders_away(self):
        # rot_x(-0.4) tips the printed top away from the camera (robot +Y);
        # the viewer maps +Y -> +Z so the top must keep +Z, not fold to -Z
        r_truth = rot_x(-0.4) @ R_FRONT
        self.assertGreater((r_truth @ [0, 1, 0])[1], 0.0)  # sanity: seed is away
        R_view = viewer_matrix_for_tag(r_truth)
        np.testing.assert_allclose(R_view, M @ r_truth, atol=2e-2)
        top_view = R_view @ [0, 1, 0]
        self.assertGreater(top_view[2], 0.0)

    def test_tag_top_toward_camera_renders_toward(self):
        r_truth = rot_x(0.4) @ R_FRONT
        self.assertLess((r_truth @ [0, 1, 0])[1], 0.0)
        R_view = viewer_matrix_for_tag(r_truth)
        np.testing.assert_allclose(R_view, M @ r_truth, atol=2e-2)
        top_view = R_view @ [0, 1, 0]
        self.assertLess(top_view[2], 0.0)

    def test_fronto_parallel_tag_still_lands_at_right_position(self):
        # a dead-frontal tag is orientation-ambiguous to a square's corners,
        # but the PnP position stays rock solid - the user-visible guarantee
        # ("position is correct") the viewer builds on
        corners, obj_pts = project_tag(R_FRONT, TAG_POS)
        cam_mat = np.array(
            [[FOCAL_PX, 0, IMG_W / 2.0], [0, FOCAL_PX, IMG_H / 2.0], [0, 0, 1]],
            dtype=np.float64,
        )
        ok, _, tvec = cv2.solvePnP(
            obj_pts, corners, cam_mat, np.zeros(5),
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        self.assertTrue(ok)
        robot = triangulation.camera_point_to_robot(
            tuple(tvec.reshape(3)),
            CAM_POS[0], CAM_POS[1], CAM_POS[2],
            CAM_YAW, CAM_PITCH,
        )
        np.testing.assert_allclose(robot, TAG_POS, atol=2e-2)

    def test_od_pnp_euler_lands_on_robot_frame(self):
        # genericYolo reports camera-frame eulers; the OD pipeline converts
        # them to robot frame before they reach the viewer
        corners, obj_pts = project_tag(R_TILT, TAG_POS)
        cam_mat = np.array(
            [[FOCAL_PX, 0, IMG_W / 2.0], [0, FOCAL_PX, IMG_H / 2.0], [0, 0, 1]],
            dtype=np.float64,
        )
        ok, rvec, _ = cv2.solvePnP(obj_pts, corners, cam_mat, np.zeros(5))
        assert ok
        od = ObjectDetectionPipeline.__new__(ObjectDetectionPipeline)
        od.camera_bot_relative_yaw = CAM_YAW
        od.camera_pitch_angle = CAM_PITCH
        cam_euler = GenericYolo.__new__(GenericYolo)._rvec_to_euler(rvec)
        roll, pitch, yaw = od._euler_to_robot_frame(*cam_euler)
        R_view = quat_to_matrix(robotpose_quat(roll, pitch, yaw))
        np.testing.assert_allclose(R_view, M @ R_TILT, atol=2e-2)


if __name__ == "__main__":
    unittest.main()