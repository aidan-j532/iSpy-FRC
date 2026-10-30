"""Tests for the `ispy-optimize` CLI (`iSpy.validations.optimize`).

All tests mock the heavy conversion path and hardware probes so they run without
a GPU, RKNN toolchain, network access, or real Ultralytics exports.  They use
temporary project roots (cwd + `_PROJECT_ROOT` patches) so no repo files are
touched.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Ensure the repo root is importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iSpy import validations  # noqa: F401
from iSpy.validations import optimize as _opt


def _make_root():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    (root / "YoloModels" / "pytorch").mkdir(parents=True)
    (root / "Outputs").mkdir(parents=True)
    return tmp, root


class _OptBase(unittest.TestCase):
    def setUp(self):
        self._tmp, self.root = _make_root()
        self._old_cwd = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(os.chdir, self._old_cwd)
        # patch optimizer's captured project root so artifact paths land in tmp
        self._patches = [
            mock.patch("iSpy.vision.optimizer._PROJECT_ROOT", self.root),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)
        # make offline mode off by default
        self._offline_patch = mock.patch(
            "iSpy.validations.optimize.network_policy.is_offline", return_value=False
        )
        self._offline_patch.start()
        self.addCleanup(self._offline_patch.stop)
        # silence logging config (no file writes)
        self._log_patch = mock.patch("iSpy.validations.optimize._configure_quiet_logging")
        self._log_patch.start()
        self.addCleanup(self._log_patch.stop)
        self._worker_output_patch = mock.patch(
            "iSpy.validations.optimize._worker_output", side_effect=lambda v: contextlib.nullcontext()
            if False
            else __import__("contextlib").nullcontext()
        )
        self._worker_output_patch.start()
        self.addCleanup(self._worker_output_patch.stop)
        for _name in ("_backup_artifact", "_restore_artifact", "_discard_backup"):
            _pp = mock.patch(f"iSpy.validations.optimize.{_name}", return_value=None)
            _pp.start()
            self.addCleanup(_pp.stop)

    def _write_pt(self, name: str, content: bytes = b"fake pt") -> Path:
        p = self.root / "YoloModels" / "pytorch" / name
        p.write_bytes(content)
        return p

    def _run(self, *argv):
        from iSpy.validations.optimize import main

        return main(list(argv))

    def _run_capture(self, *argv):
        from iSpy.validations.optimize import main

        import io

        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            rc = main(list(argv))
        return rc, buf.getvalue()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
class DiscoveryTests(_OptBase):
    def test_all_dedupes_world_excludes_parts_and_defaults(self):
        self._write_pt("a.pt")
        self._write_pt("b.pt", b"fake pt")  # same fingerprint as a.pt
        self._write_pt("_default_detect.pt")
        (self.root / "YoloModels" / "pytorch" / "world").mkdir(parents=True, exist_ok=True)
        (self.root / "YoloModels" / "pytorch" / "world" / "w.pt").write_bytes(b"world")
        (self.root / "YoloModels" / "pytorch" / "x.part").write_bytes(b"part")

        discovered = _opt._discover_all(excludes=[], skip_defaults=True)
        names = [p.name for p in discovered]
        # world + .part excluded; default skipped
        self.assertNotIn("_default_detect.pt", names)
        self.assertNotIn("w.pt", names)
        # a.pt and b.pt dedupe by fingerprint
        self.assertEqual(len(discovered), 1)

    def test_exclude_filters(self):
        self._write_pt("a.pt", b"aaa")
        self._write_pt("b.pt", b"bbb")
        discovered = _opt._discover_all(excludes=["a*"], skip_defaults=False)
        names = [p.name for p in discovered]
        self.assertNotIn("a.pt", names)
        self.assertIn("b.pt", names)

    def test_skip_defaults(self):
        self._write_pt("_default_detect.pt")
        self._write_pt("real.pt")
        discovered = _opt._discover_all(excludes=[], skip_defaults=False)
        names = [p.name for p in discovered]
        self.assertIn("_default_detect.pt", names)
        discovered2 = _opt._discover_all(excludes=[], skip_defaults=True)
        names2 = [p.name for p in discovered2]
        self.assertNotIn("_default_detect.pt", names2)
        self.assertIn("real.pt", names2)


# ---------------------------------------------------------------------------
# Failure report handling
# ---------------------------------------------------------------------------
class FailureReportTests(_OptBase):
    def test_failure_report_is_failed_not_ok(self):
        self._write_pt("m.pt")
        # mock converter to return a report (failure)
        with mock.patch(
            "iSpy.vision.optimizer._convert_model_subprocess_report",
            return_value=(self.root / "YoloModels" / "pytorch" / "m.pt", {"status": "dependency", "error": "missing rknn"}),
        ):
            with mock.patch("iSpy.validations.optimize._ensure_sidecar", return_value=True):
                with mock.patch("iSpy.validations.optimize._resolve_dataset", return_value=True):
                    rc = self._run("--all", "--format", "rknn", "--no-verify")
        self.assertEqual(rc, 1)

    def test_fail_fast_stops_at_first_failure(self):
        self._write_pt("a.pt", b"aaa")
        self._write_pt("b.pt", b"bbb")
        call_count = {"n": 0}

        def fake_converter(*a, **kw):
            call_count["n"] += 1
            return (Path(kw["model_file"]), {"status": "conversion", "error": "boom"})

        with mock.patch(
            "iSpy.vision.optimizer._convert_model_subprocess_report",
            side_effect=fake_converter,
        ):
            with mock.patch("iSpy.validations.optimize._ensure_sidecar", return_value=True):
                with mock.patch("iSpy.validations.optimize._resolve_dataset", return_value=True):
                    rc = self._run("--all", "--format", "onnx", "--fail-fast", "--no-verify")
        self.assertEqual(rc, 1)
        # only first job ran
        self.assertEqual(call_count["n"], 1)


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------
class FreshnessTests(_OptBase):
    def test_fresh_artifact_skipped(self):
        pt = self._write_pt("m.pt")
        fmt = "onnx"
        art = _opt._artifact_path(pt, fmt)
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_bytes(b"artifact")
        # ensure sidecar
        from iSpy.vision.metadata import metadata_path_for, write_metadata

        write_metadata(metadata_path_for(art), {"quantize": False, "quantization": "none"})
        # make artifact newer
        os.utime(art, (pt.stat().st_mtime + 100, pt.stat().st_mtime + 100))

        with mock.patch(
            "iSpy.vision.optimizer._convert_model_subprocess_report"
        ) as conv:
            with mock.patch("iSpy.validations.optimize._ensure_sidecar", return_value=True):
                rc = self._run("--all", "--format", "onnx", "--no-verify")
            conv.assert_not_called()
        self.assertEqual(rc, 0)

    def test_force_rebuilds(self):
        pt = self._write_pt("m.pt")
        fmt = "onnx"
        art = _opt._artifact_path(pt, fmt)
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_bytes(b"artifact")
        from iSpy.vision.metadata import metadata_path_for, write_metadata

        write_metadata(metadata_path_for(art), {"quantize": False, "quantization": "none"})
        os.utime(art, (pt.stat().st_mtime + 100, pt.stat().st_mtime + 100))

        with mock.patch(
            "iSpy.vision.optimizer._convert_model_subprocess_report",
            return_value=(art, None),
        ) as conv:
            with mock.patch("iSpy.validations.optimize._ensure_sidecar", return_value=True):
                with mock.patch("iSpy.validations.optimize._resolve_dataset", return_value=True):
                    rc = self._run("--all", "--format", "onnx", "--force", "--no-verify")
            conv.assert_called()
        self.assertEqual(rc, 0)


# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------
class FormatResolutionTests(_OptBase):
    def _patch_autoopt(self, **flags):
        import iSpy.config.AutoOpt as AutoOpt

        for name, val in flags.items():
            patch = mock.patch.object(AutoOpt, name, val)
            patch.start()
            self.addCleanup(patch.stop)

    def test_auto_uses_recommend_format(self):
        self._write_pt("m.pt")
        self._patch_autoopt(
            has_rockchip_npu=mock.MagicMock(return_value=True),
        )
        with mock.patch(
            "iSpy.validations.optimize._recommended_format",
            return_value="rknn",
        ):
            art2 = self.root / "YoloModels" / "rknn" / "m.rknn"
            art2.parent.mkdir(parents=True, exist_ok=True)
            art2.write_bytes(b"artifact")
            with mock.patch(
                "iSpy.vision.optimizer._convert_model_subprocess_report",
                return_value=(art2, None),
            ):
                with mock.patch("iSpy.validations.optimize._ensure_sidecar", return_value=True):
                    with mock.patch("iSpy.validations.optimize._resolve_dataset", return_value=True):
                        rc = self._run("--all", "--format", "auto", "--no-verify")
            self.assertEqual(rc, 0)

    def test_all_formats_resolution(self):
        self._patch_autoopt(
            has_rockchip_npu=mock.MagicMock(return_value=False),
            has_nvidia=mock.MagicMock(return_value=True),
            has_tensorrt=mock.MagicMock(return_value=True),
            has_intel_gpu=mock.MagicMock(return_value=False),
            has_intel_vpu=mock.MagicMock(return_value=False),
            has_qualcomm_npu=mock.MagicMock(return_value=False),
            has_arm=mock.MagicMock(return_value=False),
            has_edge_tpu=mock.MagicMock(return_value=False),
            has_tpu=mock.MagicMock(return_value=False),
        )
        fmts = _opt._all_formats()
        self.assertIn("onnx", fmts)
        self.assertIn("engine", fmts)
        self.assertNotIn("coreml", fmts)
        self.assertNotIn("hailo", fmts)


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------
class GuardRailTests(_OptBase):
    def test_hailo_pose_skipped(self):
        pt = self._write_pt("m.pt")
        # generate a sidecar claiming pose task
        from iSpy.vision.metadata import write_metadata, metadata_path_for

        write_metadata(metadata_path_for(pt), {"task": "pose", "nc": 1, "input_size": [640, 640]})
        jobs = _opt._plan_jobs(
            [_opt._ModelUse(_opt._model_info(pt))],
            ["hailo"],
            _opt._build_parser().parse_args([]),
        )
        self.assertEqual(jobs[0].action, "unsupported")

    def test_hailo_non640_skipped(self):
        pt = self._write_pt("m.pt")
        from iSpy.vision.metadata import write_metadata, metadata_path_for

        write_metadata(metadata_path_for(pt), {"task": "detect", "nc": 1, "input_size": [320, 320]})
        jobs = _opt._plan_jobs(
            [_opt._ModelUse(_opt._model_info(pt))],
            ["hailo"],
            _opt._build_parser().parse_args([]),
        )
        self.assertEqual(jobs[0].action, "unsupported")

    def test_qnn_pose_skipped(self):
        pt = self._write_pt("m.pt")
        from iSpy.vision.metadata import write_metadata, metadata_path_for

        write_metadata(metadata_path_for(pt), {"task": "pose", "nc": 1, "input_size": [640, 640]})
        jobs = _opt._plan_jobs(
            [_opt._ModelUse(_opt._model_info(pt))],
            ["qnn"],
            _opt._build_parser().parse_args([]),
        )
        self.assertEqual(jobs[0].action, "unsupported")

    def test_tpu_passthrough_ok(self):
        pt = self._write_pt("m.pt")
        from iSpy.vision.metadata import write_metadata, metadata_path_for

        write_metadata(metadata_path_for(pt), {"task": "detect", "nc": 1, "input_size": [640, 640]})
        jobs = _opt._plan_jobs(
            [_opt._ModelUse(_opt._model_info(pt))],
            ["tpu"],
            _opt._build_parser().parse_args([]),
        )
        self.assertEqual(jobs[0].action, "passthrough")


# ---------------------------------------------------------------------------
# Offline preflight
# ---------------------------------------------------------------------------
class OfflineTests(_OptBase):
    def test_offline_skips_needing_pip(self):
        self._write_pt("m.pt")
        self._offline_patch.stop()
        with mock.patch("iSpy.validations.optimize.network_policy.is_offline", return_value=True):
            with mock.patch("iSpy.validations.optimize._offline_reason", return_value="missing build dependencies offline: ultralytics"):
                with mock.patch("iSpy.validations.optimize._ensure_sidecar", return_value=True):
                    rc = self._run("--all", "--format", "onnx", "--no-verify")
        self.assertEqual(rc, 0)


# ---------------------------------------------------------------------------
# Dry-run and list
# ---------------------------------------------------------------------------
class DryRunAndListTests(_OptBase):
    def test_dry_run_never_calls_converter(self):
        self._write_pt("m.pt")
        with mock.patch("iSpy.vision.optimizer._convert_model_subprocess_report") as conv:
            with mock.patch("iSpy.validations.optimize._ensure_sidecar", return_value=True):
                rc = self._run("--all", "--dry-run")
            conv.assert_not_called()
        self.assertEqual(rc, 0)

    def test_list_never_calls_converter_and_no_torch(self):
        self._write_pt("m.pt")
        with mock.patch("iSpy.vision.optimizer._convert_model_subprocess_report") as conv:
            rc = self._run("--list")
            conv.assert_not_called()
        self.assertEqual(rc, 0)
        # subprocess check for torch import
        code = (
            "import sys; import iSpy.validations.optimize; "
            "print(int('torch' in sys.modules), int('cv2' in sys.modules), int('onnx' in sys.modules), int('ultralytics' in sys.modules))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "0 0 0 0")

    def test_no_selection_exits_2(self):
        rc = self._run()
        self.assertEqual(rc, 2)


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
class ApplyTests(_OptBase):
    def _make_config(self):
        import json

        cfg_path = self.root / "Config" / "config.json"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "camera_configs": {
                "cam_a": {
                    "name": "cam_a",
                    "camera_type": "opencv",
                    "source": 0,
                    "pipeline": {
                        "name": "object_detection",
                        "settings": {
                            "vision_model": {
                                "file_path": "YoloModels/pytorch/m.pt",
                                "source_pt": "YoloModels/pytorch/m.pt",
                                "optimize": "off",
                                "quantize": False,
                            }
                        },
                    },
                },
                "cam_b": {
                    "name": "cam_b",
                    "camera_type": "opencv",
                    "source": 0,
                    "pipeline": {
                        "name": "object_detection",
                        "settings": {
                            "vision_model": {
                                "file_path": "YoloModels/pytorch/other.pt",
                                "source_pt": "YoloModels/pytorch/other.pt",
                                "optimize": "off",
                                "quantize": False,
                            }
                        },
                    },
                },
            }
        }
        cfg_path.write_text(json.dumps(data))
        return cfg_path

    def test_apply_only_matching_cameras_on_success(self):
        pt = self._write_pt("m.pt")
        self._make_config()
        art = _opt._artifact_path(pt, "onnx")
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_bytes(b"artifact")
        from iSpy.vision.metadata import write_metadata, metadata_path_for

        write_metadata(metadata_path_for(art), {"quantize": False, "quantization": "none"})

        # build a fake successful job
        from iSpy.validations.optimize import Job, ModelInfo

        info = _opt._model_info(pt)
        job = Job(model=info, fmt="onnx", quantize=False, input_size=(640, 640), status="OK", artifact=art)
        # simulate apply
        with mock.patch("iSpy.validations.optimize._ensure_sidecar", return_value=True):
            applied = _opt._apply_results([job], None, [])
        # config not passed, so nothing applied
        self.assertEqual(applied, [])


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------
class ExitCodeTests(_OptBase):
    def test_failed_job_returns_1(self):
        self._write_pt("m.pt")
        with mock.patch(
            "iSpy.vision.optimizer._convert_model_subprocess_report",
            return_value=(self.root / "YoloModels" / "pytorch" / "m.pt", {"status": "conversion", "error": "boom"}),
        ):
            with mock.patch("iSpy.validations.optimize._ensure_sidecar", return_value=True):
                with mock.patch("iSpy.validations.optimize._resolve_dataset", return_value=True):
                    rc = self._run("--all", "--format", "onnx", "--no-verify")
        self.assertEqual(rc, 1)

    def test_exit_130_on_interrupt(self):
        self._write_pt("m.pt")

        def raise_kb(*a, **kw):
            raise KeyboardInterrupt()

        with mock.patch(
            "iSpy.vision.optimizer._convert_model_subprocess_report",
            side_effect=raise_kb,
        ):
            with mock.patch("iSpy.validations.optimize._ensure_sidecar", return_value=True):
                with mock.patch("iSpy.validations.optimize._resolve_dataset", return_value=True):
                    rc = self._run("--all", "--format", "onnx", "--no-verify")
        self.assertEqual(rc, 130)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
class JsonTests(_OptBase):
    def test_json_schema(self):
        self._write_pt("m.pt")
        art = self.root / "YoloModels" / "onnx" / "m.onnx"
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_bytes(b"artifact")
        with mock.patch(
            "iSpy.vision.optimizer._convert_model_subprocess_report",
            return_value=(art, None),
        ):
            with mock.patch("iSpy.validations.optimize._ensure_sidecar", return_value=True):
                with mock.patch("iSpy.validations.optimize._resolve_dataset", return_value=True):
                    rc, out = self._run_capture("--all", "--format", "onnx", "--no-verify", "--json", "-")
        self.assertEqual(rc, 0)
        # extract JSON from stdout
        data = json.loads(out)
        self.assertIn("jobs", data)
        self.assertIn("summary", data)
        self.assertIn("exit_code", data)
        job = data["jobs"][0]
        for key in ("model", "format", "status", "artifact", "size", "seconds", "quantize", "verify", "compare", "error"):
            self.assertIn(key, job)


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
class CleanTests(_OptBase):
    def test_clean_only_orphans(self):
        self._write_pt("m.pt")
        # orphan artifact
        orphan_dir = self.root / "YoloModels" / "onnx"
        orphan_dir.mkdir(parents=True, exist_ok=True)
        orphan_art = orphan_dir / "orphan.onnx"
        orphan_art.write_bytes(b"x")
        from iSpy.vision.metadata import metadata_path_for, write_metadata

        write_metadata(metadata_path_for(orphan_art), {"quantize": False})
        # valid artifact (source exists)
        valid_dir = self.root / "YoloModels" / "onnx"
        valid_art = valid_dir / "m.onnx"
        valid_art.write_bytes(b"y")
        write_metadata(metadata_path_for(valid_art), {"quantize": False})
        # rknn_ready leftover
        leftover = self.root / "YoloModels" / "onnx" / "m_rknn_ready.onnx"
        leftover.write_bytes(b"z")
        # .pt in pytorch should survive
        # QuantizeDataset file
        qd = self.root / "QuantizeDataset" / "default"
        qd.mkdir(parents=True, exist_ok=True)
        (qd / "img.jpg").write_bytes(b"img")

        rc = self._run("--clean")
        self.assertEqual(rc, 0)
        self.assertFalse(orphan_art.exists())
        self.assertTrue(valid_art.exists())
        self.assertTrue((self.root / "YoloModels" / "pytorch" / "m.pt").exists())
        self.assertTrue((qd / "img.jpg").exists())


import contextlib


if __name__ == "__main__":
    unittest.main()