"""Tests for the generalized conversion-environment registry.

All provisioning/network operations are mocked.  These tests assert the
registry contract, per-format cache layout/locking, manual-only Hailo behavior,
CLI rendering, optimizer routing for TFLite, and Depth Anything's TFLite worker.
"""

import io
import json
import logging
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from iSpy.boot import converter_envs as envs
from iSpy import cli

_REPO_ROOT = Path(__file__).resolve().parents[1]


class RegistryTests(unittest.TestCase):
    def test_registry_has_expected_entries_and_reasons(self):
        expected = {
            "rknn",
            "tflite",
            "hailo",
            "engine",
            "openvino",
            "onnx",
            "qnn",
            "tpu",
            "coreml",
        }
        self.assertEqual(set(envs.CONVERTER_ENVS), expected)
        for fmt, spec in envs.CONVERTER_ENVS.items():
            self.assertTrue(envs.converter_env_reason(fmt), fmt)
            if spec is None:
                self.assertIn(fmt, envs.CONVERTER_ENV_REASONS)
                continue
            self.assertEqual(spec.format, fmt)
            self.assertTrue(spec.reason)
            self.assertTrue(spec.verify_script)
            self.assertTrue(spec.light_verify_script)
            if fmt == "tflite":
                # TensorFlow/onnx2tf/torch fit in ~4 GiB; the old 8 GiB
                # floor rejected boards with 5-7 GiB free.
                self.assertGreaterEqual(spec.min_free_bytes, 4 * 1024 ** 3)
            else:
                self.assertGreaterEqual(spec.min_free_bytes, 6 * 1024 ** 3)
            if spec.manual_install_hint:
                self.assertFalse(spec.supported)
            else:
                self.assertTrue(spec.supported)

    def test_needs_env_boundaries(self):
        self.assertTrue(envs.needs_env("rknn"))
        self.assertTrue(envs.needs_env("tflite"))
        self.assertTrue(envs.needs_env("hailo"))
        for fmt in ("engine", "openvino", "onnx", "qnn", "tpu", "coreml"):
            self.assertFalse(envs.needs_env(fmt), fmt)
        self.assertFalse(envs.needs_env("not-a-format"))
        self.assertFalse(envs.needs_env(None))

    def test_tflite_spec_contains_ultralytics_export_stack(self):
        spec = envs.get_env_spec("tflite")
        self.assertIsNotNone(spec)
        blob = " ".join(spec.requirements)
        for name in (
            "tensorflow==2.19.0",
            "tf_keras==2.19.0",
            "onnx2tf",
            "ai-edge-litert",
            "onnx_graphsurgeon",
            "sng4onnx",
            "onnxslim",
            "torch",
            "ultralytics",
            "opencv-python",
            "ruamel.yaml",
        ):
            self.assertIn(name, blob)
        self.assertIn("numpy==1.26.4", spec.requirements)
        self.assertIn("protobuf==4.25.4", spec.requirements)
        self.assertIn("torch==2.2.0", spec.requirements)
        self.assertIn("torchvision==0.17.0", spec.requirements)
        self.assertIn("onnx==1.16.1", spec.requirements)

    def test_rknn_pins_match_toolkit_metadata(self):
        """Exact pins keep a resolver/auto-install from drifting the env."""
        spec = envs.get_env_spec("rknn")
        blob = " ".join(spec.requirements)
        for requirement in (
            "numpy==1.26.4",
            "onnx==1.16.1",
            "protobuf==4.25.4",
            "torch==2.2.0",
            "torchvision==0.17.0",
        ):
            self.assertIn(requirement, blob)
        # v9 forces a rebuild of envs built before the toolkit's remaining
        # METADATA deps (onnxruntime et al.) were listed.
        self.assertGreaterEqual(spec.depset_version, 9)

    def test_rknn_requirements_cover_toolkit_metadata(self):
        """The --no-deps vendor wheel needs its METADATA deps listed explicitly."""
        spec = envs.get_env_spec("rknn")
        for requirement in (
            "onnxruntime==1.23.0",
            "fast-histogram==0.14",
            "psutil>=5.9.0",
            "Pillow>=10.0.1",
        ):
            self.assertIn(requirement, spec.requirements)
        for probe in (
            "import onnxruntime",
            "import fast_histogram",
            "import psutil",
            "from PIL import Image",
        ):
            self.assertIn(probe, spec.verify_script)
        # rknn.build() imports onnxruntime lazily, so the cheap daily check
        # must catch a missing wheel before the next conversion.
        self.assertIn("import onnxruntime", spec.light_verify_script)

    def test_tflite_export_is_gated_on_arm64(self):
        with mock.patch.object(envs.sys, "platform", "linux"), mock.patch.object(
            envs, "_arch_key", return_value="aarch64"
        ):
            self.assertFalse(envs.exporter_supported("tflite"))
        with mock.patch.object(envs.sys, "platform", "linux"), mock.patch.object(
            envs, "_arch_key", return_value="x86_64"
        ):
            self.assertTrue(envs.exporter_supported("tflite"))
        # macOS ARM64 is supported by the LiteRT exporter.
        with mock.patch.object(envs.sys, "platform", "darwin"), mock.patch.object(
            envs, "_arch_key", return_value="aarch64"
        ):
            self.assertTrue(envs.exporter_supported("tflite"))
        self.assertTrue(envs.exporter_supported("rknn"))

    def test_identity_fingerprints_dependency_set(self):
        """A pin or probe change must invalidate the env automatically."""
        from dataclasses import replace

        spec = envs.get_env_spec("rknn")
        before = envs._identity(spec)
        changed = replace(spec, requirements=spec.requirements + ("example==1",))
        after = envs._identity(changed)

        self.assertIn("spec_hash", before)
        self.assertNotEqual(before["spec_hash"], after["spec_hash"])
        self.assertNotEqual(
            envs._identity_hash(before), envs._identity_hash(after)
        )

    def test_disk_floor_override(self):
        """Per-format overrides work without changing the registry."""
        default = envs.get_env_spec("tflite").min_free_bytes
        with mock.patch.dict(
            os.environ, {"ISPY_TFLITE_MIN_FREE_GIB": "2.5"}, clear=False
        ):
            self.assertEqual(envs.min_free_bytes("tflite"), int(2.5 * 1024 ** 3))
        with mock.patch.dict(
            os.environ, {"ISPY_TFLITE_MIN_FREE_GIB": "not-a-number"}, clear=False
        ):
            self.assertEqual(envs.min_free_bytes("tflite"), default)

    def test_disk_check_accepts_7_9_gib_for_both_envs(self):
        """7.9 GiB is enough for either converter stack."""
        free = int(7.9 * 1024 ** 3)
        for fmt in ("rknn", "tflite"):
            with mock.patch.object(
                envs, "_free_bytes", return_value=(free, Path("/tmp"))
            ):
                envs._check_disk_space(
                    envs.get_env_spec(fmt), Path("/tmp/env"), purge_cache=lambda: False
                )

    def test_disk_check_rejects_too_little_space(self):
        free = int(4.5 * 1024 ** 3)
        with mock.patch.object(
            envs, "_free_bytes", return_value=(free, Path("/tmp"))
        ):
            with self.assertRaises(envs.ConverterEnvError):
                envs._check_disk_space(
                    envs.get_env_spec("tflite"),
                    Path("/tmp/env"),
                    purge_cache=lambda: False,
                )

    def test_optimize_preflight_uses_env_floors(self):
        """Serial builds use the largest floor; concurrent builds add them."""
        from iSpy.validations.optimize import _estimated_disk_bytes

        rknn = envs.min_free_bytes("rknn")
        tflite = envs.min_free_bytes("tflite")
        self.assertEqual(
            _estimated_disk_bytes(["rknn", "tflite"], jobs=1), max(rknn, tflite)
        )
        self.assertEqual(
            _estimated_disk_bytes(["rknn", "tflite"], jobs=2), rknn + tflite
        )

    def test_runtime_inference_is_not_moved_into_tflite_env(self):
        from iSpy.vision.optimizer import BACKEND_DEPENDENCIES

        targets = [entry[1] for entry in BACKEND_DEPENDENCIES["tflite"]]
        self.assertTrue(
            any("tflite" in target or "litert" in target for target in targets),
            targets,
        )
        self.assertFalse(any("tensorflow" in target for target in targets))

    def test_cache_override_prefers_general_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            general = Path(tmp) / "general"
            legacy = Path(tmp) / "legacy"
            with mock.patch.dict(
                os.environ,
                {
                    "ISPY_CONVERTER_ENV_HOME": str(general),
                    "ISPY_RKNN_CONVERTER_HOME": str(legacy),
                },
            ):
                self.assertEqual(envs._cache_root(), general.resolve())
            with mock.patch.dict(
                os.environ,
                {"ISPY_RKNN_CONVERTER_HOME": str(legacy)},
                clear=False,
            ):
                os.environ.pop("ISPY_CONVERTER_ENV_HOME", None)
                self.assertEqual(envs._cache_root(), legacy.resolve())

    def test_env_dirs_are_format_specific(self):
        rknn = envs._env_dir(envs.get_env_spec("rknn"))
        tflite = envs._env_dir(envs.get_env_spec("tflite"))
        self.assertEqual(rknn.parent.name, "rknn_env")
        self.assertEqual(tflite.parent.name, "tflite_env")
        self.assertNotEqual(rknn, tflite)

    def test_per_format_build_locks_are_independent(self):
        rknn = envs._lock_path(envs.get_env_spec("rknn"))
        tflite = envs._lock_path(envs.get_env_spec("tflite"))
        self.assertNotEqual(rknn, tflite)
        self.assertEqual(rknn.parent, rknn.parent.parent / "rknn_env")
        self.assertEqual(tflite.parent, tflite.parent.parent / "tflite_env")

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ, {"ISPY_CONVERTER_ENV_HOME": tmp}, clear=False
            ):
                rknn = envs._lock_path(envs.get_env_spec("rknn"))
                tflite = envs._lock_path(envs.get_env_spec("tflite"))
                rknn.parent.mkdir(parents=True, exist_ok=True)
                tflite.parent.mkdir(parents=True, exist_ok=True)
                # Holding the RKNN lock must not block a TFLite build.
                with envs._build_lock(rknn, timeout=2.0, label="rknn"):
                    with envs._build_lock(tflite, timeout=2.0, label="tflite"):
                        self.assertTrue(rknn.exists())
                        self.assertTrue(tflite.exists())
                self.assertFalse(rknn.exists())
                self.assertFalse(tflite.exists())

    def test_none_entry_status_has_reason_and_no_path(self):
        status = envs.converter_env_status("onnx")
        self.assertFalse(status["needs_env"])
        self.assertFalse(status["exists"])
        self.assertEqual(status["env_dir"], "")
        self.assertTrue(status["reason"])

    def test_hailo_is_manual_only_and_never_pip_installs(self):
        with mock.patch.object(envs, "_provision") as provision:
            with self.assertRaises(envs.ConverterEnvError) as ctx:
                envs.converter_python("hailo")
        self.assertIn("Dataflow Compiler", str(ctx.exception))
        provision.assert_not_called()

    def test_offline_missing_env_fails_before_provisioning(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ, {"ISPY_CONVERTER_ENV_HOME": tmp}, clear=False
            ), mock.patch.object(envs, "_env_is_current", return_value=False), \
                 mock.patch.object(envs, "_light_verify", return_value=False), \
                 mock.patch.object(envs, "_cleanup_orphans"), \
                 mock.patch.object(envs, "_ensure_supported"), \
                 mock.patch.object(envs, "_provision") as provision, \
                 mock.patch(
                     "iSpy.core.network_policy.is_offline", return_value=True
                 ):
                with self.assertRaises(envs.ConverterEnvError) as ctx:
                    envs.converter_python("tflite")
        self.assertIn("offline mode", str(ctx.exception).lower())
        provision.assert_not_called()

    def test_provision_records_manifest_and_verified_stamp(self):
        spec = envs.get_env_spec("rknn")
        with tempfile.TemporaryDirectory() as tmp:
            env_dir = Path(tmp) / "rknn_env" / "abc"
            py = envs._is_py(env_dir)

            def fake_run(cmd, **kwargs):
                env_dir.mkdir(parents=True, exist_ok=True)
                py.parent.mkdir(parents=True, exist_ok=True)
                py.write_text("#!/bin/sh\n")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.object(envs.subprocess, "run", side_effect=fake_run), \
                 mock.patch.object(envs, "_arch_key", return_value="x86_64"), \
                 mock.patch.object(envs, "_python_tag", return_value="cp312"), \
                 mock.patch.object(envs, "_check_disk_space"), \
                 mock.patch.object(envs, "_pip_install") as pip_install, \
                 mock.patch.object(envs, "_verify"):
                result = envs._provision(spec, env_dir)

            self.assertEqual(result, py)
            manifest = envs._read_manifest(env_dir)
            self.assertEqual(manifest["format"], "rknn")
            self.assertEqual(manifest["toolkit_version"], spec.toolkit_version)
            self.assertTrue((env_dir / ".verified").exists())
            # Vendor wheel first with --no-deps, then the pinned set.
            self.assertTrue(pip_install.call_args_list[0].kwargs["no_deps"])
            self.assertEqual(
                pip_install.call_args_list[1].args[2:], spec.requirements
            )


class TfliteExportGateTests(unittest.TestCase):
    """ARM64 keeps TFLite inference but cannot export a .pt to .tflite."""

    def _detect_plan(self, *, arm, supported):
        from iSpy.validations import benchmarking

        with mock.patch.object(
            benchmarking, "has_rockchip_npu", return_value=False
        ), mock.patch.object(benchmarking, "has_tpu", return_value=False), mock.patch.object(
            benchmarking, "has_nvidia", return_value=False
        ), mock.patch.object(
            benchmarking, "has_tensorrt", return_value=False
        ), mock.patch.object(
            benchmarking, "_has_openvino", return_value=False
        ), mock.patch.object(
            benchmarking, "has_arm", return_value=arm
        ), mock.patch.object(
            benchmarking, "has_edge_tpu", return_value=False
        ), mock.patch.object(
            benchmarking, "exporter_supported", return_value=supported
        ):
            with redirect_stdout(io.StringIO()):
                return benchmarking.detect_test_plan()

    def test_all_backends_skips_tflite_when_export_unsupported(self):
        plan = self._detect_plan(arm=True, supported=False)
        self.assertNotIn("tflite", plan)
        self.assertIn("pt_cpu", plan)
        self.assertIn("onnx_cpu", plan)

    def test_all_backends_keeps_tflite_when_export_supported(self):
        plan = self._detect_plan(arm=True, supported=True)
        self.assertIn("tflite", plan)

    def test_recommended_plan_falls_back_to_onnx(self):
        from iSpy.validations import benchmarking

        with mock.patch(
            "iSpy.config.AutoOpt.recommend_format", return_value="tflite"
        ), mock.patch.object(
            benchmarking, "exporter_supported", return_value=False
        ), mock.patch.object(
            benchmarking, "has_nvidia", return_value=False
        ):
            with redirect_stdout(io.StringIO()):
                plan = benchmarking._recommended_backend_plan()

        self.assertEqual(set(plan), {"onnx"})
        self.assertEqual(plan["onnx"][1], "cpu")

    def test_optimize_all_formats_skips_tflite(self):
        from iSpy.validations import optimize

        with mock.patch.object(
            optimize, "exporter_supported", return_value=False
        ), mock.patch(
            "iSpy.config.AutoOpt.has_arm", return_value=True
        ), mock.patch(
            "iSpy.config.AutoOpt.has_nvidia", return_value=False
        ), mock.patch(
            "iSpy.config.AutoOpt.has_edge_tpu", return_value=False
        ), mock.patch(
            "iSpy.config.AutoOpt.has_rockchip_npu", return_value=False
        ), mock.patch(
            "iSpy.config.AutoOpt.has_qualcomm_npu", return_value=False
        ), mock.patch(
            "iSpy.config.AutoOpt.has_tpu", return_value=False
        ), mock.patch(
            "iSpy.config.AutoOpt.has_intel_gpu", return_value=False
        ), mock.patch(
            "iSpy.config.AutoOpt.has_intel_vpu", return_value=False
        ), mock.patch.object(
            optimize, "_rknn_env_ready", return_value=False
        ), mock.patch.object(
            optimize, "_module_installed", return_value=False
        ):
            formats = optimize._all_formats()

        self.assertNotIn("tflite", formats)
        self.assertIn("onnx", formats)


class OptimizerRoutingTests(unittest.TestCase):
    def test_tflite_uses_converter_env_python(self):
        from iSpy.vision import optimizer as opt

        fake_env = mock.Mock()
        fake_env.converter_python.return_value = Path("/fake/tflite/bin/python")
        fake_env.ispy_source_root.return_value = _REPO_ROOT

        def fake_run(cmd, **kwargs):
            args_path = Path(cmd[-1])
            result_path = Path(str(args_path) + ".result.json")
            result_path.write_text('{"result": "out.tflite"}')
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.object(
            opt, "_cached_converter_artifact", return_value=None
        ), mock.patch.object(opt, "_converter_envs", fake_env), mock.patch(
            "subprocess.run", side_effect=fake_run
        ) as run:
            result = opt._convert_model_subprocess(
                "model.pt", "tflite", [640, 640], quantize=False
            )

        self.assertEqual(result, Path("out.tflite"))
        fake_env.converter_python.assert_called_once_with("tflite")
        cmd = run.call_args[0][0]
        self.assertEqual(cmd[0], str(Path("/fake/tflite/bin/python")))
        self.assertIn("iSpy.boot._convert_worker", cmd[1:])
        worker_env = run.call_args[1]["env"]
        self.assertEqual(worker_env["PYTHONPATH"], str(_REPO_ROOT))
        self.assertEqual(worker_env["PYTHONNOUSERSITE"], "1")

    def test_convert_model_refuses_new_tflite_on_arm64(self):
        from iSpy.vision import optimizer as opt

        with tempfile.TemporaryDirectory() as tmp:
            pt = Path(tmp) / "model.pt"
            pt.write_bytes(b"pt")
            with mock.patch.object(
                opt, "exporter_supported", return_value=False
            ), mock.patch.object(opt, "_ensure_export_deps") as deps:
                result = opt.convert_model(
                    str(pt), "tflite", (640, 640), quantize=False
                )

        self.assertEqual(Path(result), pt)
        deps.assert_not_called()

    def test_cached_tflite_skips_provisioning_and_worker(self):
        from iSpy.vision import optimizer as opt

        with tempfile.TemporaryDirectory() as tmp:
            pt = Path(tmp) / "model.pt"
            pt.write_text("weights")
            artifact = Path(tmp) / "model.tflite"
            artifact.write_bytes(b"tflite")

            fake_env = mock.Mock()
            with mock.patch.object(
                opt, "_desired_output_path", return_value=artifact
            ), mock.patch.object(opt, "_converter_envs", fake_env), mock.patch(
                "subprocess.run"
            ) as run:
                result, report = opt._convert_model_subprocess_report(
                    str(pt), "tflite", [640, 640], quantize=False
                )

        self.assertEqual(result, artifact)
        self.assertIsNone(report)
        fake_env.converter_python.assert_not_called()
        run.assert_not_called()

    def test_cached_hailo_skips_manual_toolchain(self):
        from iSpy.vision import optimizer as opt

        with tempfile.TemporaryDirectory() as tmp:
            pt = Path(tmp) / "model.pt"
            pt.write_text("weights")
            artifact = Path(tmp) / "model.hef"
            artifact.write_bytes(b"hef")

            fake_env = mock.Mock()
            with mock.patch.object(
                opt, "_desired_output_path", return_value=artifact
            ), mock.patch.object(opt, "_converter_envs", fake_env), mock.patch(
                "subprocess.run"
            ) as run:
                result, report = opt._convert_model_subprocess_report(
                    str(pt), "hailo", [640, 640], quantize=False
                )

        self.assertEqual(result, artifact)
        self.assertIsNone(report)
        fake_env.converter_python.assert_not_called()
        run.assert_not_called()

    def test_converter_env_failure_is_reported_as_dependency(self):
        from iSpy.vision import optimizer as opt

        fake_env = mock.Mock()
        fake_env.converter_python.side_effect = RuntimeError("no internet")
        with mock.patch.object(
            opt, "_cached_converter_artifact", return_value=None
        ), mock.patch.object(opt, "_converter_envs", fake_env), mock.patch(
            "subprocess.run"
        ) as run:
            result, report = opt._convert_model_subprocess_report(
                "model.pt", "tflite", [640, 640], quantize=False
            )

        self.assertEqual(result, Path("model.pt"))
        self.assertEqual(report["status"], "dependency")
        self.assertEqual(report["category"], "provision")
        self.assertIn("no internet", report["error"])
        run.assert_not_called()


class DepthTfliteRoutingTests(unittest.TestCase):
    """Depth Anything TFLite builds run in the TFLite converter env."""

    @staticmethod
    def _pipeline():
        from iSpy.vision.pipelines.depth_anything import DepthAnythingPipeline

        da = DepthAnythingPipeline.__new__(DepthAnythingPipeline)
        da.logger = logging.getLogger("test.depth.tflite")
        da.quantize = False
        da._input_size = 518
        return da

    def test_depth_build_spawns_tflite_worker_in_env(self):
        da = self._pipeline()
        with tempfile.TemporaryDirectory() as tmp:
            onnx = Path(tmp) / "model.onnx"
            onnx.write_bytes(b"onnx")
            output = Path(tmp) / "model.tflite"

            with mock.patch.object(
                envs,
                "converter_python",
                return_value=Path("/fake/tflite/bin/python"),
            ) as converter_python, mock.patch.object(
                envs, "ispy_source_root", return_value=_REPO_ROOT
            ), mock.patch("subprocess.run") as run:

                def fake_run(cmd, **kwargs):
                    args_path = Path(cmd[-1])
                    payload = json.loads(args_path.read_text())
                    self.assertEqual(payload["onnx"], str(onnx))
                    self.assertEqual(payload["output"], str(output))
                    output.write_bytes(b"tflite")
                    Path(str(args_path) + ".result.json").write_text(
                        json.dumps({"result": str(output)})
                    )
                    return mock.Mock(returncode=0, stdout="", stderr="")

                run.side_effect = fake_run
                da._build_tflite_subprocess(onnx, output)

            converter_python.assert_called_once_with("tflite")
            cmd = run.call_args[0][0]
            self.assertEqual(cmd[0], str(Path("/fake/tflite/bin/python")))
            self.assertIn("iSpy.boot._tflite_onnx_worker", cmd[1:])
            worker_env = run.call_args[1]["env"]
            self.assertEqual(worker_env["PYTHONPATH"], str(_REPO_ROOT))
            self.assertEqual(worker_env["PYTHONNOUSERSITE"], "1")

    def test_depth_build_failure_keeps_real_worker_error(self):
        da = self._pipeline()
        with mock.patch.object(
            envs, "converter_python", return_value=Path("/fake/tflite/bin/python")
        ), mock.patch.object(
            envs, "ispy_source_root", return_value=_REPO_ROOT
        ), mock.patch("subprocess.run") as run:

            def fake_run(cmd, **kwargs):
                args_path = Path(cmd[-1])
                Path(str(args_path) + ".result.json").write_text(
                    json.dumps(
                        {
                            "error": "onnx2tf conversion failed: unsupported op",
                            "error_type": "RuntimeError",
                            "traceback": "...",
                        }
                    )
                )
                return mock.Mock(returncode=1, stdout="", stderr="")

            run.side_effect = fake_run
            with self.assertRaises(RuntimeError) as ctx:
                da._build_tflite_subprocess(
                    Path("model.onnx"), Path("model.tflite")
                )

        self.assertIn("onnx2tf conversion failed", str(ctx.exception))

    def test_cached_tflite_loads_without_provisioning_env(self):
        da = self._pipeline()
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "model.tflite"
            artifact.write_bytes(b"tflite")
            interpreter = mock.Mock()

            with mock.patch.object(
                da, "_artifact_path", return_value=artifact
            ), mock.patch.object(da, "_export_onnx_source") as export, mock.patch.object(
                envs, "converter_python"
            ) as converter_python, mock.patch.object(
                da, "_tflite_interpreter", return_value=interpreter
            ):
                da._load_tflite(force=False)

        export.assert_not_called()
        converter_python.assert_not_called()
        interpreter.allocate_tensors.assert_called_once_with()
        self.assertIs(da._session, interpreter)
        self.assertEqual(da._infer, da._infer_depth_tflite)
        self.assertIsNone(da._load_error)

    def test_missing_tflite_builds_through_env(self):
        da = self._pipeline()
        with tempfile.TemporaryDirectory() as tmp:
            onnx = Path(tmp) / "model.onnx"
            output = Path(tmp) / "model.tflite"
            interpreter = mock.Mock()

            with mock.patch.object(
                da, "_artifact_path", return_value=output
            ), mock.patch.object(
                da, "_export_onnx_source", return_value=(onnx, True)
            ) as export, mock.patch.object(
                da, "_build_tflite_subprocess"
            ) as build, mock.patch.object(
                da, "_tflite_interpreter", return_value=interpreter
            ):
                da._load_tflite(force=False)

        export.assert_called_once_with(False, quantize=False)
        build.assert_called_once_with(onnx, output)
        interpreter.allocate_tensors.assert_called_once_with()
        self.assertEqual(da._infer, da._infer_depth_tflite)

    def test_build_failure_falls_back_to_onnx(self):
        da = self._pipeline()
        with tempfile.TemporaryDirectory() as tmp:
            onnx = Path(tmp) / "model.onnx"
            output = Path(tmp) / "model.tflite"

            with mock.patch.object(
                da, "_artifact_path", return_value=output
            ), mock.patch.object(
                da, "_export_onnx_source", return_value=(onnx, True)
            ), mock.patch.object(
                da, "_build_tflite_subprocess", side_effect=RuntimeError("no env")
            ), mock.patch.object(da, "_load_onnx") as load_onnx:
                da._load_tflite(force=False)

        load_onnx.assert_called_once_with(False)
        self.assertIsNone(da._session)

    def test_worker_moves_onnx2tf_output_to_requested_path(self):
        from iSpy.boot import _tflite_onnx_worker as worker

        with tempfile.TemporaryDirectory() as tmp:
            onnx = Path(tmp) / "model.onnx"
            onnx.write_bytes(b"onnx")
            output = Path(tmp) / "requested.tflite"
            args_path = Path(tmp) / "args.json"
            args_path.write_text(json.dumps({"onnx": str(onnx), "output": str(output)}))
            fake_onnx2tf = mock.Mock()

            def fake_convert(**kwargs):
                folder = Path(kwargs["output_folder_path"])
                (folder / f"{onnx.stem}_float32.tflite").write_bytes(b"tflite")

            fake_onnx2tf.convert.side_effect = fake_convert
            with mock.patch.dict(sys.modules, {"onnx2tf": fake_onnx2tf}), mock.patch.object(
                sys, "argv", ["_tflite_onnx_worker", str(args_path)]
            ):
                worker.main()

            self.assertEqual(output.read_bytes(), b"tflite")
            result = json.loads(
                Path(str(args_path) + ".result.json").read_text()
            )
            self.assertEqual(result["result"], str(output))
            fake_onnx2tf.convert.assert_called_once_with(
                input_onnx_file_path=str(onnx),
                output_folder_path=str(output.parent),
                non_verbose=True,
            )


class CliTests(unittest.TestCase):
    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_parser_accepts_env_subcommands(self):
        parser = cli._build_parser()
        self.assertEqual(parser.parse_args(["env", "list"]).env_command, "list")
        self.assertEqual(
            parser.parse_args(["env", "rebuild", "rknn"]).format, "rknn"
        )
        self.assertEqual(
            parser.parse_args(["env", "remove", "tflite"]).format, "tflite"
        )
        self.assertEqual(
            parser.parse_args(["env", "path", "hailo"]).format, "hailo"
        )

    def test_env_list_renders_required_fields(self):
        status = {
            "format": "rknn",
            "needs_env": True,
            "current": True,
            "env_dir": "C:/cache/rknn_env/abc",
            "size_bytes": 1536,
            "last_verified": "2026-10-21T12:00:00",
            "identity_hash": "abc",
            "reason": "pinned toolkit conflict",
        }
        with mock.patch(
            "iSpy.boot.converter_envs.list_converter_envs", return_value=[status]
        ):
            code, out, _ = self._run(["env", "list"])
        self.assertEqual(code, 0)
        self.assertIn("FORMAT", out)
        self.assertIn("rknn", out)
        self.assertIn("yes", out)
        self.assertIn("C:/cache/rknn_env/abc", out)
        self.assertIn("1.5 KiB", out)
        self.assertIn("2026-10-21T12:00:00", out)
        self.assertIn("abc", out)
        self.assertIn("pinned toolkit conflict", out)

    def test_env_rebuild_and_remove_dispatch(self):
        with mock.patch(
            "iSpy.boot.converter_envs.rebuild_env",
            return_value=Path("/cache/rknn_env/abc/python"),
        ) as rebuild:
            code, out, _ = self._run(["env", "rebuild", "rknn"])
        self.assertEqual(code, 0)
        rebuild.assert_called_once_with("rknn")
        self.assertIn("Rebuilt", out)

        with mock.patch(
            "iSpy.boot.converter_envs.remove_env", return_value=True
        ) as remove:
            code, out, _ = self._run(["env", "remove", "tflite"])
        self.assertEqual(code, 0)
        remove.assert_called_once_with("tflite")
        self.assertIn("Removed", out)

    def test_env_path_explains_no_env_formats(self):
        status = {
            "format": "onnx",
            "needs_env": False,
            "reason": "uses the main environment",
        }
        with mock.patch(
            "iSpy.boot.converter_envs.converter_env_status", return_value=status
        ):
            code, out, _ = self._run(["env", "path", "onnx"])
        self.assertEqual(code, 0)
        self.assertIn("no converter environment", out)
        self.assertIn("main environment", out)

    def test_env_errors_return_exit_code(self):
        with mock.patch(
            "iSpy.boot.converter_envs.rebuild_env",
            side_effect=envs.ConverterEnvError("offline"),
        ):
            code, _, err = self._run(["env", "rebuild", "rknn"])
        self.assertEqual(code, 1)
        self.assertIn("offline", err)


if __name__ == "__main__":
    unittest.main()
