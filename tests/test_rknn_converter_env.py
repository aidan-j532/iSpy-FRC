"""Tests for the isolated RKNN conversion environment.

Covers environment provisioning, routing, error handling, and the
conversion/inference boundary.  All tests mock subprocess execution and
package installation so they run without an RK3588, network downloads, or a
real RKNN conversion.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Ensure the repo root is importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class ConverterEnvProvisioningTests(unittest.TestCase):
    """Environment creation, reuse, and identity validation."""

    def _import_env_module(self):
        from iSpy.boot import _rknn_converter_env

        return _rknn_converter_env

    def test_env_dir_is_outside_project_and_main_venv(self):
        """The converter env must live outside the source tree and main venv."""
        env = self._import_env_module()
        with mock.patch.object(sys, "executable", "/fake/venv/bin/python"):
            status = env.converter_env_status()
        env_dir = Path(status["env_dir"])
        # Not inside the repo
        self.assertFalse(str(env_dir).startswith(str(_REPO_ROOT)))
        # Not inside the main venv
        self.assertNotIn(".venv", str(env_dir))
        self.assertNotIn("site-packages", str(env_dir))

    def test_identity_keyed_to_python_arch_toolkit_and_depset(self):
        env = self._import_env_module()
        identity = env._identity()
        self.assertIn("python_tag", identity)
        self.assertIn("arch", identity)
        self.assertIn("rknn_toolkit_version", identity)
        self.assertIn("depset_version", identity)
        self.assertEqual(identity["rknn_toolkit_version"], "2.3.2")
        self.assertTrue(identity["python_tag"].startswith("cp"))

    def test_mismatched_manifest_is_not_reused(self):
        """An env built with a different depset version is not reused."""
        env = self._import_env_module()
        identity = env._identity()
        env_dir = env._env_dir(identity)

        # Simulate a stale env with an old depset version
        with tempfile.TemporaryDirectory() as tmp:
            stale = Path(tmp) / "stale"
            stale.mkdir()
            manifest = dict(identity)
            manifest["depset_version"] = "1"  # old version
            (stale / "env-manifest.json").write_text(json.dumps(manifest))
            py = env._is_py(stale)
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("#!/bin/sh\n")

            self.assertFalse(env._env_is_current(stale))

    def test_matching_manifest_is_current(self):
        env = self._import_env_module()
        identity = env._identity()

        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good"
            good.mkdir()
            manifest = dict(identity)
            (good / "env-manifest.json").write_text(json.dumps(manifest))
            py = env._is_py(good)
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("#!/bin/sh\n")

            self.assertTrue(env._env_is_current(good))

    def test_missing_manifest_is_not_current(self):
        env = self._import_env_module()

        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            self.assertFalse(env._env_is_current(empty))

    def test_unsupported_arch_raises_clear_error(self):
        env = self._import_env_module()
        with mock.patch.object(env, "_arch_key", return_value="riscv64"):
            with self.assertRaises(Exception) as ctx:
                env._toolkit_wheel_url()
            self.assertIn("riscv64", str(ctx.exception))
            self.assertIn("not possible", str(ctx.exception).lower())

    def test_reuse_does_not_reinstall(self):
        """A valid existing converter env is reused without pip installs."""
        env = self._import_env_module()
        identity = env._identity()

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            with mock.patch.object(env, "_cache_root", return_value=cache):
                # Converter envs are now grouped as <format>_env/<identity>.
                env_dir = env._env_dir(identity)
                py = env._is_py(env_dir)

                with mock.patch.object(env, "_env_is_current", return_value=True), \
                     mock.patch.object(env, "_light_verify", return_value=True):
                    with mock.patch.object(env, "_provision") as mock_provision:
                        result_py = env.converter_python()
                        mock_provision.assert_not_called()
                        self.assertEqual(result_py, py)

    def test_missing_env_is_created_on_first_use(self):
        """converter_python() provisions when no valid env exists."""
        env = self._import_env_module()
        identity = env._identity()

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"

            with mock.patch.object(env, "_cache_root", return_value=cache):
                with mock.patch.object(env, "_provision", return_value=env._is_py(env._env_dir(identity))) as mock_provision:
                    py = env.converter_python()
                    mock_provision.assert_called_once()
                    self.assertTrue(Path(py).exists() or "python" in str(py))

    def test_concurrent_setup_releases_lock(self):
        """The build lock is released after provisioning (no deadlock)."""
        env = self._import_env_module()
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            cache.mkdir()
            lock = cache / "build.lock"
            lock.write_text("999999\n")  # stale PID

            # Should reclaim the stale lock without raising; the lock file
            # is removed in the finally block after the context exits.
            with env._build_lock(lock, timeout=2.0):
                pass
            self.assertFalse(lock.exists())


class ConverterEnvDependencyTests(unittest.TestCase):
    """The pinned dependency set is correct and minimal."""

    def test_numpy_pinned_below_2(self):
        """NumPy must be 1.26.4 (toolkit2 2.3.2 ABI compatibility)."""
        from iSpy.boot._rknn_converter_env import _CONVERTER_REQUIREMENTS

        numpy_req = next(r for r in _CONVERTER_REQUIREMENTS if r.startswith("numpy"))
        # Exact pin matching rknn-toolkit2 2.3.2's METADATA; also satisfies
        # the old "numpy<2" contract.
        self.assertIn("1.26.4", numpy_req)

    def test_setuptools_pinned_below_82(self):
        """setuptools must be <82 (provides pkg_resources)."""
        from iSpy.boot._rknn_converter_env import _CONVERTER_REQUIREMENTS

        st_req = next(r for r in _CONVERTER_REQUIREMENTS if r.startswith("setuptools"))
        self.assertIn("<82", st_req)

    def test_scipy_pinned_for_numpy1_compat(self):
        """scipy must be compatible with numpy 1.x."""
        from iSpy.boot._rknn_converter_env import _CONVERTER_REQUIREMENTS

        scipy_req = next(r for r in _CONVERTER_REQUIREMENTS if r.startswith("scipy"))
        self.assertIn("<1.15", scipy_req)

    def test_legacy_identity_includes_spec_hash(self):
        """A pin/probe change must invalidate the legacy RKNN env too."""
        from iSpy.boot import _rknn_converter_env

        identity = _rknn_converter_env._identity()
        self.assertIn("spec_hash", identity)
        self.assertIn("spec_hash", _rknn_converter_env._canonical_identity(identity))

    def test_pyproject_has_no_platform_wide_numpy_pin(self):
        """Runtime NumPy is resolved per host, not assumed for all aarch64."""
        import tomllib

        data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
        deps = " ".join(data["project"]["dependencies"])
        self.assertNotIn("platform_machine == 'aarch64'", deps)
        self.assertNotIn("platform_system == 'Linux'", deps)

    def test_no_ultralytics_in_main_runtime_deps(self):
        """Ultralytics must NOT be a main runtime dependency."""
        import tomllib

        pyproject = _REPO_ROOT / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text())

        core_deps = data["project"]["dependencies"]
        for dep in core_deps:
            self.assertNotIn("ultralytics", dep.lower())

        # It IS allowed in the optional 'optimizer' extra (build-time only)
        optimizer_deps = data["project"]["optional-dependencies"]["optimizer"]
        self.assertTrue(
            any("ultralytics" in d.lower() for d in optimizer_deps),
            "ultralytics should be in the optimizer extra",
        )

    def test_converter_deps_include_full_toolkit_url(self):
        """The converter env wheel URL points at the toolkit2 release."""
        from iSpy.boot import _rknn_converter_env

        with mock.patch("platform.machine", return_value="aarch64"), \
             mock.patch.object(_rknn_converter_env, "_python_tag", return_value="cp312"):
            url = _rknn_converter_env._toolkit_wheel_url()
        self.assertIn("rknn_toolkit2-2.3.2", url)
        self.assertIn("aarch64", url)
        self.assertTrue(url.endswith(".whl"))


class ConversionRoutingTests(unittest.TestCase):
    """RKNN conversion runs under the converter env's Python interpreter."""

    def test_rknn_uses_converter_python(self):
        """_convert_model_subprocess routes RKNN through the converter env."""
        from iSpy.boot import _rknn_converter_env
        from iSpy.vision import optimizer as opt

        fake_converter_py = "/fake/converter/bin/python"

        with mock.patch.object(
            _rknn_converter_env, "converter_python", return_value=Path(fake_converter_py)
        ), mock.patch.object(
            _rknn_converter_env, "ispy_source_root", return_value=_REPO_ROOT
        ), mock.patch("subprocess.run") as mock_run:

            # Write a result file so _run_once returns it
            def fake_run(cmd, **kwargs):
                args_path = Path(cmd[-1])
                result_path = Path(str(args_path) + ".result.json")
                result_path.write_text(json.dumps({"result": "out.rknn"}))
                return mock.Mock(returncode=0, stdout="", stderr="")

            mock_run.side_effect = fake_run

            result = opt._convert_model_subprocess(
                "model.pt", "rknn", [640, 640], quantize=False
            )

            # The worker was launched with the converter env's Python
            call_args = mock_run.call_args
            cmd = call_args[0][0]
            self.assertEqual(os.fspath(cmd[0]), fake_converter_py.replace("/", os.sep))
            self.assertIn("iSpy.boot._convert_worker", cmd[1:])

            # PYTHONPATH was set so the worker finds the iSpy source
            env = call_args[1].get("env")
            self.assertIsNotNone(env)
            self.assertIn(str(_REPO_ROOT), env.get("PYTHONPATH", ""))

            self.assertEqual(result, Path("out.rknn"))

    def test_rknn_worker_env_does_not_inherit_pythonpath(self):
        """Runtime paths must not shadow the converter env's pinned packages."""
        from iSpy.boot import _rknn_converter_env
        from iSpy.vision import optimizer as opt

        with mock.patch.dict(
            os.environ, {"PYTHONPATH": os.path.join("main", "site-packages")}
        ), mock.patch.object(
            _rknn_converter_env,
            "converter_python",
            return_value=Path("/fake/converter/bin/python"),
        ), mock.patch.object(
            _rknn_converter_env, "ispy_source_root", return_value=_REPO_ROOT
        ), mock.patch("subprocess.run") as mock_run:

            def fake_run(cmd, **kwargs):
                args_path = Path(cmd[-1])
                result_path = Path(str(args_path) + ".result.json")
                result_path.write_text(json.dumps({"result": "out.rknn"}))
                return mock.Mock(returncode=0, stdout="", stderr="")

            mock_run.side_effect = fake_run

            opt._convert_model_subprocess(
                "model.pt", "rknn", [640, 640], quantize=False
            )

            env = mock_run.call_args[1].get("env")
            self.assertEqual(env.get("PYTHONPATH"), str(_REPO_ROOT))
            self.assertEqual(env.get("PYTHONNOUSERSITE"), "1")
            self.assertEqual(env.get("ISPY_CONVERTER_ENV"), "rknn")

    def test_protobuf_mismatch_rebuilds_env_and_retries_once(self):
        """A drifted env is rebuilt once, then the conversion is retried."""
        from iSpy.boot import _rknn_converter_env
        from iSpy.vision import optimizer as opt

        calls = {"n": 0}

        def fake_run(cmd, **kwargs):
            calls["n"] += 1
            args_path = Path(cmd[-1])
            result_path = Path(str(args_path) + ".result.json")
            if calls["n"] == 1:
                result_path.write_text(
                    json.dumps(
                        {
                            "error": (
                                "'google._upb._message.FieldDescriptor' object "
                                "has no attribute 'label'"
                            ),
                            "error_type": "AttributeError",
                            "traceback": "Traceback ...",
                        }
                    )
                )
                return mock.Mock(returncode=1, stdout="", stderr="")
            result_path.write_text(json.dumps({"result": "out.rknn"}))
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.object(
            _rknn_converter_env,
            "converter_python",
            return_value=Path("/fake/converter/bin/python"),
        ), mock.patch.object(
            _rknn_converter_env, "ispy_source_root", return_value=_REPO_ROOT
        ), mock.patch(
            "iSpy.boot.converter_envs.rebuild_env",
            return_value=Path("/fake/converter/bin/python"),
        ) as rebuild, mock.patch("subprocess.run", side_effect=fake_run):
            result, report = opt._convert_model_subprocess_report(
                "model.pt", "rknn", [640, 640], quantize=False, force=True
            )

        self.assertIsNone(report)
        self.assertEqual(result, Path("out.rknn"))
        rebuild.assert_called_once_with("rknn")
        self.assertEqual(calls["n"], 2)

    def test_non_rknn_uses_main_python(self):
        """Non-RKNN conversion continues using sys.executable."""
        from iSpy.vision import optimizer as opt

        with mock.patch("subprocess.run") as mock_run:
            def fake_run(cmd, **kwargs):
                args_path = Path(cmd[-1])
                result_path = Path(str(args_path) + ".result.json")
                result_path.write_text(json.dumps({"result": "out.onnx"}))
                return mock.Mock(returncode=0, stdout="", stderr="")

            mock_run.side_effect = fake_run

            result = opt._convert_model_subprocess(
                "model.pt", "onnx", [640, 640], quantize=False
            )

            call_args = mock_run.call_args
            cmd = call_args[0][0]
            self.assertEqual(cmd[0], sys.executable)
            self.assertEqual(result, Path("out.onnx"))


class ErrorHandlingTests(unittest.TestCase):
    """Error handling preserves real causes and doesn't mislabel failures."""

    def _mock_rknn_import_error(self, original_error):
        """Return a fake __import__ that raises *original_error* for rknn."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "rknn.api" or name.startswith("rknn"):
                raise original_error
            return real_import(name, *args, **kwargs)

        return fake_import

    def test_nested_import_error_preserved(self):
        """A nested numpy/scipy ImportError is not reported as missing toolkit."""
        from iSpy.vision import optimizer as opt

        original_error = ImportError(
            "cannot import name '_center' from 'numpy.core._multiarray_umath'"
        )

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "model.onnx"
            onnx_path.write_bytes(b"fake")

            with mock.patch(
                "iSpy.vision.optimizer._ensure_export_deps", return_value=True
            ), mock.patch(
                "iSpy.vision.optimizer._export_ultralytics", return_value=str(onnx_path)
            ), mock.patch(
                "iSpy.vision._safe_imports.repair_standard_log_levels"
            ), mock.patch(
                "builtins.__import__",
                side_effect=self._mock_rknn_import_error(original_error),
            ):
                with self.assertRaises(ImportError) as ctx:
                    opt._convert_rknn("model.pt", [640, 640])

                msg = str(ctx.exception)
                # The real cause is preserved
                self.assertIn("_center", msg)
                self.assertIn("numpy", msg.lower())
                # It is NOT misreported as a missing toolkit
                self.assertNotIn("RKNN Toolkit not found", msg)

    def test_pkg_resources_missing_preserved(self):
        """ModuleNotFoundError for pkg_resources is preserved, not hidden."""
        from iSpy.vision import optimizer as opt

        original_error = ModuleNotFoundError("No module named 'pkg_resources'")

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "model.onnx"
            onnx_path.write_bytes(b"fake")

            with mock.patch(
                "iSpy.vision.optimizer._ensure_export_deps", return_value=True
            ), mock.patch(
                "iSpy.vision.optimizer._export_ultralytics", return_value=str(onnx_path)
            ), mock.patch(
                "iSpy.vision._safe_imports.repair_standard_log_levels"
            ), mock.patch(
                "builtins.__import__",
                side_effect=self._mock_rknn_import_error(original_error),
            ):
                with self.assertRaises(ImportError) as ctx:
                    opt._convert_rknn("model.pt", [640, 640])

                msg = str(ctx.exception)
                self.assertIn("pkg_resources", msg)
                self.assertNotIn("RKNN Toolkit not found", msg)

    def _mock_converter_env(self):
        """Patch the converter-env functions used by _convert_model_subprocess."""
        from iSpy.boot import _rknn_converter_env

        return mock.patch.object(
            _rknn_converter_env, "converter_python", return_value=Path(sys.executable)
        ), mock.patch.object(
            _rknn_converter_env, "ispy_source_root", return_value=_REPO_ROOT
        )

    def test_python_exception_not_retried_as_oom(self):
        """A Python exception (exit code 1) is not retried as an OOM kill."""
        from iSpy.vision import optimizer as opt

        calls = []

        def fake_run(cmd, **kwargs):
            args_path = Path(cmd[-1])
            payload = json.loads(args_path.read_text())
            calls.append(payload["quantize"])
            return mock.Mock(returncode=1, stdout="", stderr="Traceback...")

        conv_py, src_root = self._mock_converter_env()
        with conv_py, src_root, mock.patch("subprocess.run", side_effect=fake_run):
            result = opt._convert_model_subprocess(
                "model.pt", "rknn", [640, 640], quantize=True
            )

        # Should NOT retry: Python exception is not an OOM kill
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0])
        # Falls back to .pt
        self.assertEqual(result, Path("model.pt"))

    def test_signal_kill_retries_unquantized(self):
        """A signal kill (negative return code) still retries unquantized."""
        from iSpy.vision import optimizer as opt

        calls = []

        def fake_run(cmd, **kwargs):
            args_path = Path(cmd[-1])
            payload = json.loads(args_path.read_text())
            calls.append(payload["quantize"])
            if payload["quantize"]:
                return mock.Mock(returncode=-9, stdout="", stderr="")  # SIGKILL
            result_path = Path(str(args_path) + ".result.json")
            result_path.write_text(json.dumps({"result": "out_unquant.rknn"}))
            return mock.Mock(returncode=0, stdout="", stderr="")

        conv_py, src_root = self._mock_converter_env()
        with conv_py, src_root, mock.patch("subprocess.run", side_effect=fake_run):
            result = opt._convert_model_subprocess(
                "model.pt", "rknn", [640, 640], quantize=True
            )

        # Should retry unquantized after signal kill
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0])
        self.assertFalse(calls[1])
        self.assertEqual(result, Path("out_unquant.rknn"))

    def test_structured_error_surfaced_by_parent(self):
        """The parent surfaces the worker's structured error, not a generic one."""
        from iSpy.vision import optimizer as opt

        conv_py, src_root = self._mock_converter_env()
        with conv_py, src_root, mock.patch("subprocess.run") as mock_run:
            def fake_run(cmd, **kwargs):
                args_path = Path(cmd[-1])
                result_path = Path(str(args_path) + ".result.json")
                result_path.write_text(json.dumps({
                    "error": "cannot import name '_center' from numpy",
                    "error_type": "ImportError",
                    "traceback": "...",
                }))
                return mock.Mock(returncode=1, stdout="", stderr="")

            mock_run.side_effect = fake_run

            with self.assertLogs("iSpy.vision.optimizer", level="ERROR") as logs:
                result = opt._convert_model_subprocess(
                    "model.pt", "rknn", [640, 640], quantize=False
                )

            self.assertEqual(result, Path("model.pt"))
            # The real error is in the log, not a generic exit-code message
            log_output = "\n".join(logs.output)
            self.assertIn("_center", log_output)


class BenchmarkAccuracyTests(unittest.TestCase):
    """Benchmarks cannot report successful RKNN when conversion failed."""

    def test_failed_conversion_returns_none_plus_report(self):
        """get_or_convert returns (None, report) when conversion falls back.""" 
        from iSpy.validations import benchmarking

        with mock.patch.object(
            benchmarking,
            "_convert_model_subprocess_report",
            return_value=(
                Path("model.pt"),
                {"status": "conversion", "error": "worker fell back to source"},
            ),
        ):
            model_path, report = benchmarking.get_or_convert(
                Path("model.pt"), "rknn"
            )

        self.assertIsNone(model_path)
        self.assertIsNotNone(report)
        self.assertEqual(report["status"], "conversion")

    def test_successful_conversion_returns_artifact(self):
        from iSpy.validations import benchmarking

        with tempfile.TemporaryDirectory() as tmp:
            rknn_path = Path(tmp) / "model.rknn"
            rknn_path.write_bytes(b"fake")

            with mock.patch.object(
                benchmarking,
                "_convert_model_subprocess_report",
                return_value=(rknn_path, None),
            ):
                model_path, report = benchmarking.get_or_convert(
                    Path("model.pt"), "rknn"
                )

            self.assertIsNotNone(model_path)
            self.assertTrue(str(model_path).endswith(".rknn"))
            self.assertIsNone(report)

    def test_failure_result_entry_is_explicitly_not_ok(self):
        """A failed conversion yields an ok=False entry, never a fake FPS."""
        from iSpy.validations import benchmarking

        entry = benchmarking._conversion_failure_result(
            "yolo", "rknn", 0, {"status": "terminated", "error": "killed", "exit_code": -9}
        )
        self.assertFalse(entry["ok"])
        self.assertIsNone(entry["fps"])
        self.assertIn("killed", entry["error"])
        self.assertEqual(entry["conversion_report"]["status"], "terminated")

    def test_non_rknn_backend_plan_unchanged(self):
        """Non-RKNN backend detection is not affected by converter env changes."""
        from iSpy.validations import benchmarking

        with mock.patch.object(benchmarking, "has_rockchip_npu", return_value=False), \
             mock.patch.object(benchmarking, "has_tpu", return_value=False), \
             mock.patch.object(benchmarking, "has_nvidia", return_value=False):
            plan = benchmarking.detect_test_plan()

        self.assertNotIn("rknn", plan)
        self.assertIn("pt_cpu", plan)
        self.assertIn("onnx_cpu", plan)


class InferenceIsolationTests(unittest.TestCase):
    """Normal .rknn inference does not provision the converter environment."""

    def test_lite_import_does_not_trigger_provisioning(self):
        """Importing rknnlite must not provision the converter env."""
        import types

        from iSpy.boot import _rknn_converter_env

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(_rknn_converter_env, "_cache_root", return_value=Path(tmp)):
                with mock.patch.object(_rknn_converter_env, "_provision") as mock_prov:
                    # Simulate a successful rknnlite import (as genericYolo does)
                    fake_lite = types.ModuleType("rknnlite")
                    fake_api = types.ModuleType("rknnlite.api")
                    fake_api.RKNNLite = object
                    fake_lite.api = fake_api
                    with mock.patch.dict(sys.modules, {"rknnlite": fake_lite, "rknnlite.api": fake_api}):
                        from rknnlite.api import RKNNLite  # noqa: F401
                    mock_prov.assert_not_called()

    def test_converter_env_status_is_json_serializable(self):
        from iSpy.boot import _rknn_converter_env

        status = _rknn_converter_env.converter_env_status()
        # Should not raise
        json.dumps(status)
        self.assertIn("env_dir", status)
        self.assertIn("identity", status)


class MainEnvSeparationTests(unittest.TestCase):
    """The main runtime env does not receive conversion-only packages."""

    def test_backend_deps_rknn_only_lite(self):
        """BACKEND_DEPENDENCIES['rknn'] contains only the lite wheel, not the full toolkit."""
        from iSpy.boot import backend_deps

        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / (
                "rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_"
                "aarch64.manylinux2014_aarch64.whl"
            )
            wheel.write_bytes(b"")
            platform = backend_deps.PlatformInfo(
                system="Linux", machine="aarch64", python_tag="cp312"
            )
            with mock.patch.object(
                backend_deps, "rknn_lite_wheel_dir", return_value=Path(tmp)
            ):
                deps = backend_deps.legacy_dependencies(platform).get("rknn", [])

        # Should have exactly one entry: the lite wheel
        self.assertEqual(len(deps), 1)
        mod, target = deps[0]
        self.assertEqual(mod, "rknnlite")
        # Should NOT contain the full toolkit URL
        self.assertNotIn("http", target)
        self.assertNotIn("rknn_toolkit2", target)

    def test_no_numpy_scipy_setuptools_in_main_rknn_deps(self):
        """The main env RKNN deps must not include numpy/scipy/setuptools."""
        from iSpy.vision import optimizer as opt

        deps = opt.BACKEND_DEPENDENCIES.get("rknn", [])
        targets = [t for _, t in deps]
        for target in targets:
            self.assertNotIn("numpy", target)
            self.assertNotIn("scipy", target)
            self.assertNotIn("setuptools", target)
            self.assertNotIn("ultralytics", target)
            self.assertNotIn("onnx", target)
            self.assertNotIn("protobuf", target)

    def test_install_special_deps_does_not_install_full_toolkit(self):
        """install_special_dependencies delegates to the central registry.

        The registry resolves RKNN only to the bundled lite wheel (see
        test_backend_deps_rknn_only_lite); this test pins the delegation so the
        optimizer cannot quietly grow its own installer again.
        """
        from iSpy.vision import optimizer as opt

        report = mock.Mock()
        report.lines.return_value = []
        report.results = []
        with mock.patch(
            "iSpy.boot.backend_deps.ensure_backends", return_value=report
        ) as ensure:
            result = opt.install_special_dependencies(
                auto_install=True, backend="rknn"
            )

        self.assertIs(result, report)
        ensure.assert_called_once_with(["rknn"], auto_install=True)

    def test_ensure_export_deps_noop_in_converter_env(self):
        """_ensure_export_deps is a no-op when deps are already present."""
        from iSpy.vision import optimizer as opt

        with mock.patch.object(opt, "_is_installed", return_value=True):
            result = opt._ensure_export_deps()
        self.assertTrue(result)

    def test_converter_env_never_installs_unpinned_export_deps(self):
        """A missing dep must rebuild the env, not pip-mutate its pins."""
        from iSpy.vision import optimizer as opt

        with mock.patch.object(opt, "_in_converter_env", return_value=True), \
             mock.patch.object(opt, "_is_installed", return_value=False), \
             mock.patch.object(opt, "_pip_install") as pip_install:
            result = opt._ensure_export_deps()

        self.assertFalse(result)
        pip_install.assert_not_called()

    def test_rknn_wheel_targets_never_offers_full_toolkit(self):
        """The normal-env wheel helper returns lite only, never the full toolkit."""
        from iSpy.vision import optimizer as opt

        with mock.patch.object(opt, "_IS_AARCH64", True), \
             mock.patch.object(opt, "_PY_TAG", "cp312"), \
             mock.patch.object(
                 opt, "_RKNN_LITE_DIR", _REPO_ROOT / "iSpy" / "rknn_wheels"
             ):
            targets = opt._rknn_wheel_targets()

        self.assertEqual(len(targets), 1)
        mod, target = targets[0]
        self.assertEqual(mod, "rknnlite")
        self.assertIn("rknn_toolkit_lite2", target)
        self.assertNotIn("rknn_toolkit2-", target)
        self.assertNotIn("http", target)


class ConverterEnvFullToolkitOnlyTests(unittest.TestCase):
    """The converter env gets the full toolkit2 and never the lite wheel."""

    def test_provision_installs_full_toolkit_not_lite(self):
        import types

        from iSpy.boot import _rknn_converter_env as env

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            env_dir = cache / "rknn-converter" / "abc"
            py = env._is_py(env_dir)
            full_url = (
                "https://example.invalid/rknn_toolkit2-2.3.2-cp312-cp312-"
                "manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
            )
            free = types.SimpleNamespace(free=10**12, used=0, total=0)

            def fake_run(cmd, **kwargs):
                env_dir.mkdir(parents=True, exist_ok=True)
                py.parent.mkdir(parents=True, exist_ok=True)
                py.write_text("#!/bin/sh\n")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.object(env, "_cache_root", return_value=cache), \
                 mock.patch.object(env.shutil, "disk_usage", return_value=free), \
                 mock.patch.object(env, "_toolkit_wheel_url", return_value=full_url), \
                 mock.patch.object(env.subprocess, "run", side_effect=fake_run), \
                 mock.patch.object(env, "_pip_install") as mock_pip, \
                 mock.patch.object(env, "_verify"):
                env._provision(env_dir)

            installed = [
                arg
                for call in mock_pip.call_args_list
                for arg in call.args[1:]
                if isinstance(arg, str)
            ]
            self.assertIn(full_url, installed)
            for target in installed:
                self.assertNotIn("rknn_toolkit_lite2", target)
                self.assertNotIn("rknnlite", target)


class CachedArtifactTests(unittest.TestCase):
    """A cached .rknn artifact is reused without provisioning anything."""

    def _write_sidecar(self, rknn: Path, quantize: bool) -> None:
        meta = rknn.parent / f"{rknn.stem}_metadata.yaml"
        meta.write_text(f"quantize: {'true' if quantize else 'false'}\n")

    def test_matching_cached_artifact_shortcircuits_provisioning(self):
        from iSpy.vision import optimizer as opt

        with tempfile.TemporaryDirectory() as tmp:
            rknn = Path(tmp) / "model.rknn"
            rknn.write_bytes(b"rknn")
            self._write_sidecar(rknn, True)

            with mock.patch.object(opt, "_desired_output_path", return_value=rknn), \
                 mock.patch.object(opt, "_rknn_converter_env") as mock_env, \
                 mock.patch("subprocess.run") as mock_run:
                result, report = opt._convert_model_subprocess_report(
                    "model.pt", "rknn", [640, 640], quantize=True
                )

            self.assertEqual(result, rknn)
            self.assertIsNone(report)
            # No worker spawn, no converter environment build
            mock_env.converter_python.assert_not_called()
            mock_run.assert_not_called()

    def test_cached_artifact_wrong_quantize_reconverts(self):
        from iSpy.vision import optimizer as opt

        with tempfile.TemporaryDirectory() as tmp:
            rknn = Path(tmp) / "model.rknn"
            rknn.write_bytes(b"rknn")
            self._write_sidecar(rknn, False)

            with mock.patch.object(opt, "_desired_output_path", return_value=rknn):
                cached = opt._cached_rknn_artifact(Path("model.pt"), quantize=True)

            self.assertIsNone(cached)

    def test_missing_sidecar_does_not_block_reuse(self):
        """No sidecar = treat the existing artifact as valid (like convert_model)."""
        from iSpy.vision import optimizer as opt

        with tempfile.TemporaryDirectory() as tmp:
            rknn = Path(tmp) / "model.rknn"
            rknn.write_bytes(b"rknn")

            with mock.patch.object(opt, "_desired_output_path", return_value=rknn), \
                 mock.patch.object(opt, "_rknn_converter_env") as mock_env, \
                 mock.patch("subprocess.run") as mock_run:
                result, report = opt._convert_model_subprocess_report(
                    "model.pt", "rknn", [640, 640], quantize=True
                )

            self.assertEqual(result, rknn)
            self.assertIsNone(report)
            mock_env.converter_python.assert_not_called()
            mock_run.assert_not_called()

    def test_force_ignores_cache(self):
        from iSpy.vision import optimizer as opt

        with tempfile.TemporaryDirectory() as tmp:
            rknn = Path(tmp) / "model.rknn"
            rknn.write_bytes(b"rknn")
            self._write_sidecar(rknn, True)

            with mock.patch.object(opt, "_desired_output_path", return_value=rknn):
                cached = opt._cached_rknn_artifact(
                    Path("model.pt"), quantize=True, force=True
                )

            self.assertIsNone(cached)


class ProvisioningFailureTests(unittest.TestCase):
    """A converter-environment provision failure is a classified dependency
    error that is never retried and never spawns a worker."""

    def test_provision_error_is_dependency_and_no_worker(self):
        from iSpy.boot._rknn_converter_env import ConverterEnvError
        from iSpy.vision import optimizer as opt

        def fake_run(cmd, **kwargs):
            raise AssertionError("worker must not be spawned")

        with mock.patch.object(opt, "_rknn_converter_env") as mock_env, \
             mock.patch("subprocess.run", side_effect=fake_run):
            mock_env.converter_python.side_effect = ConverterEnvError(
                "No rknn-toolkit2 2.3.2 wheel for x86_64 (Python cp314)"
            )
            result, report = opt._convert_model_subprocess_report(
                "model.pt", "rknn", [640, 640], quantize=True
            )

        self.assertEqual(result, Path("model.pt"))
        self.assertEqual(report["status"], "dependency")
        self.assertIn("converter environment unavailable", report["error"])
        self.assertIn("cp314", report["error"])


class WorkerFailureClassificationTests(unittest.TestCase):
    """Failure classification is accurate and drives the retry decision."""

    def _classify(self, *args):
        from iSpy.vision.optimizer import _classify_worker_failure

        return _classify_worker_failure(*args)

    def test_signal_kill_is_terminated(self):
        self.assertEqual(self._classify(-9, "", None), "terminated")

    def test_import_error_is_dependency(self):
        self.assertEqual(
            self._classify(
                1,
                "cannot import name '_center' from 'numpy.core._multiarray_umath'",
                "ImportError",
            ),
            "dependency",
        )

    def test_missing_module_is_dependency(self):
        self.assertEqual(
            self._classify(1, "No module named 'pkg_resources'", "ModuleNotFoundError"),
            "dependency",
        )

    def test_protobuf_onnx_label_mismatch_is_dependency(self):
        """A drifted protobuf must be reported as a broken converter env."""
        self.assertEqual(
            self._classify(
                1,
                "'google._upb._message.FieldDescriptor' object has no attribute 'label'",
                "AttributeError",
            ),
            "dependency",
        )

    def test_protobuf_onnx_mismatch_helper(self):
        """Only the onnx/protobuf crash triggers the env self-heal retry."""
        from iSpy.vision.optimizer import _is_protobuf_onnx_mismatch

        self.assertTrue(
            _is_protobuf_onnx_mismatch(
                "'google._upb._message.FieldDescriptor' object has no attribute 'label'"
            )
        )
        self.assertFalse(_is_protobuf_onnx_mismatch("rknn build failed with code 8"))
        self.assertFalse(_is_protobuf_onnx_mismatch("No module named 'onnx'"))

    def test_memory_error_is_resource(self):
        self.assertEqual(
            self._classify(1, "cannot allocate memory", "MemoryError"), "resource"
        )

    def test_plain_build_error_is_conversion(self):
        self.assertEqual(self._classify(1, "rknn build failed with code 8"), "conversion")


class DepthWorkerRoutingTests(unittest.TestCase):
    """Depth Anything RKNN builds route through the converter env's worker."""

    def test_depth_rknn_build_spawns_converter_worker(self):
        import logging

        from iSpy.boot import _rknn_converter_env
        from iSpy.vision.pipelines.depth_anything import DepthAnythingPipeline

        with tempfile.TemporaryDirectory() as tmp:
            onnx = Path(tmp) / "model.onnx"
            onnx.write_bytes(b"onnx")
            produced = Path(tmp) / "model.rknn"
            produced.write_bytes(b"rknn")

            da = DepthAnythingPipeline.__new__(DepthAnythingPipeline)
            da.logger = logging.getLogger("test.da")
            da.quantize = False
            da._input_size = 518

            with mock.patch.object(
                _rknn_converter_env,
                "converter_python",
                return_value=Path("/conv/bin/python"),
            ), mock.patch.object(
                _rknn_converter_env, "ispy_source_root", return_value=_REPO_ROOT
            ), mock.patch.object(da, "_rknn_calibration_txt", return_value=None), mock.patch(
                "subprocess.run"
            ) as mock_run:

                def fake_run(cmd, **kwargs):
                    args_path = Path(cmd[-1])
                    payload = json.loads(args_path.read_text())
                    self.assertEqual(payload["onnx"], str(onnx))
                    self.assertFalse(payload["quantize"])
                    result_path = Path(str(args_path) + ".result.json")
                    result_path.write_text(
                        json.dumps({"result": str(produced)})
                    )
                    return mock.Mock(returncode=0, stdout="", stderr="")

                mock_run.side_effect = fake_run
                da._build_rknn_subprocess(onnx, produced)

            call = mock_run.call_args
            cmd = call[0][0]
            expected = str(Path("/conv/bin/python")).replace("/", os.sep)
            self.assertEqual(os.fspath(cmd[0]), expected)
            self.assertIn("iSpy.boot._rknn_onnx_worker", cmd[1:])
            env = call[1].get("env", {})
            self.assertIn(str(_REPO_ROOT), env.get("PYTHONPATH", ""))
            self.assertEqual(env.get("PYTHONNOUSERSITE"), "1")

    def test_depth_rknn_worker_failure_raises_with_real_error(self):
        from iSpy.boot import _rknn_converter_env
        from iSpy.vision.pipelines.depth_anything import DepthAnythingPipeline

        da = DepthAnythingPipeline.__new__(DepthAnythingPipeline)
        import logging

        da.logger = logging.getLogger("test.da")
        da.quantize = False
        da._input_size = 518

        with mock.patch.object(
            _rknn_converter_env,
            "converter_python",
            return_value=Path("/conv/bin/python"),
        ), mock.patch.object(
            _rknn_converter_env, "ispy_source_root", return_value=_REPO_ROOT
        ), mock.patch.object(da, "_rknn_calibration_txt", return_value=None), mock.patch(
            "subprocess.run"
        ) as mock_run:

            def fake_run(cmd, **kwargs):
                args_path = Path(cmd[-1])
                result_path = Path(str(args_path) + ".result.json")
                result_path.write_text(
                    json.dumps(
                        {
                            "error": "rknn build failed with code -1",
                            "error_type": "RuntimeError",
                            "traceback": "...",
                        }
                    )
                )
                return mock.Mock(returncode=1, stdout="", stderr="")

            mock_run.side_effect = fake_run
            with self.assertRaises(RuntimeError) as ctx:
                da._build_rknn_subprocess(Path("model.onnx"), Path("model.rknn"))

            self.assertIn("rknn build failed with code -1", str(ctx.exception))


class ProvisionFullFlowTests(unittest.TestCase):
    """The full provisioning flow stamps verified and records a manifest."""

    def test_provision_success_stamps_verified_and_manifest(self):
        import types

        from iSpy.boot import _rknn_converter_env as env

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            env_dir = cache / "rknn-converter" / "abc"
            py = env._is_py(env_dir)

            free = types.SimpleNamespace(free=10**12, used=0, total=0)

            def fake_run(cmd, **kwargs):
                # Simulate `python -m venv <env_dir>` creating the tree.
                env_dir.mkdir(parents=True, exist_ok=True)
                py = env._is_py(env_dir)
                py.parent.mkdir(parents=True, exist_ok=True)
                py.write_text("#!/bin/sh\n")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.object(env, "_cache_root", return_value=cache), \
                 mock.patch.object(env.shutil, "disk_usage", return_value=free), \
                 mock.patch.object(env, "_toolkit_wheel_url", return_value=(
                     "https://example.invalid/rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
                 )), \
                 mock.patch.object(env.subprocess, "run", side_effect=fake_run), \
                 mock.patch.object(env, "_pip_install"), \
                 mock.patch.object(env, "_verify"):
                result = env._provision(env_dir)

            self.assertEqual(result, py)
            self.assertTrue((env_dir / "env-manifest.json").exists())
            self.assertTrue((env_dir / ".verified").exists())

    def test_provision_refuses_on_low_disk(self):
        """A full disk still refuses the build - but only after a cache purge."""
        import types

        from iSpy.boot import _rknn_converter_env as env

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            low = types.SimpleNamespace(free=10, used=0, total=0)

            with mock.patch.object(env, "_cache_root", return_value=cache), \
                 mock.patch.object(env.shutil, "disk_usage", return_value=low), \
                 mock.patch.object(env, "_purge_pip_cache", return_value=False) as mock_purge, \
                 mock.patch.object(env, "_pip_install") as mock_pip:
                with self.assertRaises(Exception) as ctx:
                    env._provision(cache / "rknn-converter" / "abc")

            self.assertIn("disk space", str(ctx.exception).lower())
            # the pip cache is the first thing we try to reclaim, the env is not
            mock_purge.assert_called_once()
            mock_pip.assert_not_called()

    def test_provision_low_disk_recovers_after_cache_purge(self):
        """If purging the pip cache frees enough room, the build just continues."""
        import types

        from iSpy.boot import _rknn_converter_env as env

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            env_dir = cache / "rknn-converter" / "abc"
            py = env._is_py(env_dir)

            low = types.SimpleNamespace(free=10, used=0, total=0)
            roomy = types.SimpleNamespace(free=10**12, used=0, total=0)

            def fake_run(cmd, **kwargs):
                env_dir.mkdir(parents=True, exist_ok=True)
                py.parent.mkdir(parents=True, exist_ok=True)
                py.write_text("#!/bin/sh\n")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.object(env, "_cache_root", return_value=cache), \
                 mock.patch.object(env.shutil, "disk_usage", side_effect=[low, roomy]), \
                 mock.patch.object(env, "_purge_pip_cache", return_value=True) as mock_purge, \
                 mock.patch.object(env, "_toolkit_wheel_url", return_value=(
                     "https://example.invalid/rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
                 )), \
                 mock.patch.object(env.subprocess, "run", side_effect=fake_run), \
                 mock.patch.object(env, "_pip_install"), \
                 mock.patch.object(env, "_verify"):
                result = env._provision(env_dir)

            self.assertEqual(result, py)
            mock_purge.assert_called_once()

    def test_pip_install_retries_once_after_cache_purge_on_space_error(self):
        """An out-of-space pip failure purges the cache and retries exactly once."""
        from iSpy.boot import _rknn_converter_env as env

        py = Path(sys.executable)
        full = mock.Mock(returncode=1, stdout="ERROR: No space left on device", stderr="")
        ok = mock.Mock(returncode=0, stdout="Successfully installed", stderr="")

        with mock.patch.object(env, "_purge_pip_cache", return_value=True) as mock_purge, \
             mock.patch.object(env.subprocess, "run", side_effect=[full, ok]) as mock_run:
            env._pip_install(py, "some-package")  # must not raise

        self.assertEqual(mock_run.call_count, 2)
        mock_purge.assert_called_once()

    def test_pip_install_does_not_purge_on_unrelated_failure(self):
        """A normal pip error is not mistaken for a full disk."""
        from iSpy.boot import _rknn_converter_env as env

        py = Path(sys.executable)
        bad = mock.Mock(returncode=1, stdout="ERROR: No matching distribution found", stderr="")

        with mock.patch.object(env, "_purge_pip_cache") as mock_purge, \
             mock.patch.object(env.subprocess, "run", side_effect=[bad]) as mock_run:
            with self.assertRaises(env.ConverterEnvError):
                env._pip_install(py, "some-package")

        self.assertEqual(mock_run.call_count, 1)
        mock_purge.assert_not_called()


class LightVerifyTests(unittest.TestCase):
    """The daily light-verify gate on env reuse."""

    def test_light_verify_rebuilds_broken_env(self):
        """A cached env that fails light verification is rebuilt, not reused."""
        from iSpy.boot import _rknn_converter_env as env

        identity = env._identity()

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            with mock.patch.object(env, "_cache_root", return_value=cache):
                # Converter envs are now grouped as <format>_env/<identity>.
                env_dir = env._env_dir(identity)
                py = env._is_py(env_dir)

                with mock.patch.object(env, "_env_is_current", return_value=True), \
                     mock.patch.object(env, "_light_verify", return_value=False), \
                     mock.patch.object(env, "_provision", return_value=py) as mock_provision:
                    result = env.converter_python()

            self.assertEqual(result, py)
            # The broken env was not trusted: provisioning was triggered.
            mock_provision.assert_called_once()

    def test_light_verify_stamps_ok_env(self):
        """A fresh .verified stamp satisfies verification without a subprocess."""
        from iSpy.boot import _rknn_converter_env as env

        with tempfile.TemporaryDirectory() as tmp:
            env_dir = Path(tmp)
            py = env._is_py(env_dir)
            stamp = env_dir / ".verified"
            stamp.write_text("stamp")  # just written -> fresh mtime

            with mock.patch.object(env, "subprocess") as mock_sub:
                result = env._light_verify(py, env_dir)

            self.assertTrue(result)
            mock_sub.run.assert_not_called()

    def test_verify_scripts_probe_protobuf_onnx_compat(self):
        """Both checks exercise the onnx API rknn.build() depends on."""
        from iSpy.boot import _rknn_converter_env as env

        for script in (env._VERIFY_SCRIPT, env._LIGHT_VERIFY_SCRIPT):
            self.assertIn("onnx.helper.strip_doc_string", script)


if __name__ == "__main__":
    unittest.main()
