"""Focused tests for the centralized runtime-backend dependency registry."""

from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from iSpy.boot import backend_deps as bd


LINUX_ARM = bd.PlatformInfo(
    system="Linux", machine="aarch64", python_tag="cp312"
)
LINUX_X86 = bd.PlatformInfo(
    system="Linux", machine="x86_64", python_tag="cp312"
)
WINDOWS_X86 = bd.PlatformInfo(
    system="Windows", machine="x86_64", python_tag="cp312"
)
JETSON = bd.PlatformInfo(
    system="Linux",
    machine="aarch64",
    python_tag="cp310",
    is_jetson=True,
    has_nvidia=True,
    cuda_major=12,
)
NVIDIA_X86 = bd.PlatformInfo(
    system="Linux",
    machine="x86_64",
    python_tag="cp312",
    has_nvidia=True,
    cuda_major=12,
)


class TorchSelectionTests(unittest.TestCase):
    def test_linux_aarch64_uses_cpu_index(self):
        resolution = bd.resolve_backend("pytorch", LINUX_ARM)

        self.assertEqual(resolution.mode, "install")
        pkg = resolution.packages[0]
        self.assertEqual(pkg.dist, "torch")
        self.assertEqual(pkg.index_url, bd.TORCH_CPU_INDEX)
        self.assertIn("torch.version.cuda is None", pkg.verify_script)

    def test_torch_verify_exercises_numpy_bridge(self):
        """A torch built for another NumPy ABI must fail verification."""
        pkg = bd.resolve_backend("pytorch", LINUX_ARM).packages[0]
        self.assertIn("from_numpy", pkg.verify_script)
        self.assertIn("NumPy", pkg.verify_script)

    def test_linux_cpu_x86_never_uses_generic_index(self):
        resolution = bd.resolve_backend("pytorch", LINUX_X86)

        self.assertEqual(resolution.packages[0].index_url, bd.TORCH_CPU_INDEX)

    def test_nvidia_linux_tries_cuda_then_cpu_fallback(self):
        resolution = bd.resolve_backend("pytorch", NVIDIA_X86)

        self.assertEqual(resolution.mode, "install")
        first, fallback = resolution.packages
        self.assertEqual(first.index_url, "https://download.pytorch.org/whl/cu126")
        self.assertIn("torch.cuda.is_available", first.verify_script)
        self.assertEqual(fallback.index_url, bd.TORCH_CPU_INDEX)

    def test_nvidia_without_driver_stays_cpu(self):
        platform = bd.PlatformInfo(
            system="Linux",
            machine="x86_64",
            python_tag="cp312",
            has_nvidia=True,
            cuda_major=None,
        )

        resolution = bd.resolve_backend("pytorch", platform)

        self.assertEqual(len(resolution.packages), 1)
        self.assertEqual(resolution.packages[0].index_url, bd.TORCH_CPU_INDEX)

    def test_jetson_is_system_managed(self):
        resolution = bd.resolve_backend("pytorch", JETSON)

        self.assertEqual(resolution.mode, "system")
        self.assertEqual(resolution.packages, ())
        self.assertIn("NVIDIA", resolution.reason)

    def test_armv7_is_explicitly_unsupported(self):
        platform = bd.PlatformInfo(
            system="Linux", machine="armv7l", python_tag="cp311"
        )

        resolution = bd.resolve_backend("pytorch", platform)

        self.assertEqual(resolution.mode, "unsupported")
        self.assertIn("armv7l", resolution.reason)

    def test_windows_arm64_is_unsupported(self):
        platform = bd.PlatformInfo(
            system="Windows", machine="aarch64", python_tag="cp312"
        )
        self.assertEqual(
            bd.resolve_backend("pytorch", platform).mode, "unsupported"
        )

    def test_macos_x86_is_unsupported(self):
        platform = bd.PlatformInfo(
            system="Darwin", machine="x86_64", python_tag="cp312"
        )
        self.assertEqual(
            bd.resolve_backend("pytorch", platform).mode, "unsupported"
        )


class OnnxRuntimeSelectionTests(unittest.TestCase):
    def test_linux_arm_cpu(self):
        pkg = bd.resolve_backend("onnx", LINUX_ARM).packages[0]
        self.assertEqual(pkg.dist, "onnxruntime")
        self.assertIn("CPUExecutionProvider", pkg.verify_script)

    def test_nvidia_linux_gpu(self):
        pkg = bd.resolve_backend("onnx", NVIDIA_X86).packages[0]
        self.assertEqual(pkg.dist, "onnxruntime-gpu")
        self.assertIn("CUDAExecutionProvider", pkg.verify_script)
        self.assertIn("onnxruntime", pkg.uninstall_first)

    def test_windows_directml_without_nvidia(self):
        pkg = bd.resolve_backend("onnx", WINDOWS_X86).packages[0]
        self.assertEqual(pkg.dist, "onnxruntime-directml")
        self.assertIn("DmlExecutionProvider", pkg.verify_script)

    def test_windows_nvidia_prefers_gpu(self):
        platform = bd.PlatformInfo(
            system="Windows",
            machine="x86_64",
            python_tag="cp312",
            has_nvidia=True,
            cuda_major=12,
        )
        pkg = bd.resolve_backend("onnx", platform).packages[0]
        self.assertEqual(pkg.dist, "onnxruntime-gpu")

    def test_jetson_is_system_managed(self):
        self.assertEqual(bd.resolve_backend("onnx", JETSON).mode, "system")


class TfliteSelectionTests(unittest.TestCase):
    def test_old_python_uses_tflite_runtime(self):
        platform = bd.PlatformInfo(
            system="Linux", machine="aarch64", python_tag="cp311"
        )
        pkg = bd.resolve_backend("tflite", platform).packages[0]
        self.assertEqual(pkg.dist, "tflite-runtime")

    def test_new_python_uses_litert(self):
        pkg = bd.resolve_backend("tflite", LINUX_ARM).packages[0]
        self.assertEqual(pkg.dist, "ai-edge-litert")
        self.assertIn("ai_edge_litert.interpreter", pkg.verify_script)

    def test_windows_new_python_uses_litert(self):
        pkg = bd.resolve_backend("tflite", WINDOWS_X86).packages[0]
        self.assertEqual(pkg.dist, "ai-edge-litert")

    def test_windows_arm64_is_unsupported(self):
        platform = bd.PlatformInfo(
            system="Windows", machine="aarch64", python_tag="cp312"
        )
        self.assertEqual(
            bd.resolve_backend("tflite", platform).mode, "unsupported"
        )

    def test_macos_x86_unsupported(self):
        platform = bd.PlatformInfo(
            system="Darwin", machine="x86_64", python_tag="cp312"
        )
        self.assertEqual(
            bd.resolve_backend("tflite", platform).mode, "unsupported"
        )


class RknnSelectionTests(unittest.TestCase):
    def test_only_bundled_lite_wheel_is_offered(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / (
                "rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_"
                "aarch64.manylinux2014_aarch64.whl"
            )
            wheel.write_bytes(b"")
            with mock.patch.object(
                bd, "rknn_lite_wheel_dir", return_value=Path(tmp)
            ):
                resolution = bd.resolve_backend("rknn", LINUX_ARM)

        self.assertEqual(resolution.mode, "install")
        pkg = resolution.packages[0]
        self.assertEqual(pkg.dist, "rknn-toolkit-lite2")
        self.assertEqual(pkg.requirement, str(wheel))
        self.assertTrue(pkg.no_deps)
        self.assertNotIn("toolkit2-", pkg.requirement.replace("lite2", ""))

    def test_x86_rknn_runtime_is_unsupported(self):
        self.assertEqual(
            bd.resolve_backend("rknn", LINUX_X86).mode, "unsupported"
        )


class EngineAndVendorSelectionTests(unittest.TestCase):
    def test_tensorrt_uses_cuda_specific_index_and_generic_fallback(self):
        resolution = bd.resolve_backend("engine", NVIDIA_X86)

        self.assertEqual(resolution.mode, "install")
        self.assertGreaterEqual(len(resolution.packages), 2)
        first = resolution.packages[0]
        self.assertEqual(first.dist, "tensorrt-cu12")
        self.assertIn(bd.NVIDIA_INDEX, first.extra_index_urls)
        self.assertEqual(resolution.packages[-1].dist, "tensorrt")

    def test_tensorrt_without_nvidia_is_unsupported(self):
        self.assertEqual(
            bd.resolve_backend("engine", LINUX_X86).mode, "unsupported"
        )

    def test_tensorrt_on_jetson_is_system_managed(self):
        self.assertEqual(bd.resolve_backend("engine", JETSON).mode, "system")

    def test_hailo_is_manual_only(self):
        self.assertEqual(bd.resolve_backend("hailo", LINUX_ARM).mode, "manual")

    def test_coreml_only_on_macos(self):
        self.assertEqual(
            bd.resolve_backend("coreml", LINUX_X86).mode, "unsupported"
        )
        mac = bd.PlatformInfo(
            system="Darwin", machine="aarch64", python_tag="cp312"
        )
        self.assertEqual(bd.resolve_backend("coreml", mac).mode, "install")

    def test_qnn_requires_qualcomm_hardware(self):
        self.assertEqual(bd.resolve_backend("qnn", LINUX_ARM).mode, "unsupported")
        qualcomm = bd.PlatformInfo(
            system="Linux",
            machine="aarch64",
            python_tag="cp312",
            has_qualcomm=True,
        )
        pkg = bd.resolve_backend("qnn", qualcomm).packages[0]
        self.assertEqual(pkg.dist, "onnxruntime-qnn")

    def test_qnn_requires_python_311_or_newer(self):
        qualcomm = bd.PlatformInfo(
            system="Linux",
            machine="aarch64",
            python_tag="cp310",
            has_qualcomm=True,
        )
        self.assertEqual(bd.resolve_backend("qnn", qualcomm).mode, "unsupported")

    def test_openvino_uses_only_published_wheel_platforms(self):
        windows_arm = bd.PlatformInfo(
            system="Windows", machine="aarch64", python_tag="cp312"
        )
        mac_x86 = bd.PlatformInfo(
            system="Darwin", machine="x86_64", python_tag="cp312"
        )
        self.assertEqual(
            bd.resolve_backend("openvino", windows_arm).mode, "unsupported"
        )
        self.assertEqual(
            bd.resolve_backend("openvino", mac_x86).mode, "unsupported"
        )

    def test_tpu_installs_cpu_torch_first(self):
        platform = bd.PlatformInfo(
            system="Linux", machine="x86_64", python_tag="cp312"
        )
        resolution = bd.resolve_backend("tpu", platform)

        self.assertEqual(resolution.mode, "install")
        self.assertEqual(resolution.packages[0].dist, "torch")
        self.assertEqual(resolution.packages[0].index_url, bd.TORCH_CPU_INDEX)
        self.assertEqual(resolution.packages[1].dist, "torch-xla")


class ConfigSelectionTests(unittest.TestCase):
    def test_pt_file_selects_pytorch_and_ignores_source_pt_for_converted(self):
        config = {
            "camera_configs": {
                "cam": {
                    "pipeline": {
                        "settings": {
                            "vision_model": {
                                "file_path": "YoloModels/rknn/model.rknn",
                                "source_pt": "YoloModels/pytorch/model.pt",
                            }
                        }
                    }
                }
            }
        }

        self.assertEqual(bd.backends_for_config(config, LINUX_ARM), ["rknn"])

    def test_onnx_plus_qualcomm_adds_qnn_plugin(self):
        config = {"file_path": "YoloModels/onnx/model.onnx"}
        qualcomm = bd.PlatformInfo(
            system="Linux",
            machine="aarch64",
            python_tag="cp312",
            has_qualcomm=True,
        )

        self.assertEqual(
            bd.backends_for_config(config, qualcomm), ["onnx", "qnn"]
        )

    def test_openvino_directory_is_detected(self):
        config = {"file_path": "YoloModels/openvino/model_openvino_model/model.xml"}
        self.assertEqual(
            bd.backends_for_config(config, LINUX_X86), ["openvino"]
        )


class ProvisioningTests(unittest.TestCase):
    def setUp(self):
        self.installer = mock.Mock(return_value=True)
        self.verifier = mock.Mock(return_value=(True, "ok"))

    def test_valid_install_is_reused_without_pip(self):
        with mock.patch.object(bd, "installed_version", return_value="1.20.0"):
            result = bd.ensure_backend(
                "onnx",
                platform_info=LINUX_ARM,
                auto_install=True,
                installer=self.installer,
                verifier=self.verifier,
            )

        self.assertEqual(result.status, "reused")
        self.installer.assert_not_called()
        self.verifier.assert_called_once()

    def test_missing_package_is_installed_and_verified(self):
        with mock.patch.object(bd, "installed_version", return_value=None):
            result = bd.ensure_backend(
                "onnx",
                platform_info=LINUX_ARM,
                auto_install=True,
                installer=self.installer,
                verifier=self.verifier,
            )

        self.assertEqual(result.status, "installed")
        self.installer.assert_called_once()
        self.assertEqual(result.packages[0].status, "installed")

    def test_version_mismatch_installs_upgrade(self):
        with mock.patch.object(bd, "installed_version", return_value="1.0.0"):
            result = bd.ensure_backend(
                "onnx",
                platform_info=LINUX_ARM,
                auto_install=True,
                installer=self.installer,
                verifier=self.verifier,
            )

        self.assertEqual(result.status, "installed")
        pkg, force = self.installer.call_args.args
        self.assertEqual(pkg.dist, "onnxruntime")
        self.assertFalse(force)

    def test_broken_import_forces_reinstall(self):
        self.verifier.return_value = (False, "broken import")

        def _install(pkg, force):
            self.assertTrue(force)
            self.verifier.return_value = (True, "ok")
            return True

        with mock.patch.object(bd, "installed_version", return_value="1.20.0"):
            result = bd.ensure_backend(
                "onnx",
                platform_info=LINUX_ARM,
                auto_install=True,
                installer=_install,
                verifier=self.verifier,
            )

        self.assertEqual(result.status, "installed")

    def test_verify_only_reuses_cpu_torch_when_cuda_build_is_missing(self):
        def verifier(pkg):
            if "torch.cuda.is_available" in (pkg.verify_script or ""):
                return False, "torch cannot access the NVIDIA GPU"
            return True, "torch cpu ok"

        with mock.patch.object(bd, "installed_version", return_value="2.0.0+cpu"):
            result = bd.ensure_backend(
                "pytorch",
                platform_info=NVIDIA_X86,
                auto_install=False,
                verifier=verifier,
            )

        self.assertEqual(result.status, "reused")
        self.assertEqual(
            [pkg.status for pkg in result.packages], ["missing", "reused"]
        )
        self.assertTrue(result.packages[1].fallback)

    def test_manual_backend_is_verified_when_present(self):
        with mock.patch.object(bd, "_module_present", return_value=True):
            result = bd.ensure_backend(
                "hailo",
                platform_info=LINUX_ARM,
                auto_install=False,
                verifier=lambda pkg: (True, "vendor runtime ok"),
            )

        self.assertEqual(result.status, "system")

    def test_verify_only_does_not_uninstall_anything(self):
        with mock.patch.object(bd, "installed_version", return_value="2.0.0"), \
             mock.patch.object(
                 bd, "remove_forbidden_cuda_packages"
             ) as remove:
            result = bd.ensure_backend(
                "pytorch",
                platform_info=LINUX_ARM,
                auto_install=False,
                verifier=self.verifier,
            )

        self.assertTrue(result.ok)
        remove.assert_not_called()

    def test_missing_without_auto_install_is_reported_not_installed(self):
        with mock.patch.object(bd, "installed_version", return_value=None):
            result = bd.ensure_backend(
                "onnx",
                platform_info=LINUX_ARM,
                auto_install=False,
                installer=self.installer,
                verifier=self.verifier,
            )

        self.assertEqual(result.status, "missing")
        self.installer.assert_not_called()

    def test_install_failure_is_not_success(self):
        self.installer.return_value = False
        with mock.patch.object(bd, "installed_version", return_value=None):
            result = bd.ensure_backend(
                "onnx",
                platform_info=LINUX_ARM,
                auto_install=True,
                installer=self.installer,
                verifier=self.verifier,
            )

        self.assertEqual(result.status, "failed")
        self.assertFalse(result.ok)
        report = bd.ProvisionReport(platform=LINUX_ARM, results=[result])
        with self.assertRaises(bd.DependencyProvisionError):
            bd.raise_for_actionable_failures(report)

    def test_low_disk_blocks_install(self):
        with mock.patch.object(bd, "installed_version", return_value=None), \
             mock.patch.object(
                 bd, "check_disk_space", return_value=(False, "only 100 MiB free")
             ):
            result = bd.ensure_backend(
                "onnx",
                platform_info=LINUX_ARM,
                auto_install=True,
                installer=self.installer,
                verifier=self.verifier,
            )

        self.assertEqual(result.status, "disk")
        self.installer.assert_not_called()

    def test_conflicting_variants_are_uninstalled_before_switch(self):
        pkg = bd.resolve_backend("onnx", WINDOWS_X86).packages[0]
        with mock.patch.object(bd, "uninstall_packages", return_value=True) as remove, \
             mock.patch.object(bd, "pip_install", return_value=True) as pip:
            ok = bd._default_install(pkg, force=False)

        self.assertTrue(ok)
        remove.assert_called_once_with(pkg.uninstall_first)
        pip.assert_called_once()
        _, kwargs = pip.call_args
        self.assertFalse(kwargs["no_deps"])

    def test_engine_failure_provisions_onnx_fallback(self):
        def installed(dist):
            return "1.20.0" if dist == "onnxruntime" else None

        with mock.patch.object(bd, "installed_version", side_effect=installed):
            report = bd.ensure_backends(
                ["engine"],
                platform_info=LINUX_X86,
                auto_install=True,
                installer=self.installer,
                verifier=self.verifier,
            )

        self.assertEqual([r.backend for r in report.results], ["engine", "onnx"])
        self.assertEqual(report.results[0].status, "unsupported")
        self.assertEqual(report.results[1].status, "reused")


class ArmCudaCleanupTests(unittest.TestCase):
    def test_cuda_only_packages_removed_on_cpu_arm(self):
        with mock.patch.object(
            bd,
            "_cuda_related_distributions",
            return_value=["nvidia-cublas-cu12", "triton"],
        ), mock.patch.object(bd, "uninstall_packages", return_value=True) as remove:
            removed = bd.remove_forbidden_cuda_packages(LINUX_ARM)

        self.assertEqual(removed, ["nvidia-cublas-cu12", "triton"])
        remove.assert_called_once_with(["nvidia-cublas-cu12", "triton"])

    def test_cleanup_skipped_for_jetson_and_nvidia(self):
        with mock.patch.object(bd, "uninstall_packages") as remove:
            self.assertEqual(bd.remove_forbidden_cuda_packages(JETSON), [])
            nvidia_arm = bd.PlatformInfo(
                system="Linux",
                machine="aarch64",
                python_tag="cp312",
                has_nvidia=True,
                cuda_major=12,
            )
            self.assertEqual(bd.remove_forbidden_cuda_packages(nvidia_arm), [])
        remove.assert_not_called()


class PlatformOverrideTests(unittest.TestCase):
    def tearDown(self):
        bd.clear_platform_cache()

    def test_target_overrides_and_hardware_skip(self):
        bd.clear_platform_cache()
        with mock.patch.dict(
            os.environ,
            {
                "ISPY_TARGET_ARCH": "aarch64",
                "ISPY_TARGET_OS": "Linux",
                "ISPY_SKIP_HARDWARE_DETECT": "1",
            },
            clear=False,
        ):
            platform = bd.detect_platform()

        self.assertEqual(platform.machine, "aarch64")
        self.assertEqual(platform.system, "Linux")
        self.assertFalse(platform.has_nvidia)
        self.assertIsNone(platform.cuda_major)


class BootSelectionTests(unittest.TestCase):
    def _config(self, path):
        return types.SimpleNamespace(config={"file_path": path})

    def test_runtime_backends_follow_configured_model(self):
        from iSpy.boot import boot

        self.assertEqual(
            boot._runtime_dependency_backends(self._config("a/model.pt")),
            ["pytorch"],
        )
        self.assertEqual(
            boot._runtime_dependency_backends(self._config("a/model.onnx")),
            ["onnx"],
        )

    def test_runtime_backends_default_to_pytorch(self):
        from iSpy.boot import boot

        self.assertEqual(
            boot._runtime_dependency_backends(types.SimpleNamespace(config={})),
            ["pytorch"],
        )

    def test_all_backend_names_excludes_coreml(self):
        from iSpy.boot import backend_deps

        names = backend_deps.all_backend_names()
        self.assertIn("pytorch", names)
        self.assertIn("rknn", names)
        self.assertIn("tflite", names)
        self.assertIn("onnx", names)
        self.assertNotIn("coreml", names)

    def test_fresh_boot_provisions_all_backends(self):
        from iSpy.boot import boot

        report = bd.ProvisionReport(
            platform=LINUX_X86,
            results=[bd.BackendResult("pytorch", "PyTorch", "installed", "ok")],
        )
        with mock.patch(
            "iSpy.boot.backend_deps.ensure_backends", return_value=report
        ) as ensure, mock.patch(
            "iSpy.core.network_policy.is_offline", return_value=False
        ):
            boot._provision_runtime_dependencies(
                self._config("a/model.pt"), fresh=True
            )
        # Fresh install should provision ALL backends, not just pytorch
        called_backends = ensure.call_args[0][0]
        self.assertIn("rknn", called_backends)
        self.assertIn("tflite", called_backends)
        self.assertIn("onnx", called_backends)
        self.assertIn("pytorch", called_backends)

    def test_fresh_boot_raises_on_actionable_failure(self):
        from iSpy.boot import boot

        report = bd.ProvisionReport(
            platform=LINUX_X86,
            results=[
                bd.BackendResult(
                    "onnx", "ONNX Runtime", "failed", "pip failed"
                )
            ],
        )
        with mock.patch(
            "iSpy.boot.backend_deps.ensure_backends", return_value=report
        ), mock.patch(
            "iSpy.core.network_policy.is_offline", return_value=False
        ):
            with self.assertRaises(bd.DependencyProvisionError):
                boot._provision_runtime_dependencies(
                    self._config("a/model.onnx"), fresh=True
                )

    def test_normal_boot_does_not_raise_on_missing_backend(self):
        from iSpy.boot import boot

        report = bd.ProvisionReport(
            platform=LINUX_X86,
            results=[
                bd.BackendResult(
                    "onnx", "ONNX Runtime", "missing", "run -f"
                )
            ],
        )
        with mock.patch(
            "iSpy.boot.backend_deps.ensure_backends", return_value=report
        ), mock.patch(
            "iSpy.core.network_policy.is_offline", return_value=False
        ):
            result = boot._provision_runtime_dependencies(
                self._config("a/model.onnx"), fresh=False
            )

        self.assertIs(result, report)


class LegacyViewTests(unittest.TestCase):
    def test_legacy_mapping_is_derived_from_registry(self):
        deps = bd.legacy_dependencies(LINUX_ARM)

        self.assertIn("pytorch", deps)
        self.assertIn("onnx", deps)
        self.assertEqual(deps["tflite"][0][0], "ai_edge_litert")
        # tflite-runtime is only selected for cp310/cp311.
        old = bd.legacy_dependencies(
            bd.PlatformInfo(
                system="Linux", machine="aarch64", python_tag="cp311"
            )
        )
        self.assertEqual(old["tflite"][0][0], "tflite_runtime")

    def test_backend_dependencies_view_is_lazy(self):
        """Importing the optimizer must not probe hardware or import torch."""
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root)
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import iSpy.vision.optimizer; "
                "print('torch' in sys.modules)",
            ],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "False")

    def test_has_nvidia_probe_does_not_import_torch(self):
        """A hardware yes/no must not pull torch into the main process."""
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root)
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from iSpy.config import AutoOpt; "
                "AutoOpt.has_nvidia.cache_clear(); AutoOpt.has_nvidia(); "
                "print('torch' in sys.modules)",
            ],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
