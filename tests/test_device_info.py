import unittest
from types import SimpleNamespace
from unittest.mock import patch

from iSpy.config import deviceInfo


class TestDeviceInfo(unittest.TestCase):
    def test_expected_fps_matches_device_name_variants(self):
        result = deviceInfo.get_expected_inf_ms("Orange Pi 5 Model B")
        self.assertAlmostEqual(result["expected_inf_ms"], 1000.0 / 30.0)

    def test_unknown_device_has_zero_inference_without_type_error(self):
        self.assertEqual(
            deviceInfo.get_expected_inf_ms("Completely Unknown Board"),
            {"expected_inf_ms": 0.0},
        )

    def test_formatted_description_has_every_section(self):
        with patch.object(
            deviceInfo, "get_device_model", return_value="Orange Pi 5"
        ), patch.object(
            deviceInfo, "get_raw_cpu_name", return_value="ARMv8"
        ), patch.object(
            deviceInfo, "recommend_format", return_value="rknn"
        ), patch.object(
            deviceInfo,
            "get_accelerators_name_from_type",
            return_value="Rockchip RK3588",
        ), patch.object(
            deviceInfo, "get_total_storage", return_value="64.00 GB"
        ):
            text = deviceInfo.get_formatted_description()

        for label in (
            "Device: Orange Pi 5",
            "CPU:",
            "Accelerator (rknn): Rockchip RK3588",
            "Memory:",
            "Storage: 64.00 GB",
            "OS:",
            "Architecture:",
            "Python:",
            "Backends:",
            "Expected FPS: 30.0",
            "Expected inference: 33.33 ms",
        ):
            self.assertIn(label, text)

    def test_windows_11_is_detected_from_build_number(self):
        with patch.object(deviceInfo.platform, "system", return_value="Windows"), patch.object(
            deviceInfo.platform,
            "win32_ver",
            return_value=("10", "10.0.22000", "", ""),
        ):
            self.assertTrue(deviceInfo.get_os_version().startswith("11"))

    def test_linux_uses_os_release(self):
        with patch.object(deviceInfo.platform, "system", return_value="Linux"), patch.object(
            deviceInfo,
            "_os_release",
            return_value={"NAME": "Ubuntu", "VERSION_ID": "22.04"},
        ):
            self.assertEqual(deviceInfo.get_os_name(), "Ubuntu")
            self.assertEqual(deviceInfo.get_os_version(), "22.04")

    def test_macos_device_model_uses_sysctl(self):
        with patch.object(deviceInfo.platform, "system", return_value="Darwin"), patch.object(
            deviceInfo.subprocess, "check_output", return_value="Mac14,2\n"
        ):
            self.assertEqual(deviceInfo.get_device_model(), "Mac14,2")

    def test_storage_sums_unique_partitions(self):
        partitions = [
            SimpleNamespace(device="/dev/sda1", mountpoint="/"),
            SimpleNamespace(device="/dev/sda1", mountpoint="/boot"),
            SimpleNamespace(device="/dev/sdb1", mountpoint="/mnt/usb"),
        ]
        sizes = {
            "/": 100 * 1024**3,
            "/boot": 999 * 1024**3,
            "/mnt/usb": 50 * 1024**3,
        }

        with patch.object(
            deviceInfo.psutil, "disk_partitions", return_value=partitions
        ), patch.object(
            deviceInfo.psutil,
            "disk_usage",
            side_effect=lambda mountpoint: SimpleNamespace(total=sizes[mountpoint]),
        ):
            self.assertEqual(deviceInfo.get_total_storage(), "150.00 GB")


if __name__ == "__main__":
    unittest.main()
