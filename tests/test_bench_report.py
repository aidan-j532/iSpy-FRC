import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from iSpy.validations import benchmarking
from iSpy.validations.bench_to_matplotlib import (
    PLOT_SECTIONS,
    collect_system_info,
    render_report,
)


def _row(**overrides):
    row = {
        "model": "yolov8n",
        "backend": "Auto/ONNX-CPU",
        "format": "onnx",
        "device": "cpu",
        "batch_size": 1,
        "mode": "serial",
        "streams": 1,
        "fps": 42.3,
        "inference_ms": 23.2,
        "latency_ms": {
            "mean": 23.2,
            "p50": 22.9,
            "p95": 26.4,
            "p99": 31.2,
            "min": 21.8,
            "max": 34.0,
            "stdev": 1.7,
            "samples": 211,
        },
        "stage_ms": {"preprocess": 3.2, "device": 17.1, "postprocess": 2.9},
        "provider": "CPUExecutionProvider",
        "detections": 187,
        "frames": 211,
        "elapsed": 5.0,
        "ok": True,
    }
    row.update(overrides)
    return row


def _payload():
    return {
        "timestamp": "2026-11-12 16:42:08",
        "system": {
            "platform": "Windows-11-10.0.22631-SP0",
            "python": "3.11.2",
            "cpu_count": 16,
            "cpu": "Test CPU",
            "memory_gb": 32.0,
            "torch": "2.6.0+cu124",
            "cuda": "12.4",
            "gpu": "Test GPU",
            "onnxruntime": "1.20.1",
            "onnxruntime_providers": ["CPUExecutionProvider"],
            "ispy": "2.2.15",
            "hostname": "TESTBOX",
        },
        "settings": {
            "duration": 5.0,
            "batch_sizes": [1],
            "streams": 1,
            "modes": ["serial"],
            "backends": ["onnx"],
        },
        "all": [
            _row(),
            _row(
                backend="Auto/TensorRT",
                format="engine",
                device="0",
                fps=249.8,
                inference_ms=3.9,
                latency_ms={
                    "mean": 3.9,
                    "p50": 3.8,
                    "p95": 4.4,
                    "p99": 5.1,
                    "min": 3.4,
                    "max": 6.0,
                    "stdev": 0.4,
                    "samples": 1249,
                },
                stage_ms={
                    "preprocess": 0.7,
                    "device": 2.8,
                    "postprocess": 0.4,
                },
                provider="TensorRT",
                detections=1198,
                frames=1249,
                model_forward_fps=312.4,
                model_forward_ms=3.2,
            ),
            _row(
                model="yolov8s",
                backend="Auto/HAILO",
                format="hailo",
                fps=None,
                inference_ms=None,
                latency_ms={},
                stage_ms={},
                provider="unavailable",
                detections=0,
                frames=None,
                elapsed=None,
                ok=False,
                error="conversion failed: hailo_sdk_client is not installed",
            ),
        ],
    }


class PlotFlagTests(unittest.TestCase):
    def test_plot_is_on_by_default(self):
        args = benchmarking._build_parser().parse_args([])
        self.assertTrue(args.plot)
        self.assertEqual(args.plot_sections, PLOT_SECTIONS)
        self.assertEqual(args.plot_theme, "light")
        self.assertEqual(args.plot_dpi, 180)
        self.assertEqual(args.plot_sort, "run")
        self.assertEqual(args.plot_latency_scale, "linear")

    def test_no_plot_disables_report(self):
        args = benchmarking._build_parser().parse_args(["--no-plot"])
        self.assertFalse(args.plot)

    def test_plot_sections_add_and_remove_panels(self):
        args = benchmarking._build_parser().parse_args(
            ["--plot-sections", "latency,table"]
        )
        self.assertEqual(args.plot_sections, ("latency", "table"))
        args = benchmarking._build_parser().parse_args(["--plot-sections", "none"])
        self.assertEqual(args.plot_sections, ())
        args = benchmarking._build_parser().parse_args(["--plot-sections", "all"])
        self.assertEqual(args.plot_sections, PLOT_SECTIONS)

    def test_unknown_plot_section_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                benchmarking._build_parser().parse_args(
                    ["--plot-sections", "throughput,nope"]
                )

    def test_plot_options_are_parsed(self):
        args = benchmarking._build_parser().parse_args(
            [
                "--plot-theme",
                "dark",
                "--plot-dpi",
                "320",
                "--plot-sort",
                "fps",
                "--plot-latency-scale",
                "log",
                "--plot-title",
                "Bench night",
            ]
        )
        self.assertEqual(args.plot_theme, "dark")
        self.assertEqual(args.plot_dpi, 320)
        self.assertEqual(args.plot_sort, "fps")
        self.assertEqual(args.plot_latency_scale, "log")
        self.assertEqual(args.plot_title, "Bench night")

    def test_plot_output_without_plot_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                benchmarking.main(["--no-plot", "--plot-output", "chart.png"])


class RenderReportTests(unittest.TestCase):
    def test_collect_system_info_has_platform_fields(self):
        info = collect_system_info()
        self.assertIn("platform", info)
        self.assertIn("python", info)

    def test_render_report_writes_png_with_all_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.png"
            result = render_report(_payload(), path)
            self.assertEqual(result, path)
            self.assertGreater(path.stat().st_size, 10_000)
            with Image.open(path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 1000)
                self.assertGreater(image.height, 800)

    def test_render_report_supports_theme_sections_and_sort(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dark.png"
            result = render_report(
                _payload(),
                path,
                title="Dark report",
                sections=("summary", "throughput", "latency", "table"),
                theme="dark",
                sort="fps",
                latency_scale="log",
                dpi=110,
            )
            self.assertEqual(result, path)
            self.assertGreater(path.stat().st_size, 5_000)
            with Image.open(path) as image:
                self.assertEqual(image.format, "PNG")

    def test_render_report_handles_header_only_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "header.png"
            self.assertIsNotNone(
                render_report(_payload(), path, sections="none", dpi=80)
            )
            self.assertTrue(path.exists())

    def test_render_report_handles_all_failed_runs(self):
        payload = _payload()
        payload["all"] = [
            _row(
                backend="Auto/RKNN",
                format="rknn",
                fps=None,
                inference_ms=None,
                latency_ms={},
                stage_ms={},
                detections=0,
                frames=None,
                elapsed=None,
                ok=False,
                error="conversion failed: rknn-toolkit2 is not installed",
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failed.png"
            self.assertIsNotNone(render_report(payload, path, dpi=80))
            self.assertTrue(path.exists())

    def test_render_report_skips_empty_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = render_report({"all": []}, Path(tmp) / "empty.png")
            self.assertIsNone(path)


class BenchMainIntegrationTests(unittest.TestCase):
    def test_main_writes_json_and_png_with_plot_flags(self):
        result = _row()
        result.update(
            model="model",
            backend="CPU",
            format="pt",
            device="cpu",
            core_mask=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "bench.json"
            png_path = Path(tmp) / "bench.png"
            with (
                mock.patch.object(
                    benchmarking,
                    "_recommended_backend_plan",
                    return_value={"pt": ("pt", "cpu", [(None, "CPU")])},
                ),
                mock.patch.object(
                    benchmarking,
                    "find_pt_files",
                    return_value=[Path("model.pt")],
                ),
                mock.patch.object(
                    benchmarking,
                    "get_or_convert",
                    return_value=(Path("model.pt"), None),
                ),
                mock.patch.object(
                    benchmarking,
                    "fill_missing_config",
                    side_effect=lambda config: config,
                ),
                mock.patch.object(
                    benchmarking,
                    "_run_benchmark_task",
                    return_value=result,
                ),
                mock.patch.object(
                    benchmarking,
                    "collect_system_info",
                    return_value={"platform": "test", "python": "3.11"},
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = benchmarking.main(
                    [
                        "--model",
                        "model.pt",
                        "--output",
                        str(json_path),
                        "--plot-output",
                        str(png_path),
                        "--plot-sections",
                        "summary,table",
                        "--plot-theme",
                        "dark",
                        "--plot-dpi",
                        "80",
                        "--duration",
                        "1",
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(json_path.read_text())
            self.assertEqual(payload["settings"]["duration"], 1.0)
            self.assertEqual(payload["settings"]["backends"], ["pt"])
            self.assertEqual(payload["all"][0]["backend"], "CPU")
            self.assertTrue(png_path.exists())
            with Image.open(png_path) as image:
                self.assertEqual(image.format, "PNG")


if __name__ == "__main__":
    unittest.main()
