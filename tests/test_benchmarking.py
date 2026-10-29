import contextlib
import io
import unittest
from unittest import mock

from iSpy.validations import benchmarking


class BenchmarkingCliTests(unittest.TestCase):
    def test_parallel_defaults_to_one(self):
        args = benchmarking._build_parser().parse_args([])
        self.assertEqual(args.parallel, 1)

    def test_parallel_accepts_worker_count(self):
        args = benchmarking._build_parser().parse_args(["--parallel", "3"])
        self.assertEqual(args.parallel, 3)

    def test_batch_pipeline_and_stream_options(self):
        args = benchmarking._build_parser().parse_args(
            ["--batch-sizes", "1,4,8", "--pipelined", "--streams", "3"]
        )
        self.assertEqual(args.batch_sizes, (1, 4, 8))
        self.assertTrue(args.pipelined)
        self.assertEqual(args.streams, 3)

    def test_serial_and_pipelined_comparison_are_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                benchmarking.main(["--serial", "--pipelined"])

    def test_parallel_must_be_positive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                benchmarking.main(["--parallel", "0"])


class BenchmarkTaskTests(unittest.TestCase):
    def setUp(self):
        self.task = {
            "model": "detector",
            "backend": "TPU",
            "format": "tpu",
            "device": "tpu",
            "core_mask": None,
            "config": {"file_path": "model.pt"},
            "duration": 2.0,
        }

    def test_worker_returns_benchmark_metrics(self):
        with mock.patch.object(
            benchmarking, "benchmark", return_value=(12.3, 81.3, 25, 2.03)
        ):
            result = benchmarking._run_benchmark_task(self.task)

        self.assertEqual(result["model"], "detector")
        self.assertEqual(result["backend"], "TPU")
        self.assertEqual(result["fps"], 12.3)
        self.assertEqual(result["inference_ms"], 81.3)
        self.assertEqual(result["frames"], 25)
        self.assertEqual(result["elapsed"], 2.03)
        self.assertTrue(result["ok"])

    def test_worker_records_benchmark_errors(self):
        with mock.patch.object(
            benchmarking, "benchmark", side_effect=RuntimeError("unavailable")
        ):
            result = benchmarking._run_benchmark_task(self.task)

        self.assertIsNone(result["fps"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "unavailable")

    def test_zero_inferences_are_not_a_valid_result(self):
        with mock.patch.object(
            benchmarking, "benchmark", return_value=(0.0, 0.0, 0, 2.0)
        ):
            result = benchmarking._run_benchmark_task(self.task)

        self.assertFalse(result["ok"])
        self.assertIsNone(result["fps"])

    def test_model_forward_benchmark_skips_pre_and_postprocessing(self):
        import numpy as np

        class FakeOnnxSession:
            def __init__(self):
                self.calls = 0

            def get_providers(self):
                return ["CPUExecutionProvider"]

            def run(self, output_names, inputs):
                self.calls += 1
                return [np.zeros((1, 1))]

        class FakeModel:
            model_type = "onnx"
            _onnx_out_names = ["output"]
            _onnx_inp_name = "input"

            def __init__(self):
                self.model = FakeOnnxSession()
                self.preprocess_calls = 0

            def _preprocess_frame(self, frame):
                self.preprocess_calls += 1
                return np.asarray(frame, dtype=np.float32)[None]

        model = FakeModel()
        result = benchmarking._benchmark_model_forward(
            model, [np.zeros((2, 2), dtype=np.float32)], batch_size=1
        )

        self.assertEqual(model.preprocess_calls, 1)
        self.assertEqual(model.model.calls, 18)
        self.assertGreater(result["model_forward_ms"], 0)
        self.assertAlmostEqual(
            result["model_forward_fps"],
            1000 / result["model_forward_ms"],
        )

    def test_result_format_distinguishes_predict_and_vision_only(self):
        result = benchmarking._fmt_result(
            {
                "ok": True,
                "backend": "CUDA-0",
                "batch_size": 1,
                "mode": "serial",
                "streams": 1,
                "fps": 40.0,
                "inference_ms": 25.0,
                "model_forward_fps": 125.0,
                "model_forward_ms": 8.0,
            }
        )

        self.assertIn("predict", result)
        self.assertIn("vision-only 125.0 FPS 8.00 ms/frame", result)

    def test_combined_result_preserves_both_speed_metrics(self):
        result = benchmarking._combine_benchmark_runs(
            [
                {
                    "frames": 40,
                    "elapsed": 1.0,
                    "fps": 40.0,
                    "latencies": [25.0],
                    "stage_ms": {},
                    "detections": 0,
                    "provider": "CUDAExecutionProvider",
                    "model_forward_fps": 125.0,
                    "model_forward_ms": 8.0,
                }
            ]
        )

        self.assertEqual(result["predict_fps"], 40.0)
        self.assertEqual(result["model_forward_fps"], 125.0)
        self.assertEqual(result["model_forward_ms"], 8.0)

    def test_model_forward_benchmark_measures_tpu_forward(self):
        import contextlib
        import numpy as np
        import torch
        import torch.nn as nn

        class FakeModel:
            model_type = "tpu"
            _tpu_device = torch.device("cpu")
            tpu_dtype = "fp32"
            model = nn.Identity()

            @staticmethod
            def _preprocess_tpu_frame(frame):
                return frame

            @staticmethod
            def _tpu_autocast(_torch):
                return contextlib.nullcontext()

            @staticmethod
            def _tpu_sync():
                return None

        result = benchmarking._benchmark_model_forward(
            FakeModel(), [np.zeros((2, 2, 3), dtype=np.uint8)], batch_size=1
        )

        self.assertGreater(result["model_forward_ms"], 0)
        self.assertAlmostEqual(
            result["model_forward_fps"],
            1000 / result["model_forward_ms"],
        )

    def test_model_forward_benchmark_measures_pytorch_forward(self):
        import numpy as np
        import torch.nn as nn

        class FakeModel:
            model_type = "yolo"
            device = "cpu"
            model = mock.Mock(model=nn.Identity())

            @staticmethod
            def _preprocess_frame(_frame):
                return np.zeros((1, 2, 2, 3), dtype=np.uint8)

        result = benchmarking._benchmark_model_forward(
            FakeModel(), [np.zeros((2, 2, 3), dtype=np.uint8)], batch_size=1
        )

        self.assertGreater(result["model_forward_ms"], 0)
        self.assertAlmostEqual(
            result["model_forward_fps"],
            1000 / result["model_forward_ms"],
        )

    def test_model_forward_benchmark_skips_unimplemented_backend(self):
        class FakeModel:
            model_type = "rknn"

        self.assertIsNone(
            benchmarking._benchmark_model_forward(FakeModel(), [], batch_size=1)
        )


if __name__ == "__main__":
    unittest.main()
