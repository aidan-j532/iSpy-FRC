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


if __name__ == "__main__":
    unittest.main()
