import sys
import types
import unittest
from unittest import mock


class _FakeEngine:
    def __init__(self, names):
        self._names = list(names)
        self.num_io_tensors = len(self._names)

    def get_tensor_name(self, index):
        return self._names[index]


class _FakeContext:
    def __init__(self, fail_address_for=None, execute_ok=True):
        self.addresses = {}
        self._fail = fail_address_for or set()
        self._execute_ok = execute_ok
        self.executed = False

    def set_tensor_address(self, name, ptr):
        if name in self._fail:
            return False
        self.addresses[name] = ptr
        return True

    def execute_async_v3(self, stream_handle=None):
        self.executed = True
        return self._execute_ok


class BindTensorAddressTests(unittest.TestCase):
    # execute_async_v3 relies on set_tensor_address having bound every tensor;
    # forgetting one silently reads garbage, so these are the regression guard.

    def test_binds_every_tensor(self):
        from iSpy.vision.engine_utils import bind_tensor_addresses

        engine = _FakeEngine(["images", "output0"])
        context = _FakeContext()
        bind_tensor_addresses(context, engine, {"images": 111, "output0": 222})
        self.assertEqual(context.addresses, {"images": 111, "output0": 222})

    def test_missing_address_raises(self):
        from iSpy.vision.engine_utils import bind_tensor_addresses

        engine = _FakeEngine(["images", "output0"])
        with self.assertRaises(ValueError):
            bind_tensor_addresses(
                engine=engine, context=_FakeContext(), addresses={"images": 111}
            )

    def test_false_set_tensor_address_raises(self):
        from iSpy.vision.engine_utils import bind_tensor_addresses

        engine = _FakeEngine(["images", "output0"])
        context = _FakeContext(fail_address_for={"output0"})
        with self.assertRaises(RuntimeError):
            bind_tensor_addresses(
                context, engine, {"images": 111, "output0": 222}
            )


class ExecuteBoundAsyncTests(unittest.TestCase):
    def test_false_execute_raises(self):
        from iSpy.vision.engine_utils import execute_bound_async_v3

        with self.assertRaises(RuntimeError):
            execute_bound_async_v3(_FakeContext(execute_ok=False), 0)

    def test_true_execute_passes(self):
        from iSpy.vision.engine_utils import execute_bound_async_v3

        context = _FakeContext()
        execute_bound_async_v3(context, 0)
        self.assertTrue(context.executed)


class EngineIoNamesTests(unittest.TestCase):
    def test_splits_inputs_and_outputs(self):
        from iSpy.vision.engine_utils import engine_io_names

        mode = types.SimpleNamespace(INPUT="input", OUTPUT="output")
        engine = _FakeEngine(["images", "output0"])
        engine.get_tensor_mode = (
            lambda name: mode.INPUT if name == "images" else mode.OUTPUT
        )
        fake_trt = types.ModuleType("tensorrt")
        fake_trt.TensorIOMode = mode
        with mock.patch.dict(sys.modules, {"tensorrt": fake_trt}):
            inputs, outputs = engine_io_names(engine)
        self.assertEqual(inputs, ["images"])
        self.assertEqual(outputs, ["output0"])


class TpuSyncBlockingTests(unittest.TestCase):
    # torch_xla dispatch is async; without a blocking wait the benchmark times
    # dispatch overhead, not inference. verify _tpu_sync asks for wait=True.
    #
    # GenericYolo is imported once, OUTSIDE the torch_xla patch, so that
    # mock.patch.dict's sys.modules restore doesn't evict cv2/numpy (reloading
    # numpy's C extension twice in one process blows up).

    @classmethod
    def setUpClass(cls):
        from iSpy.vision.genericYolo import GenericYolo

        cls._GenericYolo = GenericYolo

    def _wrapper(self):
        wrapper = self._GenericYolo.__new__(self._GenericYolo)
        wrapper.logger = mock.Mock()
        return wrapper

    @staticmethod
    def _fake_xla(sync=None):
        fake_xla = types.ModuleType("torch_xla")
        if sync is not None:
            fake_xla.sync = sync
        fake_core = types.ModuleType("torch_xla.core")
        fake_xm = types.ModuleType("torch_xla.core.xla_model")
        fake_core.xla_model = fake_xm
        return {
            "torch_xla": fake_xla,
            "torch_xla.core": fake_core,
            "torch_xla.core.xla_model": fake_xm,
        }, fake_xm

    def test_sync_called_with_wait_true(self):
        calls = {}

        def fake_sync(wait=True):
            calls["wait"] = wait

        mods, _ = self._fake_xla(sync=fake_sync)
        with mock.patch.dict(sys.modules, mods):
            self._wrapper()._tpu_sync()
        self.assertTrue(calls.get("wait"))

    def test_legacy_sync_without_wait_kwarg_falls_back(self):
        # old torch_xla.sync() takes no wait kwarg but still blocks
        calls = {"n": 0}

        def old_sync():
            calls["n"] += 1

        mods, _ = self._fake_xla(sync=old_sync)
        with mock.patch.dict(sys.modules, mods):
            self._wrapper()._tpu_sync()
        self.assertEqual(calls["n"], 1)

    def test_legacy_mark_step_materializes_output(self):
        # without torch_xla.sync, mark_step does not block - we force a read
        marked = {}

        def mark_step():
            marked["stepped"] = True

        mods, fake_xm = self._fake_xla(sync=None)
        fake_xm.mark_step = mark_step
        output = mock.Mock()
        with mock.patch.dict(sys.modules, mods):
            self._wrapper()._tpu_sync(output=output)
        self.assertTrue(marked.get("stepped"))
        output.cpu.assert_called_once()


class OrtGpuProviderTests(unittest.TestCase):
    def _provider(self, available):
        import iSpy.validations.benchmarking as bench

        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.get_available_providers = lambda: available
        with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
            return bench._ort_gpu_provider()

    def test_cuda_provider_detected(self):
        self.assertEqual(
            self._provider(["CPUExecutionProvider", "CUDAExecutionProvider"]),
            "CUDAExecutionProvider",
        )

    def test_cpu_only_returns_none(self):
        self.assertIsNone(self._provider(["CPUExecutionProvider"]))

    def test_missing_ort_returns_none(self):
        import iSpy.validations.benchmarking as bench

        with mock.patch.dict(sys.modules, {"onnxruntime": None}):
            self.assertIsNone(bench._ort_gpu_provider())


class DetectTestPlanOnnxLabelTests(unittest.TestCase):
    def _detect(self, ort_provider):
        import iSpy.validations.benchmarking as bench

        with mock.patch.object(bench, "has_rockchip_npu", return_value=False), \
            mock.patch.object(bench, "has_tpu", return_value=False), \
            mock.patch.object(bench, "has_nvidia", return_value=True), \
            mock.patch.object(bench, "has_tensorrt", return_value=False), \
            mock.patch.object(bench, "_has_openvino", return_value=False), \
            mock.patch.object(bench, "has_arm", return_value=False), \
            mock.patch.object(bench, "has_edge_tpu", return_value=False), \
            mock.patch.object(bench, "_cuda_devices", return_value=[(0, "GPU0")]), \
            mock.patch.object(bench, "_ort_gpu_provider", return_value=ort_provider):
            return bench.detect_test_plan()

    def test_onnx_skipped_without_gpu_provider(self):
        plan = self._detect(None)
        self.assertIn("pt_gpu0", plan)
        self.assertNotIn("onnx_gpu0", plan)

    def test_onnx_labelled_with_gpu_provider(self):
        plan = self._detect("CUDAExecutionProvider")
        self.assertIn("onnx_gpu0", plan)
        self.assertEqual(plan["onnx_gpu0"][2][0][1], "ONNX-GPU0")


class RecommendedBackendLabelTests(unittest.TestCase):
    def _plan(self, ort_provider):
        import iSpy.validations.benchmarking as bench

        with mock.patch.object(
            bench, "_recommended_backend_plan", wraps=bench._recommended_backend_plan
        ), mock.patch(
            "iSpy.config.AutoOpt.recommend_format", return_value="onnx"
        ), mock.patch.object(
            bench, "has_nvidia", return_value=True
        ), mock.patch.object(
            bench, "_ort_gpu_provider", return_value=ort_provider
        ), mock.patch.object(
            bench, "_cuda_devices", return_value=[(0, "GPU0")]
        ):
            return bench._recommended_backend_plan()

    def test_no_gpu_provider_labelled_cpu(self):
        plan = self._plan(None)
        fmt, dev, masks = plan["onnx"]
        self.assertEqual(dev, "cpu")
        self.assertEqual(masks[0][1], "Auto/ONNX-CPU")

    def test_gpu_provider_labelled_cuda(self):
        plan = self._plan("CUDAExecutionProvider")
        fmt, dev, masks = plan["onnx"]
        self.assertEqual(dev, 0)
        self.assertEqual(masks[0][1], "Auto/ONNX-CUDA")


if __name__ == "__main__":
    unittest.main()
