import json
from pathlib import Path


def read_engine_plan(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) < 5:
        return data

    header_size = int.from_bytes(data[:4], "little")
    header_end = 4 + header_size
    if header_size < 1 or header_end >= len(data):
        return data
    try:
        header = json.loads(data[4:header_end])
    except (UnicodeDecodeError, json.JSONDecodeError):
        return data
    if not isinstance(header, dict):
        return data
    return data[header_end:]


def execute_engine(engine, context, inputs: dict[str, object], device=0) -> list:
    import numpy as np
    import tensorrt as trt
    import torch

    try:
        device_index = int(device)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"TensorRT inference requires a CUDA device, got {device!r}"
        ) from exc
    if not torch.cuda.is_available() or device_index < 0:
        raise RuntimeError(
            f"TensorRT inference requires an available CUDA device, got {device!r}"
        )
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {device_index} is unavailable "
            f"(device count: {torch.cuda.device_count()})"
        )

    buffers = []
    output_buffers = []
    bindings = []
    cuda_device = torch.device(f"cuda:{device_index}")
    with torch.cuda.device(device_index):
        for idx in range(engine.num_io_tensors):
            name = engine.get_tensor_name(idx)
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if name not in inputs:
                    raise ValueError(
                        f"No input data supplied for TensorRT input {name!r}"
                    )
                array = np.ascontiguousarray(inputs[name], dtype=dtype)
                buffer = torch.from_numpy(array).to(device=cuda_device)
                buffers.append(buffer)
                bindings.append(buffer.data_ptr())
            else:
                shape = tuple(context.get_tensor_shape(name))
                if any(size < 0 for size in shape):
                    raise RuntimeError(
                        f"TensorRT output {name!r} has unresolved shape {shape}"
                    )
                torch_dtype = torch.from_numpy(np.empty((), dtype=dtype)).dtype
                buffer = torch.empty(shape, dtype=torch_dtype, device=cuda_device)
                buffers.append(buffer)
                bindings.append(buffer.data_ptr())
                output_buffers.append(buffer)

        if not context.execute_v2(bindings):
            raise RuntimeError("TensorRT execute_v2 returned false")

        return [buffer.cpu().numpy() for buffer in output_buffers]


def engine_io_names(engine) -> tuple[list[str], list[str]]:
    # split engine tensors into (input_names, output_names) once so callers
    # don't each re-walk num_io_tensors
    import tensorrt as trt

    inputs, outputs = [], []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(name)
        else:
            outputs.append(name)
    return inputs, outputs


def bind_tensor_addresses(context, engine, addresses: dict) -> None:
    # execute_async_v3 (the tensor-name API) reads/writes whatever device
    # address was last bound to each tensor name. It does NOT use the old
    # bindings list, so if we forget set_tensor_address for even one tensor it
    # silently reads garbage / writes to a dangling pointer instead of failing.
    # Bind every tensor and check each return so a false one raises.
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if name not in addresses:
            raise ValueError(
                f"No device address prepared for TensorRT tensor {name!r}"
            )
        if not context.set_tensor_address(name, addresses[name]):
            raise RuntimeError(
                f"TensorRT set_tensor_address returned false for {name!r}"
            )


def bind_engine_io(
    engine, context, inputs: dict, buffers: dict, device=0
) -> list[str]:
    # Prepare device buffers for one execute_async_v3 pass and bind every
    # tensor address. Shared by the runtime (_run_engine) and the forward-only
    # benchmark so both exercise the exact same execution path. Input/output
    # buffers are allocated lazily into `buffers` (reused across calls, resized
    # when a shape changes) so we don't reallocate every frame. Returns the
    # ordered output tensor names.
    import numpy as np
    import tensorrt as trt
    import torch

    try:
        device_index = int(device)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"TensorRT inference requires a CUDA device, got {device!r}"
        ) from exc
    if not torch.cuda.is_available() or device_index < 0:
        raise RuntimeError(
            f"TensorRT inference requires an available CUDA device, got {device!r}"
        )
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {device_index} is unavailable "
            f"(device count: {torch.cuda.device_count()})"
        )

    cuda_device = torch.device(f"cuda:{device_index}")
    output_names = []
    with torch.cuda.device(device_index):
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if name not in inputs:
                    raise ValueError(
                        f"No input data supplied for TensorRT input {name!r}"
                    )
                array = np.ascontiguousarray(inputs[name], dtype=dtype)
                torch_dtype = torch.from_numpy(np.empty((), dtype=dtype)).dtype
                buffer = buffers.get(name)
                if buffer is None or tuple(buffer.shape) != array.shape:
                    # shape changed (batch/resolution) - the old buffer would
                    # be the wrong size, so reallocate instead of trusting it
                    buffer = torch.empty(
                        array.shape, dtype=torch_dtype, device=cuda_device
                    )
                    buffers[name] = buffer
                buffer.copy_(torch.from_numpy(array).to(device=cuda_device))
                if not context.set_input_shape(name, array.shape):
                    raise RuntimeError(
                        f"TensorRT rejected input shape {array.shape} for "
                        f"tensor {name!r}"
                    )
            else:
                output_names.append(name)

        for name in output_names:
            shape = tuple(context.get_tensor_shape(name))
            if any(size < 0 for size in shape):
                raise RuntimeError(
                    f"TensorRT output {name!r} has unresolved shape {shape}"
                )
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            torch_dtype = torch.from_numpy(np.empty((), dtype=dtype)).dtype
            buffer = buffers.get(name)
            if buffer is None or tuple(buffer.shape) != shape:
                buffers[name] = torch.empty(
                    shape, dtype=torch_dtype, device=cuda_device
                )

        bind_tensor_addresses(
            context,
            engine,
            {name: buffers[name].data_ptr() for name in buffers},
        )

    return output_names


def execute_bound_async_v3(context, stream_handle) -> None:
    # run one already-bound execute_async_v3 pass, checking the return so a
    # failed enqueue raises instead of leaving stale outputs to be read
    if not context.execute_async_v3(stream_handle=stream_handle):
        raise RuntimeError("TensorRT execute_async_v3 returned false")