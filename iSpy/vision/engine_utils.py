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
            else:
                shape = tuple(context.get_tensor_shape(name))
                if any(size < 0 for size in shape):
                    raise RuntimeError(
                        f"TensorRT output {name!r} has unresolved shape {shape}"
                    )
                torch_dtype = torch.from_numpy(np.empty((), dtype=dtype)).dtype
                buffer = torch.empty(shape, dtype=torch_dtype, device=cuda_device)
                output_buffers.append(buffer)
            buffers.append(buffer)
            bindings.append(buffer.data_ptr())

        if not context.execute_v2(bindings):
            raise RuntimeError("TensorRT execute_v2 returned false")

        return [buffer.cpu().numpy() for buffer in output_buffers]