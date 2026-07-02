import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from iSpy.boot.boot import _normalize_box_coords_for_quantization


def test_normalizes_box_coords_before_concat():
    box_input = helper.make_tensor_value_info("box_input", TensorProto.FLOAT, [1, 4, 8400])
    conf_input = helper.make_tensor_value_info("conf_input", TensorProto.FLOAT, [1, 1, 8400])
    output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, 8400])

    concat_node = helper.make_node(
        "Concat",
        inputs=["box_input", "conf_input"],
        outputs=["output0"],
        axis=1,
        name="concat_out",
    )

    graph = helper.make_graph(
        [concat_node],
        "box_norm_test",
        [box_input, conf_input],
        [output],
    )
    model = helper.make_model(graph)

    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "source.onnx"
        dst = Path(tmpdir) / "patched.onnx"
        onnx.save(model, src)

        divisor = _normalize_box_coords_for_quantization(str(src), str(dst), 640)

        patched = onnx.load(dst)
        assert divisor == 640.0
        assert any(node.op_type == "Div" for node in patched.graph.node)
        concat_nodes = [node for node in patched.graph.node if node.op_type == "Concat"]
        assert len(concat_nodes) == 1
        assert concat_nodes[0].input[0].endswith("_iSpy_normalized")
