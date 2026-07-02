import onnx
import onnx.shape_inference

model = onnx.load("YoloModels/onnx/_default_v26_detect_for_fuel.onnx")
inferred = onnx.shape_inference.infer_shapes(model)
graph = inferred.graph

output_names = {o.name for o in graph.output}
value_info = {vi.name: vi for vi in list(graph.value_info) + list(graph.input) + list(graph.output)}

for node in graph.node:
    if node.op_type == "Concat" and any(o in output_names for o in node.output):
        print("Concat node:", node.name, "-> outputs:", list(node.output))
        for inp in node.input:
            vi = value_info.get(inp)
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim] if vi else "unknown"
            print(f"  input '{inp}': shape={dims}")