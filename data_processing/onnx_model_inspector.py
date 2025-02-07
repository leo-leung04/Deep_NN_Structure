import onnx
from onnx import shape_inference

# Model loading
model = onnx.load("convit_small_Opset16.onnx")

# Shape inference
model = shape_inference.infer_shapes(model)

# Obtain the graph
graph = model.graph

# Obtain input info
print("Model inputs:")
for i in graph.input:
    print(f"Name: {i.name}, Type: {i.type.tensor_type.elem_type}, Shape: ", end="")
    dim_info = i.type.tensor_type.shape.dim
    print([dim.dim_value for dim in dim_info])

# Obtain output info
print("\nModel outputs:")
for o in graph.output:
    print(f"Name: {o.name}, Type: {o.type.tensor_type.elem_type}, Shape: ", end="")
    dim_info = o.type.tensor_type.shape.dim
    print([dim.dim_value for dim in dim_info])

# List all nodes (operators) in the graph
print("\nNodes in the graph:")
for idx, node in enumerate(graph.node):
    print(f"\nNode {idx+1}:")
    print("  OpType: ", node.op_type)
    print("  Inputs: ", node.input)
    print("  Outputs:", node.output)

# iterate over graph.value_info
print("\nValue info:")
for value in graph.value_info:
    print(f"Name: {value.name}")
    typ = value.type.tensor_type
    shape = [dim.dim_value for dim in typ.shape.dim]
    print(f"  Shape: {shape}")
    print(f"  Dtype: {typ.elem_type}")