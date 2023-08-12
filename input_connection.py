import onnx
from onnx import checker


def remove(onnx_path):
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph

    for i, input in enumerate(onnx_model.graph.input):
        if 'timesteps' == input.name:
            print("remove input timesteps")
            onnx_model.graph.input.remove(graph.input[i])
    for i, input in enumerate(onnx_model.graph.input):
        if 'x' == input.name:
            print("remove input x")
            onnx_model.graph.input.remove(graph.input[i])
    for i, input in enumerate(onnx_model.graph.input):
        if 'context' == input.name:
            print("remove input context")
            onnx_model.graph.input.remove(graph.input[i])

    for node in onnx_model.graph.node:
        for i, name in enumerate(node.input):
            if name == "x":
                print("change input x name")
                node.input[i] = 'm1_x_in'
            if name == "timesteps":
                print("change input timesteps name")
                node.input[i] = 'm1_timesteps_in'
            if name == "context":
                print("change input context name")
                node.input[i] = 'm1_context_in'

    onnx.save(onnx_model, "./connection.onnx", save_as_external_data=True)
    checker.check_model("./connection.onnx")
    onnx.shape_inference.infer_shapes_path("./connection.onnx", "./connection_infer.onnx")