import onnx
import numpy as np


def clip_fp16():
    new_onnx_model = onnx.load("clip.onnx")
    new_onnx_path2 = "clip_fp16.onnx"
    for node in new_onnx_model.graph.node:
        # if node.name == "/text_model/ConstantOfShape_1":
        if node.op_type == "ConstantOfShape":
            print(node)
            attr = node.attribute[0]
            print(attr)
            if attr.name == "value" and attr.t.data_type == 7:
                np_array = np.frombuffer(attr.t.raw_data, dtype=np.float32).copy()
                print("raw array", np_array)
                np_array[np_array == -np.inf] = -100000  # 将所有负无穷的值改为-100000
                attr.t.raw_data = np_array.tobytes()
                print("new array", np_array)
            print(attr)
    onnx.save_model(
        new_onnx_model,
        new_onnx_path2,
    )


