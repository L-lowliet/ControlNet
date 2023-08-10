import onnx
import onnx_graphsurgeon as gs
import os
from onnx import checker
from typing import List, MutableMapping, Optional, Set, Tuple

from onnx import ModelProto
from onnx import checker, helper
from onnx.compose import add_prefix, merge_graphs


def merge_models(
    m1: ModelProto,
    m2: ModelProto,
    io_map: List[Tuple[str, str]],
    inputs: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None,
    prefix1: Optional[str] = None,
    prefix2: Optional[str] = None,
    name: Optional[str] = None,
    doc_string: Optional[str] = None,
    producer_name: Optional[str] = "onnx.compose.merge_models",
    producer_version: Optional[str] = "1.0",
    domain: Optional[str] = "",
    model_version: Optional[int] = 1,
) -> ModelProto:
    if type(m1) is not ModelProto:
        raise ValueError("m1 argument is not an ONNX model")
    if type(m2) is not ModelProto:
        raise ValueError("m2 argument is not an ONNX model")

    if m1.ir_version != m2.ir_version:
        raise ValueError(
            f"IR version mismatch {m1.ir_version} != {m2.ir_version}."
            " Both models should have the same IR version"
        )
    ir_version = m1.ir_version

    opset_import_map: MutableMapping[str, int] = {}
    opset_imports = [entry for entry in m1.opset_import] + [
        entry for entry in m2.opset_import
    ]

    for entry in opset_imports:
        if entry.domain in opset_import_map:
            found_version = opset_import_map[entry.domain]
            if entry.version != found_version:
                raise ValueError(
                    "Can't merge two models with different operator set ids for a given domain. "
                    f"Got: {m1.opset_import} and {m2.opset_import}"
                )
        else:
            opset_import_map[entry.domain] = entry.version

    # Prefixing names in the graph if requested, adjusting io_map accordingly
    if prefix1 or prefix2:
        if prefix1:
            m1_copy = ModelProto()
            m1_copy.CopyFrom(m1)
            m1 = m1_copy
            m1 = add_prefix(m1, prefix=prefix1)
        if prefix2:
            m2_copy = ModelProto()
            m2_copy.CopyFrom(m2)
            m2 = m2_copy
            m2 = add_prefix(m2, prefix=prefix2)
        io_map = [
            (
                prefix1 + io[0] if prefix1 else io[0],
                prefix2 + io[1] if prefix2 else io[1],
            )
            for io in io_map
        ]

    graph = merge_graphs(
        m1.graph,
        m2.graph,
        io_map,
        inputs=inputs,
        outputs=outputs,
        name=name,
        doc_string=doc_string,
    )
    model = helper.make_model(
        graph,
        producer_name=producer_name,
        producer_version=producer_version,
        domain=domain,
        model_version=model_version,
        opset_imports=opset_imports,
        ir_version=ir_version,
    )

    # Merging model metadata props
    model_props = {}
    for meta_entry in m1.metadata_props:
        model_props[meta_entry.key] = meta_entry.value
    for meta_entry in m2.metadata_props:
        if meta_entry.key in model_props:
            value = model_props[meta_entry.key]
            if value != meta_entry.value:
                raise ValueError(
                    "Can't merge models with different values for the same model metadata property."
                    f" Found: property = {meta_entry.key}, with values {value} and {meta_entry.value}."
                )
        else:
            model_props[meta_entry.key] = meta_entry.value
    helper.set_model_props(model, model_props)

    # Merging functions
    function_overlap = list(
        {f.name for f in m1.functions} & {f.name for f in m2.functions}
    )
    if function_overlap:
        raise ValueError(
            "Can't merge models with overlapping local function names."
            " Found in both graphs: " + ", ".join(function_overlap)
        )
    model.functions.MergeFrom(m1.functions)
    model.functions.MergeFrom(m2.functions)

    # checker.check_model(model)
    return model


def merge(controlnet_path, unet_path):
    controlnet = "controlnetsim.onnx"
    unet = "./unet/unet.onnx"

    controlnet_onnx = onnx.load(controlnet_path)
    unet_onnx = onnx.load(unet_path)

    # 避免两模型节点名字重复
    new_model = onnx.compose.add_prefix(controlnet_onnx, prefix="m1_")

    controlnet_gs = gs.import_onnx(new_model)
    unet_gs = gs.import_onnx(unet_onnx)

    graph1_outputs = [o.name for o in controlnet_gs.outputs]
    graph1_inputs = [i.name for i in controlnet_gs.inputs]
    graph1_inputs.extend(graph1_outputs)
    print(graph1_inputs)

    graph2_inputs = [i.name for i in unet_gs.inputs]
    print(graph2_inputs)

    io_map = [("m1_out_0", "control1"),
              ("m1_out_1", "control2"), ("m1_out_2", "control3"), ("m1_out_3", "control4"), ("m1_out_4", "control5"),
              ("m1_out_5", "control6"), ("m1_out_6", "control7"), ("m1_out_7", "control8"), ("m1_out_8", "control9"),
              ("m1_out_9", "control10"), ("m1_out_10", "control11"), ("m1_out_11", "control12"), ("m1_out_12", "control13")]

    combined_model = merge_models(
        new_model, unet_onnx,
        io_map=io_map)

    onnx.save(combined_model, "./combine/combine.onnx", save_as_external_data=True)
    checker.check_model("./combine/combine.onnx")
    onnx.shape_inference.infer_shapes_path("./combine/combine.onnx", "./combine/combine_infer.onnx")
    os.system('onnxsim ./combine/combine.onnx ./combine/combinesim.onnx')