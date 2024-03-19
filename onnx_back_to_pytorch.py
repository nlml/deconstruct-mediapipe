import torch
from onnx.numpy_helper import to_array


def get_model_weight(onnx_model, name):
    for _ in onnx_model.graph.initializer:
        if _.name == name:
            break
    return _


def conv_node_to_w_b(onnx_model, node):
    w_node = node.input[1]
    w = to_array(get_model_weight(onnx_model, w_node))
    assert len(w.shape) == 4
    b_node = node.input[2]
    b = to_array(get_model_weight(onnx_model, b_node))
    assert len(b.shape) == 1
    w, b = [torch.from_numpy(_).float() for _ in [w, b]]
    return w, b


def get_state_dict(onnx_model):
    w, b = conv_node_to_w_b(onnx_model, onnx_model.graph.node[11])
    state_dict = {"conv1.weight": w, "conv1.bias": b}
    return state_dict
