"""
This script is used to convert the mediapipe blendshapes model to PyTorch.
Usage:
    python mediapipe_blendshapes_model_to_pytorch.py --tflite_path ./face_blendshapes.tflite --output_path ./face_blendshapes.pth

See README.md for more details.
"""

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tf2onnx
import requests
import zipfile
import argparse
import numpy as np
import onnx
import torch
from onnx.numpy_helper import to_array as _to_array

from mlp_mixer import MediaPipeBlendshapesMLPMixer


def to_array(tensor):
    # Override to_array to suppress PyTorch warning about non-writable tensor
    return np.copy(_to_array(tensor))


def get_model_weight(onnx_model, name):
    for _ in onnx_model.graph.initializer:
        if _.name == name:
            break
    return _


def conv_node_to_w_b(onnx_model, node):
    w_node = node.input[1]
    w = to_array(get_model_weight(onnx_model, w_node))
    assert len(w.shape) == 4, w.shape
    b_node = node.input[2]
    b = to_array(get_model_weight(onnx_model, b_node))
    assert len(b.shape) == 1, b.shape
    w, b = [torch.from_numpy(_).float() for _ in [w, b]]
    return w, b


def get_node_weight_by_output_name(onnx_model, search_str, input_idx):
    fold_op_name = [n for n in onnx_model.graph.node if search_str == n.output[0]]
    fold_op_name = fold_op_name[0].input[input_idx]
    return get_model_weight(onnx_model, fold_op_name)


def get_layernorm_weight(onnx_model, mixer_block_idx, norm_idx):
    search_str = f"model_1/GhumMarkerPoserMlpMixerGeneral/MLPMixer/MixerBlock_{mixer_block_idx}/layer_norm{norm_idx}/batchnorm/mul"
    # if norm_idx == 2:
    #     print(to_array(get_node_weight_by_output_name(onnx_model, search_str, 1)).shape)
    #     import pdb; pdb.set_trace()
    return get_node_weight_by_output_name(onnx_model, search_str, 1)


def get_conv_layer_weight_bias(onnx_model, mixer_block_idx, is_token_mixer, mlp_idx):
    assert mlp_idx in (1, 2)
    assert mixer_block_idx in (0, 1, 2, 3)
    search_str = """
    model_1/GhumMarkerPoserMlpMixerGeneral/MLPMixer/MixerBlock_0/mlp_token_mixing/Mlp_1/Relu;
    model_1/GhumMarkerPoserMlpMixerGeneral/MLPMixer/MixerBlock_0/mlp_token_mixing/Mlp_1/BiasAdd;
    model_1/GhumMarkerPoserMlpMixerGeneral/MLPMixer/MixerBlock_3/mlp_token_mixing/Mlp_1/Conv2D;
    model_1/GhumMarkerPoserMlpMixerGeneral/MLPMixer/MixerBlock_0/mlp_token_mixing/Mlp_1/Conv2D;
    model_1/GhumMarkerPoserMlpMixerGeneral/MLPMixer/MixerBlock_0/mlp_token_mixing/Mlp_1/BiasAdd/ReadVariableOp
    """
    if mixer_block_idx == 3:
        search_str = """
        model_1/GhumMarkerPoserMlpMixerGeneral/MLPMixer/MixerBlock_0/mlp_token_mixing/Mlp_1/Relu;
        model_1/GhumMarkerPoserMlpMixerGeneral/MLPMixer/MixerBlock_0/mlp_token_mixing/Mlp_1/BiasAdd;
        model_1/GhumMarkerPoserMlpMixerGeneral/MLPMixer/MixerBlock_0/mlp_token_mixing/Mlp_1/Conv2D;
        model_1/GhumMarkerPoserMlpMixerGeneral/MLPMixer/MixerBlock_0/mlp_token_mixing/Mlp_1/BiasAdd/ReadVariableOp
        """
    search_str = search_str.replace("\n", "").replace(" ", "").strip()

    search_str = search_str.replace("MixerBlock_0", f"MixerBlock_{mixer_block_idx}")
    if mlp_idx == 2:
        replace_str = "model_1/GhumMarkerPoserMlpMixerGeneral/MLPMixer/MixerBlock_0/mlp_token_mixing/Mlp_1/Relu;"
        replace_str = replace_str.replace(
            "MixerBlock_0", f"MixerBlock_{mixer_block_idx}"
        )
        search_str = search_str.replace(replace_str, "")
    search_str = search_str.replace("Mlp_1", f"Mlp_{mlp_idx}")
    if not is_token_mixer:
        search_str = search_str.replace("mlp_token_mixing", "mlp_channel_mixing")
    ii, node = [
        (i, n) for i, n in enumerate(onnx_model.graph.node) if search_str == n.output[0]
    ][0]
    w, b = conv_node_to_w_b(onnx_model, node)
    mlpname = "mlp_token_mixing" if is_token_mixer else "mlp_channel_mixing"
    idx = 0 if mlp_idx == 1 else 3
    return {
        f"mlpmixer_blocks.{mixer_block_idx}.{mlpname}.{idx}.weight": w,
        f"mlpmixer_blocks.{mixer_block_idx}.{mlpname}.{idx}.bias": b,
    }


def get_state_dict_mlp_mixer_layer(onnx_model, mixer_block_idx):
    state_dict = {}
    norm1_weight = get_layernorm_weight(onnx_model, mixer_block_idx, 1)
    state_dict.update(
        {
            f"mlpmixer_blocks.{mixer_block_idx}.norm1.weight": torch.from_numpy(
                to_array(norm1_weight).reshape(-1)
            ).float()
        }
    )
    state_dict.update(get_conv_layer_weight_bias(onnx_model, mixer_block_idx, True, 1))
    state_dict.update(get_conv_layer_weight_bias(onnx_model, mixer_block_idx, True, 2))
    norm2_weight = get_layernorm_weight(onnx_model, mixer_block_idx, 2)
    state_dict.update(
        {
            f"mlpmixer_blocks.{mixer_block_idx}.norm2.weight": torch.from_numpy(
                to_array(norm2_weight).reshape(-1)
            ).float()
        }
    )
    state_dict.update(get_conv_layer_weight_bias(onnx_model, mixer_block_idx, False, 1))
    state_dict.update(get_conv_layer_weight_bias(onnx_model, mixer_block_idx, False, 2))
    return state_dict


def conv_w_b_from_search_str(onnx_model, search_str):
    _, node = [
        (i, n) for i, n in enumerate(onnx_model.graph.node) if search_str == n.output[0]
    ][0]
    return conv_node_to_w_b(onnx_model, node)


def get_state_dict(onnx_model):
    state_dict = {}
    search_str = "model_1/GhumMarkerPoserMlpMixerGeneral/conv2d/BiasAdd;model_1/GhumMarkerPoserMlpMixerGeneral/conv2d/Conv2D;model_1/GhumMarkerPoserMlpMixerGeneral/conv2d/BiasAdd/ReadVariableOp"
    w, b = conv_w_b_from_search_str(onnx_model, search_str)
    state_dict.update({"conv1.weight": w, "conv1.bias": b})
    search_str = "model_1/GhumMarkerPoserMlpMixerGeneral/MLPMixer/input_tokens_embedding/BiasAdd;model_1/GhumMarkerPoserMlpMixerGeneral/MLPMixer/MixerBlock_3/mlp_channel_mixing/Mlp_2/Conv2D;model_1/GhumMarkerPoserMlpMixerGeneral/MLPMixer/input_tokens_embedding/Conv2D;model_1/GhumMarkerPoserMlpMixerGeneral/MLPMixer/input_tokens_embedding/BiasAdd/ReadVariableOp"
    w, b = conv_w_b_from_search_str(onnx_model, search_str)
    state_dict.update({"conv2.weight": w, "conv2.bias": b})
    search_str = "model_1/GhumMarkerPoserMlpMixerGeneral/MLPMixer/AddExtraTokens/concat"
    extra_token = get_node_weight_by_output_name(onnx_model, search_str, 0)
    state_dict.update({"extra_token": torch.from_numpy(to_array(extra_token)).float()})
    # MLP Mixer layers
    state_dict.update(get_state_dict_mlp_mixer_layer(onnx_model, 0))
    state_dict.update(get_state_dict_mlp_mixer_layer(onnx_model, 1))
    state_dict.update(get_state_dict_mlp_mixer_layer(onnx_model, 2))
    state_dict.update(get_state_dict_mlp_mixer_layer(onnx_model, 3))
    search_str = "model_1/GhumMarkerPoserMlpMixerGeneral/output_blendweights/BiasAdd;model_1/GhumMarkerPoserMlpMixerGeneral/output_blendweights/Conv2D;model_1/GhumMarkerPoserMlpMixerGeneral/output_blendweights/BiasAdd/ReadVariableOp"
    w, b = conv_w_b_from_search_str(onnx_model, search_str)
    state_dict.update({"output_mlp.weight": w, "output_mlp.bias": b})
    return state_dict


def download_and_unzip_blendshapes_model():
    # Equivalent to:
    # wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
    # unzip face_landmarker_v2_with_blendshapes.task
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    if not os.path.exists("face_landmarker_v2_with_blendshapes.task"):
        print("Downloading face_landmarker_v2_with_blendshapes.task")
        with open("face_landmarker_v2_with_blendshapes.task", "wb") as f:
            f.write(requests.get(url, allow_redirects=True).content)
    print("Unzipping face_landmarker_v2_with_blendshapes.task")
    with zipfile.ZipFile("face_landmarker_v2_with_blendshapes.task", "r") as zip_ref:
        zip_ref.extractall("./")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite_path", type=str, default="./face_blendshapes.tflite")
    parser.add_argument("--output_path", type=str, default="./face_blendshapes.pth")
    args = parser.parse_args()
    if not os.path.exists(args.tflite_path):
        download_and_unzip_blendshapes_model()
    model_proto, external_tensor_storage = tf2onnx.convert.from_tflite(args.tflite_path)
    checker = onnx.checker.check_model(model_proto)
    state_dict = get_state_dict(model_proto)
    MediaPipeBlendshapesMLPMixer().load_state_dict(state_dict)
    torch.save(state_dict, args.output_path)
    print(f"Saved model to {args.output_path}")
