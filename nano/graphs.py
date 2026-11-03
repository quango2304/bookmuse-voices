"""Nano's ONNX graphs as PyTorch modules (onnx2torch), with the vector estimator given a frame mask so it can run on a
padded, fixed number of frames: the padded frames are zeroed before every depthwise conv (its only mixing across
frames), and the frame count it normalises positions by is the mask's sum. Shared by convert.py and the checks."""
import glob

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
from onnx2torch import convert

SNAPSHOT = glob.glob('/Users/leongo/.cache/huggingface/hub/models--pnnbao-ump--VieNeu-TTS-v3-Nano/snapshots/*/')[0]


def masked_estimator() -> onnx.ModelProto:
    m = onnx.load(SNAPSHOT + 'vector_estimator.onnx')
    m.graph.input.append(helper.make_tensor_value_info('x_mask', TensorProto.FLOAT, ['B', 1, 'T']))
    m.graph.initializer.append(onnx.numpy_helper.from_array(np.array([1], np.int64), 'mask_axes'))
    ones = next(n for n in m.graph.node if n.name == '/ConstantOfShape')
    dtype = ones.attribute[0].t.data_type if ones.attribute else TensorProto.FLOAT
    nodes = []
    for n in m.graph.node:
        if n is ones:
            nodes.append(helper.make_node('Squeeze', ['x_mask', 'mask_axes'], ['x_mask_2d']))
            nodes.append(helper.make_node('Cast', ['x_mask_2d'], [ones.output[0]], to=dtype))
            continue
        if n.op_type == 'Conv' and '/dw/' in n.name:
            masked = n.input[0] + '_masked'
            nodes.append(helper.make_node('Mul', [n.input[0], 'x_mask'], [masked]))
            n.input[0] = masked
        nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    return m


def tidy(m: onnx.ModelProto) -> onnx.ModelProto:
    """onnx2torch reads a Clip's min and max as constants: an omitted max ("") becomes +inf, a bound made by a Constant
    node becomes an initializer."""
    consts = {n.output[0]: n for n in m.graph.node if n.op_type == 'Constant'}
    for n in m.graph.node:
        if n.op_type != 'Clip':
            continue
        for i, name in enumerate(list(n.input)[1:], 1):
            if name == '':
                value = np.array(np.inf if i == 2 else -np.inf, np.float32)
                n.input[i] = f'{n.name}_bound{i}'
                m.graph.initializer.append(onnx.numpy_helper.from_array(value, n.input[i]))
            elif name in consts:
                m.graph.initializer.append(onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(consts[name].attribute[0].t), name))
                m.graph.node.remove(consts[name])
                del consts[name]
    return m


def torch_graph(name: str) -> torch.nn.Module:
    model = masked_estimator() if name == 'vector_estimator' else onnx.load(SNAPSHOT + name + '.onnx')
    return convert(tidy(model)).eval()
