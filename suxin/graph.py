"""Suxin-Light's ONNX graph made deterministic for conversion: both random draws (the duration predictor's noise and the
prior's sample) become zeros, which is what noise 0 / 0 gives anyway (the card's settings); `scales` becomes a constant
(noise 0, length 1, noise_w 0: the pace is applied on the audio, as for Phonon); and each phoneme's frame count (the
ceil of its predicted duration) is an extra output, so the app knows when every character starts."""
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

WORK = Path(__file__).resolve().parent.parent / "work/suxin"
FP32 = WORK / "zh_en-suxin-light.onnx"
FLOW_CHANNELS = 192  # the prior z the flow reverses


def deterministic(path: Path = FP32) -> onnx.ModelProto:
    m = onnx.load(str(path))
    g = m.graph
    nodes = []
    for n in g.node:
        if n.op_type == "RandomNormalLike":
            # zeros shaped like the input: x - x
            nodes.append(helper.make_node("Sub", [n.input[0], n.input[0]], [n.output[0]], name=n.name + "_zeros"))
            continue
        nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)
    keep = [i for i in g.input if i.name != "scales"]
    del g.input[:]
    g.input.extend(keep)
    g.initializer.append(numpy_helper.from_array(np.array([0.0, 1.0, 0.0], np.float32), "scales"))
    g.output.append(helper.make_tensor_value_info("durations", onnx.TensorProto.FLOAT, None))
    g.node.append(helper.make_node("Identity", ["/Ceil_output_0"], ["durations"]))
    return m


# The same fix as nano/graphs.py.
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
    # The flow's channel flip is a Slice from -1 to -2^63 with step -1; Core ML squeezes -2^63 into 32 bits and keeps
    # 190 of the 192 channels. A Gather of the channels in reverse does the same and converts cleanly.
    values = lambda name: (onnx.numpy_helper.to_array(consts[name].attribute[0].t) if name in consts
                           else next((onnx.numpy_helper.to_array(t) for t in m.graph.initializer if t.name == name), None))
    for n in list(m.graph.node):
        if n.op_type != 'Slice' or len(n.input) < 5 or not n.name.startswith('/flow/flows.'):
            continue
        steps = values(n.input[4])
        if steps is None or steps.tolist() != [-1]:
            continue
        assert values(n.input[3]).tolist() == [1], n.name
        rev = f'{n.name}_reversed'
        m.graph.initializer.append(onnx.numpy_helper.from_array(np.arange(FLOW_CHANNELS - 1, -1, -1, dtype=np.int64), rev))
        i = list(m.graph.node).index(n)
        m.graph.node.remove(n)
        m.graph.node.insert(i, helper.make_node('Gather', [n.input[0], rev], list(n.output), name=n.name + '_gather', axis=1))
    # The decoder's attention works in 128-frame blocks and masks padded keys with -inf; a block that is all padding (the
    # app decodes in fixed windows, so the last window can run past the end) would give NaN. -1e4 gives the same weights
    # wherever a block has a real frame, and no NaN where it has none.
    for n in m.graph.node:
        if n.op_type == 'Where' and n.name.startswith('/dec/') and n.input[1] in consts:
            v = onnx.numpy_helper.to_array(consts[n.input[1]].attribute[0].t)
            if np.isneginf(v).all():
                consts[n.input[1]].attribute[0].t.CopyFrom(onnx.numpy_helper.from_array(np.full_like(v, -1e4), n.input[1]))
    # onnx2torch wants a conv's weight and bias as initializers; the iSTFT's are Constant nodes.
    for n in m.graph.node:
        if n.op_type not in ('Conv', 'ConvTranspose'):
            continue
        for name in list(n.input)[1:]:
            if name in consts:
                m.graph.initializer.append(onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(consts[name].attribute[0].t), name))
                m.graph.node.remove(consts[name])
                del consts[name]
    return m
