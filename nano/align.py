"""Word times for the apps (bookmuse docs/listening.md, Nano "Word times"): the vector estimator's text cross-attention in block 1
as an extra graph output. Every latent frame weighs the chunk's phoneme tokens there; the apps follow it with a monotonic path to
each phoneme word's first frame. Block 1 was the closest of the four against Whisper's word times (align_eval.py).

`with_align` adds `align`: block 1's softmax for the first item of the batch (the conditional pass in the apps' batch of 2),
averaged over the heads → (1, T, L). Nothing else in the graph changes, so the audio doesn't either.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SOFTMAX = "/est/blocks.1/ctext/Softmax"


def with_align(m: onnx.ModelProto) -> onnx.ModelProto:
    g = m.graph
    source = next(n for n in g.node if n.name == SOFTMAX).output[0]  # (B, heads, T, L)
    for name, value in [("align_start", 0), ("align_end", 1), ("align_axis", 0)]:
        g.initializer.append(numpy_helper.from_array(np.array([value], np.int64), name))
    g.node.extend([
        helper.make_node("Slice", [source, "align_start", "align_end", "align_axis"], ["align_first"], name="/align/Slice"),
        # opset 17: ReduceMean takes its axes as an attribute.
        helper.make_node("ReduceMean", ["align_first"], ["align"], axes=[1], keepdims=0, name="/align/ReduceMean"),
    ])
    g.output.append(helper.make_tensor_value_info("align", TensorProto.FLOAT, [1, "T", "L"]))
    return m
