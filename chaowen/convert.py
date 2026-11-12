#!/usr/bin/env python3
"""Chao Wen → three Core ML models (iOS 18, fixed sizes, CPU), written to work/chaowen/coreml:

- ChaowenEncoder: ids (1, 384) int32, lengths (1,) int32 → x, m (1, 192, 384)
- ChaowenDurations: x (1, 192, 384), mask (1, 1, 384), scale (1,) → frames (1, 1, 384)
- ChaowenDecoder: z (1, 192, 512), mask (1, 1, 512) → audio (1, 1, 256 × 512) at 22.05 kHz

    uv run python convert.py [--fp32]
"""
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from graph import WORK
from model import Decoder, Durations, Encoder

OUT = WORK / "coreml"
OUT.mkdir(parents=True, exist_ok=True)
PRECISION = ct.precision.FLOAT32 if "--fp32" in sys.argv else ct.precision.FLOAT16
SUFFIX = "-fp32" if "--fp32" in sys.argv else ""
TOKENS, WINDOW = 384, 512  # phonemes per call (padded); frames per decoder window

# onnx2torch writes some comparisons with their long names, which coremltools only knows by the short ones.
from coremltools.converters.mil.frontend.torch import ops as _ops  # noqa: E402,F401
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY as _R  # noqa: E402

for _alias, _op in {"less": "lt", "greater": "gt", "less_equal": "le", "greater_equal": "ge", "not_equal": "ne"}.items():
    if _alias not in _R.name_to_func_mapping:
        _R.name_to_func_mapping[_alias] = _R.name_to_func_mapping[_op]


def _identity(context, node):
    # resolve_conj / resolve_neg do nothing to a real tensor
    context.add(context[node.inputs[0]], torch_name=node.name)


for _name in ("resolve_conj", "resolve_neg"):
    _R.name_to_func_mapping.setdefault(_name, _identity)


def save(traced, inputs, outputs, name):
    ml = ct.convert(traced, inputs=inputs, outputs=outputs, minimum_deployment_target=ct.target.iOS18,
                    compute_precision=PRECISION, convert_to="mlprogram", compute_units=ct.ComputeUnit.CPU_ONLY)
    path = OUT / f"{name}{SUFFIX}.mlpackage"
    ml.save(str(path))
    print("saved", path)


class EncoderI32(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.e = Encoder()

    def forward(self, ids, lengths):
        return self.e(ids.long(), lengths.long())


if __name__ == "__main__":
    # Fixed sizes: flexible lengths don't load (the decoder's plan fails with -7, the encoder can't resize).
    T, F_ = TOKENS, WINDOW
    with torch.no_grad():
        ids = torch.randint(1, 100, (1, T), dtype=torch.int32)
        enc = torch.jit.trace(EncoderI32().eval(), (ids, torch.tensor([T // 2], dtype=torch.int32)))
        save(enc, [ct.TensorType("ids", (1, T), dtype=np.int32), ct.TensorType("lengths", (1,), dtype=np.int32)],
             [ct.TensorType("x"), ct.TensorType("m")], "ChaowenEncoder")

        dur = torch.jit.trace(Durations().eval(), (torch.randn(1, 192, T), torch.ones(1, 1, T), torch.ones(1)))
        save(dur, [ct.TensorType("x", (1, 192, T)), ct.TensorType("mask", (1, 1, T)), ct.TensorType("scale", (1,))],
             [ct.TensorType("frames")], "ChaowenDurations")

        dec = torch.jit.trace(Decoder().eval(), (torch.randn(1, 192, F_), torch.ones(1, 1, F_)))
        save(dec, [ct.TensorType("z", (1, 192, F_)), ct.TensorType("mask", (1, 1, F_))], [ct.TensorType("audio")],
             "ChaowenDecoder")
