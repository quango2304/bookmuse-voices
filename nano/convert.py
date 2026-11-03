#!/usr/bin/env python3
"""VieNeu-TTS v3 Nano → three Core ML models (fp16, iOS 18), written to work/nano/coreml:

- NanoText: ids (1, 256) int32 padded with 0, their mask, the voice (style 50 × 256, x-vector 192) → ctx (1, 256, 256)
  and the chunk's log seconds (the text encoder and the duration predictor in one call).
- NanoStep: one Euler step with classifier-free guidance, the conditional and the null pass as a batch of 2 →
  x + dt · (v_null + cfg · (v − v_null)), padded frames zeroed. One function per frame bucket (`f<T>`).
- NanoCodec: latents (1, 144, T) → 24 kHz audio (1, 1, 1536 T). Causal, so the padding is cut off. One function per bucket.

    uv run python convert.py
"""
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

from graphs import torch_graph

OUT = Path(__file__).resolve().parent.parent / "work/nano/coreml"
OUT.mkdir(parents=True, exist_ok=True)
TOKENS = 320
BUCKETS = [32, 64, 96, 128, 160, 192, 240]
F16 = np.float16


class Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.te, self.dp = torch_graph("text_encoder"), torch_graph("duration_predictor")

    def forward(self, ids, mask, style, spk):
        ctx = self.te(ids.long(), style)
        return ctx, self.dp(ctx, mask > 0.5, spk)


class Step(nn.Module):
    def __init__(self):
        super().__init__()
        self.ve = torch_graph("vector_estimator")

    def forward(self, x, x_mask, t, dt, cfg, ctx, ctx_mask, spk, style):
        v = self.ve(x.expand(2, -1, -1), t.expand(2), ctx, ctx_mask > 0.5, spk, style, x_mask.expand(2, -1, -1))
        v = v[1:2] + cfg * (v[0:1] - v[1:2])
        return (x + dt * v) * x_mask


def save(module, example, inputs, outputs, path):
    with torch.no_grad():
        traced = torch.jit.trace(module.eval(), example)
    ml = ct.convert(traced, inputs=inputs, outputs=[ct.TensorType(name=n, dtype=F16) for n in outputs],
                    minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16)
    ml.save(str(path))
    return path


def multifunction(paths, name):
    d = ct.utils.MultiFunctionDescriptor()
    for T, p in paths:
        d.add_function(str(p), src_function_name="main", target_function_name=f"f{T}")
    d.default_function_name = f"f{paths[-1][0]}"
    ct.utils.save_multifunction(d, str(OUT / f"{name}.mlpackage"))
    print("wrote", name, [T for T, _ in paths], flush=True)


if __name__ == "__main__":
    r = torch.randn
    save(Text(), (torch.ones(1, TOKENS, dtype=torch.int32), torch.ones(1, TOKENS), r(1, 50, 256), r(1, 192)),
         [ct.TensorType(name="ids", shape=(1, TOKENS), dtype=np.int32), ct.TensorType(name="mask", shape=(1, TOKENS), dtype=F16),
          ct.TensorType(name="style", shape=(1, 50, 256), dtype=F16), ct.TensorType(name="spk", shape=(1, 192), dtype=F16)],
         ["ctx", "log_seconds"], OUT / "NanoText.mlpackage")
    print("wrote NanoText", flush=True)

    step, codec = Step(), torch_graph("codec_decoder")
    steps, codecs = [], []
    for T in BUCKETS:
        ex = (r(1, 144, T), torch.ones(1, 1, T), torch.tensor([0.5]), torch.tensor([0.1]), torch.tensor([3.0]),
              r(2, TOKENS, 256), torch.ones(2, TOKENS), r(2, 192), r(2, 50, 256))
        names = [("x", (1, 144, T)), ("x_mask", (1, 1, T)), ("t", (1,)), ("dt", (1,)), ("cfg", (1,)), ("ctx", (2, TOKENS, 256)),
                 ("ctx_mask", (2, TOKENS)), ("spk", (2, 192)), ("style", (2, 50, 256))]
        steps.append((T, save(step, ex, [ct.TensorType(name=n, shape=s, dtype=F16) for n, s in names], ["x_next"], OUT / f"_step{T}.mlpackage")))
        codecs.append((T, save(codec, (r(1, 144, T),), [ct.TensorType(name="x", shape=(1, 144, T), dtype=F16)], ["wav"], OUT / f"_codec{T}.mlpackage")))
        print("bucket", T, flush=True)
    multifunction(steps, "NanoStep")
    multifunction(codecs, "NanoCodec")
